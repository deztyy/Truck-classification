import atexit
import threading
import mlflow
import onnxruntime as ort
import cv2
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import pytz

from psycopg2 import pool
import numpy as np
import psycopg2
import psycopg2.extras
import redis
from minio import Minio
from minio.error import S3Error
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import supervision as sv
from collections import defaultdict
import csv
import atexit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

QUEUE_NAME = "frame_batches"
PROCESSED_BUCKET = "process-frames"
JPG_QUALITY = 95
THAI_TIMEZONE = pytz.timezone("Asia/Bangkok")
_worker_service = None

# === MinIO Retry Configuration ===
RETRY_DELAY = 0.01  # Seconds to wait between retries (10ms)
MAX_WAIT_TIME = 30  # Maximum seconds to wait before logging warning (not stopping)
MAX_RETRIES = 3


def worker_service(config: Dict[str, Any]):
    global _worker_service
    redis_conf = config.get('redis', {'host': 'redis', 'port': 6379})
    _worker_service = ProcessingService(
        db_config=config['postgres'], 
        minio_config=config['minio'],
        redis_config=redis_conf
    )
    atexit.register(lambda: _worker_service.db.close())

def task_handler(task_json: str):
    global _worker_service
    try:
        task = ProcessingTask.from_json(task_json)
        return _worker_service.process_task(task)
    except Exception as e:
        logging.error(f"Task processing error: {e}")
        logging.error(f"Worker PID {os.getpid()} failed task: {e}")
        return {"status": "error", "error": str(e)}

class VehicleClass(Enum):
    CAR = 0
    OTHER = 1
    OTHER_TRUCK = 2
    PICKUP_TRUCK = 3
    TRUCK_20_BACK = 4
    TRUCK_20_FRONT = 5
    TRUCK_20X2 = 6
    TRUCK_40 = 7
    TRUCK_RORO = 8
    TRUCK_TAIL = 9
    MOTORCYCLE = 10
    TRUCK_HEAD = 11


@dataclass
class VehicleTransaction:
    camera_id: str
    track_id: str
    class_id: int
    total_fee: float = 0.00
    time_stamp: Optional[datetime.datetime] = None
    img_path: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "total_fee": round(self.total_fee, 2),
            "time_stamp": self.time_stamp or datetime.datetime.now(datetime.timezone.utc),
            "img_path": self.img_path,
            "confidence": round(self.confidence, 4) if self.confidence else None,
        }

@dataclass
class ProcessingTask:
    task_id: str
    camera_id: str
    video_file: str
    minio_bucket: str
    object_key_or_prefix: str
    timestamp: Optional[datetime.datetime] = None
    # NEW: Optional detection coordinates for tracking
    track_id: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "camera_id": self.camera_id,
            "video_file": self.video_file,
            "minio_bucket": self.minio_bucket,
            "object_key_or_prefix": self.object_key_or_prefix,
            "timestamp": (self.timestamp or datetime.datetime.now(datetime.timezone.utc)).isoformat(),
            "track_id": self.track_id,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "ProcessingTask":
        try:
            data = json.loads(json_str)
            return cls(
                task_id=data.get("task_id") or data.get("batch_id") or "unknown",
                camera_id=data.get("camera_id", "unknown"),
                video_file=data.get("video_file") or data.get("video_path", ""),
                minio_bucket=data.get("bucket_name") or data.get("minio_bucket") or data.get("bucket", "video-frames"),
                object_key_or_prefix=data.get("object_name") or data.get("object_key_or_prefix") or data.get("key", ""),
                timestamp=datetime.datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
                track_id=data.get("track_id"),
                retry_count=data.get("retry_count", 0),
            )
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON: {json_str}")
            raise

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class RedisQueueManager:
    """Manages Redis queue for processing tasks"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        queue_name: str = QUEUE_NAME,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.queue_name = queue_name
        self.notification_channel = f"{queue_name}:notifications"

        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            # logging.info(f"✓ Redis connected: {host}:{port}/{db}")
        except Exception as e:
            # logging.error(f"✗ Redis connection failed: {e}")
            raise

    def push_task(self, task: ProcessingTask) -> bool:
        try:
            self.client.rpush(self.queue_name, task.to_json())
            # logging.info(f"✓ Task pushed: {task.task_id}")
            return True
        except Exception as e:
            # logging.error(f"✗ Push failed: {e}")
            return False

class MinIOManager:
    """Manages MinIO operations"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.endpoint = endpoint
        self.client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        # logging.info(f"✓ MinIO connected: {endpoint}")

    def create_bucket(self, bucket_name: str) -> bool:
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            return True
        except S3Error as e:
            # logging.error(f"✗ Create bucket failed: {e}")
            return False

    def upload_from_bytes(self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        try:
            self.client.put_object(bucket, object_name, BytesIO(data), length=len(data), content_type=content_type)
            # logging.info(f"✓ Uploaded: {bucket}/{object_name}")
            return True
        except S3Error as e:
            # logging.error(f"✗ Upload failed: {e}")
            return False

    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            return [{"name": obj.object_name, "size": obj.size, "last_modified": obj.last_modified} for obj in objects]
        except S3Error as e:
            # logging.error(f"✗ List objects failed: {e}")
            return []

    def delete_object(self, bucket: str, object_name: str) -> bool:
        try:
            self.client.remove_object(bucket, object_name)
            # logging.info(f"✓ Deleted: {bucket}/{object_name}")
            return True
        except S3Error as e:
            # logging.error(f"✗ Delete failed: {e}")
            return False

    def get_object_data(self, bucket: str, object_name: str) -> bytes:
        response = None
        try:
            response = self.client.get_object(bucket, object_name)
            return response.read()
        except S3Error as e:
            # logging.error(f"✗ Direct download failed: {e}")
            raise
        finally:
            if response:
                response.close()
                response.release_conn()
    
    def wait_for_npy_file(self, bucket: str, prefix: str, retry_delay: float = RETRY_DELAY, timeout_seconds: int = 10) -> Optional[str]:
        logging.info(f"   - prefix: '{prefix}'")
        try:
            if not self.client.bucket_exists(bucket):
                logging.error(f"✗ Bucket '{bucket}' does not exist")
                return None
        except Exception as e:
            logging.error(f"✗ Cannot access bucket: {e}")
            return None

        if not prefix or len(prefix) < 5:
            logging.error(f"✗ Invalid prefix: '{prefix}'")
            return None
        
        attempt = 0
        start_time = time.time()
        last_warning_time = start_time
        
        while True:
            attempt += 1
            elapsed = time.time() - start_time
            
            # ✅ ADD THIS: Check timeout FIRST
            if elapsed > timeout_seconds:
                logging.error(
                    f"✗ TIMEOUT: .npy file not found after {timeout_seconds}s "
                    f"({attempt} attempts) in {bucket}/{prefix}"
                )
                return None  # ← CRITICAL: Return None instead of blocking forever
            
            try:
                # If prefix is already a direct .npy file path, check if it exists
                if prefix.endswith(".npy"):
                    try:
                        self.client.stat_object(bucket, prefix)
                        if attempt > 1:
                            logging.info(f"✓ Found .npy file after {attempt} attempts ({elapsed:.1f}s): {prefix}")
                        else:
                            logging.info(f"✓ Found .npy file: {prefix}")
                        return prefix
                    except S3Error:
                        pass  # File doesn't exist yet, continue waiting
                
                # Otherwise, list objects under prefix
                listed_objects = self.list_objects(bucket=bucket, prefix=prefix)
                npy_files = [obj for obj in listed_objects 
                        if obj["name"].endswith(".npy") and not obj["name"].endswith("/")]
                
                if npy_files:
                    # Sort by last_modified to get the newest file
                    npy_files.sort(key=lambda x: x["last_modified"], reverse=True)
                    found_file = npy_files[0]["name"]
                    if attempt > 1:
                        logging.info(f"✓ Found .npy file after {attempt} attempts ({elapsed:.1f}s): {found_file}")
                    else:
                        logging.info(f"✓ Found .npy file: {found_file}")
                    return found_file
                
                # Periodic logging to show we're still waiting
                if elapsed - (last_warning_time - start_time) >= MAX_WAIT_TIME:
                    remaining = timeout_seconds - elapsed
                    logging.warning(
                        f"⏳ Still waiting for .npy file... "
                        f"({attempt} attempts, {elapsed:.1f}s elapsed, {remaining:.1f}s remaining)"
                    )
                    last_warning_time = time.time()
                
                # Short debug log for troubleshooting (every 50 attempts)
                if attempt % 50 == 0:
                    logging.debug(f"Waiting for .npy file in {bucket}/{prefix} (attempt {attempt}, {elapsed:.1f}s)")
                
                # Wait before next retry
                time.sleep(retry_delay)
                    
            except Exception as e:
                logging.error(f"Error checking for .npy file (attempt {attempt}): {e}")
                time.sleep(retry_delay)

class PostgreSQLDatabase:
    """Manages PostgreSQL database operations"""
    _pool = None
    _lock = threading.Lock()
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.connection_params = {"host": host, "port": port, "database": database, "user": user, "password": password}
        with PostgreSQLDatabase._lock:
            if PostgreSQLDatabase._pool is None:
                PostgreSQLDatabase._pool = pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=20,
                    **self.connection_params
                )
        
        logging.info(f"✓ PostgreSQL pool initialized: {host}:{port}/{database}")
    
    def get_vehicle_class(self, class_id: int) -> Optional[Dict]:
        conn = None
        try:
            conn = self._pool.getconn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM vehicle_classes WHERE class_id = %s", (class_id,))
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logging.error(f"✗ Get vehicle class failed: {e}")
            return None
        finally:
            if conn:
                self._pool.putconn(conn)

    def insert_transaction(self, transaction: VehicleTransaction) -> bool:
        """Insert a vehicle transaction record and log to CSV"""
        
        # === [ส่วนที่ 1] เขียนลงไฟล์ CSV (เพิ่มใหม่) ===
        try:
            # หา Path ของโฟลเดอร์ที่ไฟล์ Python นี้อยู่
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_file = os.path.join(current_dir, "debug_transactions.csv")
            # เช็คว่ามีไฟล์อยู่แล้วไหม (เพื่อเขียน Header แค่ครั้งเดียว)
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # ถ้าเพิ่งสร้างไฟล์ใหม่ ให้เขียนหัวตารางก่อน
                if not file_exists:
                    writer.writerow(['Timestamp', 'Camera ID', 'Track ID', 'Class ID', 'Fee', 'Confidence', 'Image Path'])
                
                # เขียนข้อมูล Transaction ลงไป
                writer.writerow([
                    transaction.time_stamp,
                    transaction.camera_id,
                    transaction.track_id,
                    transaction.class_id,
                    transaction.total_fee,
                    transaction.confidence,
                    transaction.img_path
                ])
        except Exception as e:
            logging.error(f"⚠️ Failed to write CSV log: {e}")
        # =================================================
        
        conn = self._pool.getconn()
        try:
            # ... (โค้ด Insert ลง DB เดิมของคุณอยู่ตรงนี้) ...
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vehicle_transactions 
                    (camera_id, track_id, class_id, total_fee, time_stamp, img_path, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    transaction.camera_id, transaction.track_id, transaction.class_id,
                    transaction.total_fee, transaction.time_stamp, transaction.img_path,
                    transaction.confidence,
                ))
            conn.commit()
            logging.info(f"✅ Transaction saved: {transaction.track_id}")
            return True
        except Exception as e:
            conn.rollback()
            logging.error(f"✗ Insert failed: {e}")
            raise
        finally:
            self._pool.putconn(conn)

    def close(self):
        """Close all connections in the pool"""
        with PostgreSQLDatabase._lock:
            if PostgreSQLDatabase._pool:
                try:
                    PostgreSQLDatabase._pool.closeall()
                    PostgreSQLDatabase._pool = None
                    logging.info("✓ PostgreSQL pool closed")
                except Exception as e:
                    logging.error(f"✗ Failed to close PostgreSQL pool: {e}")


class ProcessingService:
    """Service that processes tasks with tracking to prevent double-counting"""

    def __init__(self, db_config: Dict[str, Any], minio_config: Dict[str, Any], redis_config: Dict[str, Any]):
        # 1. เชื่อมต่อ Database & MinIO
        self.db = PostgreSQLDatabase(**db_config) # ใช้ **db_config เพื่อแตก dict เข้า params
        self.minio_client = Minio(**minio_config)
        self.bucket_name = PROCESSED_BUCKET
        
        # 2. เชื่อมต่อ Redis (สำหรับกันซ้ำ)
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'redis'),
            port=redis_config.get('port', 6379),
            db=0,
            decode_responses=True
        )

        self.minio_manager = MinIOManager(
            endpoint=minio_config['endpoint'],
            access_key=minio_config['access_key'],
            secret_key=minio_config['secret_key'],
            secure=minio_config.get('secure', False)
        )
        
        # โหลด Model
        try:
            model_path = "model/truck_classification.onnx"
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            logging.info(f"✓ Model loaded: {model_path}")
        except Exception as e:
            logging.error(f"✗ Model load failed: {e}")
            raise

        # ตัวแปรสำหรับ Tracker
        self.trackers = {}
        self.track_positions = {}
        self.line1_crossings = {}
        self.counted_ids = {} # Local cache (optional)
        
        self.track_last_seen = defaultdict(dict)
        self.track_classes = defaultdict(dict)
        self.line_configs = {
            "line1_pos": 0.40,
            "line2_pos": 0.70,
        }
        self.npy_timeout = int(os.getenv("NPY_TIMEOUT_SECONDS", 10))

    def _run_inference(self, frame: np.ndarray) -> Tuple[int, float, float, Optional[Tuple[int, int, int, int]]]:
        try:
            # 1. เก็บขนาดภาพจริงไว้ก่อน
            original_h, original_w = frame.shape[:2]
            
            input_tensor = self._preprocess_frame(frame)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            class_id, confidence, bbox = self._postprocess_outputs(outputs)
            
            # === [FIX] แปลงสเกล bbox จาก 640x640 กลับเป็นขนาดภาพจริง ===
            if bbox is not None:
                # คำนวณอัตราส่วน (Scale Factor)
                scale_x = original_w / 640.0
                scale_y = original_h / 640.0
                
                x1, y1, x2, y2 = bbox
                
                # คูณสเกลกลับเข้าไป
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # กันไม่ให้กล่องล้นขอบภาพ (Clip)
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))
                
                bbox = (x1, y1, x2, y2)
            # =========================================================

            # ... (ส่วนกรอง OTHER class เหมือนเดิม) ...
            if class_id == VehicleClass.OTHER.value:
                # logging.warning(f"⏭️  Ignoring OTHER class detection - not a real vehicle")
                return class_id, 0.0, 0.0, None
            
            vehicle_info = self.db.get_vehicle_class(class_id)
            total_fee = vehicle_info["total_fee"] if vehicle_info else 0.00
            
            return class_id, total_fee, confidence, bbox
        except Exception as e:
            logging.error(f"✗ Inference failed: {e}")
            return 0, 0.0, 0.0, None

    def _preprocess_frame(self, frame_bgr: np.ndarray, input_size=(640, 640)) -> np.ndarray:
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.shape[2] == 1:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, input_size)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_chw = np.transpose(frame_norm, (2, 0, 1))
        return np.expand_dims(frame_chw, axis=0)

    def _postprocess_outputs(self, outputs) -> Tuple[int, float, Optional[Tuple[int, int, int, int]]]:
        
        output = outputs[0][0]
        boxes = output[:4, :].T  # Shape: (N, 4) - [x1, y1, x2, y2]
        class_probs = output[4:, :].T
        
        class_ids = np.argmax(class_probs, axis=1)
        confidences = np.max(class_probs, axis=1)
        
        if len(confidences) > 0:
            best_idx = np.argmax(confidences)
            class_id = int(class_ids[best_idx])
            confidence = float(confidences[best_idx])
            
            # Extract bounding box for best detection
            bbox = boxes[best_idx]  # [x1, y1, x2, y2]
            
            if class_id >= len(VehicleClass):
                class_id = VehicleClass.CAR.value
            
            # Return box as integers
            return class_id, confidence, tuple(map(int, bbox))
        
        return 0, 0.0, None

    def _get_detection_center_from_bbox(self, bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int]:
        
        if bbox is None:
            # Fallback to frame center
            return 320, 320
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        return center_x, center_y

    # @staticmethod
    # def _select_frame(batch: np.ndarray) -> Tuple[np.ndarray, int]:
    #     if batch.ndim == 4:
    #         frame_idx = len(batch) // 2
    #         return batch[frame_idx], frame_idx
    #     return batch, 0

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                return (arr * 255).astype(np.uint8)
            return arr.astype(np.uint8)
        if arr.dtype != np.uint8:
            return arr.astype(np.uint8)
        return arr

    def convert_npy_to_jpg(self, npy_array: np.ndarray, frame_index: int, camera_id: str, task_id: str, quality: int = 85) -> Optional[str]:
        try:
            now = datetime.datetime.now(THAI_TIMEZONE)
            date_str = now.strftime("%Y-%m-%d")
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
            jpg_filename = f"{timestamp}_f{frame_index}.jpg"

            if npy_array.ndim != 3:
                logging.error(f"Invalid array shape: {npy_array.shape}")
                return None

            if npy_array.dtype != np.uint8:
                if npy_array.dtype in (np.float32, np.float64) and npy_array.max() <= 1.0:
                    npy_array = (npy_array * 255).astype(np.uint8)
                else:
                    npy_array = npy_array.astype(np.uint8)

            if npy_array.shape[2] == 3:
                frame_rgb = cv2.cvtColor(npy_array, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = npy_array

            image = Image.fromarray(frame_rgb)
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=quality, optimize=True)
            img_bytes = buf.getvalue()

            object_name = f"{date_str}/{camera_id}/{jpg_filename}"
            self.minio_manager.create_bucket(PROCESSED_BUCKET)
            
            success = self.minio_manager.upload_from_bytes(
                bucket=PROCESSED_BUCKET,
                object_name=object_name,
                data=img_bytes,
                content_type="image/jpeg",
            )

            if success:
                return f"{PROCESSED_BUCKET}/{object_name}"
            return None

        except Exception as e:
            logging.error(f"✗ Error converting frame: {e}")
            return None

def _run_truck_tracker(self, camera_id, class_id, confidence, bbox, frame_idx, frame_height):
        """
        Track vehicles and determine if they cross the defined lines.
        Returns: (track_id, class_id, confidence, line1_frame, line1_time, status_count)
        """
        if bbox is None:
            return None, class_id, confidence, None, None, False

        # แปลง bbox เป็น format ที่ supervision ต้องการ [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        detections = sv.Detections(
            xyxy=np.array([[x1, y1, x2, y2]]),
            confidence=np.array([confidence]),
            class_id=np.array([class_id])
        )

        # อัปเดต Tracker (ByteTrack)
        # หมายเหตุ: ในโค้ดจริงต้องมีการจัดการ Tracker แยกตามกล้อง (self.trackers[camera_id])
        # แต่เพื่อความง่าย ผมจะสมมติว่าใช้ Tracker กลาง หรือคุณต้องเพิ่ม Logic สร้าง Tracker ถ้ายังไม่มี
        if camera_id not in self.trackers:
            self.trackers[camera_id] = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
        
        tracker = self.trackers[camera_id]
        detections = tracker.update_with_detections(detections)

        if len(detections) == 0:
            return None, class_id, confidence, None, None, False

        # ดึงข้อมูล Track ID
        track_id = detections.tracker_id[0]
        
        # คำนวณตำแหน่งเส้น (Line Positions)
        line1_y = int(frame_height * self.line_configs["line1_pos"])
        line2_y = int(frame_height * self.line_configs["line2_pos"])
        
        # จุด Center ของรถ (ใช้ขอบล่าง y2 ตามมาตรฐาน)
        _, _, _, y_curr = detections.xyxy[0]
        
        # --- Logic การตรวจสอบการข้ามเส้น (Line Crossing) ---
        # 1. สร้าง Dictionary เก็บข้อมูลข้ามเส้นของกล้องนี้ถ้ายังไม่มี
        if camera_id not in self.line1_crossings:
            self.line1_crossings[camera_id] = {}
        
        # 2. เช็คการข้ามเส้นที่ 1 (บน)
        if y_curr > line1_y:
            if track_id not in self.line1_crossings[camera_id]:
                self.line1_crossings[camera_id][track_id] = {
                    "frame": frame_idx,
                    "time": datetime.datetime.now(THAI_TIMEZONE)
                }
        
        # 3. เช็คการข้ามเส้นที่ 2 (ล่าง) -> เพื่อยืนยันว่า "นับ" (Count)
        status_count = False
        if y_curr > line2_y:
            # ต้องเคยผ่านเส้น 1 มาก่อนถึงจะนับ
            if track_id in self.line1_crossings[camera_id]:
                status_count = True
        
        # ดึงเวลาที่ผ่านเส้น 1 มาใช้
        line1_info = self.line1_crossings[camera_id].get(track_id)
        line1_frame = line1_info["frame"] if line1_info else None
        line1_time = line1_info["time"] if line1_info else None

        return track_id, class_id, confidence, line1_frame, line1_time, status_count
    
def _cleanup_stale_tracks(self, camera_id: str, current_frame: int, max_age: int = 150):
        """ลบ ID ที่หายไปนานๆ ออกจาก Memory ของกล้องนั้นๆ"""
        if camera_id not in self.track_last_seen:
            return

        stale_ids = []
        # เช็คว่า ID ไหนไม่ได้อัปเดตเกิน max_age เฟรม
        for tid, last_seen in self.track_last_seen[camera_id].items():
            if (current_frame - last_seen) > max_age:
                stale_ids.append(tid)
        
        # ลบข้อมูลขยะ
        for tid in stale_ids:
            self.track_positions[camera_id].pop(tid, None)
            self.track_last_seen[camera_id].pop(tid, None)
            self.track_classes[camera_id].pop(tid, None)
            self.line1_crossings[camera_id].pop(tid, None)
            # ไม่ลบ counted_ids เพื่อกันนับซ้ำ (หรือแล้วแต่นโยบาย)

        if stale_ids:
            logging.info(f"🧹 Cleaned {len(stale_ids)} stale tracks for {camera_id}")

def _process_single_frame(self, frame: np.ndarray, frame_idx: int, task: ProcessingTask) -> Optional[Dict]:
        """Process a single frame and return transaction info or None if skipped"""
        frame_uint8 = self._normalize_to_uint8(frame)
        
        # 1. Inference
        class_id, total_fee, confidence, bbox = self._run_inference(frame_uint8)
        
        # 2. Tracking
        (
            track_id, 
            class_id, 
            confidence, 
            line1_frame, 
            line1_time, 
            status_count, 
        ) = self._run_truck_tracker(
            camera_id=task.camera_id,
            class_id=class_id,
            confidence=confidence,
            bbox=bbox,
            frame_idx=frame_idx,
            frame_height=frame_uint8.shape[0],
        )

        if track_id is None or not status_count:
            return None

        # === [Logic กันซ้ำด้วย Redis] ===
        # สร้าง Key: "dedup:กล้อง:คลาสรถ" (ตัด Track ID ออกไป เพราะ ID อาจเปลี่ยน)
        dedup_key = f"dedup:{task.camera_id}:{class_id}"
        
        # เช็คว่ามี Key นี้ใน Redis ไหม?
        if self.redis_client.exists(dedup_key):
            logging.warning(f"🚫 REDIS SKIP: Found duplicate vehicle (Class {class_id}) within cooldown.")
            return None
            
        # ถ้ายังไม่มี -> สั่ง Redis ให้จำไว้ 5 วินาที
        self.redis_client.set(dedup_key, "1", ex=5)
        # ==============================

        # 3. Convert Image & Upload
        minio_path = self.convert_npy_to_jpg(
            npy_array=frame_uint8,
            frame_index=frame_idx,
            camera_id=task.camera_id,
            task_id=task.task_id,
            quality=JPG_QUALITY
        )

        if not minio_path:
            return None

        # 4. Prepare Transaction
        tx = VehicleTransaction(
            camera_id=task.camera_id,
            track_id=str(track_id),
            class_id=class_id,
            total_fee=total_fee,
            time_stamp=line1_time or datetime.datetime.now(THAI_TIMEZONE),
            img_path=minio_path,
            confidence=confidence
        )

        # 5. Insert to DB (and CSV Log)
        try:
            self.db.insert_transaction(tx)
            logging.info(f"✅ Saved transaction for {track_id}")
            return tx.to_dict()
        except Exception as e:
            logging.error(f"DB insert failed: {e}")
            return None

def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Main entry point for processing a task"""
        logging.info(f"🚀 Processing task: {task.task_id} for camera {task.camera_id}")
        
        try:
            # 1. รอและโหลดไฟล์ .npy จาก MinIO
            npy_data = self.minio_manager.wait_for_npy_file(
                bucket_name=task.bucket_name,
                object_name=task.object_name
            )
            
            if npy_data is None:
                raise FileNotFoundError(f"Could not fetch {task.object_name} from MinIO")

            # 2. โหลดข้อมูลเข้า Numpy Array
            with BytesIO(npy_data) as f:
                frames = np.load(f)
            
            logging.info(f"🎞️ Loaded {len(frames)} frames. Shape: {frames.shape}")
            
            # 3. วนลูปประมวลผลทีละเฟรม
            results = []
            for i, frame in enumerate(frames):
                # เรียกใช้ _process_single_frame (ที่มี Logic Redis กันซ้ำอยู่ข้างใน)
                result = self._process_single_frame(frame, i, task)
                if result:
                    results.append(result)

            logging.info(f"✅ Task {task.task_id} completed. Generated {len(results)} transactions.")
            return {
                "status": "success",
                "task_id": task.task_id,
                "transactions": results
            }

        except Exception as e:
            logging.error(f"💥 Error processing task {task.task_id}: {e}")
            return {
                "status": "error", 
                "task_id": task.task_id, 
                "error": str(e)
            }
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            return {"status": "error", "task_id": task.task_id, "error": str(e)}

        finally:
            if batch_object and task.minio_bucket:
                try:
                    self.minio_manager.delete_object(task.minio_bucket, batch_object)
                except Exception as e:
                    logging.warning(f"Cleanup failed: {e}")
def main():
    # 1. Config Database
    db_config = {
        "host": os.getenv("DB_HOST", "postgres"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("POSTGRES_DB", "vehicle_db"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres1234"),
    }
    
    # 2. Config MinIO
    minio_config = {
        "endpoint": os.getenv("MINIO_ENDPOINT", "minio:9000"),
        "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        "secure": os.getenv("MINIO_SECURE", "false").lower() == "true"
    }

    # 3. Config Redis
    redis_config = {
        "host": os.getenv("REDIS_HOST", "redis"),
        "port": int(os.getenv("REDIS_PORT", 6379))
    }

    # 4. รวม Config (จุดที่เกิด Error)
    worker_config = {
        "postgres": db_config,  # ✅ ต้องมี Key นี้
        "minio": minio_config,
        "redis": redis_config
    }

    # Setup Redis Manager สำหรับ Main Process
    redis_manager = RedisQueueManager(
        host=redis_config["host"], 
        port=redis_config["port"]
    )
    
    # เริ่มต้น Worker Pool
    num_workers = int(os.getenv("NUM_WORKERS", 4))
    logging.info(f"🚀 Starting Worker Pool with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_service, initargs=(worker_config,)) as executor:
        # ส่วน PubSub Logic
        pubsub = redis_manager.client.pubsub()
        pubsub.subscribe(redis_manager.queue_name)
        
        logging.info(f"👂 Listening for tasks on queue: {redis_manager.queue_name}")
        
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    # ดึงงานจาก Queue
                    while True:
                        task_json = redis_manager.client.lpop(redis_manager.queue_name)
                        if not task_json:
                            break
                        
                        # ส่งงานให้ Worker
                        logging.info(f"📨 Dispatching task to worker")
                        future = executor.submit(task_handler, task_json)
                        
        except KeyboardInterrupt:
            logging.info("🛑 Shutting down...")
            pubsub.close()
        except Exception as e:
            logging.error(f"❌ Error: {e}")
            pubsub.close()
            raise
if __name__ == "__main__":
    main()