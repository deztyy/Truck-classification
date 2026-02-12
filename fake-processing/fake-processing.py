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

# ==========================================
# [NEW] Import for Aik's Tracking Logic
# ==========================================
import supervision as sv
from collections import defaultdict

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
    _worker_service = ProcessingService(**config)
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
    
    def wait_for_npy_file(self, bucket: str, prefix: str, retry_delay: float = RETRY_DELAY, 
                       timeout_seconds: int = 10) -> Optional[str]:
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
        """Insert a vehicle transaction record"""
        conn = self._pool.getconn()
        try:
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
            logging.info(f"✓ Transaction saved: {transaction.track_id}")
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

    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        db_host: str,
        db_port: int,
        db_name: str,
        db_user: str,
        db_password: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        minio_secure: bool = False,
        mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI"),
        model_uri: str = os.getenv("MODEL_URI"),
        npy_timeout_seconds: int = int(os.getenv("NPY_TIMEOUT_SECONDS")),
    ):
        # Initialize Redis for tracking
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        self.minio_manager = MinIOManager(endpoint=minio_endpoint, access_key=minio_access_key, 
                                         secret_key=minio_secret_key, secure=minio_secure)
        self.db = PostgreSQLDatabase(host=db_host, port=db_port, database=db_name, 
                                     user=db_user, password=db_password)
        self.npy_timeout = npy_timeout_seconds

        # Load model
        logging.info(f"Loading model from MLflow: {model_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        onnx_path = os.path.join(local_model_path, "model.onnx")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # =========================================================
        # [NEW] Aik's Tracking Logic State
        # =========================================================
        self.trackers = {}          # Store ByteTrack instances per camera_id
        self.track_histories = {}   # Store track positions per camera_id
        self.counted_ids = {}       # Store counted IDs per camera_id
        self.line_configs = {       # Line configurations
            "line1_pos": 0.40,
            "line2_pos": 0.70
        }
        
        logging.info("✓ ProcessingService initialized with tracking & Aik's Logic")

    def _run_inference(self, frame: np.ndarray) -> Tuple[int, float, float, Optional[Tuple[int, int, int, int]]]:
        try:
            input_tensor = self._preprocess_frame(frame)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            class_id, confidence, bbox = self._postprocess_outputs(outputs)
            
            # === NEW: Filter out OTHER class ===
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

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                return (arr * 255).astype(np.uint8)
            return arr.astype(np.uint8)
        if arr.dtype != np.uint8:
            return arr.astype(np.uint8)
        return arr

    def convert_npy_to_jpg(self, npy_array: np.ndarray, frame_index: int, 
                          camera_id: str, task_id: str, quality: int = 85) -> Optional[str]:
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

    # =========================================================
    # [NEW] Aik's Tracking Logic Function
    # =========================================================
    def _run_aik_tracking(self, camera_id: str, class_id: int, confidence: float, 
                          bbox: Optional[Tuple[int, int, int, int]], frame_idx: int) -> Tuple[Optional[str], bool]:
        """
        Aik's tracking logic: Use ByteTrack and check Line Crossing
        """
        if bbox is None:
            return None, False

        # Initialize Tracker for this camera if needed
        if camera_id not in self.trackers:
            self.trackers[camera_id] = sv.ByteTrack(
                track_activation_threshold=0.1,
                lost_track_buffer=60,
                frame_rate=30
            )
            self.track_histories[camera_id] = defaultdict(list)
            self.counted_ids[camera_id] = set()

        # Format Detection for Supervision
        xyxy = np.array([bbox])
        confidences = np.array([confidence])
        class_ids = np.array([class_id])
        
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids
        )

        # Update Tracker
        tracks = self.trackers[camera_id].update_with_detections(detections)
        
        if len(tracks) == 0:
            return None, False

        track = tracks[0] 
        track_id = str(track.tracker_id)
        
        # History and Line Crossing Logic
        x1, y1, x2, y2 = track.xyxy[0]
        center_y = (y1 + y2) / 2
        
        history = self.track_histories[camera_id][track_id]
        history.append((frame_idx, center_y))
        
        if len(history) > 30:
            history.pop(0)

        is_crossed = False
        if track_id not in self.counted_ids[camera_id] and len(history) >= 2:
            # Assume Frame Height for line calc (adjust if needed)
            FRAME_HEIGHT = 1080 
            line1_y = FRAME_HEIGHT * self.line_configs["line1_pos"]
            line2_y = FRAME_HEIGHT * self.line_configs["line2_pos"]
            
            curr_y = history[-1][1]
            
            # Check history
            for prev_frame, prev_y in reversed(history[:-1]):
                if (frame_idx - prev_frame) > 10: 
                    break
                
                if (prev_y < line1_y <= curr_y) or (prev_y < line2_y <= curr_y):
                    is_crossed = True
                    self.counted_ids[camera_id].add(track_id)
                    logging.info(f"🚀 Aik's Logic: Vehicle {track_id} crossed line at frame {frame_idx}")
                    break

        return track_id, is_crossed

    def _process_single_frame(self, frame: np.ndarray, frame_idx: int, task: ProcessingTask) -> Optional[Dict]:
        """Process a single frame and return transaction info or None if skipped"""
        frame_uint8 = self._normalize_to_uint8(frame)
        class_id, total_fee, confidence, bbox = self._run_inference(frame_uint8)

        # ==============================================================================
        # ✅ Call Aik's tracking logic function
        # ==============================================================================
        track_id_from_logic, is_crossed = self._run_aik_tracking(
            camera_id=task.camera_id,
            class_id=class_id,
            confidence=confidence,
            bbox=bbox,
            frame_idx=frame_idx
        )
        
        if track_id_from_logic is None:
            return None

        # [OPTION] Uncomment below if you only want to save when crossing line
        # if not is_crossed: return None 

        # Skip invalid detections
        if bbox is None or class_id == VehicleClass.OTHER.value:
            return None

        # Convert and upload image
        minio_path = self.convert_npy_to_jpg(
            npy_array=frame_uint8,
            frame_index=frame_idx,
            camera_id=task.camera_id,
            task_id=task.task_id,
            quality=JPG_QUALITY
        )

        if not minio_path:
            return None

        # Generate unique track_id if not provided
        # Use Track ID from Aik's Logic
        track_id = track_id_from_logic

        # Save transaction
        transaction = VehicleTransaction(
            camera_id=task.camera_id,
            track_id=track_id,
            class_id=class_id,
            total_fee=total_fee,
            time_stamp=task.timestamp or datetime.datetime.now(datetime.timezone.utc),
            img_path=minio_path,
            confidence=confidence,
        )
        
        try:
            self.db.insert_transaction(transaction)
            logging.info(f"✅ Saved transaction for {track_id}")
            return {
                "track_id": track_id,
                "minio_path": minio_path,
                "transaction": transaction.to_dict()
            }
        except Exception as e:
            logging.error(f"DB insert failed: {e}")
            return None

    def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process task - just do inference and save to DB"""
        batch_object = None

        try:
            if not task.object_key_or_prefix:
                logging.error(f"Empty object_key_or_prefix for task {task.task_id}")
                return {"status": "error", "reason": "Missing object key"}
            
            if task.camera_id not in task.object_key_or_prefix:
                logging.error(f"Camera ID mismatch in object_key")
                return {"status": "error", "reason": "Camera ID mismatch"}
            
            # Wait for .npy file
            batch_object = self.minio_manager.wait_for_npy_file(
                bucket=task.minio_bucket,
                prefix=task.object_key_or_prefix,
                timeout_seconds=self.npy_timeout
            )

            if batch_object is None:
                logging.error(f"Failed to retrieve .npy file - timeout")
                return {"status": "timeout_npy_file", "task_id": task.task_id}

            # Download to RAM
            data_bytes = self.minio_manager.get_object_data(
                bucket=task.minio_bucket,
                object_name=batch_object
            )

            try:
                with BytesIO(data_bytes) as bio:
                    batch_data = np.load(bio)
                
                if batch_data.size == 0:
                    return {"status": "empty_batch", "task_id": task.task_id}
                    
                if batch_data.ndim not in (3, 4):
                    return {"status": "invalid_shape", "task_id": task.task_id}
                    
            except Exception as e:
                logging.error(f"Failed to load batch: {e}")
                return {"status": "corrupt_batch", "task_id": task.task_id}

            # Process frame(s)
            if batch_data.ndim == 3:
                batch_data = np.expand_dims(batch_data, axis=0)  # Make it (1, H, W, C)
            
            processed_count = 0
            results = []
            
            for frame_idx, frame in enumerate(batch_data):
                result = self._process_single_frame(frame, frame_idx, task)
                if result:
                    processed_count += 1
                    results.append(result)
            
            return {
                "status": "success",
                "task_id": task.task_id,
                "frames_processed": processed_count,
                "results": results
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
    """Main entry point - single queue, handles all cameras dynamically"""
    logging.info("Initializing Processing Service with Tracking...")

    required_vars = [
        "REDIS_HOST", "REDIS_PORT", "MINIO_ENDPOINT", "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY", "DB_HOST", "DB_PORT", "POSTGRES_DB",
        "POSTGRES_USER", "POSTGRES_PASSWORD","NPY_TIMEOUT_SECONDS"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    config = {
        "redis_host": os.getenv("REDIS_HOST"),
        "redis_port": int(os.getenv("REDIS_PORT")),
        "minio_endpoint": os.getenv("MINIO_ENDPOINT"),
        "minio_access_key": os.getenv("MINIO_ACCESS_KEY"),
        "minio_secret_key": os.getenv("MINIO_SECRET_KEY"),
        "minio_secure": os.getenv("MINIO_SECURE", "false").lower() == "true",
        "db_host": os.getenv("DB_HOST"),
        "db_port": int(os.getenv("DB_PORT")),
        "db_name": os.getenv("POSTGRES_DB"),
        "db_user": os.getenv("POSTGRES_USER"),
        "db_password": os.getenv("POSTGRES_PASSWORD"),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"),
        "model_uri": os.getenv("MODEL_URI", "models:/Truck_classification_Model/Production"),
        "npy_timeout_seconds": int(os.getenv("NPY_TIMEOUT_SECONDS")),
    }

    redis_manager = RedisQueueManager(host=config["redis_host"], port=config["redis_port"])
    num_workers = int(os.getenv("NUM_WORKERS", 4))  # Scale workers based on camera count
    
    logging.info(f"Starting Worker Pool with {num_workers} processes")
    logging.info(f"Monitoring single queue: {redis_manager.queue_name}")
    logging.info("Will handle any camera_id dynamically from task metadata")

    # Create Pub/Sub client
    pubsub_client = redis.Redis(
        host=config["redis_host"], 
        port=config["redis_port"], 
        decode_responses=True
    )
    pubsub = pubsub_client.pubsub()
    pubsub.subscribe(redis_manager.notification_channel)
    
    logging.info(f"📡 Subscribed to notification channel: {redis_manager.notification_channel}")

    # Callback function for handling results
    def handle_result(future, task_json):
        try:
            result = future.result()
            
            # Log the actual result
            if result.get("status") == "success":
                logging.info(f"✅ COMPLETED: {result.get('task_id')} - Saved to DB")
            elif result.get("status") == "timeout_npy_file":
                task_data = json.loads(task_json)
                retry_count = task_data.get("retry_count", 0)
                
                if retry_count < MAX_RETRIES:
                    logging.warning(
                        f"⚠️ TIMEOUT: {result.get('task_id')} - Retrying ({retry_count + 1}/{MAX_RETRIES})"
                    )       
                    task_data["retry_count"] = retry_count + 1
                    updated_task_json = json.dumps(task_data)    
                    redis_manager.client.rpush(redis_manager.queue_name, updated_task_json)
                    redis_manager.client.publish(redis_manager.notification_channel, "new_task")
                else:
                    logging.error(
                        f"❌ FAILED: {result.get('task_id')} - Max retries exceeded"
                    )
                    redis_manager.client.rpush("failed_tasks", task_json)
            else:
                # Log other statuses
                logging.warning(f"⚠️ INCOMPLETE: {result.get('task_id')} - Status: {result.get('status')}")
                    
        except Exception as e:
            logging.error(f"❌ EXCEPTION: Task processing failed - {e}")

    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_service, initargs=(config,)) as executor:
        try:
            # Check for existing tasks on startup
            logging.info("🔍 Checking for existing tasks in queue...")
            startup_tasks = 0
            while True:
                result = redis_manager.client.lpop(redis_manager.queue_name)
                if not result:
                    break
                startup_tasks += 1
                task_json = result
                future = executor.submit(task_handler, task_json)
                future.add_done_callback(lambda f, t=task_json: handle_result(f, t))
            
            if startup_tasks > 0:
                logging.info(f"✅ Found and dispatched {startup_tasks} existing tasks")
            else:
                logging.info("✅ Queue was empty on startup")
            
            # Listen for notifications
            logging.info("👂 Listening for new task notifications...")
            for message in pubsub.listen():
                if message['type'] == 'message':
                    logging.info("🔔 Received notification: New task available!")
                    
                    # Process all available tasks
                    tasks_dispatched = 0
                    while True:
                        result = redis_manager.client.lpop(redis_manager.queue_name)
                        if not result:
                            break
                        
                        tasks_dispatched += 1
                        task_json = result
                        future = executor.submit(task_handler, task_json)
                        future.add_done_callback(lambda f, t=task_json: handle_result(f, t))
                    
                    if tasks_dispatched > 0:
                        logging.info(f"✅ Dispatched {tasks_dispatched} task(s)")
                        
        except KeyboardInterrupt:
            logging.info("\nShutting down worker pool...")
            pubsub.close()
        except Exception as e:
            logging.error(f"Worker pool error: {e}")
            pubsub.close()
            raise
if __name__ == "__main__":
    main()