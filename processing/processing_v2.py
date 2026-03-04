import traceback
import schedule
# from pyexpat import model
import threading
import mlflow
import cv2
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional
import pytz
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.engine.results import Boxes
# import onnxruntime as ort

import torch
from ultralytics import YOLO
from psycopg2 import pool
import numpy as np
import psycopg2
import psycopg2.extras
import redis
from minio import Minio
from minio.error import S3Error
from PIL import Image
from queue import Queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

QUEUE_NAME = "frame_batches"
PROCESSED_BUCKET = "process-frames"
JPG_QUALITY = 85
THAI_TIMEZONE = pytz.timezone("Asia/Bangkok")

# === MinIO Retry Configuration ===
RETRY_DELAY = 0.01  # Seconds to wait between retries (10ms)
MAX_WAIT_TIME = 30  # Maximum seconds to wait before logging warning (not stopping)
MAX_RETRIES = 3

LINE_Y1 = 270
LINE_Y2 = 335
COUNT_DIRECTION = "down"

def flush_redis_daily(redis_client, service):
    """Clear all tracking keys at midnight — blocks new tasks during flush"""
    # Step 1 — Block new tasks from starting
    service._flush_lock.clear()  # ✅ new tasks will wait
    logging.info("🔒 Flush lock acquired — no new tasks will start")

    # Step 2 — Wait for currently active tasks to finish
    wait_start = time.time()
    max_wait = 300

    while time.time() - wait_start < max_wait:
        with service._active_tasks_lock:
            active = service._active_tasks
        if active == 0:
            break
        logging.info(f"🕐 Midnight flush waiting — {active} tasks still running...")
        time.sleep(5)
    else:
        logging.warning(f"⚠️ Midnight flush forced after timeout")

    # Step 3 — Safe to flush now
    patterns = [
        "counted_tracks:*",
        "insert_lock:*",
        "pos_history:*",
        "pending_vehicle:*",
        "line1_crossed:*",
        "best_class:*",
        "last_pos:*",
    ]
    total = 0
    for pattern in patterns:
        for key in redis_client.scan_iter(pattern):
            redis_client.delete(key)
            total += 1
    logging.info(f"🌙 Midnight flush: cleared {total} Redis keys")

    # Step 4 — Unblock tasks
    service._flush_lock.set()  # ✅ tasks can run again
    logging.info("🔓 Flush lock released — resuming normal processing")

def start_midnight_scheduler(redis_client, service):
    schedule.every().day.at("00:00").do(flush_redis_daily, redis_client, service)
    
    def scheduler_loop():
        while True:
            schedule.run_pending()
            time.sleep(30)
    
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    logging.info("✅ Midnight scheduler started")

class CameraWorker:
    def __init__(self, camera_id, service, process_fn):
        self.camera_id = camera_id
        self.queue = Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.process_fn = process_fn
        self.thread.start()

    def _run(self):
        while True:
            task_json = self.queue.get()
            if task_json is None:
                break
            self.process_fn(task_json)

    def submit(self, task_json):
        self.queue.put(task_json)

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
        timeout_seconds: int = 10,) -> Optional[str]:
        
        logging.info(f"Waiting for .npy file — bucket: '{bucket}', prefix: '{prefix}'")

        # === Pre-checks ===
        if not prefix or len(prefix) < 5:
            logging.error(f"Invalid prefix: '{prefix}'")
            return None

        try:
            if not self.client.bucket_exists(bucket):
                logging.error(f"Bucket '{bucket}' does not exist")
                return None
        except Exception as e:
            logging.error(f"Cannot access bucket '{bucket}': {e}")
            return None

        # === Poll until file found or timeout ===
        attempt = 0
        start_time = time.time()

        while True:
            attempt += 1
            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
                logging.error(f"Timeout after {timeout_seconds}s ({attempt} attempts): {bucket}/{prefix}")
                return None

            try:
                found = self._find_npy_file(bucket, prefix)
                if found:
                    log_msg = f"Found .npy file after {attempt} attempts ({elapsed:.1f}s): {found}"
                    logging.info(log_msg) if attempt > 1 else logging.info(f"Found .npy file: {found}")
                    return found

            except Exception as e:
                logging.error(f"Error on attempt {attempt}: {e}")

            time.sleep(retry_delay)

    def _find_npy_file(self, bucket: str, prefix: str) -> Optional[str]:
        """Return .npy file path if it exists in MinIO, else None."""

        # Case A: prefix is a direct file path
        if prefix.endswith(".npy"):
            try:
                self.client.stat_object(bucket, prefix)
                return prefix
            except S3Error:
                return None

        # Case B: prefix is a folder — list and find newest .npy
        objects = self.list_objects(bucket=bucket, prefix=prefix)
        npy_files = [
            obj for obj in objects
            if obj["name"].endswith(".npy") and not obj["name"].endswith("/")
        ]

        if not npy_files:
            return None

        npy_files.sort(key=lambda x: x["last_modified"], reverse=True)
        return npy_files[0]["name"]

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
        time_stamp = transaction.time_stamp
        if time_stamp:
            if time_stamp.tzinfo is not None:
                time_stamp = time_stamp.astimezone(THAI_TIMEZONE).replace(tzinfo=None)
            else:
                time_stamp = time_stamp + datetime.timedelta(hours=7)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vehicle_transactions 
                    (camera_id, track_id, class_id, total_fee, time_stamp, img_path, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (camera_id, track_id, date(time_stamp)) DO NOTHING
                """, (
                    transaction.camera_id, transaction.track_id, transaction.class_id,
                    transaction.total_fee, time_stamp, transaction.img_path,
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
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        self.onnx_path = os.path.join(local_model_path, "model.onnx")
        self.models: Dict[str, YOLO] = {}
        self._model_init_lock = threading.Lock()

        # self.trackers = {} 
        self.byte_trackers: Dict[str, BYTETracker] = {}
        self._tracker_lock = threading.Lock()
        self._flush_lock = threading.Event()
        self._flush_lock.set()  # set = allowed to process
        self._active_tasks = 0
        self._active_tasks_lock = threading.Lock()

    def _get_model(self, camera_id: str) -> YOLO:
        if camera_id not in self.models:
            with self._model_init_lock:
                if camera_id not in self.models:
                    self.models[camera_id] = YOLO(self.onnx_path, task='detect')
                    logging.info(
                        f"[{camera_id}] ✅ Model instance created | "
                        f"total models loaded: {len(self.models)}"
                )
        return self.models[camera_id]
    def _increment_active(self):
        with self._active_tasks_lock:
            self._active_tasks += 1

    def _decrement_active(self):
        with self._active_tasks_lock:
            self._active_tasks -= 1

    def _get_byte_tracker(self, camera_id: str) -> BYTETracker:
        with self._tracker_lock:
            if camera_id not in self.byte_trackers:
                args = IterableSimpleNamespace(
                    track_high_thresh=0.5,
                    track_low_thresh=0.1,
                    new_track_thresh=0.6,
                    track_buffer=30,
                    match_thresh=0.8,
                    fuse_score=True,
                )
                self.byte_trackers[camera_id] = BYTETracker(args, frame_rate=30)
        return self.byte_trackers[camera_id]
    
    def _crop_with_padding(self, frame: np.ndarray, box: np.ndarray, scale: float = 1.0) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(float)

        # Bbox dimensions
        box_w = x2 - x1
        box_h = y2 - y1

        # Expand obliquely:
        # - Bottom-left corner: expand LEFT and DOWN
        # - Top-right corner: expand RIGHT and UP
        expand_x = box_w * (scale - 1)
        expand_y = box_h * (scale - 1)

        new_x1 = x1 - expand_x   # bottom-left goes further LEFT
        new_y2 = y2 + expand_y   # bottom-left goes further DOWN
        new_x2 = x2 + expand_x   # top-right goes further RIGHT
        new_y1 = y1 - expand_y   # top-right goes further UP

        # Clamp to frame bounds
        new_x1 = int(max(0, new_x1))
        new_y1 = int(max(0, new_y1))
        new_x2 = int(min(w, new_x2))
        new_y2 = int(min(h, new_y2))

        return frame[new_y1:new_y2, new_x1:new_x2]
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
                   camera_id: str, task_id: str, task_timestamp: datetime.datetime = None,
                   quality: int = 85, crop_box: Optional[np.ndarray] = None,track_id: Optional[str] = None) -> Optional[str]:
        try:
            now = task_timestamp or datetime.datetime.now(THAI_TIMEZONE)
            if isinstance(now, str):
                now = datetime.datetime.fromisoformat(now)
            # Convert to Thai timezone for filename
            if now.tzinfo is not None:
                now = now.astimezone(THAI_TIMEZONE)
            date_str = now.strftime("%Y-%m-%d")
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
            jpg_filename = f"{timestamp}_f{frame_index}_t{track_id}.jpg" if track_id else f"{timestamp}_f{frame_index}.jpg"

            if npy_array.ndim != 3: return None
            # ... conversion logic ...
            img_array = self._crop_with_padding(npy_array, crop_box) if crop_box is not None else npy_array

            image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=quality, optimize=True)
            img_bytes = buf.getvalue()

            object_name = f"{date_str}/{camera_id}/{jpg_filename}"
            self.minio_manager.create_bucket(PROCESSED_BUCKET)

            if self.minio_manager.upload_from_bytes(PROCESSED_BUCKET, object_name, img_bytes, "image/jpeg"):
                return f"{PROCESSED_BUCKET}/{object_name}"
            return None
        except Exception as e:
            logging.error(f"Error converting frame: {e}")
            return None
        
    def _is_counted(self, camera_id, track_id):
        key = f"counted_tracks:{camera_id}:{track_id}"
        return self.redis_client.exists(key)

    def _mark_counted(self, camera_id, track_id):
        key = f"counted_tracks:{camera_id}:{track_id}"
        self.redis_client.setex(key, 3600, 1)  # expires in 1 hour

    def _save_position_history(self, camera_id, track_id, center_y):
        """Store last 10 center_y positions per track"""
        key = f"pos_history:{camera_id}:{track_id}"
        
        # Get existing history
        raw = self.redis_client.get(key)
        history = json.loads(raw) if raw else []
        
        # Append new position and keep last 10 only
        history.append(center_y)
        if len(history) > 10:
            history = history[-10:]
        
        self.redis_client.setex(key, 3600, json.dumps(history))
        return history

    def _get_position_history(self, camera_id, track_id) -> list:
        key = f"pos_history:{camera_id}:{track_id}"
        raw = self.redis_client.get(key)
        return json.loads(raw) if raw else []
 
    def _save_pending_vehicle(self, camera_id, track_id, data: dict):
        """Save frame + timestamp when vehicle crosses Line 1"""
        key = f"pending_vehicle:{camera_id}:{track_id}"
        self.redis_client.setex(key, 300, json.dumps(data))

    def _get_pending_vehicle(self, camera_id, track_id) -> Optional[dict]:
        """Get saved data when vehicle crosses Line 2"""
        key = f"pending_vehicle:{camera_id}:{track_id}"
        val = self.redis_client.get(key)
        return json.loads(val) if val else None

    def _delete_pending_vehicle(self, camera_id, track_id):
        """Clean up after committing to DB"""
        key = f"pending_vehicle:{camera_id}:{track_id}"
        self.redis_client.delete(key)
        
    def _save_last_known_position(self, camera_id, track_id, center_x, center_y):
        """Save last known position when track is active"""
        key = f"last_pos:{camera_id}:{track_id}"
        self.redis_client.setex(key, 10, json.dumps({  # 10 sec TTL
            "x": center_x,
            "y": center_y,
            "track_id": track_id
        }))

    def _get_all_lost_tracks(self, camera_id) -> list:
        """Get all recently lost tracks for this camera"""
        pattern = f"last_pos:{camera_id}:*"
        lost_tracks = []
        for key in self.redis_client.scan_iter(pattern):
            val = self.redis_client.get(key)
            if val:
                lost_tracks.append(json.loads(val))
        return lost_tracks

    def _find_matching_lost_track(self, camera_id, center_x, center_y, 
                                current_track_id, max_distance=35) -> Optional[int]:
        """
        Check if current detection matches a recently lost track by position.
        Returns old track_id if match found, None otherwise.
        """
        lost_tracks = self._get_all_lost_tracks(camera_id)
        
        best_match = None
        best_distance = max_distance  # pixel threshold
        
        for lost in lost_tracks:
            # Skip if same track_id (still active)
            if lost["track_id"] == current_track_id:
                continue
                
            # Skip if already counted
            if self._is_counted(camera_id, lost["track_id"]):
                continue
                
            # Calculate Euclidean distance
            dist = ((center_x - lost["x"]) ** 2 + (center_y - lost["y"]) ** 2) ** 0.5
            
            if dist < best_distance:
                best_distance = dist
                best_match = lost["track_id"]
        
        return best_match
    def _mark_line1_crossed(self, camera_id, track_id):
        key = f"line1_crossed:{camera_id}:{track_id}"
        self.redis_client.setex(key, 3600, 1)

    def _is_line1_crossed(self, camera_id, track_id) -> bool:
        key = f"line1_crossed:{camera_id}:{track_id}"
        return bool(self.redis_client.exists(key))

    def _transfer_track_state(self, camera_id, old_track_id, new_track_id):
        """Transfer all Redis state from old track_id to new track_id"""
        
        # Transfer position history
        old_history = self._get_position_history(camera_id, old_track_id)
        if old_history:
            key = f"pos_history:{camera_id}:{new_track_id}"
            self.redis_client.setex(key, 3600, json.dumps(old_history))
        
        # Transfer pending vehicle (Line 1 data)
        pending = self._get_pending_vehicle(camera_id, old_track_id)
        if pending:
            self._save_pending_vehicle(camera_id, new_track_id, pending)
            self._delete_pending_vehicle(camera_id, old_track_id)
        if self._is_line1_crossed(camera_id, old_track_id):
            self._mark_line1_crossed(camera_id, new_track_id)
        self.redis_client.delete(f"line1_crossed:{camera_id}:{old_track_id}")    
        
        # Clean up old position key
        self.redis_client.delete(f"last_pos:{camera_id}:{old_track_id}")
        self.redis_client.delete(f"pos_history:{camera_id}:{old_track_id}")
        
        logging.info(f"🔄 ReID: Transferred state from track_{old_track_id} → track_{new_track_id}")

    def _try_lock_for_insert(self, camera_id, track_id) -> bool:
        """Atomic lock — only first caller returns True"""
        key = f"insert_lock:{camera_id}:{track_id}"
        # SET key 1 NX EX 3600 — only sets if key doesn't exist
        result = self.redis_client.set(key, 1, nx=True, ex=3600)
        return result is True

    def _update_best_class(self, camera_id, track_id, class_id, conf):
        key = f"best_class:{camera_id}:{track_id}"
        raw = self.redis_client.get(key)
        current = json.loads(raw) if raw else None
        
        if current is None or conf > current["confidence"]:
            self.redis_client.setex(key, 3600, json.dumps({
                "class_id": class_id,
                "confidence": conf
            }))

    def _get_best_class(self, camera_id, track_id) -> dict:
        key = f"best_class:{camera_id}:{track_id}"
        raw = self.redis_client.get(key)
        return json.loads(raw) if raw else {"class_id": 0, "confidence": 0.0}   
    def _check_crossed_line(self, history: list, line_y: int, 
                            direction: str) -> bool:
        """Check last N positions not just prev frame"""
        if len(history) < 2:
            return False
        
        curr_y = history[-1]
        
        # Check against recent positions
        for prev_y in reversed(history[:-1]):
            if direction == "down" and prev_y < line_y <= curr_y:
                return True
            elif direction == "up" and prev_y > line_y >= curr_y:
                return True
        
        return False 

    def _process_single_frame(self, frame: np.ndarray, frame_idx: int, task: ProcessingTask) -> Optional[Dict]:
        torch.cuda.set_device(0)
        frame_uint8 = self._normalize_to_uint8(frame)
        logging.info(f"Frame shape: {frame_uint8.shape}")

        model = self._get_model(task.camera_id)
        predict_results = list(model.predict(
            source=frame_uint8,
            verbose=False,
            conf=0.35,
            iou=0.5,
            device=0,
            stream=True,
        ))
        logging.info(f"✅ predict done, {len(predict_results)} results")

        byte_tracker = self._get_byte_tracker(task.camera_id)
        final_results = []

        for r in predict_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Move to CPU for BYTETracker
            r = r.cpu()
            xyxy  = r.boxes.xyxy.numpy()
            confs = r.boxes.conf.numpy()
            clss  = r.boxes.cls.numpy()

            det_tensor = torch.from_numpy(
                np.concatenate([xyxy, confs[:, None], clss[:, None]], axis=1).astype(np.float32)
            )
            boxes_input = Boxes(det_tensor, orig_shape=frame_uint8.shape[:2])

            # ✅ Wrap in try/catch to see exact BYTETracker error if it fails
            try:
                tracked = byte_tracker.update(boxes_input, img=frame_uint8)
            except Exception as e:
                logging.error(f"BYTETracker.update failed: {traceback.format_exc()}")
                continue

            if tracked is None or len(tracked) == 0:
                continue

            for t in tracked:
                x1, y1, x2, y2, track_id = t[:5]
                track_id = int(track_id)
                box = np.array([x1, y1, x2, y2])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                det_centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
                det_centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
                dists = np.sqrt((det_centers_x - (x1+x2)/2)**2 + (det_centers_y - (y1+y2)/2)**2)
                best_idx = int(np.argmin(dists))

                conf = float(confs[best_idx])
                cls  = int(clss[best_idx])

                # ✅ ReID guard — only after 3 stable frames
                history = self._save_position_history(task.camera_id, track_id, center_y)
                self._save_last_known_position(task.camera_id, track_id, center_x, center_y)

                if len(history) >= 3:
                    matched_old_track_id = self._find_matching_lost_track(
                        task.camera_id, center_x, center_y, track_id
                    )
                    if matched_old_track_id is not None:
                        logging.info(f"🔍 ReID match: new track_{track_id} → old track_{matched_old_track_id}")
                        self._transfer_track_state(task.camera_id, matched_old_track_id, track_id)

                unique_track_id = f"{task.camera_id}_{track_id}"

                if history[-1] > 240:
                    logging.info(f"📊 track_{track_id} center_y history: {history}")

                if cls != VehicleClass.OTHER.value:
                    self._update_best_class(task.camera_id, track_id, cls, conf)

                if len(history) < 2:
                    continue
                if cls == VehicleClass.OTHER.value:
                    continue
                if self._is_counted(task.camera_id, track_id):
                    logging.warning(f"⚠️ track_{track_id} skipped — already counted")
                    continue

                # === LINE 1 ===
                if self._check_crossed_line(history, LINE_Y1, COUNT_DIRECTION):
                    if not self._is_line1_crossed(task.camera_id, track_id):
                        logging.info(f"📸 Line 1 crossed: {unique_track_id}")
                        minio_path = self.convert_npy_to_jpg(
                            npy_array=frame_uint8,
                            frame_index=frame_idx,
                            camera_id=task.camera_id,
                            task_id=task.task_id,
                            task_timestamp=task.timestamp,
                            crop_box=box,
                            track_id=track_id,
                        )
                        self._save_pending_vehicle(task.camera_id, track_id, {
                            "minio_path": minio_path,
                            "timestamp": (task.timestamp or datetime.datetime.now(datetime.timezone.utc)).isoformat(),
                            "class_id": cls,
                            "confidence": conf,
                            "unique_track_id": unique_track_id,
                        })
                        self._mark_line1_crossed(task.camera_id, track_id)
                    continue

                # === LINE 2 ===
                if not self._check_crossed_line(history, LINE_Y2, COUNT_DIRECTION):
                    continue

                pending = self._get_pending_vehicle(task.camera_id, track_id)
                if pending is None:
                    logging.warning(f"⚠️ Line 2 crossed but no Line 1 data for {unique_track_id}")
                    continue

                logging.info(f"✅ Line 2 crossed: {unique_track_id} - saving to DB")
                if not self._try_lock_for_insert(task.camera_id, track_id):
                    logging.warning(f"⚠️ Duplicate insert prevented for {unique_track_id}")
                    continue

                self._mark_counted(task.camera_id, track_id)
                self._delete_pending_vehicle(task.camera_id, track_id)

                best = self._get_best_class(task.camera_id, track_id)
                vehicle_info = self.db.get_vehicle_class(best["class_id"])
                total_fee = vehicle_info.get("total_fee", 0.0) if vehicle_info else 0.0

                transaction = VehicleTransaction(
                    camera_id=task.camera_id,
                    track_id=track_id,
                    class_id=best["class_id"],
                    total_fee=total_fee,
                    time_stamp=datetime.datetime.fromisoformat(pending["timestamp"]),
                    img_path=pending["minio_path"],
                    confidence=best["confidence"],
                )
                try:
                    self.db.insert_transaction(transaction)
                    final_results.append({
                        "track_id": track_id,
                        "bbox": box.tolist(),
                        "class": pending["class_id"],
                        "minio_path": pending["minio_path"],
                    })
                except Exception as e:
                    logging.error(f"DB insert failed for track {track_id}: {e}")

        return {"frame_idx": frame_idx, "detections": final_results} if final_results else None
    def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        self._flush_lock.wait() 
        self._increment_active()
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
            self._decrement_active()
            if batch_object and task.minio_bucket:
                # Queue for async deletion instead of blocking
                deletion_task = f"{task.minio_bucket}|{batch_object}"
                self.redis_client.rpush('deletion_queue', deletion_task)
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
        "model_uri": os.getenv("MODEL_URI", "models:/Truck_classification_Model_v2/Production"),
        "npy_timeout_seconds": int(os.getenv("NPY_TIMEOUT_SECONDS")),
    }

    redis_manager = RedisQueueManager(host=config["redis_host"], port=config["redis_port"])
    
    # Single service instance — no ProcessPoolExecutor needed
    service = ProcessingService(**config)
    start_midnight_scheduler(service.redis_client, service)
    logging.info("✅ Single worker service initialized")

    pubsub_client = redis.Redis(host=config["redis_host"], port=config["redis_port"], decode_responses=True)
    pubsub = pubsub_client.pubsub()
    pubsub.subscribe(redis_manager.notification_channel)
    camera_workers = {}

    def process_task_json(task_json):
        try:
            task = ProcessingTask.from_json(task_json)
            result = service.process_task(task)
            
            if result.get("status") == "success":
                logging.info(f"✅ COMPLETED: {result.get('task_id')}")
            elif result.get("status") == "timeout_npy_file":
                task_data = json.loads(task_json)
                retry_count = task_data.get("retry_count", 0)
                if retry_count < MAX_RETRIES:
                    task_data["retry_count"] = retry_count + 1
                    redis_manager.client.rpush(redis_manager.queue_name, json.dumps(task_data))
                    redis_manager.client.publish(redis_manager.notification_channel, "new_task")
                else:
                    redis_manager.client.rpush("failed_tasks", task_json)
            else:
                logging.warning(f"⚠️ INCOMPLETE: {result.get('task_id')} - {result.get('status')}")
        except Exception as e:
            logging.error(f"❌ Task failed: {e}")
    def route_task(task_json):
        """Route task to correct camera worker"""
        try:
            task = ProcessingTask.from_json(task_json)
            if task.camera_id not in camera_workers:
                logging.info(f"🆕 Creating worker for {task.camera_id}")
                camera_workers[task.camera_id] = CameraWorker(task.camera_id, service, process_task_json)
            camera_workers[task.camera_id].submit(task_json)
        except Exception as e:
            logging.error(f"❌ Route failed: {e}")

    try:
        # Drain existing tasks on startup
        while True:
            task_json = redis_manager.client.lpop(redis_manager.queue_name)
            if not task_json:
                break
            route_task(task_json)

        # Listen for new tasks
        for message in pubsub.listen():
            if message['type'] == 'subscribe':
                # ✅ Do an extra drain here on first subscription confirmation
                while True:
                    task_json = redis_manager.client.lpop(redis_manager.queue_name)
                    if not task_json:
                        break
                    route_task(task_json)
            elif message['type'] == 'message':
                while True:
                    task_json = redis_manager.client.lpop(redis_manager.queue_name)
                    if not task_json:
                        break
                    route_task(task_json)

    except KeyboardInterrupt:
        logging.info("Shutting down...")
        pubsub.close()
        service.db.close()
if __name__ == "__main__":
    main()