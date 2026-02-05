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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

QUEUE_NAME = "frame_batches"
PROCESSED_BUCKET = "process-frames"
JPG_QUALITY = 95
THAI_TIMEZONE = pytz.timezone("Asia/Bangkok")
_worker_service = None

# === NEW: Tracking Configuration ===
TRACK_EXPIRY_SECONDS = 300  # 5 minutes - tracks expire after this
MATCH_DISTANCE_THRESHOLD = 120  # pixels - max distance to match detection to track
REAPPEARANCE_DISTANCE = 20  # pixels - distance to check for re-appearing vehicles
MIN_VECTOR_STRENGTH = 5  # minimum movement frames before counting

# === MinIO Retry Configuration ===
RETRY_DELAY = 0.01  # Seconds to wait between retries (10ms)
MAX_WAIT_TIME = 30  # Maximum seconds to wait before logging warning (not stopping)


@dataclass
class VehicleTrack:
    """Represents a tracked vehicle across frames"""
    track_id: str
    camera_id: str
    last_x: int
    last_y: int
    vector_strength: int
    counted: bool
    lost_frames: int
    last_seen: datetime.datetime
    class_id: Optional[int] = None
    confidence: Optional[float] = None
    last_dx: int = 0
    last_dy: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "last_x": self.last_x,
            "last_y": self.last_y,
            "vector_strength": self.vector_strength,
            "counted": self.counted,
            "lost_frames": self.lost_frames,
            "last_seen": self.last_seen.isoformat(),
            "class_id": self.class_id,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VehicleTrack":
        data["last_seen"] = datetime.datetime.fromisoformat(data["last_seen"])
        return cls(**data)


def worker_service(config: Dict[str, Any]):
    global _worker_service
    _worker_service = ProcessingService(**config)

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
    detection_x: Optional[int] = None
    detection_y: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "camera_id": self.camera_id,
            "video_file": self.video_file,
            "minio_bucket": self.minio_bucket,
            "object_key_or_prefix": self.object_key_or_prefix,
            "timestamp": (self.timestamp or datetime.datetime.now(datetime.timezone.utc)).isoformat(),
            "detection_x": self.detection_x,
            "detection_y": self.detection_y,
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
                detection_x=data.get("detection_x"),
                detection_y=data.get("detection_y"),
            )
        except Exception as e:
            logging.error(f"Failed to parse task JSON: {json_str}")
            raise

class TrackingManager:
    """Manages vehicle tracking state in Redis"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.track_prefix = "track:"
        self.camera_tracks_prefix = "camera_tracks:"
        self.next_id_key = "next_track_id"
    
    def get_next_track_id(self, camera_id: str) -> str:
        """Generate unique track ID"""
        track_num = self.redis.incr(f"{self.next_id_key}:{camera_id}")
        return f"{camera_id}_track_{track_num}"
    
    def get_active_tracks(self, camera_id: str) -> List[VehicleTrack]:
        """Get all active tracks for a camera"""
        track_ids = self.redis.smembers(f"{self.camera_tracks_prefix}{camera_id}")
        tracks = []
        
        for track_id in track_ids:
            track_data = self.redis.get(f"{self.track_prefix}{track_id}")
            if track_data:
                try:
                    track = VehicleTrack.from_dict(json.loads(track_data))
                    tracks.append(track)
                except Exception as e:
                    logging.error(f"Failed to deserialize track {track_id}: {e}")
        
        return tracks
    
    def save_track(self, track: VehicleTrack):
        """Save track to Redis with expiry"""
        track_key = f"{self.track_prefix}{track.track_id}"
        camera_set = f"{self.camera_tracks_prefix}{track.camera_id}"
        
        # Save track data
        self.redis.setex(
            track_key,
            TRACK_EXPIRY_SECONDS,
            json.dumps(track.to_dict())
        )
        
        # Add to camera's active tracks set
        self.redis.sadd(camera_set, track.track_id)
        self.redis.expire(camera_set, TRACK_EXPIRY_SECONDS)
    
    def remove_track(self, track: VehicleTrack):
        """Remove expired track"""
        track_key = f"{self.track_prefix}{track.track_id}"
        camera_set = f"{self.camera_tracks_prefix}{track.camera_id}"
        
        self.redis.delete(track_key)
        self.redis.srem(camera_set, track.track_id)
    
    def cleanup_stale_tracks(self, camera_id: str):
        """Remove tracks that haven't been seen recently"""
        tracks = self.get_active_tracks(camera_id)
        now = datetime.datetime.now(datetime.timezone.utc)
        
        for track in tracks:
            time_diff = (now - track.last_seen).total_seconds()
            max_lost_time = 60 if track.counted else 15  # Keep counted tracks longer
            
            if time_diff > max_lost_time or track.lost_frames > 60:
                self.remove_track(track)
                logging.info(f"Cleaned up stale track: {track.track_id}")


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

        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            logging.info(f"✓ Redis connected: {host}:{port}/{db}")
        except Exception as e:
            logging.error(f"✗ Redis connection failed: {e}")
            raise

    def push_task(self, task: ProcessingTask) -> bool:
        try:
            self.client.rpush(self.queue_name, task.to_json())
            logging.info(f"✓ Task pushed: {task.task_id}")
            return True
        except Exception as e:
            logging.error(f"✗ Push failed: {e}")
            return False


class MinIOManager:
    """Manages MinIO operations"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.endpoint = endpoint
        self.client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        logging.info(f"✓ MinIO connected: {endpoint}")

    def create_bucket(self, bucket_name: str) -> bool:
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            return True
        except S3Error as e:
            logging.error(f"✗ Create bucket failed: {e}")
            return False

    def upload_from_bytes(self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        try:
            self.client.put_object(bucket, object_name, BytesIO(data), length=len(data), content_type=content_type)
            logging.info(f"✓ Uploaded: {bucket}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"✗ Upload failed: {e}")
            return False

    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            return [{"name": obj.object_name, "size": obj.size, "last_modified": obj.last_modified} for obj in objects]
        except S3Error as e:
            logging.error(f"✗ List objects failed: {e}")
            return []

    def delete_object(self, bucket: str, object_name: str) -> bool:
        try:
            self.client.remove_object(bucket, object_name)
            logging.info(f"✓ Deleted: {bucket}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"✗ Delete failed: {e}")
            return False

    def get_object_data(self, bucket: str, object_name: str) -> bytes:
        response = None
        try:
            response = self.client.get_object(bucket, object_name)
            return response.read()
        except S3Error as e:
            logging.error(f"✗ Direct download failed: {e}")
            raise
        finally:
            if response:
                response.close()
                response.release_conn()
    
    def wait_for_npy_file(self, bucket: str, prefix: str, retry_delay: float = RETRY_DELAY) -> str:
     
        attempt = 0
        start_time = time.time()
        last_warning_time = start_time
        
        while True:
            attempt += 1
            elapsed = time.time() - start_time
            
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
                    logging.warning(f"⏳ Still waiting for .npy file... ({attempt} attempts, {elapsed:.1f}s elapsed)")
                    last_warning_time = time.time()
                
                # Short debug log for troubleshooting (every 50 attempts = 5 seconds)
                if attempt % 50 == 0:
                    logging.debug(f"Waiting for .npy file in {bucket}/{prefix} (attempt {attempt})")
                
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
        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM vehicle_classes WHERE class_id = %s", (class_id,))
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logging.error(f"✗ Get vehicle class failed: {e}")
            return None
        finally:
            self._pool.putconn(conn)

    def insert_transaction(self, transaction: VehicleTransaction) -> bool:
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
            logging.error(f"✗ Insert transaction failed: {e}")
            return False
        finally:
            self._pool.putconn(conn)

    def close(self):
        if self._pool:
            self._pool.closeall()


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
    ):
        # Initialize Redis for tracking
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.tracking_manager = TrackingManager(self.redis_client)
        
        self.minio_manager = MinIOManager(endpoint=minio_endpoint, access_key=minio_access_key, 
                                         secret_key=minio_secret_key, secure=minio_secure)
        self.db = PostgreSQLDatabase(host=db_host, port=db_port, database=db_name, 
                                     user=db_user, password=db_password)

        # Load model
        logging.info(f"Loading model from MLflow: {model_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        onnx_path = os.path.join(local_model_path, "model.onnx")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        logging.info("✓ ProcessingService initialized with tracking")

    def _run_inference(self, frame: np.ndarray) -> Tuple[int, float, float, Optional[Tuple[int, int, int, int]]]:
        try:
            input_tensor = self._preprocess_frame(frame)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            class_id, confidence, bbox = self._postprocess_outputs(outputs)
            
            # === NEW: Filter out OTHER class ===
            if class_id == VehicleClass.OTHER.value:
                logging.warning(f"⏭️  Ignoring OTHER class detection - not a real vehicle")
                return class_id, 0.0, 0.0, None
            
            vehicle_info = self.db.get_vehicle_class(class_id)
            total_fee = vehicle_info["total_fee"] if vehicle_info else 0.00
            
            return class_id, total_fee, confidence, bbox
        except Exception as e:
            logging.error(f"✗ Inference failed: {e}")
            return 0, 0.0, 0.0, None

    def _preprocess_frame(self, frame_bgr: np.ndarray, input_size=(640, 640)) -> np.ndarray:
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

    def _match_or_create_track(
        self, 
        camera_id: str, 
        det_x: int, 
        det_y: int,
        class_id: int,
        confidence: float
    ) -> Tuple[VehicleTrack, bool]:
        """
        Match detection to existing track or create new one.
        Returns: (track, is_new_vehicle)
        """
        # === NEW: Validate vehicle class ===
        if not self._is_valid_vehicle_class(class_id):
            logging.warning(f"⏭️  Invalid vehicle class {class_id} - skipping")
            return None, False
        
        active_tracks = self.tracking_manager.get_active_tracks(camera_id)
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Try to match to existing track (PRIORITIZE UNCOUNTED TRACKS)
        best_match = None
        best_match_uncounted = None
        min_distance = float('inf')
        min_distance_uncounted = float('inf')
        
        for track in active_tracks:
            dist = np.sqrt((det_x - track.last_x)**2 + (det_y - track.last_y)**2)
            
            if dist <= MATCH_DISTANCE_THRESHOLD:
                # Prioritize uncounted tracks (active vehicles being tracked)
                if not track.counted and dist < min_distance_uncounted:
                    min_distance_uncounted = dist
                    best_match_uncounted = track
                # Also track best counted match as fallback
                elif track.counted and dist < min_distance:
                    min_distance = dist
                    best_match = track
        
        # Use uncounted track if available, otherwise use counted track
        if best_match_uncounted:
            track = best_match_uncounted
            logging.debug(f"Matching to UNCOUNTED track: {track.track_id}")
        elif best_match:
            track = best_match
            logging.debug(f"Matching to COUNTED track: {track.track_id}")
        else:
            track = None
        
        if track:
            # Update existing track
            dy = det_y - track.last_y
            dx = det_x - track.last_x
            track.last_x = det_x
            track.last_y = det_y
            track.lost_frames = 0
            track.last_seen = now
            track.class_id = class_id
            track.confidence = confidence
            
            # Update movement vector - require vertical movement to indicate true vehicle
            # FIXED - Track movement AND store velocity
            if not track.counted:
                # Store velocity for prediction
                track.last_dx = dx
                track.last_dy = dy
                
                # Increment vector strength if moving
                if abs(dy) >= 2:
                    track.vector_strength += 1
                    logging.info(f"📍 Track {track.track_id} - Movement: dy={dy}, strength={track.vector_strength}/{MIN_VECTOR_STRENGTH}")
                else:
                    logging.debug(f"📍 Track {track.track_id} - Minimal movement: dy={dy}")
            
            # Check if should be counted
            is_new_vehicle = False
            if not track.counted and track.vector_strength >= MIN_VECTOR_STRENGTH:
                track.counted = True
                is_new_vehicle = True
                logging.info(f"🎯 VEHICLE #{track.track_id} COUNTED! (Class: {class_id}, Movement confirmed)")
            
            self.tracking_manager.save_track(track)
            return track, is_new_vehicle
        
        # === NEW: Check for re-appearance near RECENTLY COUNTED tracks (SAME CLASS ONLY) ===
        for track in active_tracks:
            if track.counted and track.class_id == class_id:
                dist = np.sqrt((det_x - track.last_x)**2 + (det_y - track.last_y)**2)
                if dist < REAPPEARANCE_DISTANCE:
                    logging.warning(f"⚠️  Same class detection {class_id} near recently-counted track {track.track_id} - skipping (distance: {dist:.1f}px)")
                    track.lost_frames = 0
                    self.tracking_manager.save_track(track)
                    return track, False  # Don't count again
        
        # === Create new track ===
        new_track = VehicleTrack(
            track_id=self.tracking_manager.get_next_track_id(camera_id),
            camera_id=camera_id,
            last_x=det_x,
            last_y=det_y,
            vector_strength=0,
            counted=False,
            lost_frames=0,
            last_seen=now,
            class_id=class_id,
            confidence=confidence,
            last_dx=0,
            last_dy=0,
        )
        
        self.tracking_manager.save_track(new_track)
        logging.info(f"🆕 NEW TRACK: {new_track.track_id} (class: {class_id}) at ({det_x}, {det_y})")
        return new_track, False  # New track, not yet counted

    def _is_valid_vehicle_class(self, class_id: int) -> bool:
        """Check if class_id is a valid vehicle (not OTHER)"""
        if class_id == VehicleClass.OTHER.value:
            return False
        try:
            VehicleClass(class_id)
            return True
        except ValueError:
            return False
        
    def _update_all_tracks(self, camera_id: str):
        """Update all tracks - increment lost_frames for missing vehicles"""
        active_tracks = self.tracking_manager.get_active_tracks(camera_id)
        
        for track in active_tracks:
            track.lost_frames += 1
            
            # Optional: Add simple position prediction based on last movement
            # This helps bridge detection gaps
            if hasattr(track, 'last_dx') and hasattr(track, 'last_dy'):
                track.last_x += track.last_dx
                track.last_y += track.last_dy

                # Continue building vector strength during prediction
                if not track.counted and abs(track.last_dy) >= 3:
                    track.vector_strength += 1
            
            self.tracking_manager.save_track(track)
        
        logging.debug(f"Updated {len(active_tracks)} tracks for camera {camera_id}")

    @staticmethod
    def _select_frame(batch: np.ndarray) -> Tuple[np.ndarray, int]:
        if batch.ndim == 4:
            frame_idx = len(batch) // 2
            return batch[frame_idx], frame_idx
        return batch, 0

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

    def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process task with tracking to prevent double-counting"""
        batch_object = None
        
        try:
            if not task.object_key_or_prefix:
                logging.error(f"❌ CRITICAL: Empty object_key_or_prefix for task {task.task_id}")
                logging.error(f"Task data: camera_id={task.camera_id}, bucket={task.minio_bucket}")
                return {"status": "error", "reason": "Missing object key"}
            
            # ✅ VALIDATE: Ensure object key contains camera ID
            if task.camera_id not in task.object_key_or_prefix:
                logging.error(
                    f"❌ CRITICAL BUG DETECTED: "
                    f"camera_id '{task.camera_id}' not in object_key '{task.object_key_or_prefix}'"
                )
                return {"status": "error", "reason": "Camera ID mismatch"}
            
            logging.info(
                f"✅ Processing: camera={task.camera_id}, "
                f"object={task.object_key_or_prefix}"
            )
            logging.info(f"\n--- Processing Task: {task.task_id} ---")
            
            # Clean up stale tracks
            self._update_all_tracks(task.camera_id)
            self.tracking_manager.cleanup_stale_tracks(task.camera_id)
            
            # === NEW: Wait indefinitely for .npy file ===
            logging.info(f"Waiting for .npy file in {task.minio_bucket}/{task.object_key_or_prefix}...")
            batch_object = self.minio_manager.wait_for_npy_file(
                bucket=task.minio_bucket,
                prefix=task.object_key_or_prefix
            )

            # Download to RAM
            logging.info(f"Downloading {batch_object} to memory...")
            data_bytes = self.minio_manager.get_object_data(
                bucket=task.minio_bucket,
                object_name=batch_object
            )

            with BytesIO(data_bytes) as bio:
                batch_data = np.load(bio)

            # Extract frame and run inference
            selected_frame, frame_idx = self._select_frame(batch_data)
            frame_uint8 = self._normalize_to_uint8(selected_frame)
            class_id, total_fee, confidence, bbox = self._run_inference(frame_uint8)

            # === Skip if invalid detection ===
            if bbox is None or not self._is_valid_vehicle_class(class_id):
                logging.warning(f"⏭️  Skipping frame - invalid detection")
                return {
                    "status": "skipped_invalid_class",
                    "task_id": task.task_id,
                    "class_id": class_id,
                    "reason": "Invalid detection"
                }

            # Get detection center (or use provided coordinates)
            if task.detection_x is not None and task.detection_y is not None:
                det_x, det_y = task.detection_x, task.detection_y
            else:
                det_x, det_y = self._get_detection_center_from_bbox(bbox)

            logging.info(f"🎯 Detection at ({det_x}, {det_y}) - Class: {class_id}, Conf: {confidence:.2f}")

            # === TRACKING LOGIC ===
            track, is_new_vehicle = self._match_or_create_track(
                camera_id=task.camera_id,
                det_x=det_x,
                det_y=det_y,
                class_id=class_id,
                confidence=confidence
            )

            # === Handle invalid track ===
            if track is None:
                return {
                    "status": "skipped_invalid_class",
                    "task_id": task.task_id,
                    "reason": "Invalid vehicle class"
                }

            # Only save transaction if this is a NEW counted vehicle
            if not is_new_vehicle:
                logging.info(f"⏭️  NOT COUNTING YET: {track.track_id} (vector_strength: {track.vector_strength}/{MIN_VECTOR_STRENGTH})")
                return {
                    "status": "skipped_not_ready",
                    "task_id": task.task_id,
                    "track_id": track.track_id,
                    "vector_strength": track.vector_strength,
                    "reason": "Vehicle still building movement history"
                }

            # Convert and upload image
            minio_path = self.convert_npy_to_jpg(
                npy_array=frame_uint8,
                frame_index=frame_idx,
                camera_id=task.camera_id,
                task_id=task.task_id,
                quality=JPG_QUALITY
            )

            if not minio_path:
                return {"status": "upload_failed", "task_id": task.task_id}

            # Save transaction
            transaction = VehicleTransaction(
                camera_id=task.camera_id,
                track_id=track.track_id,
                class_id=class_id,
                total_fee=total_fee,
                time_stamp=task.timestamp or datetime.datetime.now(datetime.timezone.utc),
                img_path=minio_path,
                confidence=confidence,
            )
            self.db.insert_transaction(transaction)

            return {
                "status": "success",
                "task_id": task.task_id,
                "track_id": track.track_id,
                "output_image": minio_path,
                "transaction": transaction.to_dict(),
                "new_vehicle": True
            }

        except Exception as e:
            logging.error(f"✗ Batch processing failed: {e}")
            return {"status": "error", "task_id": task.task_id, "error": str(e)}

        finally:
            if batch_object and task.minio_bucket:
                try:
                    self.minio_manager.delete_object(task.minio_bucket, batch_object)
                    logging.info(f"✓ Deleted source batch: {batch_object}")
                except Exception as e:
                    logging.warning(f"⚠ Cleanup failed: {e}")


def main():
    """Main entry point - single queue, handles all cameras dynamically"""
    logging.info("Initializing Processing Service with Tracking...")

    required_vars = [
        "REDIS_HOST", "REDIS_PORT", "MINIO_ENDPOINT", "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY", "DB_HOST", "DB_PORT", "POSTGRES_DB",
        "POSTGRES_USER", "POSTGRES_PASSWORD",
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
    }

    redis_manager = RedisQueueManager(host=config["redis_host"], port=config["redis_port"])
    num_workers = int(os.getenv("NUM_WORKERS", 4))  # Scale workers based on camera count
    
    logging.info(f"Starting Worker Pool with {num_workers} processes")
    logging.info(f"Monitoring single queue: {redis_manager.queue_name}")
    logging.info("Will handle any camera_id dynamically from task metadata")

    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_service, initargs=(config,)) as executor:
        try:
            while True:
                result = redis_manager.client.blpop(redis_manager.queue_name, timeout=5)
                if result:
                    _, task_json = result
                    # Parse to log camera_id (optional, for monitoring)
                    try:
                        task_data = json.loads(task_json)
                        camera_id = task_data.get("camera_id", "unknown")
                        logging.debug(f"Processing task for camera: {camera_id}")
                    except:
                        pass
                    
                    executor.submit(task_handler, task_json)
        except KeyboardInterrupt:
            logging.info("\nShutting down worker pool...")
        except Exception as e:
            logging.error(f"Worker pool error: {e}")
            raise

if __name__ == "__main__":
    main()