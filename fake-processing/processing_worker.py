import mlflow
import onnxruntime as ort
import cv2
import datetime
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import pytz

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
PROCESSED_BUCKET = "HISTORY"
JPG_QUALITY = 95
THAI_TIMEZONE = pytz.timezone("Asia/Bangkok")
_worker_service = None

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
            "time_stamp": self.time_stamp
            or datetime.datetime.now(datetime.timezone.utc),
            "img_path": self.img_path,
            "confidence": round(self.confidence, 4) if self.confidence else None,
        }

    def __repr__(self) -> str:
        return (
            f"VehicleTransaction(camera_id={self.camera_id}, "
            f"track_id={self.track_id}, class_id={self.class_id}, "
            f"confidence={self.confidence})"
        )

@dataclass
class ProcessingTask:
    
    task_id: str
    camera_id: str
    video_file: str
    minio_bucket: str
    object_key_or_prefix: str
    timestamp: Optional[datetime.datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "camera_id": self.camera_id,
            "video_file": self.video_file,
            "minio_bucket": self.minio_bucket,
            "minio_prefix": self.object_key_or_prefix,
            "object_key_or_prefix": self.object_key_or_prefix,
            "timestamp": (
                self.timestamp or datetime.datetime.now(datetime.timezone.utc)
            ).isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ProcessingTask":
        """Create from JSON string"""
        try:
            data = json.loads(json_str)
            logging.debug(f"Parsed task data: {data}")

            return cls(
                task_id=data.get("task_id")
                or data.get("batch_id")
                or data.get("object_name", "unknown"),
                camera_id=data.get("camera_id", "unknown"),
                video_file=data.get("video_file") or data.get("video_path", ""),
                minio_bucket=data.get("minio_bucket")
                or data.get("bucket_name")
                or data.get("bucket", "video-frames"),
                object_key_or_prefix=data.get("object_key_or_prefix")
                or data.get("minio_prefix")
                or data.get("minio_key")
                or data.get("object_name")
                or data.get("key", ""),
                timestamp=datetime.datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.datetime.now(datetime.timezone.utc),
            )
        except Exception as e:
            logging.error(f"Failed to parse task JSON: {json_str}")
            logging.error(f"Error: {e}")
            raise

    def __repr__(self) -> str:
        return (
            f"ProcessingTask(task_id={self.task_id}, "
            f"camera_id={self.camera_id}, video_file={self.video_file})"
        )

class RedisQueueManager:
    """Manages Redis queue for processing tasks"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        queue_name: str = QUEUE_NAME,
    ):
        """Initialize Redis connection"""
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
        """Push task to queue"""
        try:
            self.client.rpush(self.queue_name, task.to_json())
            logging.info(f"✓ Task pushed: {task.task_id}")
            return True
        except Exception as e:
            logging.error(f"✗ Push failed: {e}")
            return False

    def pop_task(self, timeout: int = 0) -> Optional[ProcessingTask]:
        """Pop task from queue (blocking if timeout > 0)"""
        try:
            if timeout > 0:
                result = self.client.blpop(self.queue_name, timeout)
                if result:
                    _, task_json = result
                    logging.debug(f"Received task JSON: {task_json}")
                    return ProcessingTask.from_json(task_json)
            else:
                task_json = self.client.lpop(self.queue_name)
                if task_json:
                    logging.debug(f"Received task JSON: {task_json}")
                    return ProcessingTask.from_json(task_json)
            return None
        except json.JSONDecodeError as e:
            logging.error(f"✗ Invalid JSON in queue: {e}")
            return None
        except Exception as e:
            logging.error(f"✗ Pop failed: {e}")
            return None

class MinIOManager:
    """Manages MinIO operations"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ):
        """Initialize MinIO connection"""
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure

        try:
            self.client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )
            logging.info(f"✓ MinIO connected: {endpoint}")
        except Exception as e:
            logging.error(f"✗ MinIO connection failed: {e}")
            raise

    def create_bucket(self, bucket_name: str) -> bool:
        """Create bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logging.info(f"✓ Bucket created: {bucket_name}")
            return True
        except S3Error as e:
            logging.error(f"✗ Create bucket failed: {e}")
            return False

    def upload_from_bytes(
        self, bucket: str, object_name: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> bool:
        """Upload data from bytes"""
        try:
            self.client.put_object(
                bucket,
                object_name,
                BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            logging.info(f"✓ Uploaded: {bucket}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"✗ Upload failed: {e}")
            return False

    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        """List objects in bucket with prefix"""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            result = [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                }
                for obj in objects
            ]
            logging.info(f"✓ Listed {len(result)} objects in {bucket}/{prefix}")
            return result
        except S3Error as e:
            logging.error(f"✗ List objects failed: {e}")
            return []

    def delete_object(self, bucket: str, object_name: str) -> bool:
        """Delete object from bucket"""
        try:
            self.client.remove_object(bucket, object_name)
            logging.info(f"✓ Deleted: {bucket}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"✗ Delete failed: {e}")
            return False
    def get_object_data(self, bucket: str, object_name: str) -> bytes:
        """Download object directly into memory as bytes."""
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
        
class PostgreSQLDatabase:
    """Manages PostgreSQL database operations"""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """Initialize database connection"""
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.autocommit = False
            logging.info(f"✓ PostgreSQL connected: {host}:{port}/{database}")
        except Exception as e:
            logging.error(f"✗ Database connection failed: {e}")
            raise

    def get_vehicle_class(self, class_id: int) -> Optional[Dict]:
        """Get vehicle class information"""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM vehicle_classes WHERE class_id = %s",
                    (class_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logging.error(f"✗ Get vehicle class failed: {e}")
            return None

    def insert_transaction(self, transaction: VehicleTransaction) -> bool:
        """Insert vehicle transaction"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vehicle_transactions 
                    (camera_id, track_id, class_id, total_fee, time_stamp, img_path, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    transaction.camera_id,
                    transaction.track_id,
                    transaction.class_id,
                    transaction.total_fee,
                    transaction.time_stamp,
                    transaction.img_path,
                    transaction.confidence,
                ))
                self.conn.commit()
                logging.info(f"✓ Transaction saved: {transaction.track_id}")
                return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"✗ Insert transaction failed: {e}")
            return False

    def get_transaction_count(self) -> int:
        """Get total number of transactions"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vehicle_transactions")
                return cur.fetchone()[0]
        except Exception as e:
            logging.error(f"✗ Get transaction count failed: {e}")
            return 0

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("✓ Database connection closed")

class PostgreSQLDatabase:
    """Manages PostgreSQL database operations"""

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """Initialize database connection"""
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.autocommit = False
            logging.info(f"✓ PostgreSQL connected: {host}:{port}/{database}")
        except Exception as e:
            logging.error(f"✗ Database connection failed: {e}")
            raise

    def get_vehicle_class(self, class_id: int) -> Optional[Dict]:
        """Get vehicle class information"""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM vehicle_classes WHERE class_id = %s",
                    (class_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logging.error(f"✗ Get vehicle class failed: {e}")
            return None

    def insert_transaction(self, transaction: VehicleTransaction) -> bool:
        """Insert vehicle transaction"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vehicle_transactions 
                    (camera_id, track_id, class_id, total_fee, time_stamp, img_path, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    transaction.camera_id,
                    transaction.track_id,
                    transaction.class_id,
                    transaction.total_fee,
                    transaction.time_stamp,
                    transaction.img_path,
                    transaction.confidence,
                ))
                self.conn.commit()
                logging.info(f"✓ Transaction saved: {transaction.track_id}")
                return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"✗ Insert transaction failed: {e}")
            return False

    def get_transaction_count(self) -> int:
        """Get total number of transactions"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vehicle_transactions")
                return cur.fetchone()[0]
        except Exception as e:
            logging.error(f"✗ Get transaction count failed: {e}")
            return 0

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("✓ Database connection closed")
class ProcessingService:
    """Service that processes tasks from Redis queue and fetches data from MinIO"""

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
        minio_secure: bool = False,
        output_dir: str = "./processed_data",
        mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI"),
        model_uri: str = os.getenv("MODEL_URI"),
    ):
        """Initialize processing service"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.minio_manager = MinIOManager(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
        )
        self.db = PostgreSQLDatabase(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
        )

        # Load AI model from MLflow
        logging.info(f"Loading model from MLflow: {model_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        try:
            # Download model from MLflow (stored in MinIO)
            local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
            onnx_path = os.path.join(local_model_path, "model.onnx")

            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"Model not found at: {onnx_path}")

            # Load ONNX model
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            logging.info(f"✓ Model loaded successfully from {onnx_path}")
        except Exception as e:
            logging.error(f"✗ Failed to load model: {e}")
            raise

        logging.info("✓ ProcessingService initialized")

    def _run_inference(self, frame: np.ndarray) -> Tuple[int, float, float]:
        """Run inference on frame using ONNX model"""
        try:
            input_tensor = self._preprocess_frame(frame)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            class_id, confidence = self._postprocess_outputs(outputs)
            vehicle_info = self.db.get_vehicle_class(class_id)
            total_fee = vehicle_info["total_fee"] if vehicle_info else 0.00
            logging.info(
                f"🤖 Model inference: class_id={class_id} (confidence: {confidence:.4f}, fee: {total_fee})"
            )
            return class_id, total_fee, confidence
        except Exception as e:
            logging.error(f"✗ Inference failed: {e}")
            return 0, 0.0, 0.0

    def _preprocess_frame(self, frame_bgr: np.ndarray, input_size=(640, 640)) -> np.ndarray:
        """Preprocess frame for ONNX model"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, input_size)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_chw = np.transpose(frame_norm, (2, 0, 1))
        frame_chw = np.expand_dims(frame_chw, axis=0)
        return frame_chw

    def _postprocess_outputs(self, outputs) -> Tuple[int, float]:
        """Post-process ONNX model outputs"""
        output = outputs[0][0]  # [84, 8400] for YOLOv8
        boxes = output[:4, :].T  # [8400, 4]
        class_probs = output[4:, :].T  # [8400, num_classes]
        
        class_ids = np.argmax(class_probs, axis=1)
        confidences = np.max(class_probs, axis=1)
        
        if len(confidences) > 0:
            best_idx = np.argmax(confidences)
            class_id = int(class_ids[best_idx])
            confidence = float(confidences[best_idx])
            
            # Ensure class_id is within valid range
            if class_id >= len(VehicleClass):
                class_id = VehicleClass.CAR.value
                
            return class_id, confidence
        
        return 0, 0.0  # No detection

    @staticmethod
    def _select_frame(batch: np.ndarray) -> Tuple[np.ndarray, int]:
        """Select a representative frame from a batch or return the image itself."""
        if batch.ndim == 4:
            frame_idx = len(batch) // 2
            return batch[frame_idx], frame_idx
        return batch, 0

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        """Convert array to uint8, normalizing floats when appropriate."""
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                return (arr * 255).astype(np.uint8)
            return arr.astype(np.uint8)
        if arr.dtype != np.uint8:
            return arr.astype(np.uint8)
        return arr
    
    def convert_npy_to_jpg(
        self,
        npy_array: np.ndarray,
        frame_index: int,
        camera_id: str,
        task_id: str,
        quality: int = 85
    ) -> Optional[str]:
   
        try:
            now = datetime.datetime.now(THAI_TIMEZONE)
            date_str = now.strftime("%Y-%m-%d")
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
            jpg_filename = f"{timestamp}_f{frame_index}.jpg"
           
            if npy_array.ndim != 3:
                logging.error(f"Invalid array shape: {npy_array.shape}. Expected (H, W, C)")
                return None
           
            if npy_array.dtype != np.uint8:
                if npy_array.dtype in (np.float32, np.float64) and npy_array.max() <= 1.0:
                    npy_array = (npy_array * 255).astype(np.uint8)
                else:
                    npy_array = npy_array.astype(np.uint8)
            
            # Convert BGR (OpenCV) to RGB (PIL)
            if npy_array.shape[2] == 3:
                frame_rgb = cv2.cvtColor(npy_array, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = npy_array
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Encode to JPEG bytes (instead of saving to file)
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=quality, optimize=True)
            img_bytes = buf.getvalue()
            
            # Construct MinIO object path: date/camera_id/filename
            object_name = f"{date_str}/{camera_id}/{jpg_filename}"
            
            # Ensure bucket exists
            self.minio_manager.create_bucket(PROCESSED_BUCKET)
            
            # Upload to MinIO
            success = self.minio_manager.upload_from_bytes(
                bucket=PROCESSED_BUCKET,
                object_name=object_name,
                data=img_bytes,
                content_type="image/jpeg",
            )
            
            if success:
                minio_path = f"{PROCESSED_BUCKET}/{object_name}"
                logging.info(f"✓ Frame converted and uploaded: {minio_path}")
                return minio_path
            else:
                logging.error("✗ Failed to upload frame to MinIO")
                return None
                
        except Exception as e:
            logging.error(f"✗ Error converting frame to jpg: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process task: fetch from MinIO to RAM, run AI, and save results."""
        batch_object = None
        
        try:
            logging.info(f"\n--- Processing Task: {task.task_id} ---")
            
            # 1. Determine the object name
            if task.object_key_or_prefix.endswith(".npy"):
                batch_object = task.object_key_or_prefix
            else:
                listed_objects = self.minio_manager.list_objects(
                    bucket=task.minio_bucket, prefix=task.object_key_or_prefix
                )
                files = [obj for obj in listed_objects if not obj["name"].endswith("/")]
                if not files:
                    return {"status": "no_objects", "task_id": task.task_id}
                batch_object = files[0]["name"]

            # 2. Download directly to RAM (No local file created)
            logging.info(f"Downloading {batch_object} to memory...")
            data_bytes = self.minio_manager.get_object_data(
                bucket=task.minio_bucket,
                object_name=batch_object
            )

            # 3. Load from bytes using BytesIO
            with BytesIO(data_bytes) as bio:
                batch_data = np.load(bio)
            
            # Once we leave this 'with' block, 'bio' is closed. 
            # After the function ends, 'data_bytes' is cleared from RAM automatically.

            logging.info(f"Loaded batch shape: {batch_data.shape}")

            # 4. Extract frame and run inference
            selected_frame, frame_idx = self._select_frame(batch_data)
            frame_uint8 = self._normalize_to_uint8(selected_frame)
            class_id, total_fee, confidence = self._run_inference(frame_uint8)

            # 5. Convert and upload result image
            minio_path = self.convert_npy_to_jpg(
                npy_array=frame_uint8,
                frame_index=frame_idx,
                camera_id=task.camera_id,
                task_id=task.task_id,
                quality=JPG_QUALITY
            )

            if not minio_path:
                return {"status": "upload_failed", "task_id": task.task_id}

            # 6. Save Transaction to DB
            transaction = VehicleTransaction(
                camera_id=task.camera_id,
                track_id=task.task_id,
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
                "output_image": minio_path,
                "transaction": transaction.to_dict(),
            }

        except Exception as e:
            logging.error(f"✗ Batch processing failed: {e}")
            return {"status": "error", "task_id": task.task_id, "error": str(e)}

        finally:
            # ALWAYS delete the processed .npy from MinIO to prevent re-processing
            if batch_object and task.minio_bucket:
                try:
                    self.minio_manager.delete_object(task.minio_bucket, batch_object)
                    logging.info(f"✓ Deleted source batch: {batch_object}")
                except Exception as e:
                    logging.warning(f"⚠ Cleanup failed: {e}")


def main():
    """Main entry point - runs with multiprocessing"""

    logging.info("Initializing Processing Service...")

    # Get configuration from environment variables (REQUIRED)
    required_vars = [
        "REDIS_HOST",
        "REDIS_PORT",
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "DB_HOST",
        "DB_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Build configuration dictionary
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

    redis_manager = RedisQueueManager(
        host=config["redis_host"], 
        port=config["redis_port"]
    )
    
    # Determine number of worker processes
    num_workers = int(os.getenv("NUM_WORKERS", os.cpu_count() or 2))
    logging.info(f"Starting Multi-Core Worker Pool with {num_workers} processes.")

    # Start the Process Pool
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=worker_service,
        initargs=(config,)
    ) as executor:
        try:
            while True:
            
                result = redis_manager.client.blpop(redis_manager.queue_name, timeout=5)
                if result:
                    _, task_json = result
                    # Offload work to the pool!
                    executor.submit(task_handler, task_json)
        except KeyboardInterrupt:
            logging.info("\nShutting down worker pool...")
        except Exception as e:
            logging.error(f"Worker pool error: {e}")
            raise

if __name__ == "__main__":
    main()