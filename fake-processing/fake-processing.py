"""
Mock Database Writer - Simulates writing vehicle transaction data
Generates fake data matching the vehicle_transactions table schema
Integrates with Redis Queue and MinIO for data pipeline
"""
import mlflow
import onnxruntime as ort
import cv2
import datetime
import json
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
import redis
from minio import Minio
from minio.error import S3Error
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------------------------------------------
# Module constants
# ----------------------------------------------------------------------------
DEFAULT_QUEUE_NAME = "frame_batches"
PROCESSED_BUCKET_NAME = "process-frames"
JPEG_QUALITY = 95


# ============================================================================
# DATA MODELS
# ============================================================================


class VehicleClass(Enum):
    """Vehicle classification types"""

    CAR = 1
    OTHER = 2
    OTHER_TRUCK = 3
    PICKUP_TRUCK = 4
    TRUCK_20_BACK = 5
    TRUCK_20_FRONT = 6
    TRUCK_20X2 = 7
    TRUCK_40 = 8
    TRUCK_RORO = 9
    TRUCK_TAIL = 10
    MOTORCYCLE = 11
    TRUCK_HEAD = 12


@dataclass
class VehicleTransaction:
    """Represents a vehicle transaction record"""

    camera_id: str
    track_id: str
    class_id: int
    total_fee: float = 0.00
    time_stamp: Optional[datetime.datetime] = None
    img_path: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for database insert"""
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


# ============================================================================
# REDIS QUEUE MANAGER
# ============================================================================


@dataclass
class ProcessingTask:
    """Represents a processing task from Redis queue"""

    task_id: str
    camera_id: str
    video_file: str
    minio_bucket: str
    object_key_or_prefix: str
    timestamp: Optional[datetime.datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
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
        queue_name: str = DEFAULT_QUEUE_NAME,
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

    def get_queue_size(self) -> int:
        """Get number of tasks in queue"""
        try:
            return self.client.llen(self.queue_name)
        except Exception as e:
            logging.error(f"✗ Get queue size failed: {e}")
            return 0

    def clear_queue(self) -> bool:
        """Clear all tasks from queue"""
        try:
            self.client.delete(self.queue_name)
            logging.info(f"✓ Queue cleared: {self.queue_name}")
            return True
        except Exception as e:
            logging.error(f"✗ Clear queue failed: {e}")
            return False


# ============================================================================
# MINIO MANAGER (MISSING CLASS - ADDED)
# ============================================================================


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

    def download_object(self, bucket: str, object_name: str, file_path: str) -> bool:
        """Download object to file"""
        try:
            self.client.fget_object(bucket, object_name, file_path)
            logging.info(f"✓ Downloaded: {bucket}/{object_name} -> {file_path}")
            return True
        except S3Error as e:
            logging.error(f"✗ Download failed: {e}")
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


# ============================================================================
# POSTGRESQL DATABASE (MISSING CLASS - ADDED)
# ============================================================================


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

            # Initialize tables
            self._init_tables()
        except Exception as e:
            logging.error(f"✗ Database connection failed: {e}")
            raise

    def _init_tables(self):
        """Initialize required database tables"""
        try:
            with self.conn.cursor() as cur:
                # Create vehicle_classes table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vehicle_classes (
                        class_id INTEGER PRIMARY KEY,
                        class_name VARCHAR(50) UNIQUE NOT NULL,
                        entry_fee NUMERIC(10, 2) DEFAULT 0.00,
                        xray_fee NUMERIC(10, 2) DEFAULT 0.00,
                        total_fee NUMERIC(10, 2) DEFAULT 0.00
                    );
                """)

                # Create vehicle_transactions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vehicle_transactions (
                        id SERIAL PRIMARY KEY,
                        camera_id VARCHAR(50) NOT NULL,
                        track_id VARCHAR(100) NOT NULL,
                        class_id INT NOT NULL,
                        total_fee NUMERIC(10, 2) DEFAULT 0.00,
                        time_stamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        img_path TEXT,
                        confidence NUMERIC(5, 4),
                        FOREIGN KEY (class_id) REFERENCES vehicle_classes(class_id)
                    );
                """)

                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_camera_timestamp 
                    ON vehicle_transactions(camera_id, time_stamp DESC);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_track_id 
                    ON vehicle_transactions(track_id);
                """)

                # Insert vehicle class data
                vehicle_classes = [
                    (0, "car", 0.00, 0.00, 0.00),
                    (1, "other", 0.00, 0.00, 0.00),
                    (2, "other_truck", 100.00, 50.00, 150.00),
                    (3, "pickup_truck", 0.00, 0.00, 0.00),
                    (4, "truck_20_back", 100.00, 250.00, 350.00),
                    (5, "truck_20_front", 100.00, 250.00, 350.00),
                    (6, "truck_20x2", 100.00, 500.00, 600.00),
                    (7, "truck_40", 100.00, 350.00, 450.00),
                    (8, "truck_roro", 100.00, 50.00, 150.00),
                    (9, "truck_tail", 100.00, 50.00, 150.00),
                    (10, "motorcycle", 0.00, 0.00, 0.00),
                    (11, "truck_head", 100.00, 50.00, 150.00),
                ]

                for class_data in vehicle_classes:
                    cur.execute("""
                        INSERT INTO vehicle_classes (class_id, class_name, entry_fee, xray_fee, total_fee)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (class_id) DO NOTHING
                    """, class_data)

                self.conn.commit()
                logging.info("✓ Database tables initialized")

        except Exception as e:
            self.conn.rollback()
            logging.error(f"✗ Table initialization failed: {e}")
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


# ============================================================================
# PROCESSING SERVICE
# ============================================================================


class ProcessingService:
    """Service that processes tasks from Redis queue and fetches data from MinIO"""

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
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
        mlflow_tracking_uri: str = "http://mlflow-server:5000",
        model_uri: str = "models:/Truck_classification_Model/Production",
    ):
        """Initialize processing service"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize managers
        self.redis_manager = RedisQueueManager(
            host=redis_host, port=redis_port, db=0, queue_name="frame_batches"
        )
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
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)

            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})

            # Post-process outputs
            class_id, confidence = self._postprocess_outputs(outputs)

            # Get fee from database
            vehicle_info = self.db.get_vehicle_class(class_id)
            total_fee = vehicle_info["total_fee"] if vehicle_info else 0.00

            logging.info(
                f"🤖 Model inference: class_id={class_id} (confidence: {confidence:.4f}, fee: {total_fee})"
            )

            return class_id, total_fee, confidence
        except Exception as e:
            logging.error(f"✗ Inference failed: {e}")
            # Fallback to default class
            return 0, 0.0, 0.0

    def _preprocess_frame(self, frame_bgr: np.ndarray, input_size=(640, 640)) -> np.ndarray:
        """Preprocess frame for ONNX model"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Resize
        frame_resized = cv2.resize(frame_rgb, input_size)
        # Normalize to [0, 1]
        frame_norm = frame_resized.astype(np.float32) / 255.0
        # Convert to CHW format (Channel, Height, Width)
        frame_chw = np.transpose(frame_norm, (2, 0, 1))
        # Add batch dimension
        frame_chw = np.expand_dims(frame_chw, axis=0)
        return frame_chw

    def _postprocess_outputs(self, outputs) -> Tuple[int, float]:
        """Post-process ONNX model outputs"""
        # Adjust based on your model's output format
        # Example: YOLOv8 detection output [1, num_classes+4, num_predictions]
        output = outputs[0][0]  # [84, 8400] for YOLOv8
        
        # Extract boxes and class probabilities
        boxes = output[:4, :].T  # [8400, 4]
        class_probs = output[4:, :].T  # [8400, num_classes]
        
        # Get class with highest confidence
        class_ids = np.argmax(class_probs, axis=1)
        confidences = np.max(class_probs, axis=1)
        
        # Get best detection
        if len(confidences) > 0:
            best_idx = np.argmax(confidences)
            class_id = int(class_ids[best_idx])
            confidence = float(confidences[best_idx])
            
            # Ensure class_id is within valid range
            if class_id >= 12:
                class_id = 0  # Default to "car"
                
            return class_id, confidence
        
        return 0, 0.0  # No detection

    @staticmethod
    def _load_batch_from_file(file_path: str) -> np.ndarray:
        """Load a numpy batch file from disk."""
        return np.load(file_path)

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

    @staticmethod
    def _encode_jpeg(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
        """Encode PIL image to JPEG bytes with given quality."""
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def process_task(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process single task: fetch batch from MinIO, extract 1 frame, save as JPG"""
        try:
            logging.info(f"\n--- Processing Task: {task.task_id} ---")
            logging.info(
                f"Bucket: {task.minio_bucket}, Object: {task.object_key_or_prefix}"
            )

            # Check if minio_prefix is a full object name (file) or a prefix (directory)
            if task.object_key_or_prefix.endswith(".npy"):
                batch_object = task.object_key_or_prefix
                logging.info(f"Processing direct file: {batch_object}")
            else:
                # Prefix/directory - list objects
                listed_objects = self.minio_manager.list_objects(
                    bucket=task.minio_bucket, prefix=task.object_key_or_prefix
                )
                logging.info(
                    f"Found {len(listed_objects)} objects in {task.minio_bucket}/{task.object_key_or_prefix}"
                )

                # Filter out directories
                files = [obj for obj in listed_objects if not obj["name"].endswith("/")]

                if not files:
                    logging.warning(
                        f"No files found in {task.minio_bucket}/{task.object_key_or_prefix}"
                    )
                    return {
                        "status": "no_objects",
                        "task_id": task.task_id,
                        "camera_id": task.camera_id,
                        "objects_found": len(listed_objects),
                        "files_found": 0,
                    }

                batch_object = files[0]["name"]
                logging.info(f"Processing batch: {batch_object}")

            # Download numpy batch file
            batch_file = os.path.join(self.output_dir, os.path.basename(batch_object))

            success = self.minio_manager.download_object(
                bucket=task.minio_bucket,
                object_name=batch_object,
                file_path=batch_file,
            )

            if success:
                try:
                    batch_data = self._load_batch_from_file(batch_file)
                    logging.info(f"Loaded batch shape: {batch_data.shape}")

                    selected_frame, frame_idx = self._select_frame(batch_data)
                    logging.info(
                        f"Selected frame {frame_idx} with shape: {selected_frame.shape}"
                    )

                    frame_uint8 = self._normalize_to_uint8(selected_frame)

                    # Create PIL Image
                    if frame_uint8.ndim == 2:
                        image = Image.fromarray(frame_uint8, mode="L")
                    elif frame_uint8.shape[2] == 3:
                        image = Image.fromarray(frame_uint8, mode="RGB")
                    elif frame_uint8.shape[2] == 4:
                        image = Image.fromarray(frame_uint8, mode="RGBA").convert("RGB")
                    else:
                        raise ValueError(f"Unsupported image shape: {frame_uint8.shape}")

                    img_bytes = self._encode_jpeg(image, quality=JPEG_QUALITY)

                    # Create output filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{task.camera_id}_{task.task_id}_{timestamp}.jpg"

                    # Ensure bucket exists
                    self.minio_manager.create_bucket(PROCESSED_BUCKET_NAME)

                    # Upload to MinIO
                    upload_success = self.minio_manager.upload_from_bytes(
                        bucket=PROCESSED_BUCKET_NAME,
                        object_name=output_filename,
                        data=img_bytes,
                        content_type="image/jpeg",
                    )

                    if upload_success:
                        # Delete original batch file from MinIO
                        delete_success = self.minio_manager.delete_object(
                            bucket=task.minio_bucket, object_name=batch_object
                        )

                        # Clean up local file
                        if os.path.exists(batch_file):
                            os.remove(batch_file)
                            logging.info(f"✓ Cleaned up local file: {batch_file}")

                        # Run inference
                        class_id, total_fee, confidence = self._run_inference(frame_uint8)

                        # Create transaction record
                        transaction = VehicleTransaction(
                            camera_id=task.camera_id,
                            track_id=task.task_id,
                            class_id=class_id,
                            total_fee=total_fee,
                            time_stamp=task.timestamp
                            or datetime.datetime.now(datetime.timezone.utc),
                            img_path=f"{PROCESSED_BUCKET_NAME}/{output_filename}",
                            confidence=confidence,
                        )

                        # Save to database
                        self.db.insert_transaction(transaction)

                        return {
                            "status": "success",
                            "task_id": task.task_id,
                            "camera_id": task.camera_id,
                            "batch_file": batch_object,
                            "batch_deleted": delete_success,
                            "frame_selected": frame_idx,
                            "output_image": f"{PROCESSED_BUCKET_NAME}/{output_filename}",
                            "image_size_bytes": len(img_bytes),
                            "transaction": transaction.to_dict(),
                        }
                except Exception as e:
                    logging.error(f"✗ Batch processing failed: {e}")
                    return {
                        "status": "error",
                        "task_id": task.task_id,
                        "error": f"Batch processing error: {str(e)}",
                    }

            return {
                "status": "download_failed",
                "task_id": task.task_id,
                "camera_id": task.camera_id,
            }

        except Exception as e:
            logging.error(f"✗ Task processing failed: {e}")
            return {
                "status": "error",
                "task_id": task.task_id,
                "error": str(e),
            }

    def run_worker(self):
        """Run as continuous worker, processing tasks from queue"""
        logging.info("=" * 70)
        logging.info("Processing Worker Started")
        logging.info("=" * 70)
        logging.info("Waiting for tasks from Redis queue...")
        logging.info("Press Ctrl+C to stop")

        try:
            while True:
                # Block and wait for tasks (timeout 5 seconds)
                task = self.redis_manager.pop_task(timeout=5)
                if task:
                    logging.info(f"\n{'=' * 70}")
                    logging.info(f"New task received: {task.task_id}")
                    result = self.process_task(task)

                    if result["status"] == "success":
                        logging.info("✓ Task completed successfully")
                        logging.info(f"  Image saved: {result['output_image']}")
                    elif result["status"] == "no_objects":
                        logging.warning(f"⚠ No objects found for task {task.task_id}")
                    else:
                        logging.error(
                            f"✗ Task failed: {result.get('error', 'Unknown error')}"
                        )

                    logging.info(f"{'=' * 70}")

        except KeyboardInterrupt:
            logging.info("\n\nWorker stopped by user")
            self.print_summary()
        except Exception as e:
            logging.error(f"Worker error: {e}")
            raise

    def print_summary(self):
        """Print current state summary"""
        queue_size = self.redis_manager.get_queue_size()
        transaction_count = self.db.get_transaction_count()
        logging.info(f"\n{'=' * 70}")
        logging.info("SERVICE SUMMARY")
        logging.info(f"{'=' * 70}")
        logging.info(f"Redis Queue Size: {queue_size} tasks")
        logging.info(f"Processed Transactions: {transaction_count}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main entry point - runs as processing worker"""

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

    # Get configuration from environment variables
    redis_host = os.getenv("REDIS_HOST")
    redis_port = int(os.getenv("REDIS_PORT"))
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # Database configuration
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT"))
    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")

    # MLflow configuration
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    model_uri = os.getenv("MODEL_URI", "models:/Truck_classification_Model/Production")

    try:
        # Initialize processing service
        service = ProcessingService(
            redis_host=redis_host,
            redis_port=redis_port,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
            minio_secure=minio_secure,
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_uri=model_uri,
        )

        # Run as worker
        service.run_worker()

    except Exception as e:
        logging.error(f"Failed to start processing service: {e}")
        raise


if __name__ == "__main__":
    main()