"""
Mock Database Writer - Simulates writing vehicle transaction data
Generates fake data matching the vehicle_transactions table schema
Integrates with Redis Queue and MinIO for data pipeline
"""

import datetime
import json
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Dict, List, Optional

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
class VehicleClassRecord:
    """Represents a vehicle class reference record"""

    class_id: int
    class_name: str
    entry_fee: Optional[float] = None
    xray_fee: Optional[float] = None
    total_fee: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for database insert"""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "entry_fee": round(self.entry_fee, 2) if self.entry_fee else None,
            "xray_fee": round(self.xray_fee, 2) if self.xray_fee else None,
            "total_fee": round(self.total_fee, 2) if self.total_fee else None,
        }

    def __repr__(self) -> str:
        return (
            f"VehicleClassRecord(class_id={self.class_id}, "
            f"class_name={self.class_name}, total_fee={self.total_fee})"
        )


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
    minio_prefix: str
    timestamp: datetime.datetime = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "camera_id": self.camera_id,
            "video_file": self.video_file,
            "minio_bucket": self.minio_bucket,
            "minio_prefix": self.minio_prefix,
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

            # Handle ingestion service structure:
            # {
            #   "camera_id": "...",
            #   "object_name": "camera_id/batch_xxx.npy",
            #   "timestamp": "...",
            #   "bucket_name": "video-frames",
            #   "batch_size": 30,
            #   "frame_shape": [...]
            # }

            return cls(
                task_id=data.get("task_id")
                or data.get("batch_id")
                or data.get("object_name", "unknown"),
                camera_id=data.get("camera_id", "unknown"),
                video_file=data.get("video_file") or data.get("video_path", ""),
                minio_bucket=data.get("minio_bucket")
                or data.get("bucket_name")
                or data.get("bucket", "video-frames"),
                minio_prefix=data.get("minio_prefix")
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
        queue_name: str = "frame_batches",
    ):
        """Initialize Redis connection"""
        self.host = host
        self.port = port
        self.db = db
        self.queue_name = queue_name
        try:
            self.client = redis.Redis(
                host=host, port=port, db=db, decode_responses=True
            )
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
# MINIO STORAGE MANAGER
# ============================================================================


class MinIOManager:
    """Manages MinIO object storage operations"""

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False,
    ):
        """Initialize MinIO connection"""
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        try:
            self.client = Minio(
                endpoint, access_key=access_key, secret_key=secret_key, secure=secure
            )
            # Test connection
            self.client.bucket_exists("test") or True
            logging.info(f"✓ MinIO connected: {endpoint}")
        except Exception as e:
            logging.error(f"✗ MinIO connection failed: {e}")
            raise

    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        """List objects in bucket with optional prefix"""
        try:
            objects = []
            for obj in self.client.list_objects(bucket, prefix=prefix):
                objects.append(
                    {
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified.isoformat()
                        if obj.last_modified
                        else None,
                    }
                )
            logging.info(f"✓ Listed {len(objects)} objects in {bucket}/{prefix}")
            return objects
        except S3Error as e:
            logging.error(f"✗ List objects failed: {e}")
            return []

    def download_object(self, bucket: str, object_name: str, file_path: str) -> bool:
        """Download object from MinIO to local file"""
        try:
            self.client.fget_object(bucket, object_name, file_path)
            logging.info(f"✓ Downloaded {object_name} from {bucket} to {file_path}")
            return True
        except S3Error as e:
            logging.error(f"✗ Download failed: {e}")
            return False

    def upload_object(self, bucket: str, object_name: str, file_path: str) -> bool:
        """Upload file to MinIO"""
        try:
            self.client.fput_object(bucket, object_name, file_path)
            logging.info(f"✓ Uploaded {file_path} to {bucket}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"✗ Upload failed: {e}")
            return False

    def get_object_data(self, bucket: str, object_name: str) -> Optional[bytes]:
        """Get object data as bytes"""
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            logging.info(f"✓ Retrieved {object_name} from {bucket}")
            return data
        except S3Error as e:
            logging.error(f"✗ Get object failed: {e}")
            return None

    def create_bucket(self, bucket: str) -> bool:
        """Create bucket if not exists"""
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logging.info(f"✓ Bucket created: {bucket}")
            return True
        except S3Error as e:
            logging.error(f"✗ Create bucket failed: {e}")
            return False

    def delete_object(self, bucket: str, object_name: str) -> bool:
        """Delete object from MinIO"""
        try:
            self.client.remove_object(bucket, object_name)
            logging.info(f"✓ Deleted {object_name} from {bucket}")
            return True
        except S3Error as e:
            logging.error(f"✗ Delete object failed: {e}")
            return False

    def upload_from_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "image/jpeg",
    ) -> bool:
        """Upload bytes data to MinIO"""
        try:
            data_stream = BytesIO(data)
            self.client.put_object(
                bucket,
                object_name,
                data_stream,
                length=len(data),
                content_type=content_type,
            )
            logging.info(f"✓ Uploaded {len(data)} bytes to {bucket}/{object_name}")
            return True
        except S3Error as e:
            logging.error(f"✗ Upload from bytes failed: {e}")
            return False


# ============================================================================
# POSTGRESQL DATABASE
# ============================================================================


class PostgreSQLDatabase:
    """PostgreSQL database operations for vehicle transactions and classes"""

    # Vehicle class pricing reference (matches database schema)
    VEHICLE_CLASSES_REF = {
        1: {
            "class_name": "car",
            "entry_fee": 0.00,
            "xray_fee": 0.00,
            "total_fee": 0.00,
        },
        2: {
            "class_name": "other",
            "entry_fee": 0.00,
            "xray_fee": 0.00,
            "total_fee": 0.00,
        },
        3: {
            "class_name": "other_truck",
            "entry_fee": 100.00,
            "xray_fee": 50.00,
            "total_fee": 150.00,
        },
        4: {
            "class_name": "pickup_truck",
            "entry_fee": 0.00,
            "xray_fee": 0.00,
            "total_fee": 0.00,
        },
        5: {
            "class_name": "truck_20_back",
            "entry_fee": 100.00,
            "xray_fee": 250.00,
            "total_fee": 350.00,
        },
        6: {
            "class_name": "truck_20_front",
            "entry_fee": 100.00,
            "xray_fee": 250.00,
            "total_fee": 350.00,
        },
        7: {
            "class_name": "truck_20x2",
            "entry_fee": 100.00,
            "xray_fee": 500.00,
            "total_fee": 600.00,
        },
        8: {
            "class_name": "truck_40",
            "entry_fee": 100.00,
            "xray_fee": 350.00,
            "total_fee": 450.00,
        },
        9: {
            "class_name": "truck_roro",
            "entry_fee": 100.00,
            "xray_fee": 50.00,
            "total_fee": 150.00,
        },
        10: {
            "class_name": "truck_tail",
            "entry_fee": 100.00,
            "xray_fee": 50.00,
            "total_fee": 150.00,
        },
        11: {
            "class_name": "motorcycle",
            "entry_fee": 0.00,
            "xray_fee": 0.00,
            "total_fee": 0.00,
        },
        12: {
            "class_name": "truck_head",
            "entry_fee": 100.00,
            "xray_fee": 50.00,
            "total_fee": 150.00,
        },
    }

    def __init__(
        self,
        host: str = "db",
        port: int = 5432,
        database: str = "vehicle_db",
        user: str = "postgres",
        password: str = "postgres123",
    ):
        """Initialize PostgreSQL connection"""
        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        self.transaction_count = 0

        # Test connection
        try:
            conn = self._get_connection()
            conn.close()
            logging.info(f"✓ PostgreSQL connected: {host}:{port}/{database}")
        except Exception as e:
            logging.error(f"✗ PostgreSQL connection failed: {e}")
            raise

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params)

    def get_transaction_count(self) -> int:
        """Get total number of transactions"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM vehicle_transactions")
            count = cur.fetchone()[0]

            cur.close()
            conn.close()

            return count

        except Exception as e:
            logging.error(f"✗ Get transaction count failed: {e}")
            return 0

    def insert_transaction(self, transaction: VehicleTransaction) -> bool:
        """Insert a single transaction record"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            record = transaction.to_dict()

            # Skip if class_id is 0 (not yet classified)
            if record["class_id"] == 0:
                logging.warning("Skipping insert: class_id is 0 (not yet classified)")
                return False

            sql = """
                INSERT INTO vehicle_transactions
                (camera_id, track_id, class_id, total_fee, time_stamp, img_path, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            cur.execute(
                sql,
                (
                    record["camera_id"],
                    record["track_id"],
                    record["class_id"],
                    record["total_fee"],
                    record["time_stamp"],
                    record["img_path"],
                    record["confidence"],
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            self.transaction_count += 1
            logging.info(f"✓ Transaction inserted: {record['track_id']}")
            return True

        except Exception as e:
            logging.error(f"✗ Insert failed: {e}")
            if "conn" in locals():
                conn.rollback()
                conn.close()
            return False

    def get_vehicle_class(self, class_id: int) -> Optional[Dict]:
        """Get vehicle class reference data"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(
                "SELECT * FROM vehicle_classes WHERE class_id = %s", (class_id,)
            )

            result = cur.fetchone()
            cur.close()
            conn.close()

            return dict(result) if result else None

        except Exception as e:
            logging.error(f"✗ Get vehicle class failed: {e}")
            return None

    def get_all_vehicle_classes(self) -> List[Dict]:
        """Get all vehicle classes"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute("SELECT * FROM vehicle_classes ORDER BY class_id")

            results = cur.fetchall()
            cur.close()
            conn.close()

            return [dict(r) for r in results]

        except Exception as e:
            logging.error(f"✗ Get all vehicle classes failed: {e}")
            return []

    def get_records_by_camera(self, camera_id: str) -> List[Dict]:
        """Query records by camera_id"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(
                "SELECT * FROM vehicle_transactions WHERE camera_id = %s ORDER BY time_stamp DESC",
                (camera_id,),
            )

            results = cur.fetchall()
            cur.close()
            conn.close()

            return [dict(r) for r in results]

        except Exception as e:
            logging.error(f"✗ Get records by camera failed: {e}")
            return []

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get total records and fees
            cur.execute("""
                SELECT
                    COUNT(*) as total_records,
                    COALESCE(SUM(total_fee), 0) as total_fee_collected,
                    COALESCE(AVG(confidence), 0) as avg_confidence
                FROM vehicle_transactions
            """)

            stats = dict(cur.fetchone())

            # Get vehicle counts by class
            cur.execute("""
                SELECT class_id, COUNT(*) as count
                FROM vehicle_transactions
                GROUP BY class_id
                ORDER BY class_id
            """)

            vehicle_counts = {}
            for row in cur.fetchall():
                # Map class_id to class name
                for vc in VehicleClass:
                    if vc.value == row["class_id"]:
                        vehicle_counts[vc.name] = row["count"]
                        break

            cur.close()
            conn.close()

            return {
                "total_records": stats["total_records"],
                "total_fee_collected": round(float(stats["total_fee_collected"]), 2),
                "vehicle_counts": vehicle_counts,
                "avg_confidence": round(float(stats["avg_confidence"]), 4)
                if stats["avg_confidence"]
                else 0.0,
            }

        except Exception as e:
            logging.error(f"✗ Get summary stats failed: {e}")
            return {"total_records": 0}


# ============================================================================
# PROCESSING SERVICE - Redis Queue + MinIO Integration
# ============================================================================


class ProcessingService:
    """Service that processes tasks from Redis queue and fetches data from MinIO"""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        minio_endpoint: str = "localhost:9000",
        minio_access_key: str = "minioadmin",
        minio_secret_key: str = "minioadmin",
        minio_secure: bool = False,
        db_host: str = "db",
        db_port: int = 5432,
        db_name: str = "vehicle_db",
        db_user: str = "postgres",
        db_password: str = "postgres123",
        output_dir: str = "./processed_data",
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
        logging.info("✓ ProcessingService initialized")

    def _generate_fake_classification(self) -> tuple:
        """Generate fake classification result for testing"""
        # Randomly select a vehicle class (weighted towards trucks)
        vehicle_classes = [
            (VehicleClass.CAR, 0.15),
            (VehicleClass.OTHER, 0.05),
            (VehicleClass.OTHER_TRUCK, 0.10),
            (VehicleClass.PICKUP_TRUCK, 0.10),
            (VehicleClass.TRUCK_20_BACK, 0.15),
            (VehicleClass.TRUCK_20_FRONT, 0.15),
            (VehicleClass.TRUCK_20X2, 0.10),
            (VehicleClass.TRUCK_40, 0.10),
            (VehicleClass.TRUCK_RORO, 0.03),
            (VehicleClass.TRUCK_TAIL, 0.03),
            (VehicleClass.MOTORCYCLE, 0.02),
            (VehicleClass.TRUCK_HEAD, 0.02),
        ]

        classes, weights = zip(*vehicle_classes)
        selected_class = random.choices(classes, weights=weights, k=1)[0]

        # Generate confidence score (higher for common classes)
        if selected_class in [
            VehicleClass.CAR,
            VehicleClass.TRUCK_20_BACK,
            VehicleClass.TRUCK_20_FRONT,
        ]:
            confidence = random.uniform(0.85, 0.99)
        else:
            confidence = random.uniform(0.70, 0.95)

        # Get vehicle class info from database
        vehicle_info = self.db.get_vehicle_class(selected_class.value)
        total_fee = vehicle_info["total_fee"] if vehicle_info else 0.00

        logging.info(
            f"🎲 Generated fake classification: {selected_class.name} (confidence: {confidence:.4f}, fee: {total_fee})"
        )

        return selected_class.value, total_fee, confidence

    def process_task(self, task: ProcessingTask) -> Dict:
        """Process single task: fetch batch from MinIO, extract 1 frame, save as JPG"""
        try:
            logging.info(f"\n--- Processing Task: {task.task_id} ---")
            logging.info(f"Bucket: {task.minio_bucket}, Object: {task.minio_prefix}")

            # Check if minio_prefix is a full object name (file) or a prefix (directory)
            # If it looks like a file path (contains .npy), use it directly
            if task.minio_prefix.endswith(".npy"):
                # Direct file path
                batch_object = task.minio_prefix
                logging.info(f"Processing direct file: {batch_object}")
            else:
                # Prefix/directory - list objects
                objects = self.minio_manager.list_objects(
                    bucket=task.minio_bucket, prefix=task.minio_prefix
                )
                logging.info(
                    f"Found {len(objects)} objects in {task.minio_bucket}/{task.minio_prefix}"
                )

                # Filter out directories (objects ending with /)
                file_objects = [obj for obj in objects if not obj["name"].endswith("/")]

                if not file_objects:
                    logging.warning(
                        f"No files found, only directories in {task.minio_bucket}/{task.minio_prefix}"
                    )
                    return {
                        "status": "no_objects",
                        "task_id": task.task_id,
                        "camera_id": task.camera_id,
                        "objects_found": len(objects),
                        "files_found": 0,
                    }

                batch_object = file_objects[0]["name"]
                logging.info(f"Processing batch: {batch_object}")

            # Download numpy batch file
            batch_file = os.path.join(self.output_dir, os.path.basename(batch_object))

            success = self.minio_manager.download_object(
                bucket=task.minio_bucket,
                object_name=batch_object,
                file_path=batch_file,
            )

            if success:
                # Load numpy batch (30 frames)
                try:
                    batch_data = np.load(batch_file)
                    logging.info(f"Loaded batch shape: {batch_data.shape}")

                    # Select 1 frame from batch (middle frame)
                    if (
                        len(batch_data.shape) == 4
                    ):  # (batch_size, height, width, channels)
                        frame_idx = len(batch_data) // 2  # Select middle frame
                        selected_frame = batch_data[frame_idx]
                    else:  # If single image
                        selected_frame = batch_data
                        frame_idx = 0

                        logging.info(
                            f"Selected frame {frame_idx} with shape: {selected_frame.shape}"
                        )

                    # Convert to PIL Image
                    # Handle different data types
                    if (
                        selected_frame.dtype == np.float32
                        or selected_frame.dtype == np.float64
                    ):
                        # Normalize to 0-255 if float
                        if selected_frame.max() <= 1.0:
                            selected_frame = (selected_frame * 255).astype(np.uint8)
                        else:
                            selected_frame = selected_frame.astype(np.uint8)
                    elif selected_frame.dtype != np.uint8:
                        selected_frame = selected_frame.astype(np.uint8)

                    # Create PIL Image
                    if len(selected_frame.shape) == 2:  # Grayscale
                        image = Image.fromarray(selected_frame, mode="L")
                    elif selected_frame.shape[2] == 3:  # RGB
                        image = Image.fromarray(selected_frame, mode="RGB")
                    elif selected_frame.shape[2] == 4:  # RGBA
                        image = Image.fromarray(selected_frame, mode="RGBA")
                        image = image.convert("RGB")  # Convert to RGB for JPEG
                    else:
                        raise ValueError(
                            f"Unsupported image shape: {selected_frame.shape}"
                        )

                    # Save to JPEG bytes
                    img_buffer = BytesIO()
                    image.save(img_buffer, format="JPEG", quality=95)
                    img_bytes = img_buffer.getvalue()

                    # Create output filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{task.camera_id}_{task.task_id}_{timestamp}.jpg"

                    # Ensure process_frames bucket exists
                    self.minio_manager.create_bucket("process-frames")

                    # Upload to process_frames bucket
                    upload_success = self.minio_manager.upload_from_bytes(
                        bucket="process-frames",
                        object_name=output_filename,
                        data=img_bytes,
                        content_type="image/jpeg",
                    )

                    if upload_success:
                        # Delete original batch file from MinIO
                        delete_success = self.minio_manager.delete_object(
                            bucket=task.minio_bucket, object_name=batch_object
                        )

                        # Clean up local batch file
                        if os.path.exists(batch_file):
                            os.remove(batch_file)
                            logging.info(f"✓ Cleaned up local file: {batch_file}")

                        # Generate fake classification result
                        class_id, total_fee, confidence = (
                            self._generate_fake_classification()
                        )

                        # Create transaction record
                        transaction = VehicleTransaction(
                            camera_id=task.camera_id,
                            track_id=task.task_id,
                            class_id=class_id,
                            total_fee=total_fee,
                            time_stamp=task.timestamp
                            or datetime.datetime.now(datetime.timezone.utc),
                            img_path=f"process-frames/{output_filename}",
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
                            "output_image": f"process-frames/{output_filename}",
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
                "status": "no_objects",
                "task_id": task.task_id,
                "camera_id": task.camera_id,
                "objects_found": len(objects),
                "files_found": len(file_objects) if "file_objects" in locals() else 0,
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
                else:
                    # No task available, continue waiting
                    pass

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

    # Get configuration from environment variables
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # Database configuration
    db_host = os.getenv("DB_HOST", "db")
    db_port = int(os.getenv("DB_PORT", 5432))
    db_name = os.getenv("POSTGRES_DB", "vehicle_db")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres123")

    try:
        # Initialize processing service
        service = ProcessingService(
            redis_host=redis_host,
            redis_port=redis_port,
            minio_endpoint=minio_endpoint,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_secure=minio_secure,
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
        )

        # Run as worker
        service.run_worker()

    except Exception as e:
        logging.error(f"Failed to start processing service: {e}")
        logging.error("Make sure Redis and MinIO are running and accessible")
        raise


if __name__ == "__main__":
    main()
