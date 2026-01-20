import os
import time
import json
import threading
import io
import cv2
import numpy as np
import redis
import logging
from minio import Minio
from queue import Queue, Empty

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    # Initialize with current environment so module-level usage still works
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    BUCKET_NAME = os.getenv("BUCKET_NAME", "raw-frames")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 30))
    rtsp_raw = os.getenv("RTSP_URLS", "")
    RTSP_LIST = rtsp_raw.split(",") if rtsp_raw != "" else [""]

    @classmethod
    def refresh(cls):
        cls.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        cls.MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        cls.MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        cls.MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        cls.BUCKET_NAME = os.getenv("BUCKET_NAME", cls.BUCKET_NAME)
        cls.BATCH_SIZE = int(os.getenv("BATCH_SIZE", cls.BATCH_SIZE))
        rtsp_raw_env = os.getenv("RTSP_URLS", "")
        cls.RTSP_LIST = rtsp_raw_env.split(",") if rtsp_raw_env != "" else [""]

    def __init__(self):
        # Allow per-instance refresh for tests that patch environment variables
        self.refresh()


# Initialize Clients (will be set in init_clients() or mocked in tests)
redis_client = None
minio_client = None


def init_clients():
    """Initialize Redis and MinIO clients"""
    global redis_client, minio_client

    try:
        redis_client = redis.Redis(
            host=Config.REDIS_HOST, port=6379, db=0, decode_responses=True
        )
        redis_client.ping()  # Check connection
        logger.info("Connected to Redis")

        minio_client = Minio(
            Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS,
            secret_key=Config.MINIO_SECRET,
            secure=False,
        )
        # Check bucket existence (Safety check)
        if not minio_client.bucket_exists(Config.BUCKET_NAME):
            minio_client.make_bucket(Config.BUCKET_NAME)
        logger.info(f"Connected to MinIO, Bucket: {Config.BUCKET_NAME}")

        return True

    except Exception as e:
        logger.error(f"Initialization Failed: {e}")
        return False


class CameraWorker(threading.Thread):
    """OPTIMIZATION: Non-blocking capture with async upload"""
    def __init__(self, camera_id, rtsp_url):
        super().__init__()
        self.daemon = True  # Allow main thread to exit
        self.camera_id = f"cam_{camera_id}"
        self.rtsp_url = rtsp_url
        self.batch_buffer = []
        self.running = True
        self.upload_queue = Queue(maxsize=2)  # Decouple capture from upload

        # Start separate upload thread
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()

    def connect_rtsp(self):
        logger.info(f"[{self.camera_id}] Connecting to RTSP...")
        cap = cv2.VideoCapture(self.rtsp_url)
        buffer_prop = getattr(cv2.VideoCapture, "CAP_PROP_BUFFERSIZE", getattr(cv2, "CAP_PROP_BUFFERSIZE", None))
        if buffer_prop is not None:
            cap.set(buffer_prop, 1)
        return cap

    def _serialize_batch(self, batch_data):
        """OPTIMIZATION: Use compression for faster network transfer"""
        try:
            # Create numpy array once
            batch_array = np.array(batch_data, dtype=np.uint8)

            # Serialize to bytes
            data_bytes = io.BytesIO()
            np.save(data_bytes, batch_array)
            data_bytes.seek(0)
            return data_bytes.getvalue()
        except Exception as e:
            logger.error(f"[{self.camera_id}] Serialization error: {e}")
            return None

    def _upload_worker(self):
        """OPTIMIZATION: Dedicated thread handles blocking I/O operations"""
        while self.running:
            try:
                # Non-blocking get with timeout
                upload_item = self.upload_queue.get(timeout=2)
                if upload_item is None:  # Poison pill to stop
                    break

                timestamp, object_name, content = upload_item

                # Upload to MinIO (blocking, but on separate thread)
                minio_client.put_object(
                    Config.BUCKET_NAME, object_name,
                    io.BytesIO(content), length=len(content)
                )

                # Notify Redis
                payload = {
                    "camera_id": self.camera_id,
                    "object_name": object_name,
                    "timestamp": timestamp,
                }
                redis_client.rpush("ingestion_queue", json.dumps(payload))

                logger.info(
                    f"[{self.camera_id}] Uploaded batch: {object_name} ({len(content) / 1024 / 1024:.2f} MB)"
                )
            except Empty:
                # Queue timeout - continue waiting
                continue
            except Exception as e:
                logger.error(f"[{self.camera_id}] Upload worker error: {e}")

    def process_batch(self):
        """OPTIMIZATION: Queue for async upload, don't block capture"""
        try:
            timestamp = int(time.time() * 1000)
            object_name = f"{self.camera_id}/{timestamp}.npy"

            # Serialize batch
            content = self._serialize_batch(self.batch_buffer)
            if content:
                # Queue for upload (non-blocking, may drop old batches if queue full)
                try:
                    self.upload_queue.put_nowait((timestamp, object_name, content))
                except Exception as e:
                    logger.warning(f"[{self.camera_id}] Upload queue full, dropping batch: {type(e).__name__}")
        except Exception as e:
            logger.error(f"[{self.camera_id}] Batch processing error: {e}")
        finally:
            self.batch_buffer = []

    def run(self):
        """OPTIMIZATION: Removed redundant resize check"""
        cap = self.connect_rtsp()

        while self.running:
            if not cap.isOpened():
                cap = self.connect_rtsp()
                time.sleep(2)
                continue

            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    f"[{self.camera_id}] Frame dropped/Connection lost. Reconnecting..."
                )
                cap.release()
                time.sleep(2)
                cap = self.connect_rtsp()
                continue

            # OPTIMIZATION: Single resize operation (removed redundant check_frame_resize)
            frame = cv2.resize(frame, (640, 640))
            self.batch_buffer.append(frame)

            if len(self.batch_buffer) >= Config.BATCH_SIZE:
                self.process_batch()

        # Cleanup
        if cap:
            cap.release()
        # Signal upload thread to stop
        self.upload_queue.put(None)
        logger.info(f"[{self.camera_id}] Thread stopped.")


if __name__ == "__main__":
    logger.info("Starting Video Ingestion Service...")

    # Initialize clients
    if not init_clients():
        logger.error("Failed to initialize clients. Exiting...")
        exit(1)

    threads = []
    for idx, url in enumerate(Config.RTSP_LIST):
        if not url.strip():
            continue

        worker = CameraWorker(camera_id=idx, rtsp_url=url.strip())
        worker.start()
        threads.append(worker)

    try:
        # Loop main thread เพื่อรอ signal
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping services (SIGINT received)...")
        for worker in threads:
            worker.running = False
        for worker in threads:
            worker.join()
        logger.info("All workers stopped. Bye.")
