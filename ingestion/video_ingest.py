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
from datetime import datetime

# Setup Logging (เพื่อให้เห็นชัดๆ ใน Docker logs)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    BUCKET_NAME = "raw-frames"
    BATCH_SIZE = 30
    # ใช้ Get เพื่อกัน Error กรณีลืมใส่ Env
    RTSP_LIST = os.getenv("RTSP_URLS", "").split(",")


# Initialize Clients
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

except Exception as e:
    logger.error(f"Initialization Failed: {e}")
    exit(1)  # ปิดโปรแกรมทันทีถ้าต่อ Infra ไม่ได้


class CameraWorker(threading.Thread):
    def __init__(self, camera_id, rtsp_url):
        super().__init__()
        self.camera_id = f"cam_{camera_id}"
        self.rtsp_url = rtsp_url
        self.batch_buffer = []
        self.running = True

    def connect_rtsp(self):
        logger.info(f"[{self.camera_id}] Connecting to RTSP...")
        cap = cv2.VideoCapture(self.rtsp_url)
        # ตั้งค่า Buffer size ให้เล็กที่สุดเพื่อลด Latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def process_batch(self):
        try:
            timestamp = int(time.time())
            object_name = f"{self.camera_id}/{timestamp}.npy"

            # Convert to Numpy
            batch_array = np.array(
                self.batch_buffer, dtype=np.uint8
            )  # ระบุ Type เพื่อประหยัดเมม

            # Serialize
            data_bytes = io.BytesIO()
            np.save(data_bytes, batch_array)
            data_bytes.seek(0)  # Reset pointer
            content = data_bytes.getvalue()

            # Upload
            self.upload_to_minio(content, object_name)

            # Notify
            self.notify_redis(object_name, timestamp)

            logger.info(
                f"[{self.camera_id}] Processed batch: {object_name} ({len(content) / 1024 / 1024:.2f} MB)"
            )

        except Exception as e:
            logger.error(f"[{self.camera_id}] Batch processing error: {e}")
        finally:
            self.batch_buffer = []  # Clear buffer always

    def upload_to_minio(self, data, object_name):
        minio_client.put_object(
            Config.BUCKET_NAME, object_name, io.BytesIO(data), length=len(data)
        )

    def notify_redis(self, object_name, timestamp):
        payload = {
            "camera_id": self.camera_id,
            "object_name": object_name,
            "timestamp": timestamp,
        }
        redis_client.rpush("ingestion_queue", json.dumps(payload))

    def run(self):
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
                time.sleep(2)  # รอสักนิดก่อนต่อใหม่
                cap = self.connect_rtsp()
                continue

            # Resize: ลดขนาดเพื่อประหยัด Bandwidth (ปรับตามความเหมาะสม)
            # แนะนำให้ลอง 320x320 หรือ 640x640 ดู Load ของ Network
            frame = cv2.resize(frame, (640, 640))

            self.batch_buffer.append(frame)

            if len(self.batch_buffer) >= Config.BATCH_SIZE:
                self.process_batch()

        # Cleanup เมื่อหยุดลูป
        if cap:
            cap.release()
        logger.info(f"[{self.camera_id}] Thread stopped.")


if __name__ == "__main__":
    logger.info("Starting Video Ingestion Service...")

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
