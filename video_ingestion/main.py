import cv2
import time
import os
import uuid
import json
import numpy as np
import redis
import logging

# --- Setup Logging (สำคัญสำหรับคนทำงานร่วมกัน) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load Config from .env ---
RTSP_URL = os.getenv("RTSP_URL", "0") # Default เป็น 0 (Webcam)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME = os.getenv("QUEUE_NAME", "video_frames_queue")
SHARED_PATH = os.getenv("SHARED_PATH", "/dev/shm")
IMG_SIZE = int(os.getenv("IMG_SIZE", 640))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 50))

# --- Redis Connection ---
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    r.ping()
    logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.error(f"❌ Redis Connection Error: {e}")
    exit(1)

def run_ingestion():
    logger.info(f"🚀 Starting Ingestion: Source={RTSP_URL}, Size={IMG_SIZE}x{IMG_SIZE}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    # ถ้าเป็นไฟล์วิดีโอ (ไม่ใช่ stream) อาจต้องวน Loop เล่นซ้ำ
    # แต่ถ้าเป็น RTSP กล้องจริง มันจะมาเรื่อยๆ

    while True:
        ret, frame = cap.read()
        
        if not ret:
            logger.warning("⚠️ No frame / Camera disconnected. Retrying in 2s...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        # 1. Backpressure Check
        q_len = r.llen(QUEUE_NAME)
        if q_len >= MAX_QUEUE_SIZE:
            # ใช้ logging.debug หรือไม่ต้อง print บ่อยๆ เพื่อไม่ให้รก
            # logger.warning(f"🛑 Queue full ({q_len}). Dropping frame.") 
            time.sleep(0.05) # รอแป๊บนึงค่อยวนใหม่
            continue

        # 2. Resize
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # 3. Save to Shared Memory
        frame_id = str(uuid.uuid4())
        file_name = f"{frame_id}.npy"
        file_path = os.path.join(SHARED_PATH, file_name)

        try:
            np.save(file_path, resized)
        except Exception as e:
            logger.error(f"❌ Write Error: {e}")
            continue

        # 4. Notify Redis
        message = {
            "id": frame_id,
            "file_path": file_path,
            "shape": resized.shape,
            "dtype": str(resized.dtype),
            "timestamp": time.time()
        }
        
        r.rpush(QUEUE_NAME, json.dumps(message))
        # logger.info(f"Sent frame {frame_id}") # เปิดเมื่อต้องการ debug

if __name__ == "__main__":
    run_ingestion()