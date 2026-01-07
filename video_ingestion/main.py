import cv2
import time
import os
import uuid
import json
import numpy as np
import redis
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Config (ดึงจาก Environment ที่เราตั้งใน docker-compose)
RTSP_URL = os.getenv("RTSP_URL", "/app/video.mp4")
PLAY_LOOP = os.getenv("PLAY_LOOP", "true") == "true"
QUEUE_NAME = os.getenv("QUEUE_NAME", "video_frames_queue")
SHARED_PATH = "/dev/shm"

# Connect Redis
try:
    r = redis.Redis(host="redis", port=6379, db=0)
    r.ping()
    logger.info("✅ Redis Connected!")
except Exception as e:
    logger.error(f"❌ Redis Failed: {e}")
    exit(1)

def run():
    # 1. เช็คไฟล์ก่อนเริ่ม (เหมือนตัวเทส)
    if not os.path.exists(RTSP_URL):
        logger.error(f"❌ CRITICAL: File not found at {RTSP_URL}")
        return

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logger.error("❌ CRITICAL: OpenCV cannot open file.")
        return

    logger.info("🚀 System Started. Processing frames...")
    
    while True:
        ret, frame = cap.read()
        
        # ถ้าจบไฟล์ -> วนลูป
        if not ret:
            if PLAY_LOOP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # ถ้าคิวเต็ม -> รอ
        if r.llen(QUEUE_NAME) >= 50:
            time.sleep(0.01)
            continue

        # Process: Resize -> Save .npy -> Send Redis
        try:
            # Resize
            resized = cv2.resize(frame, (640, 640))
            
            # Save to RAM (/dev/shm)
            frame_id = str(uuid.uuid4())
            file_path = os.path.join(SHARED_PATH, f"{frame_id}.npy")
            np.save(file_path, resized)

            # Send Notification to Redis
            msg = {
                "id": frame_id,
                "file_path": file_path,
                "shape": resized.shape,
                "timestamp": time.time()
            }
            r.rpush(QUEUE_NAME, json.dumps(msg))
            
            # ปริ้นบอกทุกๆ 100 เฟรม ว่าระบบยังวิ่งอยู่
            if int(time.time()) % 5 == 0 and r.llen(QUEUE_NAME) < 5:
                 logger.info(f"✅ Active... Last frame: {frame_id}")

        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    run()