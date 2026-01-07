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

# --- Load Config ---
# เพิ่ม CAM_ID เพื่อระบุตัวตน
CAM_ID = os.getenv("CAM_ID", "unknown_camera")
RTSP_URL = os.getenv("RTSP_URL", "/app/video.mp4")
PLAY_LOOP = os.getenv("PLAY_LOOP", "true") == "true"
QUEUE_NAME = os.getenv("QUEUE_NAME", "video_frames_queue")
SHARED_PATH = "/dev/shm"

# Connect Redis
try:
    r = redis.Redis(host="redis", port=6379, db=0)
    r.ping()
    # logger.info(f"✅ [{CAM_ID}] Redis Connected!")
except Exception as e:
    logger.error(f"❌ [{CAM_ID}] Redis Failed: {e}")
    exit(1)

def run():
    logger.info(f"🚀 [{CAM_ID}] Starting... Source={RTSP_URL}")
    
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logger.error(f"❌ [{CAM_ID}] Cannot open file!")
        return

    frame_count = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            if PLAY_LOOP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Backpressure: เช็คคิว (3 กล้องรุม 1 คิว ต้องระวังเต็มเร็ว)
        if r.llen(QUEUE_NAME) >= 50:
            time.sleep(0.01)
            continue

        try:
            # Resize
            resized = cv2.resize(frame, (640, 640))
            
            # Save .npy
            frame_id = str(uuid.uuid4())
            # ตั้งชื่อไฟล์โดยมี cam_id นำหน้า จะได้ไม่ซ้ำกัน
            file_name = f"{CAM_ID}_{frame_id}.npy" 
            file_path = os.path.join(SHARED_PATH, file_name)
            np.save(file_path, resized)

            # Send to Redis (เพิ่ม cam_id ใน message)
            msg = {
                "cam_id": CAM_ID,   # <--- สำคัญมาก! บอกว่ามาจากกล้องไหน
                "id": frame_id,
                "file_path": file_path,
                "shape": resized.shape,
                "timestamp": time.time()
            }
            r.rpush(QUEUE_NAME, json.dumps(msg))
            
            # FPS Log (แยกของใครของมัน)
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 5.0:
                fps = frame_count / (curr_time - prev_time)
                logger.info(f"📸 [{CAM_ID}] Speed: {fps:.2f} FPS")
                frame_count = 0
                prev_time = curr_time

        except Exception as e:
            logger.error(f"Error [{CAM_ID}]: {e}")

if __name__ == "__main__":
    run()