import cv2
import redis
import numpy as np
import time
import os
import uuid

# Config รับค่าจาก docker-compose
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
INPUT_SOURCE = os.getenv('INPUT_SOURCE', 'videos/test_video.mp4')
SHARED_MEM_PATH = "/dev/shm"

# เชื่อมต่อ Redis
try:
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    print(f"Connected to Redis at {REDIS_HOST}")
except Exception as e:
    print(f"Redis connection failed: {e}")

# เปิดวิดีโอ
cap = cv2.VideoCapture(INPUT_SOURCE)

# ถ้า Input เป็นตัวเลข (เช่น 0) ให้แปลงเป็น int สำหรับ Webcam
if INPUT_SOURCE.isdigit():
    cap = cv2.VideoCapture(int(INPUT_SOURCE))

while True:
    ret, frame = cap.read()

    # LOGIC: ถ้าวิดีโอจบ ให้วนกลับไปเริ่มใหม่ (Loop) เพื่อจำลอง CCTV
    if not ret:
        print("End of video, restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. Resize (ตาม Diagram)
    frame_resized = cv2.resize(frame, (640, 640))

    # 2. จำลองการ Save ลง Shared Memory
    file_id = str(uuid.uuid4())
    file_path = os.path.join(SHARED_MEM_PATH, f"{file_id}.npy")
    
    # Save เป็น .npy (เร็วมาก)
    np.save(file_path, frame_resized)

    # 3. ส่ง Metadata ไป Redis
    job_data = {
        "file_path": file_path,
        "camera_id": "local_test_cam",
        "timestamp": time.time()
    }
    
    # Push เข้าคิว
    r.rpush("video_tasks", str(job_data))

    print(f"Ingested: {file_id} | Queue Length: {r.llen('video_tasks')}")

    # หน่วงเวลาหน่อย ไม่งั้นไฟล์วิดีโอจะรันเร็วเกินไป (Simulation Speed)
    time.sleep(0.1) 

cap.release()