import redis
import numpy as np
import time
import os
import ast
import sys

# Config
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
QUEUE_NAME = "video_tasks"

# เชื่อมต่อ Redis
try:
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
    print(f"Worker connected to Redis at {REDIS_HOST}")
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
    sys.exit(1)

print("Waiting for tasks...")

while True:
    # 1. รอรับงาน (Blocking Pop) - จะค้างบรรทัดนี้จนกว่าจะมีงานมา
    task = r.blpop(QUEUE_NAME, timeout=0)
    
    if task:
        queue_name, data_bytes = task
        try:
            # แปลงข้อมูลจาก String กลับเป็น Dictionary
            data_str = data_bytes.decode("utf-8")
            job_data = ast.literal_eval(data_str)
            
            file_path = job_data['file_path']
            camera_id = job_data.get('camera_id', 'unknown')

            # 2. อ่านไฟล์ภาพจาก Shared Memory (RAM)
            if os.path.exists(file_path):
                # โหลด Array ขึ้นมา
                image = np.load(file_path)
                
                # --- [จำลอง AI ทำงานตรงนี้] ---
                # เช่นโยนเข้า YOLO model.predict(image)
                # สมมติว่าใช้เวลา 0.05 วินาที
                time.sleep(0.05) 
                
                print(f"✅ Processed: {camera_id} | Shape: {image.shape} | File: {file_path}")

                # 3. สำคัญที่สุด! ลบไฟล์ทิ้งเพื่อคืน RAM
                os.remove(file_path)
                
            else:
                print(f"⚠️ File missing: {file_path}")

        except Exception as e:
            print(f"❌ Error: {e}")