import cv2
import numpy as np
import os
import redis
import json
import time
import sys
import signal
import datetime

# --- Configuration ---
# รับ RTSP URL จาก Environment Variable
RTSP_URL = os.getenv("RTSP_URL") 
CAMERA_ID = os.getenv("CAMERA_ID", "camera_01") # ตั้งชื่อกล้องเพื่อใช้เป็น Key ใน Redis

OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "/app/shared_memory")
REDIS_HOST = os.getenv("REDIS_HOST", "redis_broker")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

TARGET_SIZE = (640, 640)
SKIP_FRAMES = 30 # เก็บทุกๆ 30 เฟรม
HEARTBEAT_INTERVAL = 60

RUNNING = True

def handle_signal(signum, frame):
    global RUNNING
    print(f"\n🛑 Received signal {signum}. Stopping gracefully...")
    RUNNING = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# --- Helper Functions ---
def resize_with_padding(image, target_size):
    # (ใช้ฟังก์ชันเดิมได้เลย ดีอยู่แล้ว)
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized_image = cv2.resize(image, (nw, nh))
    new_image = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
    
    y_offset = (target_h - nh) // 2
    x_offset = (target_w - nw) // 2
    new_image[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized_image
    return new_image

# --- RTSP Processing ---

def process_rtsp_stream(rtsp_url, camera_id, r_client):
    print(f"📡 Connecting to {camera_id}...")
    
    # สร้างโฟลเดอร์ตามชื่อกล้องและวันเวลา
    save_dir = os.path.join(OUTPUT_FOLDER, camera_id)
    os.makedirs(save_dir, exist_ok=True)

    # Note: ตั้งค่า buffer size ให้ต่ำที่สุดเพื่อลด Latency
    # และใช้ TCP เพื่อความเสถียร (หรือ UDP ถ้าเน้นเร็วแตภาพแตกได้)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # ลด Buffer ให้เหลือน้อยที่สุด

    if not cap.isOpened():
        print(f"❌ Error: Could not connect to {camera_id}")
        return False

    frame_count = 0
    
    # Loop ตลอดกาลจนกว่าจะสั่งหยุด
    while RUNNING:
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️ Lost connection to {camera_id}. Reconnecting in 5s...")
            cap.release()
            time.sleep(5)
            # Reconnect logic
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        # Logic: Skip Frames
        # สำหรับ RTSP เราต้องอ่านทุกเฟรมเพื่อเคลียร์ Buffer แต่จะ Process แค่บางเฟรม
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue

        try:
            # --- Processing Step ---
            processed_frame = resize_with_padding(frame, TARGET_SIZE)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # ตั้งชื่อไฟล์ตาม Timestamp แทน running number เพื่อกันไฟล์ทับกันตอน restart
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{timestamp}.npy"
            full_path = os.path.join(save_dir, file_name)
            
            np.save(full_path, rgb_frame)
            
            message = {
                "camera_id": camera_id,
                "status": "processing",
                "npy_path": full_path,
                "timestamp": time.time()
            }
            
            # Push Job เข้า Redis
            r_client.rpush('video_jobs', json.dumps(message))
            
            print(f" ✅ [{camera_id}] Sent frame {file_name}", flush=True)

        except Exception as e:
            print(f"⚠️ Error processing frame: {e}")

    cap.release()
    print(f"👋 Disconnected from {camera_id}")
    return True

# --- Main Entry Point ---

def main():
    if not RTSP_URL:
        print("❌ Error: RTSP_URL environment variable is not set.")
        sys.exit(1)

    print(f"--- RTSP Service for {CAMERA_ID} ---")
    
    r = None
    while RUNNING:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_connect_timeout=2)
            r.ping()
            print("🟢 Connected to Redis successfully!")
            break 
        except redis.ConnectionError:
            print(f"🔴 Redis not ready. Retrying in 5s...")
            time.sleep(5)

    # ไม่ต้องมี Loop หาไฟล์แล้ว เรียก function ตรงๆ เลย
    # ใส่ Loop ครอบอีกชั้นเผื่อ function หลุดออกมาแบบไม่ตั้งใจ
    while RUNNING:
        try:
            process_rtsp_stream(RTSP_URL, CAMERA_ID, r)
        except Exception as e:
            print(f"🔥 Critical Error: {e}. Restarting service in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    main()