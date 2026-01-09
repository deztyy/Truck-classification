import cv2
import numpy as np
import os
import redis
import json
import time
import shutil

# --- Configuration ---
INPUT_FOLDER = "/app/videos"
OUTPUT_FOLDER = "/app/shared_memory"
REDIS_HOST = "redis_broker"

TARGET_SIZE = (640, 640)
SKIP_FRAMES = 30  # เก็บ 1 ภาพ ทุกๆ 30 เฟรม

def process_video_stream(video_filename, r_client):
    video_path = os.path.join(INPUT_FOLDER, video_filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_filename}")
        return

    # สร้างโฟลเดอร์เก็บไฟล์ย่อย
    video_name_no_ext = os.path.splitext(video_filename)[0]
    save_dir = os.path.join(OUTPUT_FOLDER, f"{video_name_no_ext}_frames")
    
    # เคลียร์โฟลเดอร์เก่าทิ้งถ้ามี (เริ่มใหม่หมด)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Streaming: {video_filename} | Total frames: {total_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % SKIP_FRAMES == 0:
            try:
                # 1. Process Image
                resized_frame = cv2.resize(frame, TARGET_SIZE)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # 2. Save .npy (Frame เล็กๆ)
                file_name = f"frame_{frame_count:06d}.npy"
                full_path = os.path.join(save_dir, file_name)
                np.save(full_path, rgb_frame)
                
                # 3. ส่งเข้า Redis ทันที! (Real-time)
                message = {
                    "video_id": video_filename,
                    "status": "processing",      # บอกว่ายังทำไม่เสร็จนะ แค่ส่งมาเฟรมนึง
                    "frame_id": frame_count,
                    "npy_path": full_path,
                    "timestamp": time.time()
                }
                r_client.rpush('video_jobs', json.dumps(message))
                
                saved_count += 1
                print(f"  -> Sent frame {frame_count} to Redis", flush=True)

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

        frame_count += 1

    cap.release()
    
    # 4. ส่งสัญญาณ "จบวิดีโอ" (End of Stream)
    # Model จะได้รับรู้นี้แล้วรู้ว่า "อ๋อ ครบแล้ว ประมวลผลรวมได้เลย"
    end_message = {
        "video_id": video_filename,
        "status": "finished",
        "total_frames_processed": saved_count
    }
    r_client.rpush('video_jobs', json.dumps(end_message))
    
    print(f"✅ Finished streaming {video_filename}. Sent 'finished' signal.")

def main():
    print("--- Video Ingestion Service (Real-time Streaming) ---")
    
    # เชื่อมต่อ Redis ครั้งเดียว แล้วส่งตัวแปร connection ไปใช้
    r = None
    while True:
        try:
            r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
            r.ping()
            print("Connected to Redis successfully!")
            break 
        except Exception as e:
            print(f"Redis not ready. Retrying...")
            time.sleep(5) 

    while True:
        if not os.path.exists(INPUT_FOLDER):
            time.sleep(5)
            continue

        files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.mp4')]
        
        if not files:
            print("Waiting for files...", flush=True)
            time.sleep(5)
            continue

        for file in files:
            print(f"\nStarting Stream: {file}")
            process_video_stream(file, r)
            
            # (Optional) ย้ายไฟล์ที่ทำเสร็จแล้วไปที่อื่น
            # processed_path = os.path.join(INPUT_FOLDER, "processed")
            # os.makedirs(processed_path, exist_ok=True)
            # shutil.move(os.path.join(INPUT_FOLDER, file), os.path.join(processed_path, file))

        time.sleep(10)

if __name__ == "__main__":
    main()