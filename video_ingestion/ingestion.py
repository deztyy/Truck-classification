import cv2
import numpy as np
import os
import redis
import json
import time
import sys
import signal
import datetime
import threading
import glob

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
# RTSP and Camera Settings
RTSP_URL = os.getenv("RTSP_URL")
CAMERA_ID = os.getenv("CAMERA_ID", "camera_01")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "/app/shared_memory")

# Redis Connection Settings
REDIS_HOST = os.getenv("REDIS_HOST", "redis_broker")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_QUEUE_LIMIT = 100  # Threshold to stop pushing jobs if the consumer is slow

# Image Processing & Storage Settings
TARGET_SIZE = (640, 640) # Target dimensions (Width, Height)
PROCESS_FPS = 1.0        # Frame capture rate (Frames Per Second)
JPG_QUALITY = 80         # JPEG compression quality (0-100)
RETENTION_SECONDS = 3600 # File retention period (1 hour)

# Global flag for the main loop
RUNNING = True

# =============================================================================
# SIGNAL HANDLING
# =============================================================================
def handle_signal(signum, frame):
    """
    Handles system signals (SIGINT, SIGTERM) for graceful shutdown.
    """
    global RUNNING
    print(f"\n🛑 Received signal {signum}. Stopping services...")
    RUNNING = False

# Register signal handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def resize_with_padding(image, target_size):
    """
    Resizes an image to the target size while maintaining the aspect ratio.
    Adds padding (letterboxing) to fit the exact dimensions.
    
    Args:
        image (numpy.ndarray): Source image.
        target_size (tuple): (width, height).
    
    Returns:
        numpy.ndarray: Resized and padded image.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    # Resize image
    resized_image = cv2.resize(image, (nw, nh))
    
    # Create a blank canvas (black background)
    new_image = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
    
    # Center the image on the canvas
    y_offset = (target_h - nh) // 2
    x_offset = (target_w - nw) // 2
    new_image[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized_image
    
    return new_image

# =============================================================================
# THREADED VIDEO CAPTURE CLASS
# =============================================================================
class RTSPStreamLoader:
    """
    Dedicated thread for reading frames from an RTSP stream.
    This prevents the main processing loop from being blocked by network latency
    or decoding delays.
    """
    def __init__(self, src):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        
        # Enforce TCP transport to reduce artifacts/packet loss on unstable networks
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        # Limit buffer size to always get the latest frame
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        """Starts the frame update thread."""
        if self.started:
            print("⚠️ Stream already started!!")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        """Background loop to continuously grab frames."""
        while self.started:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            
            # If stream fails, wait briefly before retrying logic in main loop
            if not grabbed:
                time.sleep(0.5)

    def read(self):
        """Returns the latest available frame safely."""
        with self.read_lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Stops the thread and releases resources."""
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# =============================================================================
# BACKGROUND MAINTENANCE
# =============================================================================
def cleanup_worker(folder_path, retention_sec):
    """
    Background worker that deletes files older than the retention period.
    Runs continuously while the service is active.
    """
    print(f"🧹 Cleanup service started for {folder_path} (Retention: {retention_sec}s)")
    while RUNNING:
        try:
            now = time.time()
            # Find all JPG files in the directory
            files = glob.glob(os.path.join(folder_path, "*.jpg"))
            for f in files:
                # Delete if modification time is older than retention limit
                if os.stat(f).st_mtime < now - retention_sec:
                    os.remove(f)

            # Check every 60 seconds
            time.sleep(60) 
        except Exception as e:
            print(f"⚠️ Cleanup Error: {e}")
            time.sleep(60)

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def main():
    # 1. Validation
    if not RTSP_URL:
        print("❌ Error: RTSP_URL environment variable is not set.")
        sys.exit(1)

    # 2. Redis Connection Setup
    r = None
    while RUNNING:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, socket_connect_timeout=2)
            r.ping()
            print(f"🟢 Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            break
        except redis.ConnectionError:
            print("🔴 Redis not ready. Retrying in 5s...")
            time.sleep(5)

    # 3. Directory Setup
    save_dir = os.path.join(OUTPUT_FOLDER, CAMERA_ID)
    os.makedirs(save_dir, exist_ok=True)

    # 4. Start Cleanup Thread
    cleaner_thread = threading.Thread(target=cleanup_worker, args=(save_dir, RETENTION_SECONDS))
    cleaner_thread.daemon = True
    cleaner_thread.start()

    # 5. Start RTSP Stream
    print(f"📡 Connecting to Camera: {CAMERA_ID}...")
    video_stream = RTSPStreamLoader(RTSP_URL).start()
    
    # Allow time for buffer to fill
    time.sleep(2.0)

    last_process_time = 0
    process_interval = 1.0 / PROCESS_FPS 

    print("🚀 Service Started. Processing frames...")

    # --- MAIN LOOP ---
    while RUNNING:
        current_time = time.time()
        
        # A. FPS Throttling
        if current_time - last_process_time < process_interval:
            time.sleep(0.01) # Short sleep to reduce CPU usage
            continue

        # B. Frame Retrieval (Non-blocking)
        grabbed, frame = video_stream.read()

        # Handle stream loss / Reconnection
        if not grabbed or frame is None:
            print(f"⚠️ Frame lost or Camera disconnected. Reconnecting in 5s...")
            video_stream.stop()
            time.sleep(5)
            video_stream = RTSPStreamLoader(RTSP_URL).start()
            time.sleep(2)
            continue

        try:
            # C. Image Processing
            processed_frame = resize_with_padding(frame, TARGET_SIZE)
            
            # Generate Filename
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{timestamp_str}.jpg"
            full_path = os.path.join(save_dir, file_name)

            # Save to Disk (JPEG)
            cv2.imwrite(full_path, processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
            
            # D. Redis Push (Flow Control)
            # Check queue size to prevent backpressure
            q_len = r.llen('video_jobs')
            
            if q_len < REDIS_QUEUE_LIMIT:
                message = {
                    "camera_id": CAMERA_ID,
                    "status": "pending",
                    "image_path": full_path, 
                    "timestamp": current_time
                }
                r.rpush('video_jobs', json.dumps(message))
                
                # Update process time only on success
                last_process_time = current_time 
                
                # Optional: Logging every 10 seconds
                if int(current_time) % 10 == 0:
                      print(f"✅ [{CAMERA_ID}] Pushed {file_name} (Queue: {q_len})", flush=True)
            else:
                print(f"⚠️ Redis Queue Full ({q_len}/{REDIS_QUEUE_LIMIT}). Dropping frame...", flush=True)
                # Skip this frame interval
                last_process_time = current_time 

        except Exception as e:
            print(f"🔥 Processing Error: {e}")
            time.sleep(1)

    # Cleanup before exit
    video_stream.stop()
    print("👋 Service Stopped Gracefully.")

if __name__ == "__main__":
    main()