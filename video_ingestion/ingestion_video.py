import numpy as np
import os
import cv2
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
# Video File and Camera Settings
VIDEO_PATH = os.getenv("VIDEO_PATH")  # Path to local video file
CAMERA_ID = os.getenv("CAMERA_ID", "camera_01")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "/app/shared_memory")
LOOP_VIDEO = os.getenv("LOOP_VIDEO", "true").lower() == "true"  # Loop video when it ends

# Redis Connection Settings
REDIS_HOST = os.getenv("REDIS_HOST", "redis_broker")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_QUEUE_LIMIT = 100  # Threshold to stop pushing jobs if the consumer is slow
REDIS_CONNECT_TIMEOUT = 10  # Connection timeout in seconds
REDIS_RETRY_DELAY = 5  # Delay between connection retries

# Image Processing & Storage Settings
TARGET_SIZE = (640, 640)  # Target dimensions (Width, Height)
PROCESS_FPS = 1.0         # Frame capture rate (Frames Per Second)
JPG_QUALITY = 80          # JPEG compression quality (0-100)
RETENTION_SECONDS = 3600  # File retention period (1 hour)
BATCH_SIZE = 30           # Number of frames per batch

# Global flag for the main loop
RUNNING = True

# Frame buffer for batching
frame_buffer = []

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
# REDIS CONNECTION MANAGEMENT
# =============================================================================
def connect_to_redis(max_retries=None):
    """
    Establishes connection to Redis with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts (None for infinite)
    
    Returns:
        redis.Redis: Connected Redis client or None if failed
    """
    retry_count = 0
    
    while RUNNING:
        try:
            print(f"🔌 Attempting to connect to Redis at {REDIS_HOST}:{REDIS_PORT}...")
            
            # Create Redis client with proper timeout settings
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                socket_timeout=REDIS_CONNECT_TIMEOUT,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
                decode_responses=False  # Important: keep as bytes for binary data
            )
            
            # Test connection
            redis_client.ping()
            
            print(f"✅ Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return redis_client
            
        except redis.ConnectionError as e:
            retry_count += 1
            if max_retries and retry_count >= max_retries:
                print(f"❌ Failed to connect to Redis after {retry_count} attempts")
                return None
            
            print(f"🔴 Redis connection failed (attempt {retry_count}): {e}")
            print(f"⏳ Retrying in {REDIS_RETRY_DELAY} seconds...")
            time.sleep(REDIS_RETRY_DELAY)
            
        except Exception as e:
            print(f"❌ Unexpected error connecting to Redis: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(REDIS_RETRY_DELAY)
    
    return None

def ensure_redis_connection(redis_client):
    """
    Check if Redis connection is alive, reconnect if needed.
    
    Args:
        redis_client: Existing Redis client
    
    Returns:
        redis.Redis: Working Redis client
    """
    try:
        redis_client.ping()
        return redis_client
    except:
        print("⚠️ Redis connection lost. Reconnecting...")
        return connect_to_redis()

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

def save_and_push_batch(redis_client, save_dir, batch_frames, camera_id, queue_limit):
    """
    Saves a batch of frames to disk and pushes job to Redis.
    
    Args:
        redis_client: Redis connection
        save_dir: Directory to save batch file
        batch_frames: List of processed frames
        camera_id: Camera identifier
        queue_limit: Maximum queue length before dropping
    
    Returns:
        tuple: (success: bool, redis_client: Redis) - Returns updated Redis client
    """
    if not batch_frames:
        return False, redis_client
    
    try:
        # Generate filename with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"{timestamp_str}.npy"
        full_path = os.path.join(save_dir, file_name)
        
        # Convert list to numpy array and save
        batch_array = np.array(batch_frames, dtype=np.uint8)
        np.save(full_path, batch_array)
        
        # Ensure Redis connection is alive
        redis_client = ensure_redis_connection(redis_client)
        if redis_client is None:
            print("❌ Cannot push to Redis - connection failed")
            return False, None
        
        # Check queue length
        q_len = redis_client.llen('video_jobs')
        
        if q_len < queue_limit:
            message = {
                "camera_id": camera_id,
                "file_path": full_path,
                "timestamp": time.time()
            }
            redis_client.rpush('video_jobs', json.dumps(message))
            
            print(f"✅ [{camera_id}] Pushed {file_name} (Queue: {q_len}, Frames: {len(batch_frames)})")
            return True, redis_client
        else:
            print(f"⚠️ Redis Queue Full ({q_len}/{queue_limit}). Dropping batch...")
            # Delete the saved file since we're not processing it
            try:
                os.remove(full_path)
            except:
                pass
            return False, redis_client
            
    except redis.ConnectionError as e:
        print(f"🔥 Redis connection error: {e}")
        return False, connect_to_redis()
    except Exception as e:
        print(f"🔥 Batch save error: {e}")
        import traceback
        traceback.print_exc()
        return False, redis_client

# =============================================================================
# THREADED VIDEO CAPTURE CLASS
# =============================================================================
class LocalVideoLoader:
    """
    Dedicated thread for reading frames from a local video file.
    This prevents the main processing loop from being blocked by decoding delays.
    """
    def __init__(self, src, loop=True):
        self.src = src
        self.loop = loop
        self.stream = cv2.VideoCapture(src)
        
        if not self.stream.isOpened():
            raise ValueError(f"Unable to open video file: {src}")
        
        # Get video properties
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"📹 Video Info: {self.total_frames} frames, {self.fps:.2f} FPS, {self.duration:.2f}s duration")
        
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.frame_count = 0

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
            
            # Handle end of video
            if not grabbed:
                if self.loop:
                    print("🔄 Video ended. Looping...")
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 0
                    grabbed, frame = self.stream.read()
                else:
                    print("⏹️ Video ended. No loop enabled.")
                    with self.read_lock:
                        self.grabbed = False
                        self.frame = None
                    break
            
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.frame_count += 1
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)

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

    def get_progress(self):
        """Returns current playback progress."""
        with self.read_lock:
            return self.frame_count, self.total_frames

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
            # Find all NumPy files in the directory
            files = glob.glob(os.path.join(folder_path, "*.npy"))
            deleted_count = 0
            for f in files:
                # Delete if modification time is older than retention limit
                if os.stat(f).st_mtime < now - retention_sec:
                    os.remove(f)
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"🗑️ Cleaned up {deleted_count} old files")

            # Check every 60 seconds
            time.sleep(60) 
        except Exception as e:
            print(f"⚠️ Cleanup Error: {e}")
            time.sleep(60)

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def main():
    global frame_buffer
    
    print("=" * 60)
    print("📹 Video Capture Service Starting...")
    print("=" * 60)
    
    # 1. Validation
    if not VIDEO_PATH:
        print("❌ Error: VIDEO_PATH environment variable is not set.")
        sys.exit(1)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        sys.exit(1)
    
    print(f"📁 Video Path: {VIDEO_PATH}")
    print(f"📷 Camera ID: {CAMERA_ID}")
    print(f"💾 Output Folder: {OUTPUT_FOLDER}")
    print(f"🔄 Loop Video: {LOOP_VIDEO}")
    print(f"📦 Batch Size: {BATCH_SIZE} frames")
    print(f"⚙️ Process FPS: {PROCESS_FPS}")
    print("=" * 60)

    # 2. Redis Connection Setup
    print("\n🔌 Connecting to Redis...")
    redis_client = connect_to_redis()
    
    if redis_client is None:
        print("❌ Failed to connect to Redis. Exiting...")
        sys.exit(1)

    # 3. Directory Setup
    save_dir = os.path.join(OUTPUT_FOLDER, CAMERA_ID)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Save directory created: {save_dir}")

    # 4. Start Cleanup Thread
    cleaner_thread = threading.Thread(target=cleanup_worker, args=(save_dir, RETENTION_SECONDS))
    cleaner_thread.daemon = True
    cleaner_thread.start()

    # 5. Start Video Stream
    print(f"\n📡 Loading Video: {VIDEO_PATH}...")
    try:
        video_stream = LocalVideoLoader(VIDEO_PATH, loop=LOOP_VIDEO).start()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Allow time for buffer to fill
    time.sleep(1.0)

    last_process_time = 0
    process_interval = 1.0 / PROCESS_FPS 

    print(f"\n🚀 Service Started. Processing frames in batches of {BATCH_SIZE}...")
    print("=" * 60)

    # --- MAIN LOOP ---
    while RUNNING:
        current_time = time.time()
        
        # A. FPS Throttling
        if current_time - last_process_time < process_interval:
            time.sleep(0.01)  # Short sleep to reduce CPU usage
            continue

        # B. Frame Retrieval (Non-blocking)
        grabbed, frame = video_stream.read()

        # Handle video end (when not looping)
        if not grabbed or frame is None:
            if not LOOP_VIDEO:
                print(f"\n⏹️ Video playback completed.")
                # Save remaining frames in buffer before exiting
                if frame_buffer:
                    print(f"💾 Saving final batch of {len(frame_buffer)} frames...")
                    success, redis_client = save_and_push_batch(
                        redis_client, save_dir, frame_buffer, CAMERA_ID, REDIS_QUEUE_LIMIT
                    )
                    frame_buffer = []
                break
            else:
                print(f"⚠️ Frame read failed. Retrying...")
                time.sleep(1)
                continue

        try:
            # C. Image Processing
            processed_frame = resize_with_padding(frame, TARGET_SIZE)
            
            # Add to buffer
            frame_buffer.append(processed_frame)
            
            # D. Save and Push When Batch is Full
            if len(frame_buffer) >= BATCH_SIZE:
                success, redis_client = save_and_push_batch(
                    redis_client, save_dir, frame_buffer, CAMERA_ID, REDIS_QUEUE_LIMIT
                )
                
                # Clear buffer regardless of success
                frame_buffer = []
                
                # Update process time only after batch is handled
                last_process_time = current_time
                
                # Show progress every batch
                current_frame, total_frames = video_stream.get_progress()
                progress = (current_frame / total_frames * 100) if total_frames > 0 else 0
                print(f"📊 Progress: {progress:.1f}% ({current_frame}/{total_frames} frames)")
            else:
                # Update time even if not pushing yet
                last_process_time = current_time

        except Exception as e:
            print(f"🔥 Processing Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    # Cleanup before exit
    print("\n🛑 Shutting down...")
    video_stream.stop()
    
    # Save any remaining frames in buffer
    if frame_buffer:
        print(f"💾 Saving final batch of {len(frame_buffer)} frames...")
        success, redis_client = save_and_push_batch(
            redis_client, save_dir, frame_buffer, CAMERA_ID, REDIS_QUEUE_LIMIT
        )
    
    # Close Redis connection
    if redis_client:
        redis_client.close()
        print("✅ Redis connection closed")
    
    print("👋 Service Stopped Gracefully.")
    print("=" * 60)

if __name__ == "__main__":
    main()