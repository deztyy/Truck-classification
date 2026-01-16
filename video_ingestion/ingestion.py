import datetime
import glob
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import redis

# =============================================================================
# LOGGING CONFIGURATION (OPTIMIZED)
# =============================================================================
def setup_logging(level=logging.INFO):
    log_dir = os.getenv("LOG_DIR", "/app/logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("video_ingestion")
    logger.setLevel(level)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",  # Simplified format
        datefmt="%H:%M:%S",  # Shorter timestamp
    )

    # Console handler - only show important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Hide DEBUG from console
    console_handler.setFormatter(log_formatter)

    # File handler - keep all logs
    log_file_path = os.path.join(log_dir, "ingestion.log")
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging(level=logging.INFO)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
RTSP_URL = os.getenv("RTSP_URL")
CAMERA_ID = os.getenv("CAMERA_ID", "camera_01")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "/app/shared_memory")

REDIS_HOST = os.getenv("REDIS_HOST", "redis_broker")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_QUEUE_LIMIT = 100
REDIS_CONNECT_TIMEOUT_SEC = 2
REDIS_RETRY_INTERVAL_SEC = 5

TARGET_SIZE = (640, 640)
BATCH_SIZE = 30
RAW_FPS = 30.0
RETENTION_SECONDS = 3600

THAI_TZ = ZoneInfo("Asia/Bangkok")

STREAM_STALE_TIMEOUT_SEC = 10.0
FRAME_READ_RETRY_DELAY_SEC = 0.5
STREAM_RECONNECT_DELAY_SEC = 2
STREAM_RECONNECT_WAIT_DELAY_SEC = 5
FRAME_LOSS_RECONNECT_DELAY_SEC = 5
CLEANUP_CHECK_INTERVAL_SEC = 60
INITIAL_BUFFER_FILL_SEC = 2.0

MAX_WORKER_THREADS = 2
WORKER_QUEUE_TIMEOUT_SEC = 10

# Performance monitoring
LOG_STATS_EVERY_N_BATCHES = 10  # Log stats every N batches instead of every batch

RUNNING = True

LUA_RPUSH_LIMIT_SCRIPT = """
local queue_key = KEYS[1]
local limit = tonumber(ARGV[1])
local value = ARGV[2]

if redis.call('LLEN', queue_key) >= limit then
    return 0
else
    redis.call('RPUSH', queue_key, value)
    return 1
end
"""

# =============================================================================
# SIGNAL HANDLING
# =============================================================================
def handle_signal(signum, frame):
    global RUNNING
    logger.info(f"Signal {signum} received. Shutting down...")
    RUNNING = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def resize_with_padding(image, target_size):
    """Resize image with padding to maintain aspect ratio"""
    height, width = image.shape[:2]
    target_width, target_height = target_size

    width_scale = target_width / width
    height_scale = target_height / height
    scale = min(width_scale, height_scale)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height))
    padded_image = np.full((target_height, target_width, 3), 0, dtype=np.uint8)

    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    padded_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized_image

    return padded_image


def get_redis_client():
    """Establish Redis connection with retry logic"""
    connection_attempt = 0
    while RUNNING:
        connection_attempt += 1
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SEC,
                decode_responses=False
            )
            redis_client.ping()
            logger.info(f"✅ Redis connected: {REDIS_HOST}:{REDIS_PORT}")
            return redis_client

        except (redis.ConnectionError, redis.TimeoutError) as e:
            if connection_attempt == 1 or connection_attempt % 5 == 0:  # Log every 5th attempt
                logger.warning(f"Redis connection failed (attempt {connection_attempt}), retrying...")
            time.sleep(REDIS_RETRY_INTERVAL_SEC)

        except Exception as e:
            logger.error(f"Unexpected Redis error: {e}")
            time.sleep(REDIS_RETRY_INTERVAL_SEC)

    return None

# =============================================================================
# THREADED VIDEO CAPTURE CLASS
# =============================================================================
class RTSPStreamLoader:
    """Thread-based RTSP stream reader for non-blocking frame capture"""
    
    def __init__(self, src):
        self.src = src
        self.stream = cv2.VideoCapture(src)

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()

        self.started = False
        self.read_lock = threading.Lock()
        self.last_read_time = time.time()
        self.thread = None

    def start(self):
        if self.started:
            return None

        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed_result, frame_result = self.stream.read()

            with self.read_lock:
                self.grabbed = grabbed_result
                self.frame = frame_result
                if grabbed_result:
                    self.last_read_time = time.time()

            if not grabbed_result:
                time.sleep(FRAME_READ_RETRY_DELAY_SEC)

    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self):
        self.started = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.stream.release()

    def is_stale(self, timeout=STREAM_STALE_TIMEOUT_SEC):
        time_since_last_read = time.time() - self.last_read_time
        return time_since_last_read > timeout

# =============================================================================
# BACKGROUND I/O WORKER (OPTIMIZED LOGGING)
# =============================================================================
class BatchIOWorker:
    """Non-blocking batch processing worker"""

    def __init__(self, max_workers=MAX_WORKER_THREADS, timeout=WORKER_QUEUE_TIMEOUT_SEC):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="batch-worker-",
        )
        self.timeout = timeout
        self.lua_script = None
        self.redis_client = None
        self.batch_count = 0
        logger.info(f"BatchIOWorker initialized ({max_workers} workers)")

    def set_redis_client(self, redis_client, lua_script):
        self.redis_client = redis_client
        self.lua_script = lua_script

    def process_batch_async(self, processed_frames, batch_timestamps, save_dir, camera_id):
        try:
            self.executor.submit(
                self._process_batch_internal,
                processed_frames,
                batch_timestamps,
                save_dir,
                camera_id,
            )
            return True

        except RuntimeError:
            logger.error("Failed to submit batch: executor shutting down")
            return False

        except Exception as e:
            logger.error(f"Batch submission error: {e}")
            return False

    def _process_batch_internal(self, processed_frames, batch_timestamps, save_dir, camera_id):
        try:
            self.batch_count += 1
            batch_array = np.stack(processed_frames, axis=0)

            first_timestamp = batch_timestamps[0]
            timestamp_str = first_timestamp.strftime("%Y%m%d_%H%M%S_%f")
            batch_filename = f"{timestamp_str}_batch.npy"
            batch_file_path = os.path.join(save_dir, batch_filename)

            np.save(batch_file_path, batch_array)

            # Only log every N batches to reduce spam
            if self.batch_count % LOG_STATS_EVERY_N_BATCHES == 0:
                logger.info(
                    f"📦 [{camera_id}] Batches processed: {self.batch_count} | "
                    f"Shape: {batch_array.shape} | "
                    f"Size: {batch_array.nbytes / (1024*1024):.1f}MB"
                )

            if self.redis_client and self.lua_script:
                self._push_to_redis(camera_id, batch_file_path, first_timestamp)

        except Exception as e:
            logger.error(f"[{camera_id}] Batch processing failed: {e}")

    def _push_to_redis(self, camera_id, file_path, timestamp):
        """Push job to Redis"""
        try:
            job_payload = {
                "camera_id": camera_id,
                "file_path": file_path,
                "timestamp": timestamp.isoformat(),
            }

            serialized_job = json.dumps(job_payload)

            result = self.lua_script(
                keys=["video_jobs"], 
                args=[REDIS_QUEUE_LIMIT, serialized_job]
            )

            if result == 0:
                # Only log queue full warning occasionally
                if self.batch_count % 5 == 0:
                    logger.warning(f"⚠️ [{camera_id}] Redis queue full, batch discarded")

        except (redis.ConnectionError, redis.TimeoutError):
            logger.error(f"[{camera_id}] Redis connection lost")

        except Exception as e:
            logger.error(f"[{camera_id}] Redis error: {e}")

    def shutdown(self, wait=True, timeout=5):
        logger.info("Shutting down batch worker...")
        self.executor.shutdown(wait=wait)
        logger.info(f"✅ Total batches processed: {self.batch_count}")

# =============================================================================
# BACKGROUND MAINTENANCE (OPTIMIZED)
# =============================================================================
def cleanup_worker(folder_path, retention_seconds):
    logger.info(f"🧹 Cleanup service started (retention: {retention_seconds}s)")

    cleanup_iteration = 0
    while RUNNING:
        try:
            cleanup_iteration += 1
            now = time.time()

            batch_files = glob.glob(os.path.join(folder_path, "*.npy"))
            files_deleted = 0

            for file_path in batch_files:
                try:
                    file_mod_time = os.stat(file_path).st_mtime
                    file_age_seconds = now - file_mod_time

                    if file_age_seconds > retention_seconds:
                        os.remove(file_path)
                        files_deleted += 1

                except (FileNotFoundError, OSError):
                    pass

            # Only log if files deleted or every 10th cycle
            if files_deleted > 0:
                logger.info(f"🧹 Cleanup: deleted {files_deleted} old files")
            elif cleanup_iteration % 10 == 0:
                logger.debug(f"Cleanup cycle #{cleanup_iteration}: {len(batch_files)} files checked")

            time.sleep(CLEANUP_CHECK_INTERVAL_SEC)

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            time.sleep(CLEANUP_CHECK_INTERVAL_SEC)

# =============================================================================
# STREAM RECOVERY HANDLERS
# =============================================================================
def _handle_stream_stale(video_stream):
    logger.warning("⚠️ Stream stale, reconnecting...")
    video_stream.stop()
    time.sleep(STREAM_RECONNECT_DELAY_SEC)

    try:
        new_stream = RTSPStreamLoader(RTSP_URL).start()
        time.sleep(STREAM_RECONNECT_DELAY_SEC)
        new_stream.last_read_time = time.time()
        logger.info("✅ Stream reconnected")
        return new_stream

    except Exception as e:
        logger.error(f"Reconnection failed: {e}")
        time.sleep(STREAM_RECONNECT_WAIT_DELAY_SEC)
        return None


def _handle_frame_loss(video_stream):
    logger.warning("⚠️ Frame loss detected, reconnecting...")
    video_stream.stop()
    time.sleep(FRAME_LOSS_RECONNECT_DELAY_SEC)
    new_stream = RTSPStreamLoader(RTSP_URL).start()
    return new_stream

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("🚀 Video Ingestion Service Starting")
    logger.info("=" * 60)

    if not RTSP_URL:
        logger.error("❌ RTSP_URL not set")
        sys.exit(1)

    logger.info(f"📹 Camera: {CAMERA_ID}")
    logger.info(f"📦 Batch size: {BATCH_SIZE} frames")
    logger.info(f"📏 Target size: {TARGET_SIZE}")

    redis_client = get_redis_client()
    if redis_client is None:
        logger.error("❌ Redis connection failed")
        sys.exit(1)
    
    lua_script = redis_client.register_script(LUA_RPUSH_LIMIT_SCRIPT)

    batch_worker = BatchIOWorker(max_workers=MAX_WORKER_THREADS)
    batch_worker.set_redis_client(redis_client, lua_script)

    save_directory = os.path.join(OUTPUT_FOLDER, CAMERA_ID)
    os.makedirs(save_directory, exist_ok=True)

    cleaner_thread = threading.Thread(
        target=cleanup_worker,
        args=(save_directory, RETENTION_SECONDS),
        daemon=True,
    )
    cleaner_thread.start()

    logger.info(f"🔌 Connecting to stream...")
    video_stream = RTSPStreamLoader(RTSP_URL).start()
    time.sleep(INITIAL_BUFFER_FILL_SEC)

    frame_batch = []
    batch_timestamps = []

    logger.info("✅ Service ready - processing frames")
    logger.info("=" * 60)

    frame_count = 0
    last_log_time = time.time()

    try:
        while RUNNING:
            if video_stream.is_stale(timeout=STREAM_STALE_TIMEOUT_SEC):
                new_stream = _handle_stream_stale(video_stream)
                if new_stream:
                    video_stream = new_stream
                continue

            grabbed, frame = video_stream.read()
            capture_time = datetime.datetime.now(THAI_TZ)

            if not grabbed or frame is None:
                video_stream = _handle_frame_loss(video_stream)
                continue

            processed_frame = resize_with_padding(frame, TARGET_SIZE)

            frame_batch.append(processed_frame)
            batch_timestamps.append(capture_time)
            frame_count += 1

            if len(frame_batch) >= BATCH_SIZE:
                success = batch_worker.process_batch_async(
                    frame_batch, batch_timestamps, save_directory, CAMERA_ID
                )

                frame_batch = []
                batch_timestamps = []

            # Log FPS every 5 seconds instead of every batch
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                fps = frame_count / (current_time - last_log_time)
                logger.info(f"📊 FPS: {fps:.1f} | Total frames: {frame_count}")
                last_log_time = current_time

    except KeyboardInterrupt:
        logger.info("⏹️ Keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Main loop error: {e}")
    finally:
        logger.info("=" * 60)
        logger.info("🛑 Shutting down")
        logger.info("=" * 60)

        video_stream.stop()
        batch_worker.shutdown(wait=True, timeout=5)

        logger.info(f"✅ Graceful shutdown complete | Processed {frame_count} frames")


if __name__ == "__main__":
    main()