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
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging(level=logging.INFO):
    log_dir = os.getenv("LOG_DIR", "/app/logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("video_ingestion")
    logger.setLevel(level)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(log_formatter)

    log_file_path = os.path.join(log_dir, "ingestion.log")
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(level)
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

RUNNING = True

# Fixed Lua script - no double JSON encoding
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
    logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
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
                decode_responses=False  # Keep as bytes for consistency
            )
            redis_client.ping()
            logger.info(
                f"Redis connected successfully (attempt {connection_attempt}): "
                f"{REDIS_HOST}:{REDIS_PORT}/db{REDIS_DB}"
            )
            return redis_client

        except redis.ConnectionError as ce:
            logger.warning(
                f"Redis unavailable (attempt {connection_attempt}). "
                f"Retrying in {REDIS_RETRY_INTERVAL_SEC}s... [Error: {ce}]"
            )
            time.sleep(REDIS_RETRY_INTERVAL_SEC)

        except redis.TimeoutError as te:
            logger.warning(
                f"Redis timeout (attempt {connection_attempt}). "
                f"Retrying in {REDIS_RETRY_INTERVAL_SEC}s... [Error: {te}]"
            )
            time.sleep(REDIS_RETRY_INTERVAL_SEC)

        except Exception as e:
            logger.error(
                f"Unexpected Redis error (attempt {connection_attempt}): "
                f"{type(e).__name__}: {e}"
            )
            time.sleep(REDIS_RETRY_INTERVAL_SEC)

    logger.info("Redis connection interrupted by shutdown signal")
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
            logger.warning(f"Stream already started for {self.src}. Ignoring duplicate start() call.")
            return None

        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        logger.debug(f"Reader thread started for {self.src}")
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
            logger.debug("Waiting for reader thread to exit...")
            self.thread.join(timeout=2)
        self.stream.release()
        logger.debug(f"Stream closed: {self.src}")

    def is_stale(self, timeout=STREAM_STALE_TIMEOUT_SEC):
        time_since_last_read = time.time() - self.last_read_time
        is_stale = time_since_last_read > timeout
        if is_stale:
            logger.debug(f"Stream stale: {time_since_last_read:.1f}s > {timeout}s threshold")
        return is_stale

# =============================================================================
# BACKGROUND I/O WORKER (FIXED - No Double JSON Encoding)
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
        logger.info(f"BatchIOWorker initialized with {max_workers} workers")

    def set_redis_client(self, redis_client, lua_script):
        self.redis_client = redis_client
        self.lua_script = lua_script
        logger.debug("Redis client configured for BatchIOWorker")

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

        except RuntimeError as re:
            logger.error(f"Failed to submit batch: executor not running. Likely during shutdown. [Error: {re}]")
            return False

        except Exception as e:
            logger.error(f"Unexpected error submitting batch: {type(e).__name__}: {e}")
            return False

    def _process_batch_internal(self, processed_frames, batch_timestamps, save_dir, camera_id):
        try:
            logger.debug(f"Worker processing batch: {len(processed_frames)} frames from {camera_id}")

            batch_array = np.stack(processed_frames, axis=0)

            first_timestamp = batch_timestamps[0]
            timestamp_str = first_timestamp.strftime("%Y%m%d_%H%M%S_%f")
            batch_filename = f"{timestamp_str}_batch.npy"
            batch_file_path = os.path.join(save_dir, batch_filename)

            np.save(batch_file_path, batch_array)

            logger.info(
                f"[{camera_id}] Batch saved: {os.path.basename(batch_file_path)} "
                f"(shape: {batch_array.shape}, size: {batch_array.nbytes / (1024*1024):.1f}MB)"
            )

            if self.redis_client and self.lua_script:
                self._push_to_redis(camera_id, batch_file_path, first_timestamp)
            else:
                logger.warning(f"[{camera_id}] Redis not configured, skipping job enqueue")

        except Exception as e:
            logger.error(
                f"[{camera_id}] Background batch processing failed: {type(e).__name__}: {e}",
                exc_info=True,
            )

    def _push_to_redis(self, camera_id, file_path, timestamp):
        """Push job to Redis - FIXED: Single JSON encoding only"""
        try:
            job_payload = {
                "camera_id": camera_id,
                "file_path": file_path,
                "timestamp": timestamp.isoformat(),
            }

            # FIXED: Only encode once
            serialized_job = json.dumps(job_payload)

            result = self.lua_script(
                keys=["video_jobs"], 
                args=[REDIS_QUEUE_LIMIT, serialized_job]
            )

            if result == 1:
                logger.info(
                    f"[{camera_id}] Batch queued for processing. "
                    f"File: {os.path.basename(file_path)}"
                )
            else:
                logger.warning(
                    f"[{camera_id}] Redis queue full ({REDIS_QUEUE_LIMIT}). "
                    "Batch discarded (backpressure). Downstream worker may be slow or stuck."
                )

        except (redis.ConnectionError, redis.TimeoutError) as network_err:
            logger.error(
                f"[{camera_id}] Redis push failed (network issue): "
                f"{type(network_err).__name__}: {network_err}"
            )

        except json.JSONDecodeError as json_err:
            logger.error(f"[{camera_id}] JSON serialization failed: {json_err}", exc_info=True)

        except Exception as e:
            logger.error(
                f"[{camera_id}] Unexpected Redis error: {type(e).__name__}: {e}",
                exc_info=True,
            )

    def shutdown(self, wait=True, timeout=5):
        logger.info("Shutting down batch I/O worker...")
        logger.debug(f"Waiting for {MAX_WORKER_THREADS} worker threads to finish (timeout: {timeout}s)")
        self.executor.shutdown(wait=wait)
        logger.info("Batch I/O worker shutdown complete")

# =============================================================================
# BACKGROUND MAINTENANCE (File Retention)
# =============================================================================
def cleanup_worker(folder_path, retention_seconds):
    logger.info(
        f"Cleanup service started for {folder_path} "
        f"(retention: {retention_seconds}s, scan interval: {CLEANUP_CHECK_INTERVAL_SEC}s)"
    )

    cleanup_iteration = 0
    while RUNNING:
        try:
            cleanup_iteration += 1
            now = time.time()

            batch_files = glob.glob(os.path.join(folder_path, "*.npy"))
            files_checked = len(batch_files)
            files_deleted = 0

            for file_path in batch_files:
                try:
                    file_stat = os.stat(file_path)
                    file_mod_time = file_stat.st_mtime
                    file_age_seconds = now - file_mod_time

                    if file_age_seconds > retention_seconds:
                        file_size_mb = file_stat.st_size / (1024 * 1024)
                        os.remove(file_path)
                        files_deleted += 1
                        logger.debug(
                            f"Deleted aged batch: {os.path.basename(file_path)} "
                            f"(age: {file_age_seconds:.0f}s, size: {file_size_mb:.1f}MB)"
                        )

                except FileNotFoundError:
                    pass

                except OSError as os_err:
                    logger.warning(f"Cleanup: Could not delete {file_path}: {os_err}")

            if files_deleted > 0 or cleanup_iteration % 5 == 0:
                logger.info(
                    f"Cleanup cycle #{cleanup_iteration}: "
                    f"checked {files_checked} files, deleted {files_deleted}"
                )

            time.sleep(CLEANUP_CHECK_INTERVAL_SEC)

        except Exception as e:
            logger.error(
                f"Cleanup error (cycle #{cleanup_iteration}): {type(e).__name__}: {e}",
                exc_info=True,
            )
            time.sleep(CLEANUP_CHECK_INTERVAL_SEC)

# =============================================================================
# STREAM RECOVERY HANDLERS
# =============================================================================
def _handle_stream_stale(video_stream):
    logger.warning(
        f"Stream stale detected (no frames for {STREAM_STALE_TIMEOUT_SEC}s). Reconnecting..."
    )
    video_stream.stop()
    time.sleep(STREAM_RECONNECT_DELAY_SEC)

    try:
        new_stream = RTSPStreamLoader(RTSP_URL).start()
        time.sleep(STREAM_RECONNECT_DELAY_SEC)
        new_stream.last_read_time = time.time()
        logger.info("Stream reconnected successfully")
        return new_stream

    except Exception as e:
        logger.error(f"Stream reconnection failed: {type(e).__name__}: {e}", exc_info=True)
        time.sleep(STREAM_RECONNECT_WAIT_DELAY_SEC)
        return None


def _handle_frame_loss(video_stream):
    logger.warning(
        "Frame lost or camera disconnected. "
        f"Reconnecting in {FRAME_LOSS_RECONNECT_DELAY_SEC}s..."
    )
    video_stream.stop()
    time.sleep(FRAME_LOSS_RECONNECT_DELAY_SEC)
    new_stream = RTSPStreamLoader(RTSP_URL).start()
    return new_stream

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("RTSP Video Ingestion Service - Starting")
    logger.info("=" * 70)

    if not RTSP_URL:
        logger.error("CRITICAL: RTSP_URL environment variable not set")
        logger.error("Cannot proceed without RTSP stream source URL")
        sys.exit(1)

    logger.info("Configuration loaded:")
    logger.info(f"  - RTSP_URL: {RTSP_URL}")
    logger.info(f"  - Camera ID: {CAMERA_ID}")
    logger.info(f"  - Output Folder: {OUTPUT_FOLDER}")
    logger.info(f"  - Batch Size: {BATCH_SIZE} frames @ {RAW_FPS} FPS (~1 second)")
    logger.info(f"  - Target Size: {TARGET_SIZE}")
    logger.info(f"  - Retention: {RETENTION_SECONDS}s")

    logger.info("Initializing Redis connection...")
    redis_client = get_redis_client()
    if redis_client is None:
        logger.error("CRITICAL: Failed to connect to Redis after retries")
        sys.exit(1)
    lua_script = redis_client.register_script(LUA_RPUSH_LIMIT_SCRIPT)
    logger.info("Redis connection established and Lua script registered")

    logger.info(f"Starting batch I/O worker ({MAX_WORKER_THREADS} workers)...")
    batch_worker = BatchIOWorker(max_workers=MAX_WORKER_THREADS)
    batch_worker.set_redis_client(redis_client, lua_script)

    save_directory = os.path.join(OUTPUT_FOLDER, CAMERA_ID)
    try:
        os.makedirs(save_directory, exist_ok=True)
        logger.info(f"Output directory ready: {save_directory}")
    except OSError as e:
        logger.error(f"CRITICAL: Cannot create output directory: {e}")
        sys.exit(1)

    cleaner_thread = threading.Thread(
        target=cleanup_worker,
        args=(save_directory, RETENTION_SECONDS),
        daemon=True,
        name="cleanup-worker",
    )
    cleaner_thread.start()
    logger.info("File cleanup daemon started")

    logger.info(f"Connecting to RTSP stream: {CAMERA_ID}...")
    video_stream = RTSPStreamLoader(RTSP_URL).start()

    logger.info(f"Buffering stream for {INITIAL_BUFFER_FILL_SEC}s...")
    time.sleep(INITIAL_BUFFER_FILL_SEC)

    frame_batch = []
    batch_timestamps = []

    logger.info("Service started. Buffering frames for batch processing...")
    logger.info("=" * 70)

    frame_count = 0
    batch_count = 0

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
                batch_count += 1

                success = batch_worker.process_batch_async(
                    frame_batch, batch_timestamps, save_directory, CAMERA_ID
                )

                if success:
                    logger.debug(
                        f"Batch #{batch_count} submitted to worker pool "
                        f"(Total frames: {frame_count})"
                    )
                else:
                    logger.error(f"Failed to submit batch #{batch_count} to worker pool")

                frame_batch = []
                batch_timestamps = []

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {type(e).__name__}: {e}")
    finally:
        logger.info("=" * 60)
        logger.info("Shutdown initiated")
        logger.info("=" * 60)

        video_stream.stop()
        logger.info("RTSP stream closed")

        batch_worker.shutdown(wait=True, timeout=5)

        logger.info(f"Final stats: {frame_count} frames, {batch_count} batches processed")
        logger.info("Service stopped gracefully")


if __name__ == "__main__":
    main()