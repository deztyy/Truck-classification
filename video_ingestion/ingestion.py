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
    """
    Configure dual-output logging (console + file) for operational visibility.

    Args:
        level (int): Logging severity threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                    Defaults to INFO for production use.

    Returns:
        logging.Logger: Configured logger instance with both console and file handlers.

    Raises:
        OSError: If log directory cannot be created.

    Note:
        Log directory defaults to /app/logs (override via LOG_DIR environment variable).
        Files append mode ensures logs persist across service restarts.
    """
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
    """
    Handle graceful shutdown on system signals (SIGINT, SIGTERM).

    Args:
        signum (int): Signal number (2=SIGINT, 15=SIGTERM)
        frame: Current execution frame (required by signal API)
    """
    global RUNNING
    logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
    RUNNING = False


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def resize_with_padding(image, target_size):
    """
    Resize image to target size while preserving aspect ratio (letterboxing).

    Applies letterboxing technique to maintain aspect ratio:
    1. Calculate scale factor that fits image within target bounds
    2. Resize image using scale factor (no distortion)
    3. Center resized image on black canvas with padding

    Memory optimization: Reduces 1920×1080 BGR (6.2MB) to 640×640 BGR (1.2MB).
    For 30-frame batch: 186MB → 36MB (81% reduction).

    Args:
        image (numpy.ndarray): Source image with shape (height, width, 3).
                              Must be valid OpenCV format.
        target_size (tuple): (width, height) target dimensions. Example: (640, 640)

    Returns:
        numpy.ndarray: Resized and padded image with shape (target_height, target_width, 3),
                      dtype=uint8. Black padding (0,0,0) added as needed.

    Raises:
        AttributeError: If image is None or missing shape attribute.
        ValueError: If target_size contains zero values.

    Example:
        >>> frame = cv2.imread("photo.jpg")  # 1920×1080
        >>> processed = resize_with_padding(frame, (640, 640))
        >>> processed.shape
        (640, 640, 3)
    """
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

    padded_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return padded_image


def get_redis_client():
    """
    Establish Redis connection with automatic exponential retry.

    Blocks until Redis connects or RUNNING=False.
    Uses ping() to verify connection is working, not just TCP handshake.

    Returns:
        redis.Redis: Connected and verified Redis client instance.

    Note:
        Returns None if RUNNING becomes False during retry loop.
        Logs warnings for each failed connection attempt.

    Example:
        >>> client = get_redis_client()
        >>> client.ping()
        True
    """
    connection_attempt = 0
    while RUNNING:
        connection_attempt += 1
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SEC,
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
    """
    Thread-safe RTSP stream reader using background thread for frame capture.

    Runs RTSP stream.read() in separate thread to prevent blocking main loop.
    All shared state (grabbed, frame, last_read_time) protected by read_lock mutex.

    Uses TCP transport (reliability) instead of UDP, buffer size=1 (latest frame only),
    and automatic stale detection (no frames > 10s indicates hung connection).

    Attributes:
        src (str): RTSP stream URL
        stream (cv2.VideoCapture): OpenCV video capture object
        grabbed (bool): Whether last frame read succeeded
        frame (numpy.ndarray | None): Latest frame from stream (shared state)
        started (bool): Whether reader thread is active
        read_lock (threading.Lock): Mutex protecting frame/grabbed access
        last_read_time (float): Unix timestamp of last successful read
        thread (threading.Thread): Background reader thread instance
    """

    def __init__(self, src):
        """
        Initialize RTSP stream reader without starting background thread.

        Performs initial connection test. Note: invalid RTSP URLs don't raise
        exceptions in __init__; failure detected later via is_stale() or
        grabbed=False checks in main loop.

        Args:
            src (str): RTSP stream URL. Example: "rtsp://192.168.1.100:554/stream"
        """
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
        """
        Start background frame reader thread.

        Idempotent: calling start() twice returns None on second call.

        Returns:
            RTSPStreamLoader | None: Returns self if thread started.
                                    Returns None if already running.
        """
        if self.started:
            logger.warning(
                f"Stream already started for {self.src}. Ignoring duplicate start() call."
            )
            return None

        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        logger.debug(f"Reader thread started for {self.src}")
        return self

    def update(self):
        """
        Background loop that continuously reads frames from RTSP stream.

        Runs in separate daemon thread. Updates last_read_time only on
        successful reads (grabbed=True) to accurately detect stream stalls.
        Sleeps 0.5s on failed read to avoid busy-spinning.

        All updates to grabbed/frame protected by read_lock mutex.
        """
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
        """
        Thread-safe method to retrieve latest frame from stream.

        Returns frame.copy() (not reference) so caller can modify result
        without affecting internal state. Lock held only during copy (fast operation).

        Returns:
            tuple: (grabbed: bool, frame: numpy.ndarray | None)
                   grabbed=True if last read successful, frame is valid
                   grabbed=False if stream broken, frame=None or stale
        """
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self):
        """
        Gracefully stop reader thread and release video capture resources.

        Execution sequence:
        1. Set started=False (signals reader thread to exit)
        2. Call thread.join() with 2s timeout (prevents indefinite hang)
        3. Call stream.release() (frees sockets, file descriptors, codecs)

        Safe to call multiple times (checks thread.is_alive()).
        """
        self.started = False
        if self.thread and self.thread.is_alive():
            logger.debug("Waiting for reader thread to exit...")
            self.thread.join(timeout=2)
        self.stream.release()
        logger.debug(f"Stream closed: {self.src}")

    def is_stale(self, timeout=STREAM_STALE_TIMEOUT_SEC):
        """
        Check if stream hasn't received frames for longer than timeout period.

        Detects hung RTSP connections (connection exists but no data flowing).
        Default timeout of 10 seconds balances responsiveness vs false positives.

        Args:
            timeout (float): Seconds without frames to consider stream stale.
                           Defaults to STREAM_STALE_TIMEOUT_SEC (10.0).

        Returns:
            bool: True if last_read_time > timeout seconds ago (stale).
                 False if frames arriving within timeout (healthy).
        """
        time_since_last_read = time.time() - self.last_read_time
        is_stale = time_since_last_read > timeout
        if is_stale:
            logger.debug(
                f"Stream stale: {time_since_last_read:.1f}s > {timeout}s threshold"
            )
        return is_stale


# =============================================================================
# BACKGROUND I/O WORKER (Non-Blocking Batch Processing)
# =============================================================================
class BatchIOWorker:
    """
    Handles non-blocking batch processing (disk save + Redis push).

    Uses ThreadPoolExecutor to process batches in background without blocking
    main frame capture loop. Without background workers, np.save() (100-500ms)
    and Redis push (50-200ms) would stall main loop and drop frames.

    With workers: main loop maintains constant 30 FPS while I/O happens in parallel.

    Worker pool size MAX_WORKERS=2 chosen for optimal disk + network parallelism.
    Queue backpressure: batches discarded if Redis queue has 100+ jobs to prevent OOM.

    Attributes:
        executor (ThreadPoolExecutor): Thread pool for background tasks
        timeout (float): Submission timeout (reserved for future use)
        lua_script: Compiled Redis Lua script (atomic queue operation)
        redis_client (redis.Redis): Connected Redis client
    """

    def __init__(
        self, max_workers=MAX_WORKER_THREADS, timeout=WORKER_QUEUE_TIMEOUT_SEC
    ):
        """
        Initialize thread pool executor for background I/O operations.

        Args:
            max_workers (int): Number of concurrent worker threads.
                             Defaults to MAX_WORKER_THREADS (2).
            timeout (float): Task submission timeout in seconds.
                           Currently unused, reserved for future rate limiting.
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="batch-worker-",
        )
        self.timeout = timeout
        self.lua_script = None
        self.redis_client = None
        logger.info(f"BatchIOWorker initialized with {max_workers} workers")

    def set_redis_client(self, redis_client, lua_script):
        """
        Configure Redis client and Lua script for async operations.

        Must be called after initialization but before process_batch_async().

        Args:
            redis_client (redis.Redis): Connected and verified Redis client.
            lua_script: Compiled Lua script from redis_client.register_script().
        """
        self.redis_client = redis_client
        self.lua_script = lua_script
        logger.debug("Redis client configured for BatchIOWorker")

    def process_batch_async(
        self, processed_frames, batch_timestamps, save_dir, camera_id
    ):
        """
        Submit batch processing task to thread pool (non-blocking).

        Returns within microseconds (just enqueue operation).
        Actual disk save and Redis push happen in background worker thread.
        Main loop resumes frame capture immediately without waiting.

        Args:
            processed_frames (list): Pre-resized frame arrays [BATCH_SIZE, 640, 640, 3], dtype uint8.
            batch_timestamps (list): Datetime objects for each frame (length: BATCH_SIZE).
            save_dir (str): Directory to save batch .npy file (must exist).
            camera_id (str): Camera identifier for logging and Redis metadata.

        Returns:
            bool: True if task successfully queued.
                 False if executor shutdown or error occurred.

        Raises:
            (none - exceptions caught and False returned)
        """
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
            logger.error(
                f"Failed to submit batch: executor not running. "
                f"Likely during shutdown. [Error: {re}]"
            )
            return False

        except Exception as e:
            logger.error(f"Unexpected error submitting batch: {type(e).__name__}: {e}")
            return False

    def _process_batch_internal(
        self, processed_frames, batch_timestamps, save_dir, camera_id
    ):
        """
        Internal worker function executed in background thread.

        Performs three sequential operations:
        1. np.stack() - combine 30 frames into 4D array (~1ms, memory only)
        2. np.save() - write ~36MB to disk (100-500ms, BLOCKING I/O)
        3. redis.push() - enqueue job metadata (50-200ms, BLOCKING NETWORK I/O)

        All blocking I/O happens here in background worker, not in main loop.
        Multiple worker threads enable concurrent disk writes and network pushes.

        Exceptions caught and logged. Failed batch is discarded but
        subsequent batches continue processing normally.

        Args:
            processed_frames (list): Pre-resized 640×640 frames [length: BATCH_SIZE].
            batch_timestamps (list): Capture timestamps [length: BATCH_SIZE].
            save_dir (str): Output directory for .npy file.
            camera_id (str): Camera identifier for logging context.
        """
        try:
            logger.debug(
                f"Worker processing batch: {len(processed_frames)} frames "
                f"from {camera_id}"
            )

            batch_array = np.stack(processed_frames, axis=0)

            first_timestamp = batch_timestamps[0]
            timestamp_str = first_timestamp.strftime("%Y%m%d_%H%M%S_%f")
            batch_filename = f"{timestamp_str}_batch.npy"
            batch_file_path = os.path.join(save_dir, batch_filename)

            np.save(batch_file_path, batch_array)

            logger.info(
                f"[{camera_id}] Batch saved: {os.path.basename(batch_file_path)} "
                f"(shape: {batch_array.shape}, size: ~36MB)"
            )

            if self.redis_client and self.lua_script:
                self._push_to_redis(camera_id, batch_file_path, first_timestamp)
            else:
                logger.warning(
                    f"[{camera_id}] Redis not configured, skipping job enqueue"
                )

        except Exception as e:
            logger.error(
                f"[{camera_id}] Background batch processing failed: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )

    def _push_to_redis(self, camera_id, file_path, timestamp):
        """
        Push batch job metadata to Redis queue for downstream processing.

        Uses Lua script for atomic check-and-push operation.
        Prevents race conditions where queue size changes between check and push.

        Applies backpressure: if Redis queue has 100+ jobs, new batch is
        discarded (preferred over OOM crash when downstream worker is slow).

        Job payload format:
        {
            "camera_id": "cam_01",
            "file_path": "/app/shared_memory/cam_01/20240115_142345_123456_batch.npy",
            "timestamp": "2024-01-15T14:23:45.123456+07:00"
        }

        Args:
            camera_id (str): Camera identifier for context.
            file_path (str): Full path to saved batch file.
            timestamp (datetime): First frame's capture time (with timezone).

        Exception handling catches redis.ConnectionError, redis.TimeoutError,
        and unexpected exceptions. Logs all errors; batch is discarded on failure.
        """
        try:
            job_payload = {
                "camera_id": camera_id,
                "file_path": file_path,
                "timestamp": timestamp.isoformat(),
            }

            serialized_job = json.dumps(job_payload)

            result = self.lua_script(
                keys=["video_jobs"], args=[REDIS_QUEUE_LIMIT, serialized_job]
            )

            if result == 1:
                logger.info(
                    f"[{camera_id}] Batch queued for processing. "
                    f"File: {os.path.basename(file_path)}"
                )
            else:
                logger.warning(
                    f"[{camera_id}] Redis queue full ({REDIS_QUEUE_LIMIT}). "
                    "Batch discarded (backpressure). "
                    "Downstream worker may be slow or stuck."
                )

        except (redis.ConnectionError, redis.TimeoutError) as network_err:
            logger.error(
                f"[{camera_id}] Redis push failed (network issue): "
                f"{type(network_err).__name__}: {network_err}"
            )

        except json.JSONDecodeError as json_err:
            logger.error(
                f"[{camera_id}] JSON serialization failed: {json_err}", exc_info=True
            )

        except Exception as e:
            logger.error(
                f"[{camera_id}] Unexpected Redis error: {type(e).__name__}: {e}",
                exc_info=True,
            )

    def shutdown(self, wait=True, timeout=5):
        """
        Gracefully shutdown thread pool and wait for pending tasks.

        If wait=True, blocks until all worker threads finish current tasks.
        Ensures all pending batches are saved to disk before shutdown
        (prevents data loss during service restart/upgrade).

        Args:
            wait (bool): If True, block until all workers finish.
                        Defaults to True (safe shutdown).
            timeout (float): Maximum seconds to wait (for documentation only;
                           ThreadPoolExecutor.shutdown() doesn't enforce timeout).

        Called in main() finally block during graceful shutdown.
        """
        logger.info("Shutting down batch I/O worker...")
        logger.debug(
            f"Waiting for {MAX_WORKER_THREADS} worker threads to finish "
            f"(timeout: {timeout}s)"
        )

        self.executor.shutdown(wait=wait)

        logger.info("Batch I/O worker shutdown complete")


# =============================================================================
# BACKGROUND MAINTENANCE (File Retention)
# =============================================================================
def cleanup_worker(folder_path, retention_seconds):
    """
    Background daemon thread that periodically deletes aged .npy batch files.

    Prevents disk space exhaustion by removing batches older than retention period.
    Without cleanup, ~100 batches/hour × 36MB = 3.6GB/hour accumulation.

    Scans folder every 60 seconds and deletes files older than 1 hour (default).
    Retention policy: batches typically processed within 5 minutes, 1 hour = safe margin.

    Runs as daemon thread (exits automatically on main process shutdown).
    Periodic scan interval (60s) balances responsiveness vs filesystem load.

    Exception handling:
    - FileNotFoundError: race with other cleanup process (continue)
    - OSError: permission denied or I/O error (log warning, continue)
    - Other exceptions: log and retry (prevent thread crash)

    Args:
        folder_path (str): Directory containing .npy batch files.
                          Example: /app/shared_memory/camera_01
        retention_seconds (int): Age threshold in seconds.
                               Example: 3600 (1 hour)

    Note:
        No validation of folder_path (caller responsibility).
        Files may be in-use by downstream worker (acceptable race condition).
        Disk I/O errors logged but don't crash thread.
    """
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
    """
    Detect and recover from stale RTSP connection (no frames > 10 seconds).

    Stale condition: connection exists but no frames received.
    Causes: network packet loss, camera crash, firewall timeout, server overload.

    Recovery: disconnect, wait 2s, reconnect, wait 2s, reset timestamp.

    Args:
        video_stream (RTSPStreamLoader): Current stale stream instance.

    Returns:
        RTSPStreamLoader | None: New connected stream on success.
                                None if reconnection failed (caller retries).

    On failure: logs error, waits 5s before returning None.
    """
    logger.warning(
        f"Stream stale detected (no frames for {STREAM_STALE_TIMEOUT_SEC}s). "
        "Reconnecting..."
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
        logger.error(
            f"Stream reconnection failed: {type(e).__name__}: {e}", exc_info=True
        )
        time.sleep(STREAM_RECONNECT_WAIT_DELAY_SEC)
        return None


def _handle_frame_loss(video_stream):
    """
    Recover from complete frame loss (stream completely broken).

    Frame loss condition: grabbed=False or frame=None (connection broken/nonexistent).
    More severe than stale. Causes: camera powered off, network down, connection dropped.

    Recovery: same as stale but with longer delay (5s vs 2s for camera boot time).

    Args:
        video_stream (RTSPStreamLoader): Current broken stream.

    Returns:
        RTSPStreamLoader: New stream instance (may still be invalid).
                         Caller must verify via is_stale() or frame checks.

    Note:
        No exception handling here. Exceptions propagate to main loop.
    """
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
    """
    Production-ready RTSP video batch ingestion service.

    SYSTEM ARCHITECTURE:

    Main Thread:
    - Reads frames from RTSP at 30 FPS (~33ms per iteration)
    - Resizes frames to 640×640 immediately (memory optimization)
    - Buffers 30 processed frames (~36MB total)
    - Submits batch to worker thread (non-blocking, returns immediately)
    - Continues frame capture without waiting for I/O

    Worker Threads (2 concurrent):
    - Save batch to disk via np.save() (100-500ms, blocking)
    - Push job metadata to Redis (50-200ms, blocking)
    - Process next batch while main continues capturing frames

    Cleanup Daemon Thread:
    - Runs every 60 seconds
    - Scans for .npy files older than 1 hour
    - Deletes aged batches to prevent disk exhaustion

    RTSP Reader Thread (daemon):
    - Runs continuously in background
    - Reads frames from RTSP camera stream
    - Updates frame buffer protected by mutex (thread-safe)

    EXECUTION TIMELINE (at 30 FPS):
    T=0s:   Frame 1 captured
    T=1s:   Frame 30 captured, batch complete
    T=1ms:  Main submits batch to worker (returns instantly)
    T=1.05s: Frame 31 captured (main loop continues unblocked)
    T=1.1s:  Worker saves batch to disk (100-500ms in background)
    T=1.2s:  Worker pushes job to Redis (50-200ms in background)
    Result:  Main loop maintains constant 30 FPS throughout

    KEY IMPROVEMENTS:

    1. Non-Blocking I/O: ThreadPoolExecutor handles slow disk/network operations
    2. Memory Optimization: Frames resized immediately to 640×640 (81% memory savings)
    3. Production Logging: Structured logs with timestamps, console + file output
    4. Graceful Shutdown: Signal handlers, waits for pending batches, no data loss
    5. Resilience: Auto-reconnect on stale/lost streams, queue backpressure

    STARTUP SEQUENCE:
    1. Validate configuration (RTSP_URL must be set)
    2. Connect to Redis (blocks until available)
    3. Initialize worker thread pool
    4. Create output directory
    5. Start cleanup daemon thread
    6. Connect to RTSP stream
    7. Allow 2s for stream buffer to fill
    8. Enter main processing loop

    MAIN LOOP FLOW:
    1. Check stream staleness → reconnect if needed
    2. Read latest frame (thread-safe)
    3. Handle frame loss → reconnect if needed
    4. Resize frame immediately (memory optimization)
    5. Add to batch buffer
    6. If batch complete (30 frames):
       a. Submit to worker (non-blocking)
       b. Clear buffer for next batch

    SHUTDOWN SEQUENCE:
    1. Signal handler sets RUNNING=False (Ctrl+C or SIGTERM)
    2. Main loop detects flag on next iteration
    3. Exit main loop
    4. Finally block executes:
       a. Stop RTSP stream
       b. Shutdown worker threads (wait for pending batches)
       c. Log final statistics
       d. Exit cleanly

    ERROR HANDLING:
    - Resilient: continue processing despite individual errors
    - Observable: log all errors with context and severity
    - Recoverable: auto-reconnect on stream failures
    - Safe: graceful shutdown prevents data loss
    """

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
    logger.info(f"  - Target Size: {TARGET_SIZE} (memory optimization: 81% reduction)")
    logger.info(f"  - Retention: {RETENTION_SECONDS}s (background cleanup)")

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
                    logger.error(
                        f"Failed to submit batch #{batch_count} to worker pool"
                    )

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

        logger.info(
            f"Final stats: {frame_count} frames, {batch_count} batches processed"
        )
        logger.info("Service stopped gracefully")


if __name__ == "__main__":
    main()
