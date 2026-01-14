import datetime
import glob
import json
import os
import signal
import sys
import threading
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import redis

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
REDIS_QUEUE_LIMIT = 100  # Max jobs in queue before backpressure (stop pushing)
REDIS_CONNECT_TIMEOUT_SEC = 2  # Socket timeout for Redis connections
REDIS_RETRY_INTERVAL_SEC = 5  # Seconds between Redis reconnection attempts

# Image Processing & Storage Settings
TARGET_SIZE = (640, 640)  # Model input dimensions (Width, Height)
PROCESS_FPS = 1.0  # Frame capture rate (Frames Per Second)
JPG_QUALITY = 80  # JPEG compression quality (0-100, unused but kept for future use)
RETENTION_SECONDS = 3600  # File retention period (1 hour = 3600 sec)
THAI_TZ = ZoneInfo("Asia/Bangkok")  # Timezone for capture timestamps

# Stream Management Constants
STREAM_STALE_TIMEOUT_SEC = 10.0  # Detect stale stream after 10 seconds without frames
FRAME_READ_RETRY_DELAY_SEC = 0.5  # Delay before retrying failed frame reads
STREAM_RECONNECT_DELAY_SEC = 2  # Delay before attempting stream reconnection
STREAM_RECONNECT_WAIT_DELAY_SEC = 5  # Extended delay if reconnection fails
FRAME_LOSS_RECONNECT_DELAY_SEC = 5  # Delay when frame completely lost
CLEANUP_CHECK_INTERVAL_SEC = 60  # How often cleanup worker checks for expired files
INITIAL_BUFFER_FILL_SEC = 2.0  # Time to let stream buffer fill on startup

# Loop Control
RUNNING = True  # Global flag to gracefully stop the service

# Redis Lua Script for atomic queue push with size limit check
# Returns 1 if push succeeded, 0 if queue size >= limit
# This prevents Redis memory overflow when consumer is slow
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
    Gracefully shuts down the service on system signals.

    Signal Handling:
    This handler is registered for SIGINT (Ctrl+C) and SIGTERM (kill -15).
    Sets RUNNING=False to signal main loop to exit, allowing cleanup.

    Edge Cases:
    - Signal received during file write: write may complete before shutdown
    - Redis client: not explicitly closed (will cleanup on exit)
    - Stream thread: daemon thread (will exit with main thread)

    Args:
        signum (int): Signal number (e.g., 2 for SIGINT, 15 for SIGTERM).
        frame: Current stack frame at signal time (unused but required by signal API).
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
    Resizes an image to target size while maintaining aspect ratio with padding.

    Uses "letterboxing" technique: scales image to fit target dimensions without
    distortion, then centers it on a black background.

    Edge Cases:
    - If image is None or invalid shape: will raise cv2 or numpy errors (uncaught)
    - If target_size has zero dimensions: will cause division by zero
    - Very large images may cause memory issues during resize

    Args:
        image (numpy.ndarray): Source image with shape (height, width, channels).
                               Must be valid OpenCV format.
        target_size (tuple): (width, height) dimensions for output image.
                            Both values must be > 0.

    Returns:
        numpy.ndarray: Resized and padded image with shape (target_height, target_width, 3),
                      dtype uint8, with black (0,0,0) padding.

    Raises:
        AttributeError: If image is None or doesn't have shape attribute.
        ValueError: If target_size has zero values (implicit from min/division).
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate scale factor that fits image in target without distorting it.
    # Using min() ensures the entire image fits within bounds (letterboxing).
    scale = min(target_width / width, target_height / height)
    new_width, new_height = int(width * scale), int(height * scale)

    # Resize to calculated dimensions preserving aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create black canvas as background for padding
    padded_image = np.full((target_height, target_width, 3), 0, dtype=np.uint8)

    # Calculate offset to center the resized image on the canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Place resized image at center of canvas
    padded_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return padded_image


def get_redis_client():
    """
    Establishes and returns a Redis client with automatic retry logic.

    Blocks until connection is successful or RUNNING flag is set to False.
    Each retry waits REDIS_RETRY_INTERVAL_SEC seconds.

    Edge Cases:
    - If RUNNING is set False during connection attempt, function will exit
    - Network timeouts during ping() will retry (not fail immediately)
    - Exception during Redis instantiation is caught but connection attempt retries

    Returns:
        redis.Redis: Connected Redis client instance with default encoding.
                    Returns None implicitly if RUNNING becomes False (infinite loop exits).

    Raises:
        SystemExit: Indirectly - if RUNNING is never set to True initially (shouldn't happen).
    """
    redis_client = None
    while RUNNING:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SEC,
            )
            # Test connection with ping
            redis_client.ping()
            print(f"🟢 Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return redis_client

        except redis.ConnectionError:
            print(f"🔴 Redis not ready. Retrying in {REDIS_RETRY_INTERVAL_SEC}s...")
            time.sleep(REDIS_RETRY_INTERVAL_SEC)

        except Exception as e:
            # Catch unexpected errors (not just ConnectionError)
            print(f"🔥 Redis Connection Error: {type(e).__name__}: {e}")
            time.sleep(REDIS_RETRY_INTERVAL_SEC)


# =============================================================================
# THREADED VIDEO CAPTURE CLASS
# =============================================================================
class RTSPStreamLoader:
    """
    Thread-safe RTSP stream reader that runs frame capture in background thread.

    Prevents main processing loop from blocking on network I/O or codec delays
    by continuously reading frames into a buffer. Maintains timestamp of last
    successful read to detect stale connections.

    Design Notes:
    - Uses daemon thread for automatic cleanup on application exit
    - Minimal buffer (size=1) to get latest frame, reducing latency
    - TCP transport enforced to reduce packet loss on unreliable networks
    - Thread-safe frame access via read_lock mutex

    Edge Cases:
    - If stream URL is invalid, cv2.VideoCapture will succeed but read() will fail
    - Frame can be None even if grabbed=True (codec error)
    - start() called twice will return None second time (safety check)
    - Thread may not immediately exit after setting started=False (join timeout issues)

    Attributes:
        src (str): RTSP stream URL.
        stream (cv2.VideoCapture): OpenCV video capture object.
        grabbed (bool): Flag indicating if last read was successful.
        frame (numpy.ndarray | None): Latest frame from stream or None if read failed.
        started (bool): Flag indicating if reader thread is currently active.
        read_lock (threading.Lock): Mutex protecting frame and grabbed access.
        last_read_time (float): Unix timestamp of last successful frame read.
        thread (threading.Thread): Background reader thread (created in start()).
    """

    def __init__(self, src):
        """
        Initialize RTSP stream reader without starting background thread.

        Args:
            src (str): RTSP stream URL (e.g., "rtsp://camera:554/stream")
        """
        self.src = src
        self.stream = cv2.VideoCapture(src)

        # Use TCP protocol instead of UDP to reduce packet loss on unstable networks
        # and minimize artifacting from lost data packets
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # Set buffer size to 1 to always get the latest frame instead of accumulating
        # old frames. Reduces latency significantly in real-time processing.
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try initial read - will succeed only if stream is valid
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.last_read_time = time.time()
        self.thread = None

    def start(self):
        """
        Starts the background frame reader thread.

        Thread runs continuously until stop() is called. Daemon thread ensures
        it won't prevent application shutdown.

        Returns:
            RTSPStreamLoader: Returns self for method chaining, or None if already started.

        Safety Note:
            Calling start() multiple times on same instance will skip after first call.
        """
        if self.started:
            print("⚠️ Stream already started!!")
            return None

        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        """
        Background loop that continuously reads frames from the stream.

        Runs in separate thread to prevent blocking main process loop.
        Updates last_read_time only on successful reads (grabbed=True).
        Small delay on read failures prevents busy-waiting.

        Safety Notes:
        - Uses read_lock to ensure thread-safe updates of grabbed/frame
        - If stream becomes invalid, will continue looping with delays
        - Exception during stream.read() is NOT caught (will crash thread)
        """
        while self.started:
            grabbed, frame = self.stream.read()

            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                # Only update timestamp on successful reads (not on failures)
                if grabbed:
                    self.last_read_time = time.time()

            # If read failed, brief delay before retrying to avoid busy loop
            if not grabbed:
                time.sleep(FRAME_READ_RETRY_DELAY_SEC)

    def read(self):
        """
        Safely returns the latest frame from the stream.

        Thread-safe method that copies frame data while holding lock.
        Always returns tuple (grabbed, frame_copy) for consistency.

        Returns:
            tuple: (grabbed: bool, frame: numpy.ndarray | None)
                   - grabbed: True if last read succeeded
                   - frame: Copy of latest frame, or None if not available

        Note:
            Returns frame.copy() to prevent external modifications of internal state.
        """
        with self.read_lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        """
        Gracefully stops the reader thread and releases resources.

        Signals thread to stop, waits for it to exit (with implicit timeout),
        then releases the underlying video capture resource.

        Safety Notes:
        - Sets started=False to signal thread to exit its loop
        - Calls join() to wait for thread termination (may hang if thread is stuck)
        - Always calls release() to free video capture resources
        """
        self.started = False

        if self.thread and self.thread.is_alive():
            self.thread.join()

        self.stream.release()

    def is_stale(self, timeout=STREAM_STALE_TIMEOUT_SEC):
        """
        Checks if stream hasn't received frames for longer than timeout period.

        Used to detect connection hangs or camera disconnections.

        Args:
            timeout (float): Seconds without frames to consider stream stale.
                           Defaults to STREAM_STALE_TIMEOUT_SEC.

        Returns:
            bool: True if time since last read > timeout, False otherwise.

        Note:
            This check should be performed periodically in main loop to detect
            and recover from hung connections before too much time passes.
        """
        time_since_last_read = time.time() - self.last_read_time
        return time_since_last_read > timeout


# =============================================================================
# BACKGROUND MAINTENANCE
# =============================================================================
def cleanup_worker(folder_path, retention_seconds):
    """
    Background worker thread that periodically deletes aged files.

    Scans folder for .npy files and removes any older than retention period.
    Runs continuously while RUNNING flag is True. Each scan cycle waits
    CLEANUP_CHECK_INTERVAL_SEC before checking again.

    Edge Cases:
    - If folder doesn't exist: glob returns empty list (no error)
    - File may be deleted between glob() and os.stat() (race condition)
    - Permission errors on delete will be caught as generic Exception
    - Very large folders may cause slow scans (blocking other operations)

    Args:
        folder_path (str): Directory path containing .npy files to manage.
                          Should exist or be creatable by caller.
        retention_seconds (int): Age threshold in seconds. Files older than
                                this value will be deleted.
                                Typically 3600 (1 hour) or higher.

    Returns:
        None: Function runs in infinite loop until RUNNING is False.

    Thread Safety Notes:
    - Safe to run in background thread (only reads folder, modifies files)
    - Does NOT handle concurrent modifications well (TOCTOU race condition)
    - File deletion race: if main process writes file between glob and delete,
      that file may be deleted if it's old enough
    """
    print(
        f"🧹 Cleanup service started for {folder_path} (Retention: {retention_seconds}s)"
    )

    while RUNNING:
        try:
            now = time.time()

            # Find all NumPy files in the directory
            files = glob.glob(os.path.join(folder_path, "*.npy"))

            for file_path in files:
                try:
                    # Get file modification time
                    file_mod_time = os.stat(file_path).st_mtime
                    file_age_seconds = now - file_mod_time

                    # Delete if older than retention limit
                    if file_age_seconds > retention_seconds:
                        os.remove(file_path)
                        # Silently succeed - don't spam logs with successful deletions

                except FileNotFoundError:
                    # File already deleted by another process - skip gracefully
                    pass
                except OSError as os_err:
                    # Permission errors or other OS errors
                    print(f"⚠️ Cleanup: Could not delete {file_path}: {os_err}")

            # Check again after interval
            time.sleep(CLEANUP_CHECK_INTERVAL_SEC)

        except Exception as e:
            # Broad catch for unexpected errors (glob issues, etc.)
            print(f"⚠️ Cleanup Error: {type(e).__name__}: {e}")
            time.sleep(CLEANUP_CHECK_INTERVAL_SEC)


# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def _handle_stream_stale(video_stream):
    """
    Detect and recover from stale stream connection.

    Stops current stream and attempts reconnection with timeout delays.
    Returns new stream on success, None on failure.

    Edge Cases:
    - If new stream creation throws unexpected exception, returns None
    - Initial connection delay (STREAM_RECONNECT_DELAY_SEC) may queue frames in buffer
    - If RTSP_URL is invalid, will fail but continue main loop

    Args:
        video_stream (RTSPStreamLoader): Current stream instance to disconnect.

    Returns:
        RTSPStreamLoader | None: Connected stream on success, None if reconnection failed.

    Caller Responsibility:
        Main loop must check return value - if None, should retry this function
        or implement exponential backoff.
    """
    print("⚠️ Stream stale detected. Reconnecting...")
    video_stream.stop()
    time.sleep(STREAM_RECONNECT_DELAY_SEC)

    try:
        new_stream = RTSPStreamLoader(RTSP_URL).start()
        time.sleep(STREAM_RECONNECT_DELAY_SEC)  # Let buffer fill again
        new_stream.last_read_time = time.time()
        print("✅ Reconnected to stream successfully.")
        return new_stream

    except Exception as e:
        # Catch unexpected exceptions during reconnection
        print(f"🔥 Reconnection Error: {type(e).__name__}: {e}")
        time.sleep(STREAM_RECONNECT_WAIT_DELAY_SEC)
        return None


def _handle_frame_loss(video_stream):
    """
    Recover from complete frame loss (None or grab failure).

    Stops current stream with extended delay before reconnection attempt.
    Useful when stream completely drops connection.

    Args:
        video_stream (RTSPStreamLoader): Current stream instance.

    Returns:
        RTSPStreamLoader: New stream (caller should verify connection succeeded).

    Note:
        Unlike _handle_stream_stale, this returns the new stream regardless
        of success/failure. Caller must verify grab/frame validity.
    """
    print("⚠️ Frame lost or Camera disconnected. Reconnecting in 5s...")
    video_stream.stop()
    time.sleep(FRAME_LOSS_RECONNECT_DELAY_SEC)
    return RTSPStreamLoader(RTSP_URL).start()


def _push_job_to_redis(redis_client, lua_script, camera_id, file_path, timestamp):
    """
    Push a processing job to Redis queue with automatic backpressure handling.

    Uses atomic Lua script to prevent race conditions when checking queue size.
    Returns False on connection error (caller should reconnect and retry).

    Queue Full Behavior:
    - When queue reaches REDIS_QUEUE_LIMIT, further pushes are rejected
    - This prevents Redis memory overflow when downstream consumer is slow
    - Frames are simply dropped (not retried) - acceptable for real-time stream

    Edge Cases:
    - JSON serialization errors will raise exception (uncaught)
    - Redis connection errors caught and handled gracefully
    - Large file_path strings may cause Redis memory issues if accumulated

    Args:
        redis_client (redis.Redis): Connected Redis client instance.
        lua_script: Registered Lua script for atomic queue operation.
        camera_id (str): Camera identifier for job metadata.
        file_path (str): Full path to saved .npy file.
        timestamp (datetime): Capture time from camera timezone.

    Returns:
        bool: True if job was pushed to queue.
              False if queue limit exceeded or Redis error occurred.

    Performance Note:
        JSON serialization happens twice (nested dumps) - could be optimized
        but current approach is clearer and memory impact is minimal.
    """
    try:
        # Build job metadata
        job_metadata = json.dumps(
            {
                "camera_id": camera_id,
                "file_path": file_path,
                "timestamp": timestamp.isoformat(),
            }
        )
        # Double JSON encoding for consistency with original design
        serialized_job = json.dumps(job_metadata)

        # Execute Lua script atomically: returns 1 if pushed, 0 if queue full
        result = lua_script(
            keys=["video_jobs"], args=[REDIS_QUEUE_LIMIT, serialized_job]
        )

        # Success if Lua script returned 1
        return result == 1

    except (redis.ConnectionError, redis.TimeoutError) as redis_err:
        print(f"🔥 Redis Error: {type(redis_err).__name__}: {redis_err}")
        print("   Caller should reconnect and retry")
        return False


def main():
    """
    Main execution loop for RTSP video ingestion service.

    Service Architecture:
    - Continuously reads frames from RTSP camera at specified FPS
    - Resizes/pads frames to fixed dimensions using letterboxing
    - Saves processed frames as NumPy arrays to disk
    - Publishes job metadata to Redis for downstream workers
    - Implements backpressure control when Redis queue backs up
    - Auto-recovers from stream stalls and connection losses
    - Background thread cleans up aged files per retention policy

    Main Loop Flow:
    1. Check stream staleness (no frames > STREAM_STALE_TIMEOUT_SEC)
    2. FPS throttling (maintain PROCESS_FPS rate)
    3. Read frame from stream (non-blocking)
    4. Process frame (resize with padding to TARGET_SIZE)
    5. Save to disk with timestamp filename
    6. Push job metadata to Redis with queue size check
    7. Repeat until RUNNING=False (graceful shutdown)

    Edge Cases & Error Handling:
    - RTSP_URL not set: Early exit with error message
    - Redis connection fails: Infinite retry with exponential delays
    - Stream stalls: Auto-reconnect with timeout checks
    - Frame loss: Auto-reconnect with longer delay
    - Redis queue full: Skip frame push (acceptable data loss in real-time)
    - Processing exceptions: Log and continue (resilient)

    Signal Handling:
    - SIGINT/SIGTERM: Sets RUNNING=False for graceful shutdown

    Thread Safety:
    - Main loop thread: frame processing, Redis pushes
    - Cleanup thread: background file deletion (separate from main)
    - RTSP thread: background frame reading (inside RTSPStreamLoader)

    Resource Cleanup:
    - All threads are daemon threads (will exit on app shutdown)
    - Video stream explicitly stopped on exit
    - Redis connection kept open (will auto-cleanup on exit)
    """
    # INITIALIZATION PHASE
    # ====================================================================

    # 1. Validation: Ensure required configuration is present
    if not RTSP_URL:
        print("❌ Error: RTSP_URL environment variable is not set.")
        sys.exit(1)

    # 2. Redis Connection: Establish connection with auto-retry
    print("📋 Initializing Redis connection...")
    redis_client = get_redis_client()
    lua_script = redis_client.register_script(LUA_RPUSH_LIMIT_SCRIPT)

    # 3. Storage Setup: Create output directory if needed
    save_dir = os.path.join(OUTPUT_FOLDER, CAMERA_ID)
    os.makedirs(save_dir, exist_ok=True)

    # 4. Cleanup Thread: Start background file retention worker
    cleaner_thread = threading.Thread(
        target=cleanup_worker, args=(save_dir, RETENTION_SECONDS)
    )
    cleaner_thread.daemon = True
    cleaner_thread.start()

    # 5. Stream Connection: Initialize RTSP reader with background thread
    print(f"📡 Connecting to Camera: {CAMERA_ID}...")
    video_stream = RTSPStreamLoader(RTSP_URL).start()

    # Allow RTSP stream buffer to fill before starting main loop
    # (prevents stale timeout on first few frames)
    time.sleep(INITIAL_BUFFER_FILL_SEC)

    # Main loop timing variables
    last_process_time = 0.0  # Timestamp of last processed frame
    process_interval = 1.0 / PROCESS_FPS  # Minimum seconds between frames

    print("🚀 Service Started. Processing frames...")

    # MAIN PROCESSING LOOP
    # ====================================================================

    while RUNNING:
        current_time = time.time()

        # === Stream Health Check ===
        # Monitor for stale connections (no frames received for timeout period)
        if video_stream.is_stale(timeout=STREAM_STALE_TIMEOUT_SEC):
            new_stream = _handle_stream_stale(video_stream)
            if new_stream:
                video_stream = new_stream
            continue

        # === FPS Throttling ===
        # Enforce maximum frame processing rate to:
        # - Reduce CPU/disk load
        # - Maintain consistent frame interval
        # - Allow processing other tasks
        if current_time - last_process_time < process_interval:
            time.sleep(0.01)  # Brief sleep to avoid busy loop
            continue

        # === Frame Retrieval ===
        # Non-blocking read of latest available frame
        grabbed, frame = video_stream.read()
        capture_time = datetime.datetime.now(THAI_TZ)

        # === Handle Complete Frame Loss ===
        # This differs from FPS throttling - means stream is broken
        if not grabbed or frame is None:
            video_stream = _handle_frame_loss(video_stream)
            continue

        # === Frame Processing & Storage ===
        try:
            # Resize frame to model input dimensions with aspect ratio preservation
            processed_frame = resize_with_padding(frame, TARGET_SIZE)

            # Generate unique filename using timestamp (microsecond precision)
            timestamp_str = capture_time.strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{timestamp_str}.npy"
            full_path = os.path.join(save_dir, file_name)

            # Save processed frame as NumPy array (faster I/O than JPEG)
            np.save(full_path, processed_frame)

            # === Job Queueing ===
            # Push job to Redis for downstream processing (ML inference, etc.)
            # Update timing here so we count from after disk write
            last_process_time = current_time

            is_queued = _push_job_to_redis(
                redis_client, lua_script, CAMERA_ID, full_path, capture_time
            )

            # === Status Logging ===
            # Log every 10 seconds to avoid spam but show we're working
            if is_queued:
                if int(current_time) % 10 == 0:
                    print(f"✅ [{CAMERA_ID}] Frame processed and queued", flush=True)
            else:
                # Queue is full - consumer is slow, backpressure is working
                print(
                    f"⚠️ [{CAMERA_ID}] Redis queue full. Skipping frame to prevent overload.",
                    flush=True,
                )

        except Exception as e:
            # === Error Recovery ===
            # Log processing errors but continue (resilient architecture)
            print(f"🔥 Processing Error: {type(e).__name__}: {e}")

            # Special handling for Redis connection errors
            if isinstance(e, (redis.ConnectionError, redis.TimeoutError)):
                print("   Reconnecting to Redis...")
                try:
                    redis_client = get_redis_client()
                    lua_script = redis_client.register_script(LUA_RPUSH_LIMIT_SCRIPT)
                except Exception as reconnect_err:
                    print(f"   Reconnection failed: {reconnect_err}")

            continue

    # SHUTDOWN PHASE
    # ====================================================================

    # Graceful cleanup - stop stream reader and release resources
    video_stream.stop()
    print("👋 Service Stopped Gracefully.")


if __name__ == "__main__":
    main()
