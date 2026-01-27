"""
Video Ingestion Script for Computer Vision Pipeline
====================================================
This script reads video streams from RTSP sources, batches frames,
uploads them to MinIO (S3-compatible storage), and publishes metadata to Redis.

Date: January 2026
"""

import io
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import redis
from minio import Minio
from minio.error import S3Error

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Defaults and constants
DEFAULT_FRAME_SIZE: Tuple[int, int] = (640, 640)
DEFAULT_BATCH_SIZE = 30


class VideoIngestor:
    """
    Video Ingestion Class for Computer Vision Pipeline

    This class handles:
    - Reading RTSP video streams
    - Batching frames into NumPy arrays
    - Uploading batches to MinIO storage
    - Publishing metadata to Redis queue
    - Auto-reconnection on stream failures

    Attributes:
        camera_id (str): Unique identifier for the camera
        rtsp_url (str): RTSP stream URL
        batch_size (int): Number of frames per batch
        minio_client (Minio): MinIO client instance
        redis_client (redis.Redis): Redis client instance
        bucket_name (str): MinIO bucket name for storage
        redis_queue_name (str): Redis list name for queue
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: Optional[str] = None,
        video_file: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 5,
        loop_video: bool = True,
        frame_skip: int = 0,
    ):
        """
        Initialize VideoIngestor with configuration from environment variables.

        Args:
            camera_id: Unique identifier for this camera/stream
            rtsp_url: RTSP stream URL (defaults to env variable)
            video_file: Path to video file (alternative to RTSP)
            batch_size: Number of frames to collect before uploading
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts (seconds)
            loop_video: Loop video file when it ends (default: True)
            frame_skip: Number of frames to skip between captures (0 = no skip, 1 = skip 1 frame, etc.)
        """
        self.camera_id = camera_id
        self.video_file = video_file
        self.rtsp_url = rtsp_url or os.getenv("RTSP_URL")
        self.loop_video = loop_video
        self.batch_size = batch_size
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.frame_skip = frame_skip

        # Validate required configuration
        if not self.rtsp_url and not self.video_file:
            raise ValueError("Either RTSP_URL or video_file must be provided")

        # Initialize MinIO client
        self.minio_client = self._initialize_minio()

        # Initialize Redis client
        self.redis_client = self._initialize_redis()

        # Storage configuration
        self.minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "raw-frames")
        self.redis_list_name = os.getenv("REDIS_QUEUE_NAME", "frame_batches")

        # Ensure bucket exists
        self._ensure_bucket_exists()

        # Video capture object (initialized on start)
        self.video_capture: Optional[cv2.VideoCapture] = None

        # Statistics
        self.total_frames_processed = 0
        self.total_batches_uploaded = 0
        self.total_errors = 0

        logger.info(f"VideoIngestor initialized for camera: {self.camera_id}")

    def _initialize_minio(self) -> Minio:
        """
        Initialize MinIO client from environment variables.

        Returns:
            Minio: Configured MinIO client

        Environment Variables Required:
            - MINIO_ENDPOINT: MinIO server endpoint (e.g., 'minio:9000')
            - MINIO_ACCESS_KEY: Access key for authentication
            - MINIO_SECRET_KEY: Secret key for authentication
            - MINIO_SECURE: Use HTTPS (default: 'false')
        """
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

        # Validate required credentials
        if not endpoint or not access_key or not secret_key:
            raise ValueError(
                "Missing required MinIO environment variables: "
                "MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY"
            )

        logger.info(f"Connecting to MinIO at {endpoint} (secure={secure})")

        return Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

    def _initialize_redis(self) -> redis.Redis:
        """
        Initialize Redis client from environment variables.

        Returns:
            redis.Redis: Configured Redis client

        Environment Variables Required:
            - REDIS_HOST: Redis server hostname
            - REDIS_PORT: Redis server port
            - REDIS_DB: Redis database number (default: '0')
            - REDIS_PASSWORD: Redis password (optional)
        """
        host = os.getenv("REDIS_HOST")
        port = os.getenv("REDIS_PORT")
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD", None)

        # Validate required configuration
        if not host or not port:
            raise ValueError(
                "Missing required Redis environment variables: REDIS_HOST, REDIS_PORT"
            )

        port = int(port)
        logger.info(f"Connecting to Redis at {host}:{port}")

        return redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # We'll handle binary data
        )

    def _ensure_bucket_exists(self) -> None:
        """
        Ensure the MinIO bucket exists, create if it doesn't.
        """
        try:
            if not self.minio_client.bucket_exists(self.minio_bucket_name):
                self.minio_client.make_bucket(self.minio_bucket_name)
                logger.info(f"Created MinIO bucket: {self.minio_bucket_name}")
            else:
                logger.info(f"MinIO bucket exists: {self.minio_bucket_name}")
        except S3Error as e:
            logger.error(f"Error checking/creating bucket: {e}")
            raise

    def _open_video_source(self) -> bool:
        """
        Connect to the RTSP video stream or open a local video file.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Determine source
            source = self.video_file if self.video_file else self.rtsp_url
            source_type = "video file" if self.video_file else "RTSP stream"

            logger.info(f"Connecting to {source_type}: {source}")
            self.video_capture = cv2.VideoCapture(source)

            if not self.video_capture.isOpened():
                logger.error(f"Failed to open {source_type}")
                return False

            # Read one frame to verify stream is working
            ret, _ = self.video_capture.read()
            if not ret:
                logger.error(f"Failed to read frame from {source_type}")
                return False

            # Get video info
            if self.video_file:
                total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                logger.info(f"Video file info - Frames: {total_frames}, FPS: {fps:.2f}")

            logger.info(f"Successfully connected to {source_type}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to stream: {e}")
            return False

    def _close_video_source(self) -> None:
        """
        Safely disconnect from the video stream.
        """
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            logger.info("Disconnected from video stream")

    def _read_next_frame_batch(self) -> Optional[np.ndarray]:
        """
        Read a batch of frames from the video stream.

        Returns:
            Optional[np.ndarray]: Batch of frames as NumPy array (batch_size, height, width, channels)
                                 Returns None if stream fails

        ### CUSTOMIZATION POINT ###
        You can modify this method to:
        - Apply pre-processing to frames (resize, normalize, color conversion)
        - Skip frames (e.g., process every Nth frame)
        - Add frame metadata (timestamps, frame numbers)
        """
        frames: list[np.ndarray] = []

        for i in range(self.batch_size):
            try:
                # Read frame
                ret, frame = self.video_capture.read()

                # If video file ends and loop is enabled, restart
                if not ret and self.video_file and self.loop_video:
                    logger.info("Video file ended, restarting from beginning...")
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video_capture.read()

                if not ret:
                    # Distinguish between expected EOF for non-looping local files
                    # and other read failures (e.g., RTSP/network issues).
                    if self.video_file and not self.loop_video:
                        logger.info(
                            "End of video file reached and loop_video is False; "
                            "stopping ingestion for this source."
                        )
                        # Signal EOF to the caller so it can terminate cleanly
                        raise EOFError("End of video file reached")

                    logger.warning(f"Failed to read frame {i + 1}/{self.batch_size}")
                    return None

                # Skip frames if configured (frame_skip > 0)
                for _ in range(self.frame_skip):
                    ret, _ = self.video_capture.read()
                    if not ret and self.video_file and self.loop_video:
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if not ret:
                        break

                # --- CUSTOM PREPROCESSING GOES HERE ---
                # Example: frame = cv2.resize(frame, (640, 640))
                # Example: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, DEFAULT_FRAME_SIZE)

                frames.append(frame)
                self.total_frames_processed += 1

            except EOFError:
                # Propagate EOF so the main loop can exit gracefully
                raise
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                return None

        # Convert list of frames to NumPy array
        # Shape: (batch_size, height, width, channels)
        frames_batch = np.array(frames)
        logger.debug(f"Created batch with shape: {frames_batch.shape}")

        return frames_batch

    def _serialize_numpy(self, array: np.ndarray) -> io.BytesIO:
        """
        Serialize a NumPy array into an in-memory BytesIO buffer.

        Args:
            array: The NumPy array to serialize

        Returns:
            BytesIO buffer positioned at start
        """
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        buffer.seek(0)
        return buffer

    def _upload_frame_batch(self, frame_batch: np.ndarray) -> Optional[str]:
        """
        Upload frame batch to MinIO storage using in-memory buffer.

        Args:
            frame_batch: NumPy array containing frame batch

        Returns:
            Optional[str]: Object name in MinIO if successful, None otherwise

        ### CUSTOMIZATION POINT ###
        You can modify this method to:
        - Use different serialization formats (npz, pickle, msgpack)
        - Add compression (zlib, lz4)
        - Include additional metadata in the object
        """
        try:
            # Generate unique object name with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            object_key = f"{self.camera_id}/batch_{timestamp}.npy"

            # Serialize NumPy array to bytes using io.BytesIO
            buffer = self._serialize_numpy(frame_batch)

            # Get buffer size
            buffer_size = buffer.getbuffer().nbytes

            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=self.minio_bucket_name,
                object_name=object_key,
                data=buffer,
                length=buffer_size,
                content_type="application/octet-stream",
            )

            logger.info(f"Uploaded batch to MinIO: {object_key} ({buffer_size} bytes)")
            self.total_batches_uploaded += 1

            return object_key

        except S3Error as e:
            logger.error(f"MinIO upload error: {e}")
            self.total_errors += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            self.total_errors += 1
            return None

    def _publish_batch_metadata(self, object_key: str, batch_shape: tuple) -> bool:
        """
        Publish metadata to Redis queue for downstream processing.

        Args:
            object_name: Name of the object in MinIO
            batch_shape: Shape of the uploaded batch

        Returns:
            bool: True if publish successful, False otherwise

        ### CUSTOMIZATION POINT ###
        You can modify this method to:
        - Add more metadata fields (FPS, resolution, codec info)
        - Use different Redis data structures (Streams, Pub/Sub)
        - Add priority queuing logic
        """
        try:
            # Create metadata message
            metadata = {
                "camera_id": self.camera_id,
                "object_name": object_key,
                "timestamp": datetime.utcnow().isoformat(),
                "batch_size": batch_shape[0],
                "frame_shape": list(batch_shape[1:]),
                "bucket_name": self.minio_bucket_name,
            }

            # --- ADD CUSTOM METADATA HERE ---
            # Example: metadata['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
            # Example: metadata['processing_priority'] = 'high'

            # Serialize to JSON and push to Redis List
            message_json = json.dumps(metadata)
            self.redis_client.rpush(self.redis_list_name, message_json)

            logger.info(f"Published metadata to Redis list: {self.redis_list_name}")
            logger.debug(f"Metadata: {metadata}")

            return True

        except redis.RedisError as e:
            logger.error(f"Redis publish error: {e}")
            self.total_errors += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error during publish: {e}")
            self.total_errors += 1
            return False

    def _ingest_next_batch(self) -> bool:
        """
        Main processing loop: read batch, upload, and notify.

        Returns:
            bool: True if batch processed successfully, False otherwise
        """
        # Read frame batch
        batch = self._read_next_frame_batch()
        if batch is None:
            return False

        # Upload to MinIO
        object_key = self._upload_frame_batch(batch)
        if object_key is None:
            return False

        # Publish metadata to Redis
        publish_ok = self._publish_batch_metadata(object_key, batch.shape)

        return publish_ok

    def run(self) -> None:
        """
        Main execution loop with auto-reconnection logic.

        This method:
        1. Connects to the RTSP stream
        2. Continuously reads and processes frame batches
        3. Handles errors with automatic reconnection
        4. Runs indefinitely until interrupted

        ### CUSTOMIZATION POINT ###
        You can modify this method to:
        - Add scheduling logic (process only during certain hours)
        - Implement adaptive batch sizing based on performance
        - Add health check reporting
        - Implement graceful shutdown on signals
        """
        logger.info(f"Starting video ingestion for camera: {self.camera_id}")

        retry_count = 0

        while True:
            try:
                # Connect to stream if not connected
                if self.video_capture is None or not self.video_capture.isOpened():
                    if retry_count >= self.max_reconnect_attempts:
                        logger.error(
                            f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
                        )
                        logger.info("Waiting before resetting reconnection counter...")
                        time.sleep(self.reconnect_delay * 5)
                        retry_count = 0

                    logger.info(
                        f"Reconnection attempt {retry_count + 1}/{self.max_reconnect_attempts}"
                    )

                    if self._open_video_source():
                        retry_count = 0  # Reset counter on successful connection
                    else:
                        retry_count += 1
                        time.sleep(self.reconnect_delay)
                        continue

                # Process one batch
                success = self._ingest_next_batch()

                if not success:
                    logger.warning("Batch processing failed, will attempt reconnection")
                    self._close_video_source()
                    retry_count += 1
                    time.sleep(self.reconnect_delay)
                else:
                    # Log statistics periodically
                    if self.total_batches_uploaded % 10 == 0:
                        logger.info(
                            f"Statistics - Frames: {self.total_frames_processed}, "
                            f"Batches: {self.total_batches_uploaded}, "
                            f"Errors: {self.total_errors}"
                        )

            except EOFError:
                logger.info("End of video reached; stopping ingestion loop.")
                break
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                self.total_errors += 1
                self._close_video_source()
                time.sleep(self.reconnect_delay)

        # Cleanup
        self._close_video_source()
        logger.info(
            f"Ingestion stopped. Final statistics - "
            f"Frames: {self.total_frames_processed}, "
            f"Batches: {self.total_batches_uploaded}, "
            f"Errors: {self.total_errors}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.

        Returns:
            Dict: Statistics including frames processed, batches uploaded, errors
        """
        return {
            "camera_id": self.camera_id,
            "frames_processed": self.total_frames_processed,
            "batches_uploaded": self.total_batches_uploaded,
            "errors_count": self.total_errors,
            "is_connected": self.video_capture is not None
            and self.video_capture.isOpened(),
        }


def main():
    """
    Main entry point for multi-camera video ingestion.
    Reads camera configurations from environment variables.

    Environment Variables:
    - CAMERA_CONFIGS: Multiple cameras format "camera_01:rtsp://url1,camera_02:rtsp://url2"
    - CAMERA_ID: Single camera ID (fallback)
    - RTSP_URL: Single RTSP URL (fallback)
    - VIDEO_FILE: Single video file (fallback)
    """

    # Configuration for multiple cameras
    # Format: CAMERA_CONFIGS=camera_01:rtsp://localhost:8554/stream1,camera_02:rtsp://localhost:8554/stream2
    camera_configs_str = os.getenv("CAMERA_CONFIGS", "")

    cameras = []

    if camera_configs_str:
        # Parse multiple camera configurations
        logger.info("Parsing multiple camera configurations")
        for config in camera_configs_str.split(","):
            parts = config.strip().split(":", 1)
            if len(parts) >= 2:
                camera_id = parts[0]
                rtsp_url = parts[1]
                cameras.append({"camera_id": camera_id, "rtsp_url": rtsp_url})
                logger.info(f"Added camera: {camera_id} -> {rtsp_url}")
    else:
        # Fallback to single camera (backward compatible)
        logger.info("Using single camera configuration (backward compatible mode)")
        camera_id = os.getenv("CAMERA_ID", "camera_01")
        rtsp_url = os.getenv("RTSP_URL", None)
        video_file = os.getenv("VIDEO_FILE", None)

        cameras.append(
            {"camera_id": camera_id, "rtsp_url": rtsp_url, "video_file": video_file}
        )

    if not cameras:
        logger.error("No camera configurations found")
        return 1

    logger.info(f"Starting ingestion for {len(cameras)} camera(s)")

    # Create and start threads for each camera
    threads = []
    ingestors = []

    for cam_config in cameras:
        try:
            logger.info(f"Initializing camera: {cam_config['camera_id']}")

            ingestor = VideoIngestor(
                camera_id=cam_config["camera_id"],
                rtsp_url=cam_config.get("rtsp_url"),
                video_file=cam_config.get("video_file"),
                batch_size=1,
                max_reconnect_attempts=5,
                reconnect_delay=5,
                loop_video=True,
                frame_skip=0,  # 0 = no skip, 1 = skip 1 frame, 2 = skip 2 frames, etc.
            )

            ingestors.append(ingestor)

            # Create and start thread
            thread = threading.Thread(
                target=ingestor.run,
                name=f"Ingestor-{cam_config['camera_id']}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

            logger.info(f"Started thread for camera: {cam_config['camera_id']}")

        except Exception as e:
            logger.error(
                f"Failed to initialize camera {cam_config.get('camera_id')}: {e}",
                exc_info=True,
            )

    if not threads:
        logger.error("No cameras initialized successfully")
        return 1

    logger.info(f"All {len(threads)} camera thread(s) started successfully")

    # Keep main thread alive and handle interrupts
    try:
        while True:
            # Print statistics every 60 seconds
            time.sleep(60)
            logger.info("=== Camera Statistics ===")
            for ingestor in ingestors:
                stats = ingestor.get_statistics()
                logger.info(
                    f"{stats['camera_id']}: "
                    f"Frames={stats['frames_processed']}, "
                    f"Batches={stats['batches_uploaded']}, "
                    f"Errors={stats['errors_count']}, "
                    f"Connected={stats['is_connected']}"
                )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, waiting for threads to finish...")

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)

        logger.info("All threads stopped")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    """
    Multi-camera ingestion is now supported by default.

    Usage examples:

    1. Multiple cameras via CAMERA_CONFIGS environment variable:
       CAMERA_CONFIGS="camera_01:rtsp://localhost:8554/stream1,camera_02:rtsp://localhost:8554/stream2"

    2. Single camera (backward compatible):
       CAMERA_ID="camera_01"
       RTSP_URL="rtsp://localhost:8554/stream"

    3. Video file:
       CAMERA_ID="camera_01"
       VIDEO_FILE="/path/to/video.mp4"
    """

    exit(main())
