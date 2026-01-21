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
import time
from datetime import datetime
from typing import Any, Dict, Optional

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

# Frame preprocessing target size
FRAME_TARGET_HEIGHT = 640
FRAME_TARGET_WIDTH = 640


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
        batch_size: int = 30,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 5,
        loop_video: bool = True,
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
        """
        self.camera_id = camera_id
        self.video_file = video_file
        self.rtsp_url = rtsp_url or os.getenv("RTSP_URL")
        self.loop_video = loop_video
        self.batch_size = batch_size
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        # Validate required configuration
        if not self.rtsp_url and not self.video_file:
            raise ValueError("Either RTSP_URL or video_file must be provided")

        # Initialize MinIO client
        self.minio_client = self._initialize_minio()

        # Initialize Redis client
        self.redis_client = self._initialize_redis()

        # Storage configuration
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "raw-frames")
        self.redis_queue_name = os.getenv("REDIS_QUEUE_NAME", "frame_batches")

        # Ensure bucket exists
        self._ensure_bucket_exists()

        # Video capture object (initialized on start)
        self.video_capture: Optional[cv2.VideoCapture] = None

        # Statistics
        self.frames_processed = 0
        self.batches_uploaded = 0
        self.errors_count = 0

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
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
            else:
                logger.info(f"MinIO bucket exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error checking/creating bucket: {e}")
            raise

    def _connect_to_stream(self) -> bool:
        """
        Connect to the RTSP video stream or open video file.

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

    def _disconnect_from_stream(self) -> None:
        """
        Safely disconnect from the video stream.
        """
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            logger.info("Disconnected from video stream")

    def _read_frame_batch(self) -> Optional[np.ndarray]:
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
        frames_buffer = []

        for i in range(self.batch_size):
            try:
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

                # --- CUSTOM PREPROCESSING GOES HERE ---
                # Example: frame = cv2.resize(frame, (640, 640))
                # Example: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (FRAME_TARGET_WIDTH, FRAME_TARGET_HEIGHT))

                frames_buffer.append(frame)
                self.frames_processed += 1

            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                return None

        # Convert list of frames to NumPy array
        # Shape: (batch_size, height, width, channels)
        frames_batch = np.array(frames_buffer)
        logger.debug(f"Created batch with shape: {frames_batch.shape}")

        return frames_batch

    def _upload_batch_to_minio(self, batch: np.ndarray) -> Optional[str]:
        """
        Upload frame batch to MinIO storage using in-memory buffer.

        Args:
            batch: NumPy array containing frame batch

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
            batch_object_key = f"{self.camera_id}/batch_{timestamp}.npy"

            # Serialize NumPy array to bytes using io.BytesIO
            buffer = io.BytesIO()
            np.save(buffer, batch, allow_pickle=False)
            buffer.seek(0)  # Reset buffer position to start

            # Get buffer size
            buffer_size = buffer.getbuffer().nbytes

            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=batch_object_key,
                data=buffer,
                length=buffer_size,
                content_type="application/octet-stream",
            )

            logger.info(
                f"Uploaded batch to MinIO: {batch_object_key} ({buffer_size} bytes)"
            )
            self.batches_uploaded += 1

            return batch_object_key

        except S3Error as e:
            logger.error(f"MinIO upload error: {e}")
            self.errors_count += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            self.errors_count += 1
            return None

    def _publish_to_redis(self, batch_object_key: str, batch_shape: tuple) -> bool:
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
                "object_name": batch_object_key,
                "timestamp": datetime.utcnow().isoformat(),
                "batch_size": batch_shape[0],
                "frame_shape": list(batch_shape[1:]),
                "bucket_name": self.bucket_name,
            }

            # --- ADD CUSTOM METADATA HERE ---
            # Example: metadata['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
            # Example: metadata['processing_priority'] = 'high'

            # Serialize to JSON and push to Redis List
            message_json = json.dumps(metadata)
            self.redis_client.rpush(self.redis_queue_name, message_json)

            logger.info(f"Published metadata to Redis queue: {self.redis_queue_name}")
            logger.debug(f"Metadata: {metadata}")

            return True

        except redis.RedisError as e:
            logger.error(f"Redis publish error: {e}")
            self.errors_count += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error during publish: {e}")
            self.errors_count += 1
            return False

    def _process_batch(self) -> bool:
        """
        Main processing loop: read batch, upload, and notify.

        Returns:
            bool: True if batch processed successfully, False otherwise
        """
        # Read frame batch
        batch = self._read_frame_batch()
        if batch is None:
            return False

        # Upload to MinIO
        batch_object_key = self._upload_batch_to_minio(batch)
        if batch_object_key is None:
            return False

        # Publish metadata to Redis
        publish_ok = self._publish_to_redis(batch_object_key, batch.shape)

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

                    if self._connect_to_stream():
                        retry_count = 0  # Reset counter on successful connection
                    else:
                        retry_count += 1
                        time.sleep(self.reconnect_delay)
                        continue

                # Process one batch
                success = self._process_batch()

                if not success:
                    logger.warning("Batch processing failed, will attempt reconnection")
                    self._disconnect_from_stream()
                    retry_count += 1
                    time.sleep(self.reconnect_delay)
                else:
                    # Log statistics periodically
                    if self.batches_uploaded % 10 == 0:
                        logger.info(
                            f"Statistics - Frames: {self.frames_processed}, "
                            f"Batches: {self.batches_uploaded}, "
                            f"Errors: {self.errors_count}"
                        )

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                self.errors_count += 1
                self._disconnect_from_stream()
                time.sleep(self.reconnect_delay)

        # Cleanup
        self._disconnect_from_stream()
        logger.info(
            f"Ingestion stopped. Final statistics - "
            f"Frames: {self.frames_processed}, "
            f"Batches: {self.batches_uploaded}, "
            f"Errors: {self.errors_count}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.

        Returns:
            Dict: Statistics including frames processed, batches uploaded, errors
        """
        return {
            "camera_id": self.camera_id,
            "frames_processed": self.frames_processed,
            "batches_uploaded": self.batches_uploaded,
            "errors_count": self.errors_count,
            "is_connected": self.video_capture is not None
            and self.video_capture.isOpened(),
        }


def main():
    """
    Main entry point for the video ingestion script.

    ### CUSTOMIZATION POINT ###
    Modify this function to:
    - Launch multiple VideoIngestor instances for different cameras
    - Add configuration file support (YAML, JSON)
    - Implement multi-threading for multiple streams
    - Add CLI argument parsing
    """

    # Example: Single camera ingestion
    camera_id = os.getenv("CAMERA_ID", "camera_01")
    video_file = os.getenv("VIDEO_FILE", None)

    try:
        ingestor = VideoIngestor(
            camera_id=camera_id,
            video_file=video_file,
            batch_size=30,  # Adjust based on your needs
            max_reconnect_attempts=5,
            reconnect_delay=5,
            loop_video=True,
        )

        ingestor.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    """
    Example usage for multiple cameras (commented out):

    import threading

    cameras = [
        {'camera_id': 'camera_01', 'rtsp_url': 'rtsp://cam1.example.com/stream'},
        {'camera_id': 'camera_02', 'rtsp_url': 'rtsp://cam2.example.com/stream'},
    ]

    threads = []
    for cam_config in cameras:
        ingestor = VideoIngestor(**cam_config)
        thread = threading.Thread(target=ingestor.run, daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    """

    exit(main())
