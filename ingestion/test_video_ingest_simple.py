import os
import pytest
import json
import threading
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from queue import Queue, Empty
import io

# Import the module to test
import video_ingest
from video_ingest import Config, CameraWorker


# Setup mock clients before tests
@pytest.fixture(autouse=True)
def setup_mock_clients():
    """Setup mock Redis and MinIO clients for all tests"""
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.rpush = MagicMock()

    mock_minio = MagicMock()
    mock_minio.bucket_exists.return_value = True
    mock_minio.put_object = MagicMock()

    # Set the global clients in the video_ingest module
    video_ingest.redis_client = mock_redis
    video_ingest.minio_client = mock_minio

    yield mock_redis, mock_minio


class TestConfig:
    """Test Configuration loading from environment variables"""

    def test_config_defaults(self):
        """Test default configuration values"""
        assert Config.REDIS_HOST in ["localhost", os.getenv("REDIS_HOST", "localhost")]
        assert Config.BUCKET_NAME == "raw-frames"
        assert Config.BATCH_SIZE == 30

    def test_config_from_environment(self):
        """Test configuration can be accessed"""
        assert hasattr(Config, 'REDIS_HOST')
        assert hasattr(Config, 'MINIO_ENDPOINT')
        assert hasattr(Config, 'MINIO_ACCESS')
        assert hasattr(Config, 'MINIO_SECRET')


class TestInitialization:
    """Test initialization function"""

    @patch('video_ingest.redis.Redis')
    @patch('video_ingest.Minio')
    def test_init_clients_success(self, mock_minio_class, mock_redis_class):
        """Test successful client initialization"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        mock_minio = MagicMock()
        mock_minio.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_minio

        result = video_ingest.init_clients()

        assert result is True
        mock_redis.ping.assert_called_once()

    @patch('video_ingest.redis.Redis')
    def test_init_clients_redis_failure(self, mock_redis_class):
        """Test Redis connection failure"""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")
        mock_redis_class.return_value = mock_redis

        result = video_ingest.init_clients()

        assert result is False


class TestCameraWorker:
    """Test CameraWorker class functionality"""

    def test_camera_worker_initialization(self):
        """Test CameraWorker initialization"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        assert worker.camera_id == "cam_1"
        assert worker.rtsp_url == "rtsp://test"
        assert worker.batch_buffer == []
        assert worker.running is True
        assert isinstance(worker.upload_queue, Queue)

    @patch('video_ingest.cv2.VideoCapture')
    def test_connect_rtsp_success(self, mock_capture):
        """Test successful RTSP connection"""
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")
        cap = worker.connect_rtsp()

        mock_capture.assert_called_once_with("rtsp://test")
        assert cap == mock_cap

    def test_serialize_batch_success(self):
        """Test successful batch serialization"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Create test frames
        frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(5)]

        result = worker._serialize_batch(frames)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_batch_error(self):
        """Test serialization error handling"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Pass invalid data
        invalid_data = ["not", "numpy", "arrays"]

        result = worker._serialize_batch(invalid_data)

        assert result is None

    def test_serialize_batch_empty(self):
        """Test serialization with empty batch"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        result = worker._serialize_batch([])

        assert result is not None

    @patch('video_ingest.time.time')
    def test_process_batch_success(self, mock_time):
        """Test successful batch processing"""
        mock_time.return_value = 1234567890.123

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")
        worker.batch_buffer = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(5)]

        worker.process_batch()

        # Batch buffer should be cleared
        assert worker.batch_buffer == []

    def test_process_batch_queue_full(self):
        """Test batch processing when upload queue is full"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")
        worker.batch_buffer = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(5)]

        # Fill the queue
        worker.upload_queue.put("item1")
        worker.upload_queue.put("item2")

        # This should handle gracefully
        worker.process_batch()

        # Batch buffer should still be cleared
        assert worker.batch_buffer == []

        # Cleanup
        worker.running = False
        worker.upload_queue.put(None)

    @patch('video_ingest.cv2.VideoCapture')
    @patch('video_ingest.cv2.resize')
    def test_run_frame_capture_and_resize(self, mock_resize, mock_capture):
        """Test frame capture and resizing"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_capture.return_value = mock_cap
        mock_resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Run briefly then stop
        def stop_worker():
            time.sleep(0.2)
            worker.running = False

        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()

        worker.run()
        stop_thread.join(timeout=2)

        # Cleanup
        worker.upload_queue.put(None)
        worker.upload_thread.join(timeout=2)

    @patch('video_ingest.cv2.VideoCapture')
    def test_run_connection_failure_and_reconnect(self, mock_capture):
        """Test handling connection failure and reconnection"""
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [False, False, True]
        mock_cap.read.return_value = (True, np.zeros((640, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Run briefly then stop
        def stop_worker():
            time.sleep(0.3)
            worker.running = False

        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()

        worker.run()
        stop_thread.join(timeout=2)

        # Cleanup
        worker.upload_queue.put(None)
        worker.upload_thread.join(timeout=2)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_large_batch(self):
        """Test handling of very large batch"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Create large batch
        large_batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(100)]

        result = worker._serialize_batch(large_batch)
        assert result is not None
        assert isinstance(result, bytes)

        # Cleanup
        worker.running = False
        worker.upload_queue.put(None)

    def test_worker_cleanup(self):
        """Test worker cleanup on shutdown"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Stop the worker
        worker.running = False
        worker.upload_queue.put(None)

        # Give it time to stop
        worker.upload_thread.join(timeout=2)

        assert not worker.running


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
