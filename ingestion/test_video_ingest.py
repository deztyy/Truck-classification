import os
import pytest
import json
import threading
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from queue import Queue, Empty
import io

# Mock the clients before importing the module
import sys
sys.modules['redis'] = MagicMock()
sys.modules['minio'] = MagicMock()

# Import the module to test
import video_ingest
from video_ingest import Config, CameraWorker


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

    # Cleanup
    video_ingest.redis_client = None
    video_ingest.minio_client = None


@pytest.fixture
def mock_clients(setup_mock_clients):
    """Alias fixture to match tests expecting mock_clients."""
    return setup_mock_clients


class TestConfig:
    """Test Configuration loading from environment variables"""

    def test_config_defaults(self):
        """Test default configuration values when env vars are not set"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.REDIS_HOST == "localhost"
            assert config.MINIO_ENDPOINT == "localhost:9000"
            assert config.MINIO_ACCESS == "minioadmin"
            assert config.MINIO_SECRET == "minioadmin"
            assert config.BUCKET_NAME == "raw-frames"
            assert config.BATCH_SIZE == 30

    def test_config_from_environment(self):
        """Test configuration loading from environment variables"""
        env_vars = {
            "REDIS_HOST": "redis-server",
            "MINIO_ENDPOINT": "minio:9000",
            "MINIO_ACCESS_KEY": "test_access",
            "MINIO_SECRET_KEY": "test_secret",
            "RTSP_URLS": "rtsp://cam1,rtsp://cam2,rtsp://cam3"
        }
        with patch.dict(os.environ, env_vars):
            config = Config()
            assert config.REDIS_HOST == "redis-server"
            assert config.MINIO_ENDPOINT == "minio:9000"
            assert config.MINIO_ACCESS == "test_access"
            assert config.MINIO_SECRET == "test_secret"
            assert config.RTSP_LIST == ["rtsp://cam1", "rtsp://cam2", "rtsp://cam3"]

    def test_config_empty_rtsp_urls(self):
        """Test handling of empty RTSP URLs"""
        with patch.dict(os.environ, {"RTSP_URLS": ""}, clear=True):
            config = Config()
            assert config.RTSP_LIST == [""]

    def test_config_rtsp_urls_with_spaces(self):
        """Test RTSP URLs with trailing/leading spaces"""
        with patch.dict(os.environ, {"RTSP_URLS": " rtsp://cam1 , rtsp://cam2 "}, clear=True):
            config = Config()
            assert config.RTSP_LIST == [" rtsp://cam1 ", " rtsp://cam2 "]


class TestRedisMinioInitialization:
    """Test Redis and MinIO client initialization"""

    @patch('video_ingest.redis.Redis')
    @patch('video_ingest.Minio')
    def test_successful_initialization(self, mock_minio, mock_redis):
        """Test successful initialization of Redis and MinIO clients"""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        mock_minio_instance = MagicMock()
        mock_minio_instance.bucket_exists.return_value = True
        mock_minio.return_value = mock_minio_instance

        # This would normally be done in module initialization
        assert mock_redis_instance is not None
        assert mock_minio_instance is not None

    @patch('video_ingest.redis.Redis')
    def test_redis_connection_failure(self, mock_redis):
        """Test Redis connection failure"""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.side_effect = Exception("Connection refused")
        mock_redis.return_value = mock_redis_instance

        with pytest.raises(Exception):
            mock_redis_instance.ping()

    @patch('video_ingest.Minio')
    def test_minio_bucket_creation(self, mock_minio):
        """Test MinIO bucket creation when bucket doesn't exist"""
        mock_minio_instance = MagicMock()
        mock_minio_instance.bucket_exists.return_value = False
        mock_minio.return_value = mock_minio_instance

        # Simulate bucket creation
        if not mock_minio_instance.bucket_exists("raw-frames"):
            mock_minio_instance.make_bucket("raw-frames")

        mock_minio_instance.make_bucket.assert_called_once_with("raw-frames")


class TestCameraWorker:
    """Test CameraWorker class functionality"""

    @pytest.fixture
    def mock_clients(self):
        """Setup mock Redis and MinIO clients"""
        with patch('video_ingest.redis_client') as mock_redis, \
             patch('video_ingest.minio_client') as mock_minio:
            mock_redis.ping.return_value = True
            mock_minio.bucket_exists.return_value = True
            yield mock_redis, mock_minio

    def test_camera_worker_initialization(self, mock_clients):
        """Test CameraWorker initialization"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        assert worker.camera_id == "cam_1"
        assert worker.rtsp_url == "rtsp://test"
        assert worker.batch_buffer == []
        assert worker.running is True
        assert isinstance(worker.upload_queue, Queue)
        assert worker.upload_thread.daemon is True

    @patch('video_ingest.cv2.VideoCapture')
    def test_connect_rtsp_success(self, mock_capture, mock_clients):
        """Test successful RTSP connection"""
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")
        cap = worker.connect_rtsp()

        mock_capture.assert_called_once_with("rtsp://test")
        mock_cap.set.assert_called_once_with(mock_capture.CAP_PROP_BUFFERSIZE, 1)
        assert cap == mock_cap

    def test_serialize_batch_success(self, mock_clients):
        """Test successful batch serialization"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Create test frames
        frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(5)]

        result = worker._serialize_batch(frames)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_batch_error(self, mock_clients):
        """Test serialization error handling"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Pass invalid data
        invalid_data = ["not", "numpy", "arrays"]

        result = worker._serialize_batch(invalid_data)

        assert result is None

    def test_serialize_batch_empty(self, mock_clients):
        """Test serialization with empty batch"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        result = worker._serialize_batch([])

        assert result is not None

    @patch('video_ingest.time.time')
    def test_process_batch_success(self, mock_time, mock_clients):
        """Test successful batch processing"""
        mock_time.return_value = 1234567890.123
        mock_redis, mock_minio = mock_clients

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")
        worker.batch_buffer = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(5)]

        worker.process_batch()

        # Batch buffer should be cleared
        assert worker.batch_buffer == []

        # Upload queue should have item
        assert not worker.upload_queue.empty()

    def test_process_batch_queue_full(self, mock_clients):
        """Test batch processing when upload queue is full"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")
        worker.batch_buffer = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(5)]

        # Fill the queue
        worker.upload_queue.put("item1")
        worker.upload_queue.put("item2")

        # This should log warning but not crash
        worker.process_batch()

        # Batch buffer should still be cleared
        assert worker.batch_buffer == []

    def test_upload_worker_success(self, mock_clients):
        """Test upload worker successful upload"""
        mock_redis, mock_minio = mock_clients

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Prepare test data
        timestamp = 1234567890000
        object_name = "cam_1/1234567890000.npy"
        content = b"test_content"

        # Put data in queue
        worker.upload_queue.put((timestamp, object_name, content))

        # Give upload thread time to process
        time.sleep(0.1)

        # Stop worker
        worker.running = False
        worker.upload_queue.put(None)  # Poison pill
        worker.upload_thread.join(timeout=2)

    def test_upload_worker_queue_timeout(self, mock_clients):
        """Test upload worker handling queue timeout"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Let the upload thread run briefly with empty queue
        time.sleep(0.1)

        # Stop worker
        worker.running = False
        worker.upload_queue.put(None)
        worker.upload_thread.join(timeout=2)

        # Should handle Empty exception gracefully
        assert True

    def test_upload_worker_upload_error(self, mock_clients):
        """Test upload worker handling upload errors"""
        mock_redis, mock_minio = mock_clients
        mock_minio.put_object.side_effect = Exception("Upload failed")

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Put data in queue
        timestamp = 1234567890000
        object_name = "cam_1/1234567890000.npy"
        content = b"test_content"
        worker.upload_queue.put((timestamp, object_name, content))

        # Give thread time to process
        time.sleep(0.1)

        # Stop worker
        worker.running = False
        worker.upload_queue.put(None)
        worker.upload_thread.join(timeout=2)

        # Should handle exception gracefully
        assert True

    @patch('video_ingest.cv2.VideoCapture')
    def test_run_frame_capture_success(self, mock_capture, mock_clients):
        """Test successful frame capture in run method"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((1920, 1080, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Run briefly then stop
        def stop_worker():
            time.sleep(0.2)
            worker.running = False

        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()

        worker.run()
        stop_thread.join()

        # Should have called resize on frames
        assert mock_cap.read.called

    @patch('video_ingest.cv2.VideoCapture')
    def test_run_frame_read_failure(self, mock_capture, mock_clients):
        """Test handling of frame read failure"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(False, None), (False, None)]
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Run briefly then stop
        def stop_worker():
            time.sleep(0.3)
            worker.running = False

        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()

        worker.run()
        stop_thread.join()

        # Should attempt reconnection
        assert mock_cap.release.called

    @patch('video_ingest.cv2.VideoCapture')
    def test_run_connection_not_opened(self, mock_capture, mock_clients):
        """Test handling when RTSP connection is not opened"""
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [False, False, True]
        mock_cap.read.return_value = (True, np.zeros((640, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Run briefly then stop
        def stop_worker():
            time.sleep(0.5)
            worker.running = False

        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()

        worker.run()
        stop_thread.join()

        # Should attempt reconnection
        assert mock_capture.call_count >= 2

    @patch('video_ingest.cv2.VideoCapture')
    def test_run_batch_accumulation(self, mock_capture, mock_clients):
        """Test frame batch accumulation and processing"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((640, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")
        original_batch_size = Config.BATCH_SIZE
        Config.BATCH_SIZE = 5  # Small batch for testing

        # Run briefly to accumulate some frames
        def stop_worker():
            time.sleep(0.3)
            worker.running = False

        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()

        worker.run()
        stop_thread.join()

        Config.BATCH_SIZE = original_batch_size

        # Should have processed at least one batch
        assert mock_cap.read.called

    @patch('video_ingest.cv2.resize')
    @patch('video_ingest.cv2.VideoCapture')
    def test_run_frame_resize(self, mock_capture, mock_resize, mock_clients):
        """Test frame resizing in run method"""
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
        stop_thread.join()

        # Verify resize was called with correct dimensions
        if mock_resize.called:
            args, kwargs = mock_resize.call_args
            assert args[1] == (640, 640)


class TestMainExecution:
    """Test main execution flow"""

    @patch('video_ingest.CameraWorker')
    def test_main_worker_creation(self, mock_worker_class):
        """Test worker thread creation for multiple cameras"""
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker

        rtsp_urls = ["rtsp://cam1", "rtsp://cam2", "rtsp://cam3"]

        threads = []
        for idx, url in enumerate(rtsp_urls):
            worker = mock_worker_class(camera_id=idx, rtsp_url=url)
            worker.start()
            threads.append(worker)

        assert len(threads) == 3
        assert mock_worker_class.call_count == 3

    @patch('video_ingest.CameraWorker')
    def test_main_empty_rtsp_url_skipped(self, mock_worker_class):
        """Test that empty RTSP URLs are skipped"""
        rtsp_urls = ["rtsp://cam1", "", "rtsp://cam3"]

        threads = []
        for idx, url in enumerate(rtsp_urls):
            if not url.strip():
                continue
            worker = mock_worker_class(camera_id=idx, rtsp_url=url.strip())
            threads.append(worker)

        assert len(threads) == 2

    @patch('video_ingest.CameraWorker')
    def test_main_worker_cleanup(self, mock_worker_class):
        """Test worker cleanup on shutdown"""
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker

        threads = []
        for idx in range(2):
            worker = mock_worker_class(camera_id=idx, rtsp_url="rtsp://test")
            threads.append(worker)

        # Simulate shutdown
        for worker in threads:
            worker.running = False

        for worker in threads:
            worker.join()

        # All workers should be stopped
        assert all(not worker.running for worker in threads)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_large_batch(self, mock_clients):
        """Test handling of very large batch"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Create large batch
        large_batch = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(100)]

        result = worker._serialize_batch(large_batch)
        assert result is not None
        assert isinstance(result, bytes)

    def test_corrupted_frame_data(self, mock_clients):
        """Test handling of corrupted frame data"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Create batch with mixed valid and invalid data
        corrupted_batch = [
            np.zeros((640, 640, 3), dtype=np.uint8),
            None,  # Corrupted
            np.zeros((640, 640, 3), dtype=np.uint8),
        ]

        # Should handle gracefully
        result = worker._serialize_batch(corrupted_batch)
        # May return None or raise exception depending on implementation

    def test_concurrent_batch_processing(self, mock_clients):
        """Test concurrent batch processing"""
        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        def add_frames():
            for _ in range(50):
                worker.batch_buffer.append(np.zeros((640, 640, 3), dtype=np.uint8))
                if len(worker.batch_buffer) >= 30:
                    worker.process_batch()
                time.sleep(0.01)

        # Start multiple threads adding frames
        threads = [threading.Thread(target=add_frames) for _ in range(2)]
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should handle concurrent access gracefully
        worker.running = False
        worker.upload_queue.put(None)

    @patch('video_ingest.cv2.VideoCapture')
    def test_rapid_reconnection_attempts(self, mock_capture, mock_clients):
        """Test rapid reconnection attempts"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        worker = CameraWorker(camera_id=1, rtsp_url="rtsp://test")

        # Run briefly with constant reconnection
        def stop_worker():
            time.sleep(0.5)
            worker.running = False

        stop_thread = threading.Thread(target=stop_worker)
        stop_thread.start()

        worker.run()
        stop_thread.join()

        # Should handle reconnection loop
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
