"""
Simplified Health Check Test
=============================
Tests health check functionality WITHOUT needing Redis/MinIO running.
Uses mock/dummy connections for standalone testing.

Usage:
    python simple_health_test.py
"""

import logging
import os
import threading
import time
from datetime import datetime
from threading import Lock

import cv2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleHealthChecker:
    """Simplified version focusing on health check logic only"""

    def __init__(self, camera_id: str, video_source: str):
        self.camera_id = camera_id
        self.video_source = video_source
        self.video_capture = None

        # Health check variables
        self.last_frame_read_time = None
        self.stream_is_healthy = True
        self.downtime_start_time = None
        self.total_downtime_seconds = 0.0
        self.is_running = False
        self.health_check_thread = None
        self.health_status_lock = Lock()

        # Stats
        self.total_frames_read = 0

        # Constants (shorter for testing)
        self.HEALTH_CHECK_INTERVAL = 2  # seconds
        self.FRAME_READ_TIMEOUT = 5  # seconds

    def _open_video(self):
        """Open video source"""
        logger.info(f"Opening video: {self.video_source}")
        self.video_capture = cv2.VideoCapture(self.video_source)

        if not self.video_capture.isOpened():
            logger.error("Failed to open video")
            return False

        # Test read
        ret, frame = self.video_capture.read()
        if not ret:
            logger.error("Failed to read first frame")
            return False

        logger.info("✓ Video opened successfully")
        return True

    def _close_video(self):
        """Close video source"""
        with self.health_status_lock:
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
                logger.info("Video closed")

    def _update_frame_read_time(self):
        """Update timestamp after successful frame read"""
        with self.health_status_lock:
            self.last_frame_read_time = datetime.utcnow()

            # Recovery detection
            if not self.stream_is_healthy:
                self.stream_is_healthy = True
                if self.downtime_start_time:
                    downtime = (
                        datetime.utcnow() - self.downtime_start_time
                    ).total_seconds()
                    self.total_downtime_seconds += downtime
                    logger.info(f"✓ Stream RECOVERED after {downtime:.1f}s downtime")
                    self.downtime_start_time = None

    def _health_check_loop(self):
        """Background health monitoring thread"""
        logger.info(
            f"Health check thread STARTED (interval={self.HEALTH_CHECK_INTERVAL}s)"
        )

        while self.is_running:
            time.sleep(self.HEALTH_CHECK_INTERVAL)

            with self.health_status_lock:
                # Check if video is open
                if not self.video_capture or not self.video_capture.isOpened():
                    if self.stream_is_healthy:
                        self.stream_is_healthy = False
                        self.downtime_start_time = datetime.utcnow()
                        logger.warning("⚠ Stream DISCONNECTED")
                    continue

                # Check frame read timeout
                if self.last_frame_read_time:
                    elapsed = (
                        datetime.utcnow() - self.last_frame_read_time
                    ).total_seconds()

                    if elapsed > self.FRAME_READ_TIMEOUT:
                        if self.stream_is_healthy:
                            self.stream_is_healthy = False
                            self.downtime_start_time = datetime.utcnow()
                            logger.error(
                                f"⚠ Stream STALLED (no frames for {elapsed:.1f}s, timeout={self.FRAME_READ_TIMEOUT}s)"
                            )

        logger.info("Health check thread STOPPED")

    def start_health_check(self):
        """Start health monitoring"""
        self.is_running = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self.health_check_thread.start()

    def stop_health_check(self):
        """Stop health monitoring"""
        self.is_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)

    def read_frame(self):
        """Read one frame"""
        with self.health_status_lock:
            if not self.video_capture or not self.video_capture.isOpened():
                return False
            ret, frame = self.video_capture.read()

        if ret:
            self.total_frames_read += 1
            self._update_frame_read_time()
            return True
        return False

    def get_stats(self):
        """Get current statistics"""
        with self.health_status_lock:
            return {
                "camera_id": self.camera_id,
                "frames_read": self.total_frames_read,
                "is_healthy": self.stream_is_healthy,
                "total_downtime": self.total_downtime_seconds,
            }


def test_normal_operation(video_path: str):
    """Test 1: Normal operation with healthy stream"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Normal Operation")
    logger.info("=" * 60)

    checker = SimpleHealthChecker("test_cam_01", video_path)

    if not checker._open_video():
        logger.error("✗ TEST 1 FAILED: Cannot open video")
        return False

    # Start health check
    checker.start_health_check()

    # Read frames normally for 10 seconds
    logger.info("Reading frames for 10 seconds...")
    start = time.time()
    while time.time() - start < 10:
        if checker.read_frame():
            time.sleep(0.033)  # ~30 FPS
        else:
            # End of video, restart
            checker._close_video()
            checker._open_video()

    # Check results
    stats = checker.get_stats()
    logger.info(f"Results: {stats}")

    # Cleanup
    checker.stop_health_check()
    checker._close_video()

    # Verify
    assert stats["is_healthy"], "Stream should be healthy"
    assert stats["frames_read"] > 0, "Should have read frames"
    assert stats["total_downtime"] == 0, "Should have no downtime"

    logger.info("✓ TEST 1 PASSED")
    return True


def test_stream_disconnection(video_path: str):
    """Test 2: Detect stream disconnection"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Stream Disconnection Detection")
    logger.info("=" * 60)

    checker = SimpleHealthChecker("test_cam_02", video_path)
    checker._open_video()
    checker.start_health_check()

    # Read normally for 3 seconds
    logger.info("Phase 1: Reading frames normally (3s)...")
    start = time.time()
    while time.time() - start < 3:
        checker.read_frame()
        time.sleep(0.033)

    # Simulate disconnection
    logger.info("Phase 2: Simulating disconnection...")
    checker._close_video()

    # Wait for health check to detect it
    time.sleep(6)  # Wait for 3 health checks

    stats = checker.get_stats()
    logger.info(f"Results: {stats}")

    # Cleanup
    checker.stop_health_check()

    # Verify - downtime is recorded ONLY after recovery
    assert not stats["is_healthy"], "Should detect unhealthy state"
    # Note: downtime = 0 because no recovery yet (expected behavior)
    logger.info("Note: Downtime will be recorded when stream recovers")

    logger.info("✓ TEST 2 PASSED")
    return True


def test_stream_stall(video_path: str):
    """Test 3: Detect when stream stops sending frames"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Frame Read Timeout Detection")
    logger.info("=" * 60)

    checker = SimpleHealthChecker("test_cam_03", video_path)
    checker._open_video()
    checker.start_health_check()

    # Read normally for 2 seconds
    logger.info("Phase 1: Reading frames normally (2s)...")
    start = time.time()
    while time.time() - start < 2:
        checker.read_frame()
        time.sleep(0.033)

    # Stop reading (simulate stall)
    logger.info("Phase 2: Simulating stall (no frame reads for 8s)...")
    time.sleep(8)  # Longer than FRAME_READ_TIMEOUT (5s)

    stats = checker.get_stats()
    logger.info(f"Results: {stats}")

    # Cleanup
    checker.stop_health_check()
    checker._close_video()

    # Verify - downtime is recorded ONLY after recovery
    assert not stats["is_healthy"], "Should detect stalled stream"
    # Note: downtime = 0 because no recovery yet (expected behavior)
    logger.info("Note: Downtime will be recorded when frames start reading again")

    logger.info("✓ TEST 3 PASSED")
    return True


def test_recovery(video_path: str):
    """Test 4: Detect recovery after downtime"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Recovery After Downtime")
    logger.info("=" * 60)

    checker = SimpleHealthChecker("test_cam_04", video_path)
    checker._open_video()
    checker.start_health_check()

    # Phase 1: Normal
    logger.info("Phase 1: Reading frames normally (3s)...")
    start = time.time()
    while time.time() - start < 3:
        checker.read_frame()
        time.sleep(0.033)

    # Phase 2: Disconnect
    logger.info("Phase 2: Disconnecting for 3s...")
    checker._close_video()
    time.sleep(3)

    # Phase 3: Reconnect
    logger.info("Phase 3: Reconnecting and reading frames (3s)...")
    checker._open_video()
    start = time.time()
    while time.time() - start < 3:
        checker.read_frame()
        time.sleep(0.033)

    # Wait for health check to process
    time.sleep(3)

    stats = checker.get_stats()
    logger.info(f"Results: {stats}")

    # Cleanup
    checker.stop_health_check()
    checker._close_video()

    # Verify
    assert stats["is_healthy"], "Should have recovered to healthy state"
    assert stats["total_downtime"] > 0, "Should have recorded downtime"
    logger.info(f"Total downtime recorded: {stats['total_downtime']:.1f}s")

    logger.info("✓ TEST 4 PASSED")
    return True


def main():
    """Run all health check tests"""
    logger.info("\n" + "=" * 60)
    logger.info("HEALTH CHECK TEST SUITE (Simplified)")
    logger.info("=" * 60)

    # Find test video
    video_paths = [
        "../rtsp-camera/test-vid/test_video.mp4",
        "./test_video.mp4",
        "test_video.mp4",
    ]

    video_path = None
    for path in video_paths:
        if os.path.exists(path):
            video_path = path
            break

    if not video_path:
        logger.error("No test video found!")
        logger.info("Please provide a video file at one of these locations:")
        for path in video_paths:
            logger.info(f"  - {path}")
        logger.info("\nYou can download a sample video:")
        logger.info(
            "  curl -o test_video.mp4 https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
        )
        return 1

    logger.info(f"Using video: {video_path}\n")

    # Run tests
    results = []

    try:
        results.append(("Normal Operation", test_normal_operation(video_path)))
    except Exception as e:
        logger.error(f"Test failed: {e}")
        results.append(("Normal Operation", False))

    time.sleep(1)

    try:
        results.append(("Stream Disconnection", test_stream_disconnection(video_path)))
    except Exception as e:
        logger.error(f"Test failed: {e}")
        results.append(("Stream Disconnection", False))

    time.sleep(1)

    try:
        results.append(("Frame Timeout", test_stream_stall(video_path)))
    except Exception as e:
        logger.error(f"Test failed: {e}")
        results.append(("Frame Timeout", False))

    time.sleep(1)

    try:
        results.append(("Recovery", test_recovery(video_path)))
    except Exception as e:
        logger.error(f"Test failed: {e}")
        results.append(("Recovery", False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:25s}: {status}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")
    logger.info("=" * 60 + "\n")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    exit(main())
