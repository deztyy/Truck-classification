"""
Generate Sample Video for Testing Truck Detection System
========================================================
This script generates synthetic video footage simulating:
- Moving trucks on highway
- Different lighting conditions
- Multiple vehicles
- Realistic motion patterns

Author: Senior Python Developer
Date: January 2026
"""

import os
from datetime import datetime

import cv2
import numpy as np


class VideoGenerator:
    """Generate synthetic video for testing"""

    def __init__(
        self,
        output_path: str = "sample_videos/truck_highway_01.mp4",
        duration_seconds: int = 60,
        fps: int = 30,
        resolution: tuple = (1920, 1080),
    ):
        """
        Initialize video generator

        Args:
            output_path: Output video file path
            duration_seconds: Video duration in seconds
            fps: Frames per second
            resolution: Video resolution (width, height)
        """
        self.output_path = output_path
        self.duration = duration_seconds
        self.fps = fps
        self.width, self.height = resolution
        self.total_frames = duration_seconds * fps

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.width, self.height)
        )

        # Vehicle positions and speeds
        self.vehicles = self._initialize_vehicles()

    def _initialize_vehicles(self):
        """Initialize vehicle objects with random properties"""
        vehicles = []

        # Create 3-5 vehicles
        num_vehicles = np.random.randint(3, 6)

        for i in range(num_vehicles):
            vehicle = {
                "id": i,
                "x": np.random.randint(-200, self.width),
                "y": np.random.randint(300, 700),
                "speed": np.random.uniform(2, 8),  # pixels per frame
                "width": np.random.randint(180, 280),  # truck width
                "height": np.random.randint(120, 180),  # truck height
                "color": tuple(np.random.randint(50, 200, 3).tolist()),
                "type": np.random.choice(["truck", "car", "van"]),
            }
            vehicles.append(vehicle)

        return vehicles

    def _draw_road(self, frame, frame_num):
        """Draw highway road with lane markings"""
        # Sky (gradient)
        for y in range(self.height // 2):
            color = int(135 + (y / (self.height // 2)) * 50)
            cv2.line(frame, (0, y), (self.width, y), (color, color, 200), 1)

        # Road (dark gray)
        road_start = self.height // 2
        cv2.rectangle(
            frame, (0, road_start), (self.width, self.height), (70, 70, 70), -1
        )

        # Road texture (darker at bottom for perspective)
        for y in range(road_start, self.height, 5):
            darkness = int(70 - (y - road_start) / (self.height - road_start) * 20)
            cv2.line(frame, (0, y), (self.width, y), (darkness, darkness, darkness), 2)

        # Lane markings (animated)
        lane_offset = (frame_num * 5) % 80
        for y in range(road_start + lane_offset, self.height, 80):
            # White dashed lines
            cv2.rectangle(
                frame,
                (self.width // 3, y),
                (self.width // 3 + 10, y + 40),
                (255, 255, 255),
                -1,
            )
            cv2.rectangle(
                frame,
                (2 * self.width // 3, y),
                (2 * self.width // 3 + 10, y + 40),
                (255, 255, 255),
                -1,
            )

        # Road edges (solid yellow lines)
        cv2.line(frame, (50, road_start), (50, self.height), (0, 200, 255), 5)
        cv2.line(
            frame,
            (self.width - 50, road_start),
            (self.width - 50, self.height),
            (0, 200, 255),
            5,
        )

        return frame

    def _draw_vehicle(self, frame, vehicle):
        """Draw a vehicle (truck/car) on the frame"""
        x = int(vehicle["x"])
        y = int(vehicle["y"])
        w = vehicle["width"]
        h = vehicle["height"]
        color = vehicle["color"]

        # Main body (rectangle with rounded corners simulation)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

        # Cabin (for trucks)
        if vehicle["type"] == "truck":
            cabin_w = w // 3
            cabin_h = h + 20
            cv2.rectangle(frame, (x + w - cabin_w, y - 20), (x + w, y + h), color, -1)
            cv2.rectangle(
                frame, (x + w - cabin_w, y - 20), (x + w, y + h), (0, 0, 0), 3
            )

            # Windshield
            cv2.rectangle(
                frame,
                (x + w - cabin_w + 10, y - 10),
                (x + w - 10, y + 20),
                (100, 150, 200),
                -1,
            )
        else:
            # Windshield for cars
            cv2.rectangle(
                frame, (x + 10, y + 10), (x + 50, y + h - 10), (100, 150, 200), -1
            )

        # Wheels
        wheel_y = y + h + 5
        wheel_radius = 15
        cv2.circle(frame, (x + 30, wheel_y), wheel_radius, (30, 30, 30), -1)
        cv2.circle(frame, (x + w - 30, wheel_y), wheel_radius, (30, 30, 30), -1)
        cv2.circle(frame, (x + 30, wheel_y), wheel_radius - 5, (60, 60, 60), -1)
        cv2.circle(frame, (x + w - 30, wheel_y), wheel_radius - 5, (60, 60, 60), -1)

        # Headlights
        cv2.circle(frame, (x + 10, y + h - 10), 5, (255, 255, 0), -1)

        return frame

    def _update_vehicles(self):
        """Update vehicle positions"""
        for vehicle in self.vehicles:
            # Move vehicle to the right
            vehicle["x"] += vehicle["speed"]

            # Reset position if vehicle goes off screen
            if vehicle["x"] > self.width + 200:
                vehicle["x"] = -200
                vehicle["y"] = np.random.randint(300, 700)
                vehicle["speed"] = np.random.uniform(2, 8)

    def _add_metadata(self, frame, frame_num):
        """Add timestamp and frame info"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Add text
        cv2.putText(
            frame,
            "Camera: HIGHWAY-01",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Time: {timestamp}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            f"Frame: {frame_num}/{self.total_frames}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

        return frame

    def generate(self):
        """Generate the video"""
        print(f"Generating video: {self.output_path}")
        print(
            f"Duration: {self.duration}s | FPS: {self.fps} | Resolution: {self.width}x{self.height}"
        )

        for frame_num in range(self.total_frames):
            # Create blank frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Draw road
            frame = self._draw_road(frame, frame_num)

            # Draw vehicles
            for vehicle in self.vehicles:
                frame = self._draw_vehicle(frame, vehicle)

            # Update vehicle positions
            self._update_vehicles()

            # Add metadata overlay
            frame = self._add_metadata(frame, frame_num)

            # Write frame
            self.writer.write(frame)

            # Progress indicator
            if (frame_num + 1) % 30 == 0:
                progress = (frame_num + 1) / self.total_frames * 100
                print(
                    f"Progress: {progress:.1f}% ({frame_num + 1}/{self.total_frames} frames)"
                )

        # Release writer
        self.writer.release()

        # Get file size
        file_size = os.path.getsize(self.output_path) / (1024 * 1024)
        print("\n✓ Video generated successfully!")
        print(f"  Path: {self.output_path}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Frames: {self.total_frames}")


def main():
    """Generate multiple sample videos"""

    # Video 1: Short test video (30 seconds)
    print("=" * 60)
    print("Generating Test Video 1: Short Highway Scene")
    print("=" * 60)
    gen1 = VideoGenerator(
        output_path="sample_videos/truck_highway_short.mp4",
        duration_seconds=30,
        fps=30,
        resolution=(1280, 720),
    )
    gen1.generate()

    print("\n")

    # Video 2: Longer video (2 minutes)
    print("=" * 60)
    print("Generating Test Video 2: Extended Highway Scene")
    print("=" * 60)
    gen2 = VideoGenerator(
        output_path="sample_videos/truck_highway_long.mp4",
        duration_seconds=120,
        fps=30,
        resolution=(1920, 1080),
    )
    gen2.generate()

    print("\n")
    print("=" * 60)
    print("All sample videos generated successfully!")
    print("=" * 60)
    print("\nYou can now use these videos for testing:")
    print("  1. sample_videos/truck_highway_short.mp4 (30s, 720p)")
    print("  2. sample_videos/truck_highway_long.mp4 (2min, 1080p)")


if __name__ == "__main__":
    main()
