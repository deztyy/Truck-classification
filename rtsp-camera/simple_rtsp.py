"""
Simple Video Streamer using OpenCV and HTTP
Alternative to GStreamer-based RTSP server for Windows
Supports multiple videos with separate endpoints
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify

app = Flask(__name__)

# Global configuration
VIDEO_FOLDER = "test-vid"
VIDEO_FILES = []
VIDEO_FRAMES = {}  # Store frames for each video
LOCKS = {}  # Lock for each video


def find_videos(folder):
    """Find all video files in folder"""
    video_extensions = [".mp4", ".MP4", ".avi", ".AVI", ".mkv", ".MKV"]
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"Error: Folder '{folder}' does not exist")
        return []

    videos = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix in video_extensions:
            videos.append(str(file))

    return videos


def generate_frames(video_index):
    """Generate video frames for a specific video"""
    global VIDEO_FILES, VIDEO_FRAMES, LOCKS

    if video_index >= len(VIDEO_FILES):
        return

    video_path = VIDEO_FILES[video_index]
    print(f"Starting stream {video_index}: {Path(video_path).name}")

    while True:
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Encode frame to JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            # Store current frame
            with LOCKS[video_index]:
                VIDEO_FRAMES[video_index] = buffer.tobytes()

            time.sleep(1 / 30)  # 30 FPS

        cap.release()


def get_frame(video_index):
    """Yield current frame for a specific video"""
    while True:
        with LOCKS[video_index]:
            if video_index in VIDEO_FRAMES and VIDEO_FRAMES[video_index] is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + VIDEO_FRAMES[video_index]
                    + b"\r\n"
                )
        time.sleep(1 / 30)


@app.route("/")
def index():
    """List available streams"""
    streams = []
    for i, video in enumerate(VIDEO_FILES):
        streams.append(
            {
                "index": i,
                "name": Path(video).name,
                "url": f"http://localhost:{app.config['PORT']}/stream/{i}",
            }
        )
    return jsonify({"streams": streams})


@app.route("/stream/<int:video_index>")
def video_feed(video_index):
    """Video streaming route for specific video"""
    if video_index >= len(VIDEO_FILES):
        return "Video not found", 404
    return Response(
        get_frame(video_index), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    global VIDEO_FOLDER, VIDEO_FILES, VIDEO_FRAMES, LOCKS

    parser = argparse.ArgumentParser(
        description="Simple Video Streamer - Multiple Videos"
    )
    parser.add_argument(
        "--folder", type=str, default="test-vid", help="Path to video folder"
    )
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")

    args = parser.parse_args()

    VIDEO_FOLDER = args.folder
    VIDEO_FILES = find_videos(VIDEO_FOLDER)

    if not VIDEO_FILES:
        print(f"No video files found in '{VIDEO_FOLDER}'")
        return

    print(f"\nFound {len(VIDEO_FILES)} video(s):")
    for i, video in enumerate(VIDEO_FILES):
        print(f"  [{i}] {Path(video).name}")
        # Initialize lock and frame storage for each video
        LOCKS[i] = threading.Lock()
        VIDEO_FRAMES[i] = None

    print(f"\nStarting HTTP video server on port {args.port}")
    print("\nAvailable streams:")
    for i in range(len(VIDEO_FILES)):
        print(f"  Stream {i}: http://localhost:{args.port}/stream/{i}")
    print(f"\nStream list (JSON): http://localhost:{args.port}/")
    print("\nPress Ctrl+C to stop...")

    # Start video generation thread for each video
    for i in range(len(VIDEO_FILES)):
        thread = threading.Thread(target=generate_frames, args=(i,), daemon=True)
        thread.start()

    # Store port in app config
    app.config["PORT"] = args.port

    # Run Flask server
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
