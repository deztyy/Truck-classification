import streamlit as st
import cv2
import threading
import time

st.set_page_config(layout="wide", page_title="CCTV Monitor")

# --- 1. Robust Camera Threading Class ---
class VideoStream:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.cap = None
        self.frame = None
        self.is_running = False
        self.status = "Initializing..."

    def start(self):
        """Starts the background thread to pull frames."""
        if not self.is_running:
            self.cap = cv2.VideoCapture(self.url)
            if self.cap.isOpened():
                self.is_running = True
                self.status = "Connected"
                # Daemon thread ensures it dies when the main program stops
                threading.Thread(target=self._update, daemon=True).start()
            else:
                self.status = "Failed to Connect"
        return self

    def _update(self):
        """Internal loop to keep the latest frame updated."""
        while self.is_running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                else:
                    self.status = "Connection Lost"
                    self.is_running = False
                    break
            else:
                break

    def get_frame(self):
        """Returns the most recent frame."""
        return self.frame

    def stop(self):
        """Cleanly closes the camera capture."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.status = "Stopped"

# --- 2. Sidebar & Session Configuration ---
st.sidebar.title("⚙️ Camera Control")

# Initialize URLs in session state so they don't reset
if "cam_urls" not in st.session_state:
    st.session_state.cam_urls = {
        "Camera 1": "rtsp://admin:Smarterware2025@stwcctv.autoddns.com:554/chID=1",
        "Camera 2": "rtsp://admin:Smarterware2025@stwcctv.autoddns.com:554/chID=2",
        "Camera 3": "rtsp://admin:Smarterware2025@stwcctv.autoddns.com:554/chID=3"
    }

# Input fields for URLs
for cam_name in st.session_state.cam_urls.keys():
    st.session_state.cam_urls[cam_name] = st.sidebar.text_input(f"{cam_name} URL", st.session_state.cam_urls[cam_name])

view_mode = st.sidebar.selectbox("📺 View Mode", ["Show All", "Camera 1", "Camera 2", "Camera 3"])
run_viewer = st.sidebar.checkbox("▶️ Start Monitoring")

# Initialize Stream objects in session state
if "streams" not in st.session_state:
    st.session_state.streams = {}

# --- 3. Main Dashboard UI ---
st.title("🛡️ 3-Camera Live Surveillance")



if run_viewer:
    # Start or Restart streams if they aren't running
    for name, url in st.session_state.cam_urls.items():
        if name not in st.session_state.streams or not st.session_state.streams[name].is_running:
            st.session_state.streams[name] = VideoStream(name, url).start()

    # --- SHOW ALL CAMERAS (Grid) ---
    if view_mode == "Show All":
        cols = st.columns(3)
        # Create persistent placeholders in the columns
        placeholders = [cols[i].empty() for i in range(3)]
        
        while run_viewer:
            # Iterate through the dictionary to display all 3
            for i, (name, stream) in enumerate(st.session_state.streams.items()):
                frame = stream.get_frame() # This is the method that was 'missing'
                
                if frame is not None:
                    # Optional: Downscale slightly to improve browser performance
                    display_frame = cv2.resize(frame, (640, 360))
                    placeholders[i].image(display_frame, channels="BGR", caption=f"{name} | Status: {stream.status}", use_container_width=True)
                else:
                    placeholders[i].warning(f"{name}: {stream.status}")
            
            time.sleep(0.01) # Small delay to yield to the OS

    # --- SHOW SINGLE CAMERA (Focus Mode) ---
    else:
        placeholder = st.empty()
        while run_viewer:
            stream = st.session_state.streams.get(view_mode)
            if stream:
                frame = stream.get_frame()
                if frame is not None:
                    placeholder.image(frame, channels="BGR", caption=f"LIVE FOCUS: {view_mode}", use_container_width=True)
                else:
                    placeholder.error(f"Waiting for {view_mode} signal...")
            time.sleep(0.01)

else:
    st.info("Check the 'Start Monitoring' box in the sidebar to begin. If a camera shows 'Failed to Connect', check your RTSP credentials.")
    # Stop all threads if monitoring is turned off
    for stream in st.session_state.streams.values():
        stream.stop()