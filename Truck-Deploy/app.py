import mlflow
import numpy as np
import os
import cv2
import onnxruntime as ort
from datetime import datetime, timedelta
import time
import sys
import redis
import json
import signal
import threading
from zoneinfo import ZoneInfo
from collections import defaultdict
from sqlalchemy import create_engine, text
from pathlib import Path
import glob

# ByteTrack imports (install: pip install lap)
from dataclasses import dataclass
from typing import List, Tuple
import lap

# =============================================================================
# BYTETRACK IMPLEMENTATION
# =============================================================================
@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int

class STrack:
    """Single target track with Kalman filtering"""
    count = 0
    
    def __init__(self, bbox, score, class_id):
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.track_id = None
        self.is_activated = False
        self.tracklet_len = 0
        self.frame_id = 0
        
        self._tlwh = self._bbox_to_tlwh(bbox)
        self.mean, self.covariance = self._initiate_kalman(self._tlwh)
        
    @staticmethod
    def _bbox_to_tlwh(bbox):
        return np.array([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])
    
    def _initiate_kalman(self, tlwh):
        mean = np.zeros(8)
        mean[:4] = tlwh
        covariance = np.eye(8)
        return mean, covariance
    
    def predict(self):
        self.mean[4:] = self.mean[:4] - self.mean[4:8]
        
    def update(self, new_track):
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.class_id = new_track.class_id
        self._tlwh = self._bbox_to_tlwh(new_track.bbox)
        self.mean[:4] = self._tlwh
        self.tracklet_len += 1
        
    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.is_activated = True
        
    def re_activate(self, new_track, frame_id):
        self.update(new_track)
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.is_activated = True
        
    @classmethod
    def next_id(cls):
        cls.count += 1
        return cls.count

class ByteTracker:
    """ByteTrack multi-object tracker"""
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        
    def update(self, detections: List[Detection]):
        """Update tracks with new detections"""
        self.frame_id += 1
        
        det_high = [d for d in detections if d.score >= self.track_thresh]
        det_low = [d for d in detections if d.score < self.track_thresh]
        
        detections_high = [STrack(d.bbox, d.score, d.class_id) for d in det_high]
        detections_low = [STrack(d.bbox, d.score, d.class_id) for d in det_low]
        
        for track in self.tracked_stracks:
            track.predict()
        
        tracked_stracks = []
        unmatched_tracks = []
        
        if len(self.tracked_stracks) > 0 and len(detections_high) > 0:
            matches, u_track, u_detection = self._match(
                self.tracked_stracks, detections_high
            )
            
            for itracked, idet in matches:
                track = self.tracked_stracks[itracked]
                det = detections_high[idet]
                track.update(det)
                tracked_stracks.append(track)
                
            unmatched_tracks = [self.tracked_stracks[i] for i in u_track]
            unmatched_detections = [detections_high[i] for i in u_detection]
        else:
            unmatched_tracks = self.tracked_stracks
            unmatched_detections = detections_high
        
        if len(unmatched_tracks) > 0 and len(detections_low) > 0:
            matches, u_track, u_detection = self._match(
                unmatched_tracks, detections_low, thresh=0.5
            )
            
            for itracked, idet in matches:
                track = unmatched_tracks[itracked]
                det = detections_low[idet]
                track.update(det)
                tracked_stracks.append(track)
                
            unmatched_tracks = [unmatched_tracks[i] for i in u_track]
        else:
            unmatched_detections += detections_low
        
        for track in unmatched_tracks:
            if track.frame_id < self.frame_id - self.track_buffer:
                self.removed_stracks.append(track)
            else:
                self.lost_stracks.append(track)
        
        for det in unmatched_detections:
            if det.score >= self.track_thresh:
                det.activate(self.frame_id)
                tracked_stracks.append(det)
        
        self.tracked_stracks = tracked_stracks
        
        return [t for t in self.tracked_stracks if t.is_activated]
    
    def _match(self, tracks, detections, thresh=None):
        if thresh is None:
            thresh = self.match_thresh
            
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det.bbox)
        
        if iou_matrix.max() > 0:
            cost_matrix = 1 - iou_matrix
            indices = lap.lapjv(cost_matrix, extend_cost=True)[1]
            
            matches = []
            unmatched_tracks = []
            unmatched_detections = list(range(len(detections)))
            
            for i, j in enumerate(indices):
                if j >= 0 and iou_matrix[i, j] >= thresh:
                    matches.append([i, j])
                    if j in unmatched_detections:
                        unmatched_detections.remove(j)
                else:
                    unmatched_tracks.append(i)
                    
            return matches, unmatched_tracks, unmatched_detections
        else:
            return [], list(range(len(tracks))), list(range(len(detections)))
    
    @staticmethod
    def _iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        iou = inter_area / (bbox1_area + bbox2_area - inter_area + 1e-6)
        return iou

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
Model_uri = os.getenv("MODEL_URI", "models:/Truck_classification_Model/Production")
Mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

REDIS_HOST = os.getenv("REDIS_HOST", "redis_broker")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@db:5432/mydb')

# Image storage settings
IMAGE_STORAGE_DIR = os.getenv("IMAGE_STORAGE_DIR", "./vehicle_images")
IMAGE_RETENTION_DAYS = int(os.getenv("IMAGE_RETENTION_DAYS", 15))

# Batch insert settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))
BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", 5))  # seconds

THAI_TZ = ZoneInfo("Asia/Bangkok")

RUNNING = True

# =============================================================================
# VEHICLE CLASSIFICATION DATA
# =============================================================================
VEHICLE_CLASSES = {
    0: {"name": "car", "entry_fee": 0.00, "xray_fee": 0.00},
    1: {"name": "other", "entry_fee": 0.00, "xray_fee": 0.00},
    2: {"name": "other_truck", "entry_fee": 100.00, "xray_fee": 50.00},
    3: {"name": "pickup_truck", "entry_fee": 0.00, "xray_fee": 0.00},
    4: {"name": "truck_20_back", "entry_fee": 100.00, "xray_fee": 250.00},
    5: {"name": "truck_20_front", "entry_fee": 100.00, "xray_fee": 250.00},
    6: {"name": "truck_20x2", "entry_fee": 100.00, "xray_fee": 500.00},
    7: {"name": "truck_40", "entry_fee": 100.00, "xray_fee": 350.00},
    8: {"name": "truck_roro", "entry_fee": 100.00, "xray_fee": 50.00},
    9: {"name": "truck_tail", "entry_fee": 100.00, "xray_fee": 50.00},
    10: {"name": "motorcycle", "entry_fee": 0.00, "xray_fee": 0.00},
    11: {"name": "truck_head", "entry_fee": 100.00, "xray_fee": 50.00},
}

# Track management
saved_tracks = {}
db_lock = threading.Lock()
batch_buffer = []
batch_lock = threading.Lock()
last_batch_time = time.time()

# =============================================================================
# SIGNAL HANDLING
# =============================================================================
def handle_signal(signum, frame):
    global RUNNING
    print(f"\n🛑 Received signal {signum}. Stopping consumer...")
    RUNNING = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# =============================================================================
# FILE CLEANUP
# =============================================================================
def cleanup_old_files():
    """Remove files older than IMAGE_RETENTION_DAYS"""
    try:
        cutoff_date = datetime.now(THAI_TZ) - timedelta(days=IMAGE_RETENTION_DAYS)
        deleted_count = 0
        
        # Search for jpg and npy files
        for pattern in ['*.jpg', '*.npy']:
            for file_path in glob.glob(os.path.join(IMAGE_STORAGE_DIR, '**', pattern), recursive=True):
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path), tz=THAI_TZ)
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Error deleting {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"🧹 Cleaned up {deleted_count} old files (>{IMAGE_RETENTION_DAYS} days)")
    except Exception as e:
        print(f"❌ Cleanup error: {e}")

def schedule_cleanup():
    """Run cleanup every hour"""
    while RUNNING:
        cleanup_old_files()
        time.sleep(3600)  # Run every hour

# =============================================================================
# IMAGE CONVERSION
# =============================================================================
def convert_npy_to_jpg(npy_path, output_dir=IMAGE_STORAGE_DIR, quality=85):
    """
    Convert .npy file to .jpg and return the new path
    
    Args:
        npy_path: Path to .npy file
        output_dir: Directory to save jpg files
        quality: JPEG quality (1-100)
    
    Returns:
        Path to saved .jpg file
    """
    try:
        # Load BGR image
        frame_bgr = np.load(npy_path)
        
        # Create output directory structure: IMAGE_STORAGE_DIR/YYYY-MM-DD/
        date_str = datetime.now(THAI_TZ).strftime("%Y-%m-%d")
        day_dir = os.path.join(output_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now(THAI_TZ).strftime("%Y%m%d_%H%M%S_%f")
        jpg_filename = f"{timestamp}.jpg"
        jpg_path = os.path.join(day_dir, jpg_filename)
        
        # Save as JPEG
        cv2.imwrite(jpg_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Delete original .npy file to save space
        try:
            os.remove(npy_path)
        except:
            pass
        
        return jpg_path
    except Exception as e:
        print(f"❌ Error converting {npy_path} to jpg: {e}")
        return None

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================
def init_database(engine):
    """Initialize database tables"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS vehicle_classes (
                    class_id SERIAL PRIMARY KEY,
                    class_name VARCHAR(50) UNIQUE NOT NULL,
                    entry_fee NUMERIC(10, 2),
                    xray_fee NUMERIC(10, 2),
                    total_fee NUMERIC(10, 2)
                );
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS vehicle_transactions (
                    id SERIAL PRIMARY KEY,
                    camera_id VARCHAR(50) NOT NULL,
                    track_id INTEGER NOT NULL,
                    class_id INT,
                    applied_entry_fee NUMERIC(10, 2),
                    applied_xray_fee NUMERIC(10, 2),
                    total_applied_fee NUMERIC(10, 2),
                    image_path TEXT,
                    confidence NUMERIC(5, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (class_id) REFERENCES vehicle_classes(class_id),
                    UNIQUE(camera_id, track_id)
                );
            """))
            
            # Create index for faster queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_camera_created 
                ON vehicle_transactions(camera_id, created_at DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_track_id 
                ON vehicle_transactions(track_id);
            """))
            
            # Insert vehicle classes
            for class_id, info in VEHICLE_CLASSES.items():
                conn.execute(text("""
                    INSERT INTO vehicle_classes (class_id, class_name, entry_fee, xray_fee, total_fee)
                    VALUES (:id, :name, :entry, :xray, :total)
                    ON CONFLICT (class_id) DO NOTHING
                """), {
                    "id": class_id,
                    "name": info["name"],
                    "entry": info["entry_fee"],
                    "xray": info["xray_fee"],
                    "total": info["entry_fee"] + info["xray_fee"]
                })
            
            conn.commit()
            print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

def batch_insert_to_database(engine, batch_data):
    """
    Batch insert multiple records to database
    
    Args:
        engine: SQLAlchemy engine
        batch_data: List of dicts with transaction data
    """
    if not batch_data:
        return
    
    with db_lock:
        try:
            with engine.connect() as conn:
                # Use executemany for batch insert
                conn.execute(text("""
                    INSERT INTO vehicle_transactions 
                    (camera_id, track_id, class_id, applied_entry_fee, applied_xray_fee, 
                     total_applied_fee, image_path, confidence)
                    VALUES (:cam_id, :track_id, :class_id, :entry, :xray, :total, :img, :conf)
                    ON CONFLICT (camera_id, track_id) DO NOTHING
                """), batch_data)
                conn.commit()
                
                print(f"💾 Batch inserted {len(batch_data)} records to database")
        except Exception as e:
            print(f"❌ Batch insert error: {e}")

def add_to_batch(camera_id, track_id, class_id, confidence, image_path):
    """Add record to batch buffer"""
    global batch_buffer, last_batch_time
    
    track_key = f"{camera_id}_{track_id}"
    if track_key in saved_tracks:
        return  # Already saved
    
    vehicle = VEHICLE_CLASSES[class_id]
    total_fee = vehicle["entry_fee"] + vehicle["xray_fee"]
    
    record = {
        "cam_id": camera_id,
        "track_id": track_id,
        "class_id": class_id,
        "entry": vehicle["entry_fee"],
        "xray": vehicle["xray_fee"],
        "total": total_fee,
        "img": image_path,
        "conf": confidence
    }
    
    with batch_lock:
        batch_buffer.append(record)
        saved_tracks[track_key] = True
        
        print(f"📦 Added to batch: Track #{track_id} | {vehicle['name']} | Buffer: {len(batch_buffer)}/{BATCH_SIZE}")

def flush_batch_if_needed(engine, force=False):
    """Flush batch buffer to database if conditions are met"""
    global batch_buffer, last_batch_time
    
    with batch_lock:
        should_flush = (
            force or 
            len(batch_buffer) >= BATCH_SIZE or 
            (len(batch_buffer) > 0 and time.time() - last_batch_time >= BATCH_TIMEOUT)
        )
        
        if should_flush and batch_buffer:
            batch_to_insert = batch_buffer.copy()
            batch_buffer = []
            last_batch_time = time.time()
    
    if should_flush and batch_to_insert:
        batch_insert_to_database(engine, batch_to_insert)

# =============================================================================
# PREPROCESSING & POSTPROCESSING
# =============================================================================
def preprocess_frame(frame_bgr, input_size=(640, 640)):
    """Preprocess BGR frame for ONNX model"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, input_size)
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_chw = np.transpose(frame_norm, (2, 0, 1))
    frame_chw = np.expand_dims(frame_chw, axis=0)
    return frame_chw

def postprocess_yolo_detection(outputs, conf_thresh=0.3, img_shape=(640, 640)):
    """Postprocess YOLO output to get detections with bounding boxes"""
    output = outputs[0][0]  # [84, 8400]
    
    boxes = output[:4, :].T
    class_probs = output[4:, :].T
    
    class_ids = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs, axis=1)
    
    mask = confidences > conf_thresh
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    detections = []
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        x_center, y_center, w, h = box
        x1 = max(0, min(x_center - w / 2, img_shape[1]))
        y1 = max(0, min(y_center - h / 2, img_shape[0]))
        x2 = max(0, min(x_center + w / 2, img_shape[1]))
        y2 = max(0, min(y_center + h / 2, img_shape[0]))
        
        detections.append(Detection(
            bbox=np.array([x1, y1, x2, y2]),
            score=float(conf),
            class_id=int(class_id)
        ))
    
    return detections

# =============================================================================
# INFERENCE FUNCTION
# =============================================================================
def run_inference_on_npy(session, npy_path):
    """Load .npy file and run inference"""
    frame_bgr = np.load(npy_path)
    input_tensor = preprocess_frame(frame_bgr)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    return outputs, frame_bgr.shape[:2]

# =============================================================================
# REDIS CONSUMER WITH BYTETRACK
# =============================================================================
def process_redis_queue(session, redis_client, engine):
    """Main consumer loop with ByteTrack and batch processing"""
    print("🚀 Starting Redis consumer with ByteTrack + Batch Insert...")
    
    trackers = defaultdict(lambda: ByteTracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8
    ))
    
    processed_count = 0
    
    while RUNNING:
        try:
            result = redis_client.blpop('video_jobs', timeout=1)
            
            if result is None:
                # Check if we need to flush batch on timeout
                flush_batch_if_needed(engine)
                continue
            
            job_start_time = time.perf_counter()
            
            _, job_data = result
            
            # Handle both bytes and string
            if isinstance(job_data, bytes):
                job_data = job_data.decode('utf-8')
            
            try:
                # Double decode because ingestion.py does double JSON encoding
                job_outer = json.loads(job_data)  # First decode
                
                # If it's a string, decode again (double encoded)
                if isinstance(job_outer, str):
                    job = json.loads(job_outer)  # Second decode
                else:
                    job = job_outer
                    
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in queue: {e}")
                print(f"📝 Raw data: {job_data[:200]}...")  # Show first 200 chars
                continue
            
            # Validate job structure
            if not isinstance(job, dict):
                print(f"❌ Job is not a dict, got {type(job)}: {job}")
                continue
            
            camera_id = job.get("camera_id")
            # Handle both 'file_path' (from ingestion.py) and 'image_path'
            npy_path = job.get("file_path") or job.get("image_path")
            
            if not os.path.exists(npy_path):
                print(f"⚠️ File not found: {npy_path}")
                continue
            
            # Convert .npy to .jpg
            jpg_path = convert_npy_to_jpg(npy_path)
            if not jpg_path:
                print(f"⚠️ Failed to convert {npy_path}")
                continue
            
            # Run inference
            outputs, img_shape = run_inference_on_npy(session, npy_path) if os.path.exists(npy_path) else (None, None)
            
            if outputs is None:
                # If npy was already deleted, load from jpg
                frame_bgr = cv2.imread(jpg_path)
                input_tensor = preprocess_frame(frame_bgr)
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: input_tensor})
                img_shape = frame_bgr.shape[:2]
            
            # Get detections
            detections = postprocess_yolo_detection(outputs, conf_thresh=0.3, img_shape=img_shape)
            
            # Update tracker
            tracker = trackers[camera_id]
            online_tracks = tracker.update(detections)
            
            # Process tracked objects
            for track in online_tracks:
                if track.score >= 0.5 and track.class_id < len(VEHICLE_CLASSES):
                    # Add to batch buffer
                    add_to_batch(
                        camera_id=camera_id,
                        track_id=track.track_id,
                        class_id=track.class_id,
                        confidence=track.score,
                        image_path=jpg_path
                    )
            
            # Flush batch if needed
            flush_batch_if_needed(engine)
            
            total_time = time.perf_counter() - job_start_time
            processed_count += 1
            
            print(f"✅ [{camera_id}] Processed in {total_time:.3f}s | "
                  f"Detections: {len(detections)} | Tracks: {len(online_tracks)} | "
                  f"Total: {processed_count}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
        except Exception as e:
            print(f"🔥 Processing error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
    
    # Final flush on shutdown
    print("🔄 Flushing remaining batch records...")
    flush_batch_if_needed(engine, force=True)
    
    print(f"👋 Consumer stopped. Total processed: {processed_count}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 60)
    print("🚛 Truck Classification with ByteTrack + Batch Insert")
    print("=" * 60)
    
    # 1. Create image storage directory
    os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)
    print(f"📁 Image storage: {IMAGE_STORAGE_DIR}")
    print(f"🗑️ Retention: {IMAGE_RETENTION_DAYS} days")
    
    # 2. Connect to database
    print(f"🗄️ Connecting to database...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10)
    init_database(engine)
    
    # 3. Setup MLflow
    print(f"📡 Connecting to MLflow: {Mlflow_uri}")
    mlflow.set_tracking_uri(Mlflow_uri)
    
    # 4. Download and load model
    print(f"📦 Downloading model: {Model_uri}")
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=Model_uri)
    onnx_path = os.path.join(local_path, "model.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Model not found at: {onnx_path}")
        sys.exit(1)
    
    print(f"🧠 Loading ONNX model...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"✅ Model loaded successfully")
    
    # 5. Connect to Redis
    print(f"🔌 Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    redis_client = None
    
    while RUNNING:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                socket_connect_timeout=5,
                decode_responses=False
            )
            redis_client.ping()
            print(f"✅ Connected to Redis")
            break
        except redis.ConnectionError as e:
            print(f"🔴 Redis connection failed: {e}. Retrying in 5s...")
            time.sleep(5)
    
    if not RUNNING:
        print("🛑 Shutdown requested")
        sys.exit(0)
    
    # 6. Start cleanup thread
    cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
    cleanup_thread.start()
    print(f"🧹 Cleanup thread started (runs every hour)")
    
    # 7. Start processing
    print("=" * 60)
    print(f"⚙️ Batch settings: Size={BATCH_SIZE}, Timeout={BATCH_TIMEOUT}s")
    print("=" * 60)
    process_redis_queue(session, redis_client, engine)
    
    # 8. Cleanup
    redis_client.close()
    engine.dispose()
    print("✨ Service terminated gracefully")

if __name__ == "__main__":
    main()