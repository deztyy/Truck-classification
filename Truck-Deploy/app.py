import mlflow
import numpy as np
import os
import cv2
import onnxruntime as ort
from datetime import datetime
import time
import sys
import redis
import json
import signal
import threading

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
# Model Settings
Model_uri = os.getenv("MODEL_URI", "models:/Truck_classification_Model/Production")
Mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

# Redis Connection Settings
REDIS_HOST = os.getenv("REDIS_HOST", "redis_broker")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Output Settings
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./results")

# Global flag for graceful shutdown
RUNNING = True

# =============================================================================
# SIGNAL HANDLING
# =============================================================================
def handle_signal(signum, frame):
    """Handles system signals for graceful shutdown."""
    global RUNNING
    print(f"\n🛑 Received signal {signum}. Stopping consumer...")
    RUNNING = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# =============================================================================
# PREPROCESSING & POSTPROCESSING
# =============================================================================
def preprocess_frame(frame_bgr, input_size=(640, 640)):
    """
    Preprocesses a BGR frame for ONNX model inference.
    
    Args:
        frame_bgr: OpenCV BGR image
        input_size: Target input size (width, height)
    
    Returns:
        Preprocessed tensor ready for inference
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, input_size)
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_chw = np.transpose(frame_norm, (2, 0, 1))
    frame_chw = np.expand_dims(frame_chw, axis=0)
    return frame_chw

def postprocess_classification(outputs):
    """
    Extracts class ID and confidence from model output.
    
    Args:
        outputs: Raw model output
    
    Returns:
        Tuple of (class_id, confidence)
    """
    output = outputs[0] 
    if len(output.shape) == 3:
        output = output[0]
    class_probs = output[4:, :]
    max_indices = np.unravel_index(np.argmax(class_probs), class_probs.shape)
    class_id = int(max_indices[0])
    confidence = float(class_probs[max_indices])
    return class_id, confidence

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

# =============================================================================
# RESULT SAVING
# =============================================================================
def save_result_as_text(camera_id, class_id, confidence, image_path, output_dir="./results"):
    """
    Saves classification results to a text file.
    
    Args:
        camera_id: Camera identifier
        class_id: Predicted class ID
        confidence: Prediction confidence
        image_path: Path to source image
        output_dir: Directory to save results
    
    Returns:
        Path to saved result file
    """
    os.makedirs(output_dir, exist_ok=True)
    vehicle = VEHICLE_CLASSES[class_id]
    total_fee = vehicle["entry_fee"] + vehicle["xray_fee"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Camera ID: {camera_id}\n")
        f.write(f"Class ID: {class_id}\n")
        f.write(f"Class Name: {vehicle['name']}\n")
        f.write(f"Confidence: {confidence:.4f}\n")
        f.write(f"Entry Fee: {vehicle['entry_fee']:.2f}\n")
        f.write(f"X-ray Fee: {vehicle['xray_fee']:.2f}\n")
        f.write(f"Total Fee: {total_fee:.2f}\n")
        f.write(f"Image Path: {image_path}\n")
        f.write(f"Created At: {timestamp}\n")

    return file_path

# =============================================================================
# INFERENCE FUNCTION
# =============================================================================
def run_inference_on_npy(session, npy_path):
    """
    Loads a .npy file and runs inference.
    
    Args:
        session: ONNX Runtime session
        npy_path: Path to .npy file
    
    Returns:
        Model output
    """
    # Load the numpy array (BGR format from ingestion.py)
    frame_bgr = np.load(npy_path)
    
    # Preprocess for model
    input_tensor = preprocess_frame(frame_bgr)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    return outputs

# =============================================================================
# REDIS CONSUMER
# =============================================================================
def process_redis_queue(session, redis_client):
    """
    Main consumer loop that processes jobs from Redis queue.
    
    Args:
        session: ONNX Runtime session
        redis_client: Redis connection
    """
    print("🚀 Starting Redis consumer...")
    processed_count = 0
    
    while RUNNING:
        try:
            # Blocking pop with 1-second timeout (BLPOP)
            result = redis_client.blpop('video_jobs', timeout=1)
            
            if result is None:
                # No job available, continue loop
                continue
            
            # Parse the job
            _, job_data = result
            job = json.loads(job_data)
            
            camera_id = job.get("camera_id")
            image_path = job.get("image_path")
            timestamp = job.get("timestamp")
            
            print(f"📥 Processing job from {camera_id}: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"⚠️ File not found: {image_path}")
                continue
            
            # Run inference
            start_time = time.time()
            outputs = run_inference_on_npy(session, image_path)
            class_id, confidence = postprocess_classification(outputs)
            inference_time = time.time() - start_time
            
            # Save results
            result_path = save_result_as_text(
                camera_id=camera_id,
                class_id=class_id,
                confidence=confidence,
                image_path=image_path,
                output_dir=OUTPUT_DIR
            )
            
            processed_count += 1
            
            print(f"✅ [{camera_id}] Class: {VEHICLE_CLASSES[class_id]['name']} "
                  f"(Confidence: {confidence:.4f}, Time: {inference_time:.3f}s) "
                  f"| Total processed: {processed_count}")
            
            # Optional: Delete the .npy file after processing to save space
            # os.remove(image_path)
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in queue: {e}")
        except Exception as e:
            print(f"🔥 Processing error: {e}")
            time.sleep(1)  # Brief pause before retrying
    
    print(f"👋 Consumer stopped. Total processed: {processed_count}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 60)
    print("🚛 Truck Classification Consumer Service")
    print("=" * 60)
    
    # 1. Setup MLflow
    print(f"📡 Connecting to MLflow: {Mlflow_uri}")
    mlflow.set_tracking_uri(Mlflow_uri)
    
    # 2. Download and load model
    print(f"📦 Downloading model: {Model_uri}")
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=Model_uri)
    onnx_path = os.path.join(local_path, "model.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Model not found at: {onnx_path}")
        sys.exit(1)
    
    print(f"🧠 Loading ONNX model...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"✅ Model loaded successfully")
    
    # 3. Connect to Redis
    print(f"🔌 Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    redis_client = None
    
    while RUNNING:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST, 
                port=REDIS_PORT, 
                db=REDIS_DB,
                socket_connect_timeout=5,
                decode_responses=False  # Keep binary for consistency
            )
            redis_client.ping()
            print(f"✅ Connected to Redis")
            break
        except redis.ConnectionError as e:
            print(f"🔴 Redis connection failed: {e}. Retrying in 5s...")
            time.sleep(5)
    
    if not RUNNING:
        print("🛑 Shutdown requested before Redis connection")
        sys.exit(0)
    
    # 4. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 5. Start processing
    print("=" * 60)
    process_redis_queue(session, redis_client)
    
    # 6. Cleanup
    redis_client.close()
    print("✨ Service terminated gracefully")

if __name__ == "__main__":
    main()