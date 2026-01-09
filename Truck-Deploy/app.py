import mlflow
import numpy as np
import os
import cv2
import onnxruntime as ort
from datetime import datetime
import time
import sys

Model_uri = os.getenv("MODEL_URI", "models:/Truck_classification_Model/Production")
Mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

mlflow.set_tracking_uri(Mlflow_uri)

def preprocess_frame(frame_bgr, input_size=(640, 640)):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, input_size)
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_chw = np.transpose(frame_norm, (2, 0, 1))
    frame_chw = np.expand_dims(frame_chw, axis=0)
    return frame_chw

def load_image_as_opencv_frame(image_path):
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return frame_bgr

def run_single_image_inference(session, image_path):
    frame_bgr = load_image_as_opencv_frame(image_path)
    input_tensor = preprocess_frame(frame_bgr)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    return outputs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def postprocess_classification(outputs):
    output = outputs[0] 
    if len(output.shape) == 3:
        output = output[0]
    class_probs = output[4:, :]
    max_indices = np.unravel_index(np.argmax(class_probs), class_probs.shape)
    class_id = int(max_indices[0])
    confidence = float(class_probs[max_indices])
    return class_id, confidence

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

def save_result_as_text(camera_id, class_id, confidence, image_path, output_dir="./results"):
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

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Truck Classification Worker")
    print("=" * 60)
    print(f"MLflow Tracking URI: {Mlflow_uri}")
    print(f"Model URI: {Model_uri}")
    print("=" * 60)
    
    # Test basic connectivity first
    print("\n[1/4] Testing MLflow connectivity...")
    for i in range(5):
        try:
            import requests
            response = requests.get(f"{Mlflow_uri}/health", timeout=2)
            print(f"✓ MLflow health check passed: {response.status_code}")
            break
        except requests.exceptions.ConnectionError as e:
            print(f"✗ Connection failed (Attempt {i+1}/5): {e}")
            if i < 4:
                time.sleep(5)
        except Exception as e:
            print(f"✗ Unexpected error (Attempt {i+1}/5): {type(e).__name__}: {e}")
            if i < 4:
                time.sleep(5)
    else:
        print("\n❌ FATAL: Cannot reach MLflow server")
        print("Debug steps:")
        print("1. Run: docker logs mlflow_registry")
        print("2. Run: docker exec truck_worker ping mlflow-server")
        print("3. Run: docker exec truck_worker curl http://mlflow-server:5000/health")
        sys.exit(1)
    
    # Test MLflow client
    print("\n[2/4] Testing MLflow client...")
    try:
        client = mlflow.MlflowClient(tracking_uri=Mlflow_uri)
        print("✓ MLflow client created successfully")
        
        # List registered models
        models = client.search_registered_models()
        print(f"✓ Found {len(models)} registered models:")
        for model in models:
            print(f"  - {model.name}")
            versions = client.get_latest_versions(model.name)
            for v in versions:
                aliases = v.aliases if hasattr(v, 'aliases') else []
                tags = v.tags if hasattr(v, 'tags') else {}
                print(f"    Version {v.version}: stage={v.current_stage}, aliases={aliases}")
        
        if not models:
            print("⚠ WARNING: No models found in registry!")
            print("You need to register your model first.")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ MLflow client error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Download model
    print(f"\n[3/4] Downloading model: {Model_uri}")
    local_path = None
    for i in range(5):
        try:
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=Model_uri)
            print(f"✓ Model downloaded to: {local_path}")
            
            # List downloaded files
            print(f"Downloaded files:")
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, local_path)
                    print(f"  - {rel_path}")
            break
            
        except mlflow.exceptions.MlflowException as e:
            print(f"✗ MLflow error (Attempt {i+1}/5): {e}")
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                print("\n❌ Model not found!")
                print("Available models and versions:")
                try:
                    for model in client.search_registered_models():
                        print(f"\n  Model: {model.name}")
                        versions = client.get_latest_versions(model.name)
                        for v in versions:
                            print(f"    - Version {v.version}, Stage: {v.current_stage}")
                            if hasattr(v, 'aliases'):
                                print(f"      Aliases: {v.aliases}")
                except:
                    pass
                sys.exit(1)
            if i < 4:
                time.sleep(5)
                
        except Exception as e:
            print(f"✗ Unexpected error (Attempt {i+1}/5): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            if i < 4:
                time.sleep(5)
    
    if local_path is None:
        print("\n❌ FATAL: Could not download model after 5 attempts")
        sys.exit(1)
    
    # Load ONNX model
    print("\n[4/4] Loading ONNX model...")
    onnx_path = os.path.join(local_path, "model.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"✗ ONNX file not found at: {onnx_path}")
        print("Available files:")
        for root, dirs, files in os.walk(local_path):
            for file in files:
                print(f"  - {os.path.join(root, file)}")
        sys.exit(1)
    
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        print(f"✓ Model loaded from {onnx_path}")
    except Exception as e:
        print(f"✗ Failed to load ONNX model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run inference
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)
    
    test_image = "test_images/frame_41134.jpg"
    camera_id = "CAM_01"

    if not os.path.exists(test_image):
        print(f"✗ Test image not found: {test_image}")
        sys.exit(1)

    outputs = run_single_image_inference(session, test_image)
    print(f"Output shape: {outputs[0].shape}")
    
    class_id, confidence = postprocess_classification(outputs)
    
    if class_id is None:
        print("No vehicle detected")
        sys.exit(0)

    result_path = save_result_as_text(
        camera_id=camera_id,
        class_id=class_id,
        confidence=confidence,
        image_path=test_image,
    )

    print(f"\n✓ Inference completed successfully!")
    print(f"Class: {VEHICLE_CLASSES[class_id]['name']} (ID: {class_id})")
    print(f"Confidence: {confidence:.4f}")
    print(f"Result saved to: {result_path}")