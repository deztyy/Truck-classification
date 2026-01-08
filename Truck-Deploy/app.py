import mlflow
import numpy as np
import os
import cv2
import onnxruntime as ort
from datetime import datetime

Model_uri = "models:/Truck_classification_Model/Production"
Mlflow_uri = "http://localhost:5000"

mlflow.set_tracking_uri(Mlflow_uri)

def preprocess_frame(frame_bgr, input_size=(640, 640)):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # NO letterbox for classification
    frame_resized = cv2.resize(frame_rgb, input_size)

    frame_norm = frame_resized.astype(np.float32) / 255.0

    frame_chw = np.transpose(frame_norm, (2, 0, 1))  # HWC → CHW
    frame_chw = np.expand_dims(frame_chw, axis=0)   # (1, 3, 640, 640)

    return frame_chw

def load_image_as_opencv_frame(image_path):
    """
    Returns:
        frame_bgr: np.ndarray (H, W, 3), uint8, BGR
    """
    frame_bgr = cv2.imread(image_path)  # OpenCV ALWAYS returns BGR uint8

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
    # Detection models usually output shape (1, 16, 16800) 
    # where 16 = (4 box coordinates + 12 class scores)
    output = outputs[0] 
    
    # If the shape is (1, 16, 16800), we need to remove the batch dim
    if len(output.shape) == 3:
        output = output[0] # Now shape is (16, 16800)

    # In YOLO detection, classes usually start after the first 4 indices (box coords)
    # So indices 4 to 15 are your 12 vehicle classes
    class_probs = output[4:, :] # Shape (12, 16800)
    
    # Find the maximum confidence score in the entire grid
    max_indices = np.unravel_index(np.argmax(class_probs), class_probs.shape)
    
    class_id = int(max_indices[0]) # The row index is the Class ID
    confidence = float(class_probs[max_indices]) # The value is the confidence

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

def save_result_as_text(
    camera_id,
    class_id,
    confidence,
    image_path,
    output_dir="./results"
):
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
    print("Pulling model from MLflow Registry...")
    # local_path = mlflow.artifacts.download_artifacts(artifact_uri=Model_uri)
    onnx_path = r"D:\Truck-classification\model\truck_classification.onnx"  # wherever your model is
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("Model loaded into ONNX Runtime.")

    test_image = "test_images/frame_41134.jpg"
    camera_id = "CAM_01"

    outputs = run_single_image_inference(session, test_image)
    print(f"DEBUG: Model output shape is {outputs[0].shape}")
    class_id, confidence = postprocess_classification(outputs)
    if class_id is None:
        print("No vehicle detected")
        exit()

    result_path = save_result_as_text(
        camera_id=camera_id,
        class_id=class_id,
        confidence=confidence,
        image_path=test_image,
    )

    print("Raw output:", outputs[0])
    print("Output shape:", outputs[0].shape)
    print("Prob sum:", softmax(outputs[0])[0].sum())
    print("Inference completed.")
    print(f"Result saved to: {result_path}")
