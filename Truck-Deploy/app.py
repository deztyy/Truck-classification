import mlflow
import numpy as np
import os
import cv2
import onnxruntime as ort

Model_uri = "models:/Truck_classification_Model/Production"
Mlflow_uri = "http://host.docker.internal:5000/"

mlflow.set_tracking_uri(Mlflow_uri)

def preprocess_frame(frame_bgr, input_size=(640, 640)):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, input_size)
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_chw = np.transpose(frame_norm, (2, 0, 1))
    input_tensor = np.expand_dims(frame_chw, axis=0)
    return input_tensor

if __name__ == "__main__":
    print("Pulling model from MLflow Registry...")
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=Model_uri)
    onnx_path = os.path.join(local_path, "model.onnx")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("Model loaded into ONNX Runtime.")
