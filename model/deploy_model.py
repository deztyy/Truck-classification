import mlflow.onnx
import onnx

# 1. Load the exported ONNX file
onnx_model = onnx.load("truck_classification.onnx")

# 2. Set your MLflow tracking URI (optional if running locally)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("ONNX_Model_detection")

with mlflow.start_run():
    # 3. Log and Register the model
    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path="truck_classification_model",
        registered_model_name="Truck_classification_Model"  # This adds it to the Registry
    )
    
print("Model successfully converted and registered to MLflow!")