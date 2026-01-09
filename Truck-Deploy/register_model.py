import os
import mlflow
import onnx
from mlflow.tracking import MlflowClient
import time
import sys

# 1. Setup Environment
os.environ["AWS_ACCESS_KEY_ID"] = "admin123"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

model_name = "Truck_classification_Model"
onnx_path = "truck_classification.onnx"

print(f"{'='*60}")
print(f"MLflow Model Registration (with S3/MinIO)")
print(f"{'='*60}")

# Test connectivity first
print("\n[1/5] Testing MLflow connectivity...")
try:
    import requests
    response = requests.get("http://localhost:5000/health", timeout=5)
    print(f"  ✓ MLflow server: {response.status_code}")
except Exception as e:
    print(f"  ✗ MLflow server not accessible: {e}")
    sys.exit(1)

print("\n[2/5] Testing MinIO connectivity...")
try:
    response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
    print(f"  ✓ MinIO server: {response.status_code}")
except Exception as e:
    print(f"  ✗ MinIO server not accessible: {e}")
    print("  Make sure MinIO is running: docker-compose ps")
    sys.exit(1)

# Check if model file exists
print(f"\n[3/5] Checking model file...")
if not os.path.exists(onnx_path):
    print(f"  ✗ Model file not found: {onnx_path}")
    sys.exit(1)

file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
print(f"  ✓ Found: {onnx_path} ({file_size:.2f} MB)")

print(f"\n[4/5] Loading and validating ONNX model...")
try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX model is valid")
except Exception as e:
    print(f"  ✗ Invalid ONNX model: {e}")
    sys.exit(1)

print(f"\n[5/5] Uploading model to MLflow...")
print(f"  This may take a while for large models ({file_size:.2f} MB)...")
print(f"  Note: First upload to MinIO can be slow (30-60 seconds)")

start_time = time.time()

try:
    with mlflow.start_run(run_name="truck_model_registration") as run:
        print(f"  Run ID: {run.info.run_id}")
        
        # Log with progress indicator
        print(f"  Uploading to S3/MinIO...")
        print(f"  (Please wait, this can take 1-2 minutes for large files...)")
        
        # Add a simple progress indicator
        import threading
        stop_progress = threading.Event()
        
        def show_progress():
            dots = 0
            while not stop_progress.is_set():
                print(".", end="", flush=True)
                dots += 1
                if dots % 60 == 0:
                    elapsed = time.time() - start_time
                    print(f" {elapsed:.0f}s", flush=True)
                time.sleep(1)
        
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        try:
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="model",
                registered_model_name=model_name,
                pip_requirements=["onnx", "mlflow"]
            )
        finally:
            stop_progress.set()
            progress_thread.join(timeout=1)
        
        upload_time = time.time() - start_time
        print(f"\n  ✓ Upload complete ({upload_time:.1f}s)")
        
        run_id = run.info.run_id

except Exception as e:
    print(f"\n  ✗ Upload failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get the latest version
print(f"\nSetting model to Production stage...")
try:
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"  ✓ Model version {latest_version} → Production")
    
except Exception as e:
    print(f"  ✗ Failed to set stage: {e}")
    # Don't exit - the model is still registered
    latest_version = "?"

# Verify
print(f"\nVerifying artifacts...")
try:
    artifacts = client.list_artifacts(run_id, path="model")
    print(f"  ✓ Found {len(artifacts)} artifacts:")
    for artifact in artifacts[:5]:  # Show first 5
        size_mb = artifact.file_size / (1024 * 1024) if artifact.file_size else 0
        print(f"    - {artifact.path} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"  ⚠ Could not list artifacts: {e}")

print(f"\n{'='*60}")
print(f"✓ SUCCESS!")
print(f"{'='*60}")
print(f"Model Name: {model_name}")
print(f"Version: {latest_version}")
print(f"Stage: Production")
print(f"Model URI: models:/{model_name}/Production")
print(f"Total time: {time.time() - start_time:.1f}s")
print(f"\nView in MLflow UI: http://localhost:5000")
print(f"View in MinIO UI: http://localhost:9001")
print(f"{'='*60}")