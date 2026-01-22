import os
import mlflow
import onnx
from mlflow.tracking import MlflowClient
import time
import sys

# ============================================================================
# CRITICAL: Use SAME credentials as docker-compose.yml
# ============================================================================
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"  # Changed from admin123
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"  # Changed from password123
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

model_name = "Truck_classification_Model"
onnx_path = "truck_classification.onnx"  # Make sure this file exists!

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
    print("  Run: docker-compose up -d mlflow-server")
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
    print(f"  Please make sure you have 'truck_classification.onnx' in current directory")
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
                pip_requirements=["onnx", "onnxruntime"]
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

# Get the latest version and set to Production
print(f"\nSetting model to Production stage...")
try:
    # Get all versions
    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        latest_version = max([int(v.version) for v in versions])
        
        client.transition_model_version_stage(
            name=model_name,
            version=str(latest_version),
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"  ✓ Model version {latest_version} → Production")
    else:
        print(f"  ⚠ No versions found, trying alternative method...")
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            latest_version = latest_versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"  ✓ Model version {latest_version} → Production")
        else:
            print(f"  ⚠ Could not set Production stage automatically")
            latest_version = "1"
    
except Exception as e:
    print(f"  ⚠ Failed to set stage: {e}")
    print(f"  You can set it manually in MLflow UI: http://localhost:5000")
    latest_version = "1"

# Verify
print(f"\nVerifying artifacts in MinIO...")
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
print(f"\n📍 Next Steps:")
print(f"  1. Verify in MLflow UI: http://localhost:5000")
print(f"  2. Verify in MinIO UI: http://localhost:9001")
print(f"     - Username: minioadmin")
print(f"     - Password: minioadmin")
print(f"     - Check bucket: mlflow-bucket")
print(f"  3. Start your worker: docker-compose up -d processing-worker")
print(f"{'='*60}")