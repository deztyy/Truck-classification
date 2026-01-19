# Truck Classification System - Workflow

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRUCK CLASSIFICATION SYSTEM                       │
│                    (Docker Compose Orchestration)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │                               │
        ┌───────────▼────────────┐      ┌──────────▼──────────┐
        │  VIDEO INGESTION       │      │  TRUCK DEPLOYMENT   │
        │  SUBSYSTEM             │      │  INFERENCE SYSTEM   │
        │ (video_ingestion/)     │      │  (Truck-Deploy/)    │
        └───────────┬────────────┘      └──────────┬──────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  SHARED SERVICES  │
                          │  • Redis Queue    │
                          │  • MinIO Storage  │
                          │  • MLflow Tracking│
                          │  • PostgreSQL DB  │
                          └───────────────────┘
```

---

## Component Details

### 1. VIDEO INGESTION SUBSYSTEM (`video_ingestion/`)

**Purpose**: Captures video frames and prepares them for inference

**Components**:
- **Redis Broker**: Message queue for frame batches
- **Video Worker**: Processes video files frame-by-frame

**Workflow**:
```
Video File (testvid_1.mp4)
    │
    ▼
[Video Reader - cv2.VideoCapture]
    │
    ├─→ Resize to 640x640 with padding
    ├─→ Convert BGR to RGB
    ├─→ Apply JPEG compression (quality=80)
    │
    ▼
[Frame Batching - BATCH_SIZE=30 frames]
    │
    ▼
[Redis Message Queue]
    │
    └─→ Topic: 'video_test' (CAMERA_ID)
```

**Key Configuration** (`ingestion_video.py`):
- `VIDEO_PATH`: Path to input video
- `CAMERA_ID`: Identifier (e.g., "VIDEO_TEST")
- `TARGET_SIZE`: (640, 640)
- `PROCESS_FPS`: 1.0 frame/second
- `BATCH_SIZE`: 30 frames per batch
- `RETENTION_SECONDS`: 3600 (1 hour file cleanup)

**Outputs**:
- Frame images saved to `shared_data/{CAMERA_ID}/`
- Metadata pushed to Redis queue

---

### 2. TRUCK DEPLOYMENT SUBSYSTEM (`Truck-Deploy/`)

**Purpose**: Performs truck classification inference and tracking

**Components**:
- **MinIO**: Object storage (S3-compatible) for model artifacts
- **MLflow**: Model registry and tracking
- **Truck Classification App**: Main inference engine
- **PostgreSQL DB**: Results persistence

**Workflow**:
```
Frame Batch from Redis
    │
    ▼
[Load ONNX Model - truck_classification.onnx]
    │
    ▼
[ByteTrack Object Tracking]
    ├─→ Detect truck objects
    ├─→ Assign track IDs
    ├─→ Maintain track state (tracked, lost, removed)
    │
    ▼
[Classification]
    ├─→ Classify truck type/attributes
    ├─→ Generate confidence scores
    │
    ▼
[Database Storage]
    ├─→ Save detections to PostgreSQL
    ├─→ Save results to file
    │
    ▼
[MLflow Logging]
    └─→ Log metrics, parameters, artifacts
```

**Key Features** (`Truck-Deploy/app.py`):
- **ByteTrack Integration**: Multi-object tracking with Kalman filtering
- **ONNX Runtime**: Fast inference execution
- **Detection Classes**: 
  - `Detection`: Bounding box, confidence score, class ID
  - `STrack`: Track management with Kalman state
  - `ByteTracker`: Assignment and tracking logic

**Outputs**:
- Detection results in `results/{timestamp}.txt`
- MLflow tracking with model metrics
- Database records in PostgreSQL

---

### 3. SHARED SERVICES

#### Redis Message Broker
```
Frame Ingestion  ──→  Redis Queue  ←──  Classification App
                    (Topic: CAMERA_ID)
```
- Decouples video ingestion from inference
- FIFO processing of frame batches
- Max queue size: 100 batches (backpressure)

#### MinIO Object Storage
- Stores model artifacts
- MLflow bucket: `mlflow-bucket/`
- Model path: `0/models/{model-id}/artifacts/`

#### MLflow Tracking
- Model registry and versioning
- Experiment tracking
- Metric logging
- Model artifact storage

#### PostgreSQL Database
- Stores detection results
- Persistence across sessions
- Query results for reporting

---

## Data Flow Diagram

```
[Video File]
      │
      ├─→ [Video Ingestion] ─Redis→ [Queue]
      │                               │
      │                               ▼
      │                        [Truck App]
      │                          │ │ │
      │                          ├─┴─┼─→ [PostgreSQL]
      │                          ├───┤
      │                          │   └─→ [Results Files]
      │                          │
      │                          └─────→ [MLflow]
      │                                    │
      │                                    ▼
      │                                  [MinIO]
      │
      └─→ [Shared Data Mount] ← Persistent Storage
             (shared_data/)
```

---

## Configuration & Environment

### Video Ingestion Environment Variables
```bash
REDIS_HOST=redis_broker
REDIS_PORT=6379
OUTPUT_FOLDER=/app/shared_memory
CAMERA_ID=VIDEO_TEST
VIDEO_PATH=/app/videos/testvid_1.mp4
LOOP_VIDEO=true
PROCESS_FPS=1.0
BATCH_SIZE=30
```

### Truck Deploy Environment Variables
```bash
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET_NAME=mlflow-bucket
MINIO_PORT_API=9000
MINIO_PORT_CONSOLE=9001
```

---

## Network Architecture

```
All services connected via: truck_system_network (Docker bridge)

Services:
├── redis_broker (Port 6379)
├── video_test (Worker)
├── minio (Ports 9000, 9001)
├── create-bucket (Initialization)
├── mlflow_server
├── postgres_db
└── truck_app (Inference)
```

---

## Execution Flow

### Startup Sequence
1. **Docker Compose Up** (root directory)
   - Initializes: `video_ingestion/compose.yml` and `Truck-Deploy/compose.yml`

2. **Video Ingestion Phase**
   - Redis broker starts
   - Video worker connects to video file
   - Begins frame extraction at `PROCESS_FPS`
   - Creates frame batches of size `BATCH_SIZE`
   - Publishes to Redis topic: `{CAMERA_ID}`

3. **Infrastructure Phase**
   - MinIO object storage initializes
   - MLflow bucket created
   - PostgreSQL database starts

4. **Inference Phase**
   - Truck classification app starts
   - Loads ONNX model from MinIO
   - Subscribes to Redis queue
   - Processes incoming frame batches
   - Runs ByteTrack algorithm
   - Stores results in database and files

### Continuous Processing
```
Loop (every batch):
  1. Get frame batch from Redis
  2. Preprocess frames (resize, normalize)
  3. Run ONNX inference
  4. Apply ByteTrack
  5. Classify detections
  6. Store in PostgreSQL
  7. Log to MLflow
```

### Graceful Shutdown
- Signals: SIGINT (Ctrl+C), SIGTERM
- Services stop processing new frames
- Complete in-flight batches
- Write final results to database

---

## Output Artifacts

### Files Generated
```
results/
├── 20260108_095525.txt  (Detection results with timestamps)
├── 20260108_143036.txt
└── 20260109_045242.txt

shared_data/
├── CAM_001/  (Frame images by camera)
├── CAM_002/
├── CAM_003/
└── VIDEO_TEST/
    └── frame_*.jpg (Extracted frames)
```

### Database Records
- Detection bounding boxes
- Track IDs
- Classification results
- Timestamps
- Camera IDs

### MLflow Artifacts
- Model versions
- Training/inference metrics
- Performance logs

---

## Summary

This system implements a **real-time truck detection and classification pipeline** with:
- ✅ **Decoupled Architecture**: Ingestion and inference are independent
- ✅ **Scalability**: Multiple cameras via Redis queue
- ✅ **Persistence**: PostgreSQL + MinIO storage
- ✅ **Tracking**: ByteTrack with Kalman filtering
- ✅ **Model Management**: MLflow integration
- ✅ **Container Orchestration**: Docker Compose
