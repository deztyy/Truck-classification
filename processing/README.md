# Processing Worker Service

Frame processing worker that consumes tasks from Redis queue, processes numpy batch files from MinIO, and saves processed images back to MinIO.

## Features

- **Redis Queue Consumer**: Continuously monitors Redis queue for processing tasks
- **Batch Frame Processing**: Loads numpy arrays (30 frames), selects 1 frame, converts to JPEG
- **MinIO Integration**: Downloads from source bucket, uploads to `process-frames` bucket
- **Auto Cleanup**: Deletes processed batch files from source bucket
- **Database Writer**: Saves transaction records (ready for ML model integration)
- **12 Vehicle Classes**: Supports full vehicle classification schema

## Architecture

```
Redis Queue → Processing Worker → MinIO (process-frames) → Database
                ↓
         numpy batch (30 frames)
                ↓
         select 1 frame → JPEG
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO server endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_SECURE` | `false` | Use HTTPS for MinIO |

## Build & Run

### Docker
```bash
docker build -t processing-worker ./fake-processing
docker run --rm \
  -e REDIS_HOST=redis \
  -e MINIO_ENDPOINT=minio:9000 \
  processing-worker
```

### Docker Compose
```bash
docker-compose up processing-worker
```

### Local Development
```bash
cd fake-processing
pip install -r requirements.txt
python fake-processing.py
```

## Vehicle Classes

| ID | Class Name | Entry Fee | X-Ray Fee | Total Fee |
|----|------------|-----------|-----------|-----------|
| 1 | car | ฿0 | ฿0 | ฿0 |
| 2 | other | ฿0 | ฿0 | ฿0 |
| 3 | other_truck | ฿100 | ฿50 | ฿150 |
| 4 | pickup_truck | ฿0 | ฿0 | ฿0 |
| 5 | truck_20_back | ฿100 | ฿250 | ฿350 |
| 6 | truck_20_front | ฿100 | ฿250 | ฿350 |
| 7 | truck_20x2 | ฿100 | ฿500 | ฿600 |
| 8 | truck_40 | ฿100 | ฿350 | ฿450 |
| 9 | truck_roro | ฿100 | ฿50 | ฿150 |
| 10 | truck_tail | ฿100 | ฿50 | ฿150 |
| 11 | motorcycle | ฿0 | ฿0 | ฿0 |
| 12 | truck_head | ฿100 | ฿50 | ฿150 |

## Processing Flow

1. **Pop task from Redis queue** (blocking, 5 sec timeout)
2. **List objects in MinIO** at specified bucket/prefix
3. **Download numpy batch file** (30 frames)
4. **Select middle frame** from batch
5. **Convert to PIL Image** (handles float32/uint8)
6. **Save as JPEG** (quality 95)
7. **Upload to process-frames bucket**
   - Filename: `{camera_id}_{task_id}_{timestamp}.jpg`
8. **Delete original batch file** from source bucket
9. **Create transaction record** in database
   - `class_id`: 0 (pending ML inference)
   - `confidence`: 0.0 (pending ML inference)
   - `total_fee`: 0.00 (pending calculation)
10. **Clean up local files**

## Database Schema

### vehicle_transactions
```sql
CREATE TABLE vehicle_transactions (
    id BIGSERIAL PRIMARY KEY,
    camera_id VARCHAR(50) NOT NULL,
    track_id VARCHAR(100) NOT NULL,
    class_id INT NOT NULL,
    total_fee NUMERIC(10, 2) DEFAULT 0.00,
    time_stamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    img_path TEXT,
    confidence NUMERIC(5, 4)
);
```

### vehicle_classes
```sql
CREATE TABLE vehicle_classes (
    class_id SERIAL PRIMARY KEY,
    class_name VARCHAR(50) UNIQUE NOT NULL,
    entry_fee NUMERIC(10, 2),
    xray_fee NUMERIC(10, 2),
    total_fee NUMERIC(10, 2)
);
```

## Logs

```
INFO - Processing Worker Started
INFO - Waiting for tasks from Redis queue...
INFO - New task received: TASK_1737484800_001
INFO - Found 1 objects in video-frames/camera_001
INFO - Loaded batch shape: (30, 720, 1280, 3)
INFO - Selected frame 15 with shape: (720, 1280, 3)
INFO - ✓ Uploaded 245678 bytes to process-frames/CAM_001_TASK_001_20260121_120000.jpg
INFO - ✓ Deleted batch_001.npy from video-frames
INFO - ✓ Cleaned up local file: processed_data/batch_001.npy
INFO - ✓ Task completed successfully
```

## Dependencies

- `redis==5.0.1` - Redis client
- `minio==7.2.0` - MinIO object storage client
- `numpy==1.26.0` - Array processing
- `Pillow==10.2.0` - Image conversion
- `python-dotenv==1.0.0` - Environment variables

## Notes

- Worker runs continuously, waiting for tasks
- Supports multiple workers with `docker-compose up --scale processing-worker=3`
- Transaction data saved without classification (pending ML model integration)
- Auto-creates `process-frames` bucket if not exists
