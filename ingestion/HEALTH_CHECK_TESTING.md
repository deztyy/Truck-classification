# Health Check Testing Guide
================================

เอกสารนี้อธิบายวิธีทดสอบระบบ Health Check ใน VideoIngestor

## สิ่งที่ Health Check ตรวจสอบ

1. **Stream Disconnection** - ตรวจจับเมื่อ video stream ขาดการเชื่อมต่อ
2. **Frame Read Timeout** - ตรวจจับเมื่อ stream หยุดส่งเฟรม (stalled)
3. **Downtime Tracking** - บันทึกเวลาที่ระบบไม่ทำงาน
4. **Auto Recovery** - ตรวจจับเมื่อระบบกลับมาทำงานปกติ

## วิธีการทดสอบ

### วิธีที่ 1: ใช้ Simple Test Script (แนะนำ)

```bash
# 1. ติดตั้ง dependencies
pip install opencv-python numpy

# 2. เตรียม test video
# ดาวน์โหลด video ตัวอย่าง หรือใช้ video ที่มีอยู่
mkdir -p rtsp-camera/test-vid
# วาง video file ที่ rtsp-camera/test-vid/test_video.mp4

# 3. รัน test
cd ingestion
python simple_health_test.py
```

**Output ที่ต้องเห็น:**
```
✓ Health check thread STARTED (interval=2s)
✓ Video opened successfully
✓ Stream RECOVERED after 3.2s downtime
⚠ Stream DISCONNECTED
⚠ Stream STALLED (no frames for 5.5s, timeout=5s)
```

---

### วิธีที่ 2: ทดสอบด้วย Docker Container

```bash
# 1. Start services
docker-compose up -d redis minio

# 2. เปิด 2 terminal windows

# Terminal 1: รัน ingestion
docker-compose up ingestion-service

# Terminal 2: สังเกต logs
docker logs -f vid_ingestion

# 3. ทดสอบ scenarios:

# Scenario A: หยุด RTSP server (ถ้ามี)
docker-compose stop rtsp-server
# ต้องเห็น: "Stream disconnected for camera: camera_01"
# รอ 5 วินาที
docker-compose start rtsp-server
# ต้องเห็น: "Stream recovered after X.Xs downtime"

# Scenario B: หยุด Redis
docker-compose stop redis
# ต้องเห็น: "Redis publish error"
docker-compose start redis

# Scenario C: หยุด MinIO
docker-compose stop minio
# ต้องเห็น: "MinIO upload error"
docker-compose start minio
```

---

### วิธีที่ 3: ทดสอบใน Production Environment

**3.1 Monitor Health Status**
```python
# เพิ่ม REST API endpoint สำหรับ health check
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health_check():
    stats = video_ingestor.get_statistics()
    return jsonify({
        "status": "healthy" if stats['is_healthy'] else "unhealthy",
        "camera_id": stats['camera_id'],
        "frames_processed": stats['frames_processed'],
        "downtime_seconds": stats['total_downtime_seconds'],
        "is_connected": stats['is_connected']
    })

if __name__ == '__main__':
    app.run(port=8000)
```

**3.2 ตรวจสอบด้วย curl**
```bash
# ดู health status
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "camera_id": "camera_01",
  "frames_processed": 1523,
  "downtime_seconds": 12.5,
  "is_connected": true
}
```

---

### วิธีที่ 4: Unit Testing (Automated)

**4.1 Mock Video Source**
```python
import unittest
from unittest.mock import Mock, patch
from ingestion import VideoIngestor

class TestHealthCheck(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_stream_disconnection(self, mock_capture):
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        # Create ingestor
        ingestor = VideoIngestor(
            camera_id="test",
            rtsp_url="rtsp://test"
        )

        # Should detect disconnection
        stats = ingestor.get_statistics()
        self.assertFalse(stats['is_connected'])

    def test_frame_timeout_detection(self):
        # TODO: Implement timeout test
        pass
```

---

## Log Messages ที่ต้องเห็น

### ✅ Normal Operation
```
INFO - Health check thread started for camera: camera_01
INFO - Successfully connected to RTSP stream
INFO - Uploaded batch to MinIO: camera_01/batch_20260203_120530.npy
INFO - Published metadata to Redis list: frame_batches
```

### ⚠️ Stream Disconnection
```
WARNING - Stream disconnected for camera: camera_01
INFO - Reconnection attempt 1/5
INFO - Successfully connected to RTSP stream
INFO - Stream recovered after 3.2s downtime. Total downtime: 3.2s
```

### ⚠️ Frame Read Timeout
```
ERROR - Stream stalled for 31.5s (timeout: 30s) - camera: camera_01
WARNING - Stream still stalled for 45.2s - camera: camera_01
```

### ⚠️ Upload/Queue Errors
```
ERROR - MinIO upload error: Connection refused
ERROR - Redis publish error: Connection refused
```

---

## Performance Metrics

ดูสถิติระบบทุก 60 วินาที:
```
INFO - === Camera Statistics ===
INFO - camera_01: Frames=1500, Batches=50, Errors=0, Connected=True, Healthy=True, Downtime=0.0s
INFO - camera_02: Frames=1480, Batches=49, Errors=2, Connected=True, Healthy=True, Downtime=5.3s
```

---

## Troubleshooting

### ปัญหา: Health check ไม่ทำงาน
**วิธีแก้:**
```python
# เช็คว่า thread ทำงานหรือไม่
import threading
print(threading.active_count())  # ต้อง > 1
print([t.name for t in threading.enumerate()])  # ต้องมี 'HealthCheck-camera_01'
```

### ปัญหา: Timeout ตรวจจับไม่ได้
**วิธีแก้:**
- ลด `FRAME_READ_TIMEOUT_SECONDS` เพื่อทดสอบเร็วขึ้น
- ลด `HEALTH_CHECK_INTERVAL_SECONDS` เพื่อ check บ่อยขึ้น

```python
# ใน ingestion.py (สำหรับ testing เท่านั้น)
HEALTH_CHECK_INTERVAL_SECONDS = 2  # จาก 10
FRAME_READ_TIMEOUT_SECONDS = 5     # จาก 30
```

### ปัญหา: Downtime ไม่บันทึก
**วิธีเช็ค:**
```python
# เพิ่ม debug logging
logger.setLevel(logging.DEBUG)

# ดู internal state
print(f"Last frame read: {ingestor.last_frame_read_time}")
print(f"Downtime start: {ingestor.downtime_start_time}")
print(f"Is healthy: {ingestor.stream_is_healthy}")
```

---

## Integration with Monitoring Tools

### Prometheus Metrics
```python
from prometheus_client import Counter, Gauge

frames_processed = Counter('frames_processed_total', 'Total frames processed')
stream_health = Gauge('stream_health', 'Stream health status (1=healthy, 0=unhealthy)')
downtime_seconds = Counter('downtime_seconds_total', 'Total downtime in seconds')

# Update metrics in VideoIngestor
def _update_frame_read_time(self):
    # ... existing code ...
    frames_processed.inc()
    stream_health.set(1 if self.stream_is_healthy else 0)
```

### Grafana Dashboard
```json
{
  "panels": [
    {
      "title": "Stream Health Status",
      "targets": [{
        "expr": "stream_health"
      }]
    },
    {
      "title": "Total Downtime",
      "targets": [{
        "expr": "rate(downtime_seconds_total[5m])"
      }]
    }
  ]
}
```

---

## Best Practices

1. **ตั้งค่า timeout ให้เหมาะสม**
   - RTSP stream: 30s
   - Local file: 10s
   - Mock/test: 5s

2. **Log level ตาม environment**
   - Development: DEBUG
   - Production: INFO
   - Critical only: WARNING

3. **Alert thresholds**
   - Downtime > 60s → Send alert
   - Errors > 10 in 5min → Send alert
   - Health check thread died → Critical alert

4. **Monitor ทุก 60 วินาที**
   - เก็บ metrics ลง database
   - ส่งไป monitoring service
   - สร้าง report ประจำวัน
