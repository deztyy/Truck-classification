# Video Ingestion Service Tests

Comprehensive unit tests for the video ingestion service, covering edge cases and error handling.

## Test Coverage

### 1. Configuration Tests (`TestConfig`)
- Default configuration values
- Environment variable loading
- Empty and malformed RTSP URLs
- Configuration with spaces in URLs

### 2. Initialization Tests (`TestRedisMinioInitialization`)
- Successful Redis and MinIO client initialization
- Redis connection failures
- MinIO bucket creation when missing
- Error handling during initialization

### 3. CameraWorker Tests (`TestCameraWorker`)
- Worker initialization and threading
- RTSP connection establishment
- Batch serialization (success, errors, empty batches)
- Batch processing and queue management
- Upload worker functionality
- Queue full scenarios
- Frame capture and resizing
- Reconnection logic
- Error recovery

### 4. Main Execution Tests (`TestMainExecution`)
- Multiple camera worker creation
- Empty URL filtering
- Worker cleanup on shutdown

### 5. Edge Case Tests (`TestEdgeCases`)
- Very large batch handling
- Corrupted frame data
- Concurrent batch processing
- Rapid reconnection attempts
- Thread safety

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=video_ingest --cov-report=html --cov-report=term
```

### Run specific test class
```bash
pytest test_video_ingest.py::TestCameraWorker -v
```

### Run specific test method
```bash
pytest test_video_ingest.py::TestCameraWorker::test_serialize_batch_success -v
```

### Run tests with markers
```bash
# Run only unit tests (if marked)
pytest -m unit

# Run with verbose output
pytest -v

# Run with detailed output on failures
pytest -vv
```

### Generate coverage report
```bash
pytest --cov=video_ingest --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Structure

```
ingestion/
├── video_ingest.py           # Source code
├── test_video_ingest.py      # Unit tests
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Runtime dependencies
└── requirements-test.txt      # Test dependencies
```

## Key Testing Patterns Used

1. **Mocking External Dependencies**: Redis, MinIO, and OpenCV are mocked to isolate unit tests
2. **Threading Tests**: Proper cleanup and timeout handling for threaded operations
3. **Error Injection**: Simulating failures to test error handling paths
4. **Boundary Testing**: Testing with empty data, very large data, and corrupted data
5. **Concurrent Testing**: Validating thread-safe operations

## Continuous Integration

To run tests in CI/CD pipeline:

```yaml
# .github/workflows/test.yml example
- name: Run tests
  run: |
    pip install -r requirements.txt
    pip install -r requirements-test.txt
    pytest --cov=video_ingest --cov-report=xml
```

## Notes

- Tests use mocking to avoid requiring actual Redis, MinIO, or RTSP streams
- Threading tests include timeouts to prevent hanging
- Some tests involve brief `time.sleep()` calls to allow async operations to complete
- Coverage should be >80% for production code
