# Base Image
FROM python:3.9-slim

# ตั้งค่า Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create Work Directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python Libraries
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pyzmq \
    numpy \
    redis \
    minio

# Copy Code
COPY video_ingest.py .

# Command to run
CMD ["python", "video_ingest.py"]
