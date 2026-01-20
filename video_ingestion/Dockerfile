# Update base image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY ingestion_video.py .

RUN mkdir -p /app/videos /app/shared_memory

CMD ["python", "-u","ingestion_video.py"]