# FROM python:3.11-slim

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip uninstall -y onnxruntime-gpu

# # Copy application code
# COPY fake-processing.py .
# COPY deletion_service.py .
# COPY minio_helper.py .

# # Set environment variables
# ENV PYTHONUNBUFFERED=1

# # Run the processing service
# CMD ["python", "fake-processing.py"]
FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app    

COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.26.0 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir onnxruntime-gpu==1.20.1
 
COPY processing.py .
ENV PYTHONUNBUFFERED=1

CMD ["python3", "processing.py"]
# FROM ultralytics/ultralytics:latest-python

# WORKDIR /app

# # Copy requirements first
# COPY requirements.txt .

# # We ONLY install the things NOT already in the Ultralytics image.
# # We also fix the numpy error here.
# RUN pip install --no-cache-dir "numpy<2.0.0" && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy your code last
# COPY processing.py .

# ENV PYTHONUNBUFFERED=1
# CMD ["python3", "processing.py"]