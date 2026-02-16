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

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir onnxruntime-gpu

COPY processing.py .
ENV PYTHONUNBUFFERED=1

CMD ["python3", "processing.py"]
