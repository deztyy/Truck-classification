# Update base image
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y setuptools wheel

COPY ingestion.py .

RUN mkdir -p /app/videos /app/shared_memory

CMD ["python", "ingestion.py"]
