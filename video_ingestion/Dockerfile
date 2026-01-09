FROM python:3.9-bookworm

WORKDIR /app

# --- เพิ่มบรรทัดนี้ เพื่อให้ Print Log ออกมาทันที ---
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code
COPY ingestion.py .

# Create dirs
RUN mkdir -p /app/videos /app/shared_memory

CMD ["python", "ingestion.py"]