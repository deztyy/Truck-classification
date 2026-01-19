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

# Copy the script
COPY app.py .

# We don't need to COPY images/results if we use 'volumes' in compose, 
RUN mkdir -p /app/results /app/vehicle_images /app/shared_memory
# but copying app.py is essential.
CMD ["python", "-u", "app.py"]