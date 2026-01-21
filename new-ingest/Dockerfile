# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
# libgl1 and libglib2.0-0 are critical for cv2 to work on Linux

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default command to run the script
CMD ["python", "ingestion.py"]
