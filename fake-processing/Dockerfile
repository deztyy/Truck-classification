FROM python:3.11

WORKDIR /app

# Install system libraries required by OpenCV and other dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
	&& rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fake-processing.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the processing service
CMD ["python", "fake-processing.py"]
