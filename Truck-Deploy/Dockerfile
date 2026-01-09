FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY app.py .

# We don't need to COPY images/results if we use 'volumes' in compose, 
# but copying app.py is essential.
CMD ["python", "app.py"]