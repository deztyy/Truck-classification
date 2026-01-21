FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fake-processing.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the processing service
CMD ["python", "fake-processing.py"]
