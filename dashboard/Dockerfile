FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

EXPOSE 8501

# Create Streamlit config directory
RUN mkdir -p ~/.streamlit

# Configure Streamlit
RUN echo '\
    [server]\n\
    port = 8501\n\
    address = 0.0.0.0\n\
    headless = true\n\
    enableXsrfProtection = false\n\
    ' > ~/.streamlit/config.toml

# Run Streamlit app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
