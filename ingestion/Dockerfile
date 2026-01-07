FROM python:3.9-slim-bullseye

# ลง Library ที่จำเป็นสำหรับ OpenCV
# RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]