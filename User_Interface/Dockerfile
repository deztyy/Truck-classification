# ใช้ Python 3.10
FROM python:3.10-slim

# ตั้งค่าที่ทำงานใน Container
WORKDIR /app

# ติดตั้ง Library พื้นฐานสำหรับ PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# [cite_start]Copy และติดตั้ง Library ตามที่ระบุใน requirements.txt [cite: 1]
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ไฟล์ทั้งหมดเข้า Container
COPY . .

# เปิด Port สำหรับ Streamlit
EXPOSE 8501

# คำสั่งรันแอป
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]