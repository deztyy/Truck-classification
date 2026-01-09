# ใช้ Python 3.10
FROM python:3.10-slim

# ตั้งค่าที่ทำงานใน Container
WORKDIR /app

# Copy และติดตั้ง Library ตามที่ระบุใน requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ไฟล์ทั้งหมดเข้า Container
COPY . .

# เปิด Port สำหรับ Streamlit
EXPOSE 8501

# คำสั่งรันแอป
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]