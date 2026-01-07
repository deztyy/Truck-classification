# 1. เปลี่ยนมาใช้ตัวเต็ม (ไม่ต้องลง libgl1 เองให้ยุ่งยาก)
FROM python:3.9

# 2. ตั้งค่าทั่วไป
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. ลบขั้นตอน apt-get ทิ้งไปเลย! (ข้ามจุดที่ Error)
# RUN apt-get update && ... <-- ลบทิ้งไป

# 4. Copy และลง Library Python
COPY requirements.txt .
# เพิ่ม --default-timeout เผื่อเน็ตช้า และใช้ mirror ถ้าจำเป็น (แต่ลองแบบปกติก่อน)
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# 5. Copy Code
COPY . .

# 6. Run
CMD ["python", "main.py"]