# 🚗 Vehicle Entry System

ระบบบันทึกการเข้าออกของรถยนต์ด้วย Streamlit และ PostgreSQL

## 📋 คุณสมบัติหลัก

- **User Mode**: ดูข้อมูลรถคันปัจจุบันและประวัติการเข้าออก
- **Admin Mode**: จัดการประเภทรถ, บันทึกข้อมูล, ดูสถิติ
- **Real-time Dashboard**: แสดงข้อมูลสดพร้อม Auto-refresh
- **Analytics**: วิเคราะห์รายได้และสถิติต่างๆ

## 🚀 การติดตั้งและรัน

### วิธีที่ 1: ใช้ Docker Compose (แนะนำ)

1. **เตรียมไฟล์**
   ```bash
   # ตรวจสอบว่ามีไฟล์ครบทั้งหมด
   ls -la
   # ต้องมี: docker-compose.yml, Dockerfile, .env, main.py, requirements.txt, init.sql
   ```

2. **รันระบบ**
   ```bash
   # สร้างและรัน containers
   docker-compose up -d
   
   # ดู logs
   docker-compose logs -f
   
   # หยุดระบบ
   docker-compose down
   
   # ลบข้อมูลทั้งหมด (รวม database)
   docker-compose down -v
   ```

3. **เข้าใช้งาน**
   - เปิดเบราว์เซอร์ไปที่: http://localhost:8501
   - User Mode: คลิก "User Mode" (ไม่ต้องใส่รหัส)
   - Admin Mode: คลิก "Admin Mode" → ใส่รหัส `Admin1234`

### วิธีที่ 2: รันแยก (สำหรับ Development)

1. **ติดตั้ง PostgreSQL**
   ```bash
   # macOS
   brew install postgresql@15
   brew services start postgresql@15
   
   # Ubuntu/Debian
   sudo apt-get install postgresql-15
   sudo systemctl start postgresql
   ```

2. **สร้าง Database**
   ```bash
   psql -U postgres
   CREATE DATABASE mydb;
   CREATE USER user WITH PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE mydb TO user;
   \q
   
   # รัน init.sql
   psql -U user -d mydb -f init.sql
   ```

3. **ติดตั้ง Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **รัน Streamlit**
   ```bash
   streamlit run main.py
   ```

## 🔧 การตั้งค่า

### ไฟล์ `.env`
```env
# Database
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=mydb
POSTGRES_PORT=5429

# Application
DATABASE_URL=postgresql://user:password@db:5432/mydb

# Admin
ADMIN_PASSWORD=Admin1234
```

### เปลี่ยน Port
แก้ไขใน `docker-compose.yml`:
```yaml
web:
  ports:
    - "8080:8501"  # เปลี่ยนจาก 8501 เป็น 8080
```

## 📦 โครงสร้างโปรเจกต์

```
vehicle-entry-system/
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Web app container
├── .env                    # Environment variables
├── init.sql                # Database initialization
├── main.py                 # Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # คู่มือนี้
```

## 🗄️ โครงสร้าง Database

### ตาราง `vehicle_classes`
| Column | Type | Description |
|--------|------|-------------|
| class_id | SERIAL | รหัสประเภทรถ (PK) |
| class_name | VARCHAR(50) | ชื่อประเภทรถ (UNIQUE) |
| entry_fee | NUMERIC(10,2) | ค่าผ่านทาง |
| xray_fee | NUMERIC(10,2) | ค่า X-Ray |
| total_fee | NUMERIC(10,2) | รวมค่าใช้จ่าย |

### ตาราง `vehicle_transactions`
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | รหัสธุรกรรม (PK) |
| camera_id | VARCHAR(50) | รหัสกล้อง |
| class_id | INT | รหัสประเภทรถ (FK) |
| applied_entry_fee | NUMERIC(10,2) | ค่าผ่านทางที่เก็บจริง |
| applied_xray_fee | NUMERIC(10,2) | ค่า X-Ray ที่เก็บจริง |
| total_applied_fee | NUMERIC(10,2) | รวมค่าใช้จ่ายจริง |
| image_path | TEXT | path ของรูปภาพ |
| created_at | TIMESTAMP | เวลาบันทึก (Asia/Bangkok) |

## 🐛 Troubleshooting

### ปัญหา: Cannot connect to database
```bash
# ตรวจสอบว่า database container รันอยู่
docker ps

# ดู logs ของ database
docker-compose logs db

# Restart database
docker-compose restart db
```

### ปัญหา: Port already in use
```bash
# ดู process ที่ใช้ port 8501
lsof -i :8501

# หรือเปลี่ยน port ใน docker-compose.yml
```

### ปัญหา: Permission denied
```bash
# ให้สิทธิ์ไฟล์
chmod +x init.sql

# ใน macOS/Linux อาจต้อง
sudo docker-compose up -d
```

## 📊 ข้อมูลตัวอย่าง

ระบบจะสร้างข้อมูลตัวอย่างอัตโนมัติ:
- รถเก๋ง (Sedan): 50 + 100 = 150 บาท
- รถกระบะ (Pickup): 60 + 120 = 180 บาท
- รถบรรทุก 6 ล้อ: 100 + 150 = 250 บาท
- รถบรรทุก 10 ล้อ: 150 + 200 = 350 บาท
- รถพ่วง (Trailer): 200 + 300 = 500 บาท

## 🔐 ความปลอดภัย

**สำคัญ**: เปลี่ยนรหัสผ่านก่อนใช้งานจริง!

```env
ADMIN_PASSWORD=YourStrongPasswordHere
POSTGRES_PASSWORD=YourDatabasePassword
```

## 📝 License

MIT License - ใช้งานได้อย่างอิสระ

## 🤝 Contributing

PRs are welcome! สำหรับการเปลี่ยนแปลงใหญ่ กรุณาเปิด issue ก่อน

## 📧 Support

หากมีปัญหา กรุณา:
1. ตรวจสอบ logs: `docker-compose logs -f`
2. Restart: `docker-compose restart`
3. ลบและสร้างใหม่: `docker-compose down -v && docker-compose up -d`