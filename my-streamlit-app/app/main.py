import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

# ตั้งค่าการเชื่อมต่อ Postgres (ใช้ชื่อ Service ใน Docker Compose เป็น Host)
engine = create_engine('postgresql://user:password@db:5432/mydb')

def init_db():
    """สร้าง Table users หากยังไม่มี"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100)
            );
        """))
        conn.commit()

def save_data(name, email):
    """บันทึกข้อมูลลง Database"""
    with engine.connect() as conn:
        query = text("INSERT INTO users (name, email) VALUES (:name, :email)")
        conn.execute(query, {"name": name, "email": email})
        conn.commit()

def load_data():
    """ดึงข้อมูลทั้งหมดมาแสดงผล"""
    return pd.read_sql("SELECT * FROM users ORDER BY id DESC", engine)

# --- UI ส่วนหน้าจอ Streamlit ---
st.title("User Registration System")

# เรียกใช้งานฟังก์ชันสร้าง Table
init_db()

# ส่วนของ Input Form
with st.form("user_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    submit = st.form_submit_button("Save to Database")

    if submit:
        if name and email:
            save_data(name, email)
            st.success(f"Saved {name} successfully!")
        else:
            st.error("Please fill in both name and email.")

# ส่วนของตารางแสดงผล
st.subheader("Stored Data")
df = load_data()
st.dataframe(df, use_container_width=True)