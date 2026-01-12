import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from datetime import datetime

# 1. เชื่อมต่อฐานข้อมูล (ใช้ SQLite สำหรับการทดสอบ)
# เปลี่ยนจาก PostgreSQL เป็น SQLite เพื่อให้รันได้ง่าย
engine = create_engine('sqlite:///vehicle_entry.db')

def init_db():
    with engine.connect() as conn:
        # ตารางอัตราค่าบริการ (Master Data)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS vehicle_classes (
                class_id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name VARCHAR(50) UNIQUE NOT NULL,
                entry_fee NUMERIC(10, 2),
                xray_fee NUMERIC(10, 2),
                total_fee NUMERIC(10, 2)
            );
        """))
        # ตารางบันทึกรายการใช้งาน (Transactions)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS vehicle_transactions (
                trans_id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate VARCHAR(20) NOT NULL,
                class_id INT,
                applied_entry_fee NUMERIC(10, 2),
                applied_xray_fee NUMERIC(10, 2),
                image_path TEXT,
                total_applied_fee NUMERIC(10, 2),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (class_id) REFERENCES vehicle_classes(class_id)
            );
        """))
        conn.commit()

# --- UI Setup ---
st.set_page_config(page_title="Entry System", layout="centered")
init_db()

# ดึงข้อมูล Master เพื่อมาทำ Dropdown
df_classes = pd.read_sql("SELECT * FROM vehicle_classes", engine)

st.title("📝 Entry fee system")

# กรณีฐานข้อมูลว่างเปล่า (ให้กดโหลด Master Data ก่อน)
if df_classes.empty:
    st.warning("Please load data ")
    with st.sidebar:
        if st.button("🔄 Load Master Data"):
            sample_data = [
                ('car', 0, 0), ('other', 0, 0), ('other_truck', 100, 50),
                ('pickup_truck', 0, 0), ('truck_20_back', 100, 250),
                ('truck_20_front', 100, 250), ('truck_20x2', 100, 500),
                ('truck_40', 100, 350), ('truck_roro', 100, 50),
                ('truck_tail', 100, 50), ('motorcycle', 0, 0), ('truck_head', 100, 50)
            ]
            with engine.connect() as conn:
                for name, entry, xray in sample_data:
                    conn.execute(text("""
                        INSERT INTO vehicle_classes (class_name, entry_fee, xray_fee, total_fee) 
                        VALUES (:n, :e, :x, :t)
                    """), {"n": name, "e": entry, "x": xray, "t": entry+xray})
                conn.commit()
            st.rerun()
else:
    # --- ส่วนการ Input ข้อมูล ---
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            plate = st.text_input("CAM-ID", placeholder="CAM 1")
            # เลือกประเภทรถจากรายชื่อใน Master Data
            class_options = {row['class_name']: row['class_id'] for _, row in df_classes.iterrows()}
            selected_class_name = st.selectbox("Vehicle-type", options=list(class_options.keys()))
        
        with col2:
            # แสดงรูปพรีวิวตามประเภทที่เลือกทันที
            img_path = os.path.join("image", f"{selected_class_name}.jpg")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info("🖼️ No-preview-image")

        if st.button("💾 Save-image", use_container_width=True, type="primary"):
            if plate:
                # ดึงค่าธรรมเนียมจาก master data
                selected_class = df_classes[df_classes['class_name'] == selected_class_name].iloc[0]
                
                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO vehicle_transactions 
                        (license_plate, class_id, applied_entry_fee, applied_xray_fee, total_applied_fee, image_path) 
                        VALUES (:plate, :cid, :entry, :xray, :total, :img)
                    """), {
                        "plate": plate, 
                        "cid": class_options[selected_class_name],
                        "entry": selected_class['entry_fee'],
                        "xray": selected_class['xray_fee'],
                        "total": selected_class['total_fee'],
                        "img": img_path
                    })
                    conn.commit()
                st.success("Save successfully!")
                st.rerun()
            else:
                st.error("Please enter camera id")

    st.divider()

    # --- ส่วนการแสดงผลรายการล่าสุด ---
    st.subheader("📋 History ")
    query = """
        SELECT t.trans_id, t.license_plate, c.class_name, c.total_fee, t.timestamp 
        FROM vehicle_transactions t
        JOIN vehicle_classes c ON t.class_id = c.class_id
        ORDER BY t.timestamp DESC LIMIT 5
    """
    df_recent = pd.read_sql(query, engine)
    
    if not df_recent.empty:
        for idx, row in df_recent.iterrows():
            with st.expander(f"🚗 {row['license_plate']} - {row['class_name']} ({pd.to_datetime(row['timestamp']).strftime('%H:%M')})"):
                c1, c2 = st.columns([0.3, 0.7])
                with c1:
                    img_path = os.path.join("image", f"{row['class_name']}.jpg")
                    if os.path.exists(img_path):
                        st.image(img_path, width=150)
                with c2:
                    st.write(f"**Type:** {row['class_name']}")
                    st.write(f"**Entry fee:** {row['total_fee']:.2f} bath")
                    st.write(f"**Time:** {row['timestamp']}")
    else:
        st.write("No record today")