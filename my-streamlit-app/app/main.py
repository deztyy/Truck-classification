import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from datetime import datetime

# 1. เชื่อมต่อฐานข้อมูล
engine = create_engine('postgresql://user:password@db:5432/mydb')

def init_db():
    with engine.connect() as conn:
        # ตารางอัตราค่าบริการ (Master Data)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS vehicle_classes (
                class_id SERIAL PRIMARY KEY,
                class_name VARCHAR(50) UNIQUE NOT NULL,
                entry_fee NUMERIC(10, 2),
                xray_fee NUMERIC(10, 2),
                total_fee NUMERIC(10, 2)
            );
        """))
        # ตารางบันทึกรายการใช้งาน (Transactions)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS vehicle_transactions (
                trans_id SERIAL PRIMARY KEY,
                license_plate VARCHAR(20) NOT NULL,
                class_id INTEGER REFERENCES vehicle_classes(class_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()

# --- UI Setup ---
st.set_page_config(page_title="Entry System", layout="centered")
init_db()

# ดึงข้อมูล Master เพื่อมาทำ Dropdown
df_classes = pd.read_sql("SELECT * FROM vehicle_classes", engine)

st.title("📝 บันทึกข้อมูลการเข้าสถานี")

# กรณีฐานข้อมูลว่างเปล่า (ให้กดโหลด Master Data ก่อน)
if df_classes.empty:
    st.warning("กรุณาโหลดข้อมูลอัตราค่าบริการก่อนที่ Sidebar")
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
                        VALUES (:n, :e, :x, :t) ON CONFLICT (class_name) DO NOTHING
                    """), {"n": name, "e": entry, "x": xray, "t": entry+xray})
                conn.commit()
            st.rerun()
else:
    # --- ส่วนการ Input ข้อมูล ---
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            plate = st.text_input("เลขทะเบียนรถ", placeholder="กข 1234")
            # เลือกประเภทรถจากรายชื่อใน Master Data
            class_options = {row['class_name']: row['class_id'] for _, row in df_classes.iterrows()}
            selected_class_name = st.selectbox("ประเภทรถ", options=list(class_options.keys()))
        
        with col2:
            # แสดงรูปพรีวิวตามประเภทที่เลือกทันที
            img_path = os.path.join("app", "image", f"{selected_class_name}.png")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info("🖼️ ไม่มีรูปพรีวิว")

        if st.button("💾 บันทึกข้อมูล", use_container_width=True, type="primary"):
            if plate:
                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO vehicle_transactions (license_plate, class_id) 
                        VALUES (:plate, :cid)
                    """), {"plate": plate, "cid": class_options[selected_class_name]})
                    conn.commit()
                st.success("บันทึกรายการสำเร็จ!")
            else:
                st.error("กรุณากรอกเลขทะเบียนรถ")

    st.divider()

    # --- ส่วนการแสดงผลรายการล่าสุด ---
    st.subheader("📋 รายการที่เพิ่งบันทึก")
    query = """
        SELECT t.trans_id, t.license_plate, c.class_name, c.total_fee, t.timestamp 
        FROM vehicle_transactions t
        JOIN vehicle_classes c ON t.class_id = c.class_id
        ORDER BY t.timestamp DESC LIMIT 5
    """
    df_recent = pd.read_sql(query, engine)
    
    if not df_recent.empty:
        for idx, row in df_recent.iterrows():
            with st.expander(f"🚗 {row['license_plate']} - {row['class_name']} ({row['timestamp'].strftime('%H:%M')})"):
                c1, c2 = st.columns([0.3, 0.7])
                with c1:
                    img_path = os.path.join("app", "image", f"{row['class_name']}.jpg")
                    if os.path.exists(img_path):
                        st.image(img_path, width=150)
                with c2:
                    st.write(f"**ประเภท:** {row['class_name']}")
                    st.write(f"**ยอดชำระ:** {row['total_fee']:.2f} บาท")
                    st.write(f"**เวลา:** {row['timestamp']}")
    else:
        st.write("ยังไม่มีรายการบันทึกในวันนี้")