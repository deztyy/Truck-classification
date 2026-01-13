import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from datetime import datetime
import pytz

# 1. เชื่อมต่อฐานข้อมูล PostgreSQL
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5433/mydb')

print(f"🔗 Connecting to database: {DATABASE_URL}")

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Database connected successfully!")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    st.error(f"""
    ❌ ไม่สามารถเชื่อมต่อฐานข้อมูลได้
    
    **ข้อผิดพลาด:** {e}
    
    **วิธีแก้ไข:**
    1. ตรวจสอบว่า PostgreSQL รันอยู่หรือไม่
    2. ตรวจสอบ connection: localhost:5433/mydb
    """)
    st.stop()

def init_db():
    """ตรวจสอบและสร้างตารางถ้ายังไม่มี"""
    try:
        with engine.connect() as conn:
            # ตรวจสอบว่ามีตารางอยู่แล้วหรือไม่
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'vehicle_classes'
                );
            """))
            tables_exist = result.scalar()
            
            if not tables_exist:
                st.info("Creating tables...")
                # สร้างตารางใหม่
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS vehicle_classes (
                        class_id SERIAL PRIMARY KEY,
                        class_name VARCHAR(50) UNIQUE NOT NULL,
                        entry_fee NUMERIC(10, 2),
                        xray_fee NUMERIC(10, 2),
                        total_fee NUMERIC(10, 2)
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS vehicle_transactions (
                        id SERIAL PRIMARY KEY,
                        camera_id VARCHAR(50) NOT NULL,
                        class_id INT,
                        applied_entry_fee NUMERIC(10, 2),
                        applied_xray_fee NUMERIC(10, 2),
                        image_path TEXT,
                        total_applied_fee NUMERIC(10, 2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (class_id) REFERENCES vehicle_classes(class_id)
                    );
                """))
                conn.commit()
    except Exception as e:
        st.error(f"Error checking/creating tables: {e}")

# --- UI Setup ---
st.set_page_config(page_title="Entry System", layout="centered")
init_db()

# ดึงข้อมูล Master
try:
    df_classes = pd.read_sql("SELECT * FROM vehicle_classes ORDER BY class_id", engine)
except Exception as e:
    st.error(f"Error loading vehicle classes: {e}")
    df_classes = pd.DataFrame()

# --- Header พร้อมวันที่และเวลาปัจจุบัน ---
thailand_tz = pytz.timezone('Asia/Bangkok')
now_thailand = datetime.now(thailand_tz)

# Title with modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .header-title {
        color: white;
        font-size: 2em;
        font-weight: 700;
        margin: 0;
    }
    .datetime-box {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    .date-text {
        color: white;
        font-size: 1.1em;
        font-weight: 600;
    }
    .time-text {
        color: #ffd700;
        font-size: 1.5em;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

col_header1, col_header2 = st.columns([0.6, 0.4])
with col_header1:
    st.markdown('<div class="main-header"><p class="header-title">🚗 Vehicle Entry System</p></div>', unsafe_allow_html=True)
with col_header2:
    # ใช้ HTML component สำหรับ real-time clock
    import streamlit.components.v1 as components
    
    clock_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: sans-serif;
            }}
            .main-header {{
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 10px;
            }}
            .datetime-box {{
                background: rgba(255,255,255,0.2);
                padding: 0.5rem 1rem;
                border-radius: 8px;
            }}
            .date-text {{
                color: white;
                font-size: 1.1em;
                font-weight: 600;
                margin-bottom: 5px;
            }}
            .time-text {{
                color: #ffd700;
                font-size: 1.5em;
                font-weight: 700;
                font-family: 'Courier New', monospace;
            }}
        </style>
    </head>
    <body>
        <div class="main-header">
            <div class="datetime-box">
                <div class="date-text">📅 {now_thailand.strftime('%d %B %Y')}</div>
                <div class="time-text" id="live-clock">🕐 Loading...</div>
            </div>
        </div>
        
        <script>
            function updateClock() {{
                const now = new Date();
                
                // แปลงเป็นเวลาไทย (UTC+7)
                const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                const thailandTime = new Date(utc + (3600000 * 7));
                
                const hours = String(thailandTime.getHours()).padStart(2, '0');
                const minutes = String(thailandTime.getMinutes()).padStart(2, '0');
                const seconds = String(thailandTime.getSeconds()).padStart(2, '0');
                
                const timeString = '🕐 ' + hours + ':' + minutes + ':' + seconds;
                
                const clockElement = document.getElementById('live-clock');
                if (clockElement) {{
                    clockElement.textContent = timeString;
                }}
            }}
            
            // อัพเดททุก 1 วินาที
            setInterval(updateClock, 1000);
            
            // เรียกใช้ทันที
            updateClock();
        </script>
    </body>
    </html>
    """
    
    components.html(clock_html, height=100)

st.markdown("---")

# --- Sidebar: SQL Command Input ---
if 'sql_command' not in st.session_state:
    st.session_state.sql_command = ""
with st.sidebar:
    st.subheader("🔧 Manual SQL Command")
    sql_command = st.text_area(
        "Enter SQL Command",
        placeholder="INSERT INTO vehicle_transactions ...\nor\nUPDATE vehicle_classes ...",
        height=150,
        key="sql_input"
    )
    
    if st.button("▶️ Execute SQL", type="secondary"):
        if sql_command.strip():
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql_command))
                    conn.commit()
                    st.success(f"✅ Command executed successfully!")
                    if result.rowcount > 0:
                        st.info(f"Affected rows: {result.rowcount}")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter SQL command")
    
    st.divider()
    
    with st.expander("📝 Example Commands"):
        st.code("""
-- Insert transaction (ไม่ต้องใส่ total_applied_fee)
INSERT INTO vehicle_transactions 
(camera_id, class_id, applied_entry_fee, 
 applied_xray_fee, image_path) 
VALUES ('CAM-1', 1, 0, 0, 'image/car.jpg');

-- Update vehicle class fee
UPDATE vehicle_classes 
SET entry_fee = 150, total_fee = 200
WHERE class_name = 'truck_40';

-- Delete transaction
DELETE FROM vehicle_transactions 
WHERE id = 1;

-- View all transactions today
SELECT * FROM vehicle_transactions 
WHERE DATE(created_at) = CURRENT_DATE
ORDER BY created_at DESC;

-- View all vehicle classes
SELECT * FROM vehicle_classes;
        """, language="sql")
    
    st.divider()
    
    with st.expander("🔗 Database Info"):
        st.code(f"Connection: {DATABASE_URL}")
        st.write(f"Total vehicle classes: {len(df_classes)}")

# กรณีฐานข้อมูลว่างเปล่า
if df_classes.empty:
    st.warning("⚠️ ไม่พบข้อมูลประเภทรถ กรุณาโหลด Master Data")
    
    if st.button("🔄 Load Master Data", type="primary", use_container_width=True):
        sample_data = [
            ('car', 0, 0), ('other', 0, 0), ('other_truck', 100, 50),
            ('pickup_truck', 0, 0), ('truck_20_back', 100, 250),
            ('truck_20_front', 100, 250), ('truck_20x2', 100, 500),
            ('truck_40', 100, 350), ('truck_roro', 100, 50),
            ('truck_tail', 100, 50), ('motorcycle', 0, 0), ('truck_head', 100, 50)
        ]
        try:
            with engine.connect() as conn:
                for name, entry, xray in sample_data:
                    conn.execute(text("""
                        INSERT INTO vehicle_classes (class_name, entry_fee, xray_fee, total_fee) 
                        VALUES (:n, :e, :x, :t)
                        ON CONFLICT (class_name) DO NOTHING
                    """), {"n": name, "e": entry, "x": xray, "t": entry+xray})
                conn.commit()
            st.success("✅ Master Data loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading master data: {e}")
else:
    # ดึงรายการ Camera ID ที่มีอยู่ใน database
    try:
        camera_list_query = """
            SELECT DISTINCT camera_id 
            FROM vehicle_transactions 
        """
        df_cameras = pd.read_sql(camera_list_query, engine)
        
        if not df_cameras.empty:
            # แปลงเป็น string และหาค่าที่ไม่ซ้ำ
            camera_list_raw = [str(cam).strip() for cam in df_cameras['camera_id'].unique()]
            
            # แยกเป็น 2 กลุ่ม: ตัวเลขล้วน และมีตัวอักษร
            numeric_cameras = []
            text_cameras = []
            
            for cam in camera_list_raw:
                if cam:  # ตรวจสอบว่าไม่ใช่ค่าว่าง
                    try:
                        # ลองแปลงเป็นตัวเลข
                        num = int(cam)
                        numeric_cameras.append(num)
                    except:
                        # ถ้าแปลงไม่ได้คือมีตัวอักษร
                        text_cameras.append(str(cam))
            
            # ลบค่าซ้ำ และเรียงตัวเลข
            numeric_cameras = sorted(list(set(numeric_cameras)))
            sorted_numeric = [str(x) for x in numeric_cameras]
            
            # เรียงที่มีตัวอักษร และไม่ซ้ำกัน
            text_cameras = sorted(list(set(text_cameras)))
            
            # รวมกัน
            camera_list = sorted_numeric + text_cameras
            
            # ใช้แค่ 1, 2, 3 เท่านั้น (ไม่เอาจาก database)
            camera_list = ['1', '2', '3']
        else:
            # ถ้ายังไม่มีกล้องใน database ให้ใช้ตัวเลือกเริ่มต้น
            camera_list = ['1', '2', '3']
    except Exception as e:
        st.error(f"Error loading cameras: {e}")
        camera_list = []
    
    # --- ส่วนการ Input ข้อมูล ---
    st.markdown("### 📝 Record New Entry")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            # เลือกกล้อง
            st.markdown("**📷 Select Camera**")
            
            col_cam1, col_cam2 = st.columns([3, 1])
            with col_cam1:
                if camera_list:
                    # มีกล้องในระบบแล้ว - แสดง dropdown + option เพิ่มใหม่
                    camera_options = camera_list + ["➕ Add New"]
                    
                    camera_selection = st.selectbox(
                        "Camera ID",
                        options=camera_options,
                        help="เลือกกล้องจากรายการหรือเพิ่มใหม่",
                        label_visibility="collapsed"
                    )
                    
                    # ถ้าเลือก "Add New" ให้กรอกชื่อใหม่
                    if camera_selection == "➕ Add New":
                        camera_id = st.text_input(
                            "Enter New Camera ID",
                            placeholder="Enter New Camera Id",
                            help="กรอกหมายเลขกล้องใหม่"
                        )
                    else:
                        camera_id = camera_selection
                else:
                    # ยังไม่มีกล้องในระบบ - ให้กรอกใหม่
                    st.info("ยังไม่มีกล้องในระบบ กรุณาเพิ่มกล้องแรก")
                    camera_id = st.text_input(
                        "Camera ID",
                        placeholder="1",
                        help="กรอกหมายเลขกล้อง",
                        label_visibility="collapsed"
                    )
            
            with col_cam2:
                # แสดงจำนวนกล้องทั้งหมด
                st.metric("📹", len(camera_list), help="Total Cameras")
            
            st.markdown("---")
            
            # เลือกประเภทรถ
            class_options = {row['class_name']: row['class_id'] for _, row in df_classes.iterrows()}
            selected_class_name = st.selectbox(
                "🚗 Vehicle Type", 
                options=list(class_options.keys()),
                key="vehicle_select"
            )
            
            # แสดงค่าธรรมเนียมแบบสวยงาม
            selected_class = df_classes[df_classes['class_name'] == selected_class_name].iloc[0]
            
            col_fee1, col_fee2, col_fee3 = st.columns(3)
            with col_fee1:
                st.metric("Entry Fee", f"{selected_class['entry_fee']:.0f} ฿")
            with col_fee2:
                st.metric("X-Ray Fee", f"{selected_class['xray_fee']:.0f} ฿")
            with col_fee3:
                st.metric("Total", f"{selected_class['total_fee']:.0f} ฿", delta="Final")
        
        with col2:
            # แสดงรูปพรีวิว
            img_path = os.path.join("image", f"{selected_class_name}.jpg")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True, caption=f"Preview: {selected_class_name}")
            else:
                st.info("🖼️ No preview image available")
            
            # แสดงสถิติกล้อง
            if camera_id and camera_id != "➕ Add New" and camera_id.strip():
                try:
                    camera_stats_query = text("""
                        SELECT 
                            COUNT(*) as total_count,
                            SUM(total_applied_fee) as total_fees
                        FROM vehicle_transactions
                        WHERE camera_id = :cam_id
                    """)
                    camera_stats = pd.read_sql(camera_stats_query, engine, params={"cam_id": camera_id})
                    
                    if not camera_stats.empty and camera_stats.iloc[0]['total_count'] > 0:
                        st.markdown("---")
                        st.markdown(f"**📊 Camera {camera_id} Statistics**")
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("Records", int(camera_stats.iloc[0]['total_count']))
                        with col_stat2:
                            total = camera_stats.iloc[0]['total_fees'] or 0
                            st.metric("Total", f"{total:.0f} ฿")
                except:
                    pass

        # ปุ่ม Save
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("💾 Save Transaction", use_container_width=True, type="primary"):
                if camera_id and camera_id.strip() and camera_id != "➕ Add New":
                    try:
                        selected_class = df_classes[df_classes['class_name'] == selected_class_name].iloc[0]
                        
                        with engine.connect() as conn:
                            conn.execute(text("""
                                INSERT INTO vehicle_transactions 
                                (camera_id, class_id, applied_entry_fee, applied_xray_fee, image_path) 
                                VALUES (:cam_id, :cid, :entry, :xray, :img)
                            """), {
                                "cam_id": camera_id.strip(),
                                "cid": int(class_options[selected_class_name]),
                                "entry": float(selected_class['entry_fee']),
                                "xray": float(selected_class['xray_fee']),
                                "img": img_path
                            })
                            conn.commit()
                        st.success(f"✅ Saved successfully! Camera: {camera_id}")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving data: {e}")
                else:
                    st.error("⚠️ Please enter Camera ID")

    st.divider()

    # --- ส่วนการแสดงผล History ---
    st.markdown("### 📋 Transaction History")
    
    # Filter & Sort Section
    with st.container(border=True):
        st.markdown("#### 🔍 Filter & Sort Options")
        
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            # Date Filter
            date_filter = st.date_input(
                "📅 Select Date",
                value=now_thailand.date(),
                help="เลือกวันที่ต้องการดูข้อมูล"
            )
        
        with col_filter2:
            # Vehicle Type Filter
            vehicle_types = ['All'] + list(df_classes['class_name'].values)
            selected_vehicle = st.selectbox(
                "🚗 Vehicle Type",
                options=vehicle_types,
                help="เลือกประเภทรถที่ต้องการดู"
            )
        
        with col_filter3:
            # Sort Options
            sort_options = {
                'Newest First': 'DESC',
                'Oldest First': 'ASC',
                'Highest Fee': 'fee_desc',
                'Lowest Fee': 'fee_asc'
            }
            selected_sort = st.selectbox(
                "⬇️ Sort By",
                options=list(sort_options.keys()),
                help="เรียงลำดับข้อมูล"
            )
    
    # Query ข้อมูลตาม Filter
    try:
        # สร้าง base query
        query = """
            SELECT t.id, t.camera_id, c.class_name, c.total_fee, t.created_at,
                   t.applied_entry_fee, t.applied_xray_fee
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.created_at) = :date_filter
        """
        
        # เพิ่ม vehicle type filter
        if selected_vehicle != 'All':
            query += " AND c.class_name = :vehicle_type"
        
        # เพิ่ม sort order
        if sort_options[selected_sort] == 'DESC':
            query += " ORDER BY t.created_at DESC"
        elif sort_options[selected_sort] == 'ASC':
            query += " ORDER BY t.created_at ASC"
        elif sort_options[selected_sort] == 'fee_desc':
            query += " ORDER BY c.total_fee DESC, t.created_at DESC"
        else:  # fee_asc
            query += " ORDER BY c.total_fee ASC, t.created_at DESC"
        
        # Execute query
        params = {"date_filter": date_filter}
        if selected_vehicle != 'All':
            params["vehicle_type"] = selected_vehicle
            
        df_recent = pd.read_sql(text(query), engine, params=params)
    except Exception as e:
        st.error(f"Error loading history: {e}")
        df_recent = pd.DataFrame()
    
    st.markdown("---")
    
    # แสดง Summary Dashboard
    if not df_recent.empty:
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            st.metric("🚗 Total Entries", len(df_recent))
        with col_sum2:
            total_fees = df_recent['total_fee'].sum()
            st.metric("💰 Total Fees", f"{total_fees:.0f} ฿")
        with col_sum3:
            avg_fee = df_recent['total_fee'].mean()
            st.metric("📊 Avg Fee", f"{avg_fee:.0f} ฿")
        with col_sum4:
            last_time = pd.to_datetime(df_recent.iloc[0]['created_at']).strftime('%H:%M')
            st.metric("🕐 Latest", last_time)
        
        # แสดง breakdown by vehicle type
        with st.expander("📊 Breakdown by Vehicle Type", expanded=False):
            vehicle_summary = df_recent.groupby('class_name').agg({
                'id': 'count',
                'total_fee': 'sum'
            }).reset_index()
            vehicle_summary.columns = ['Vehicle Type', 'Count', 'Total Fee']
            vehicle_summary = vehicle_summary.sort_values('Count', ascending=False)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.dataframe(
                    vehicle_summary,
                    hide_index=True,
                    use_container_width=True
                )
            with col_chart2:
                # Simple bar chart
                st.bar_chart(
                    data=vehicle_summary.set_index('Vehicle Type')['Count'],
                    use_container_width=True
                )
    
    st.markdown("---")
    
    # สร้าง scrollable container สำหรับรายการ
    if not df_recent.empty:
        st.markdown(f"**Showing {len(df_recent)} transactions**")
        
        with st.container(height=400, border=True):
            for idx, row in df_recent.iterrows():
                timestamp_str = pd.to_datetime(row['created_at']).strftime('%H:%M:%S')
                
                # สร้าง card สวยๆ สำหรับแต่ละ transaction
                with st.expander(
                    f"📷 {row['camera_id']} | {row['class_name']} | {timestamp_str} | {row['total_fee']:.0f} ฿",
                    expanded=False
                ):
                    c1, c2 = st.columns([0.35, 0.65])
                    with c1:
                        img_path = os.path.join("image", f"{row['class_name']}.jpg")
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                    with c2:
                        st.markdown(f"**🆔 Transaction ID:** #{row['id']}")
                        st.markdown(f"**📷 Camera:** {row['camera_id']}")
                        st.markdown(f"**🚗 Vehicle:** {row['class_name']}")
                        st.markdown("---")
                        
                        col_fee1, col_fee2, col_fee3 = st.columns(3)
                        with col_fee1:
                            st.metric("Entry", f"{row['applied_entry_fee']:.0f} ฿")
                        with col_fee2:
                            st.metric("X-Ray", f"{row['applied_xray_fee']:.0f} ฿")
                        with col_fee3:
                            st.metric("Total", f"{row['total_fee']:.0f} ฿")
                        
                        st.markdown(f"**🕐 Time:** {row['created_at']}")
                    
                    # เพิ่มปุ่มลบ
                    col_del1, col_del2, col_del3 = st.columns([2, 1, 2])
                    with col_del2:
                        if st.button(f"🗑️ Delete", key=f"del_{row['id']}", type="secondary", use_container_width=True):
                            try:
                                with engine.connect() as conn:
                                    conn.execute(text("DELETE FROM vehicle_transactions WHERE id = :id"), {"id": row['id']})
                                    conn.commit()
                                st.success("Deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
    else:
        st.info(f"📭 No transactions found for {date_filter.strftime('%d %B %Y')}")