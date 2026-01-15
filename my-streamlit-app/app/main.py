import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from datetime import datetime
import pytz
import streamlit.components.v1 as components

# ==================== CONFIGURATION ====================
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'Admin1234')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://vehicleuser:V3h1cl3_P@ssw0rd_2024!@db:5432/vehicle_entry_db')
THAILAND_TZ = pytz.timezone('Asia/Bangkok')

# ==================== CUSTOM CSS ====================
def load_custom_css():
    st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background-color: #0e1117;
        }
        
        /* Header Styles */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        }
        
        .header-title {
            color: white;
            font-size: 2.5em;
            font-weight: 800;
            margin: 0;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .datetime-box {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            padding: 1rem;
            border-radius: 12px;
            margin-top: 1rem;
            text-align: center;
        }
        
        .date-text {
            color: white;
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .time-text {
            color: #ffd700;
            font-size: 2em;
            font-weight: 700;
            font-family: 'Courier New', monospace;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Card Styles */
        div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Metric Styles */
        div[data-testid="stMetricValue"] {
            font-size: 1.5em;
        }
        
        /* Button Styles */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        /* Form Container */
        .form-container {
            background: rgba(255,255,255,0.03);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== DATABASE CONNECTION ====================
@st.cache_resource
def get_database_engine():
    """Create and return database engine"""
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connected successfully!")
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        st.error(f"""
        ❌ ไม่สามารถเชื่อมต่อฐานข้อมูลได้
        
        **ข้อผิดพลาด:** {e}
        
        **วิธีแก้ไข:**
        1. ตรวจสอบว่า PostgreSQL รันอยู่หรือไม่
        2. ตรวจสอบ DATABASE_URL ใน .env
        """)
        st.stop()

engine = get_database_engine()

# ==================== DATABASE INITIALIZATION ====================
def init_database():
    """Initialize database tables if not exist"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SET TIME ZONE 'Asia/Bangkok'"))
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'vehicle_classes'
                );
            """))
            
            if not result.scalar():
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
                        total_applied_fee NUMERIC(10, 2),
                        image_path TEXT,
                        created_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Bangkok'),
                        FOREIGN KEY (class_id) REFERENCES vehicle_classes(class_id)
                    );
                """))
                conn.commit()
    except Exception as e:
        st.error(f"Error initializing database: {e}")

# ==================== AUTHENTICATION ====================
def check_authentication():
    """Check and initialize authentication state"""
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'show_password_input' not in st.session_state:
        st.session_state.show_password_input = False

def switch_mode():
    """Switch between user modes"""
    st.session_state.user_role = None
    st.session_state.show_password_input = False
    st.rerun()

def render_login_page():
    """Render login page"""
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
        }
        .login-header {
            text-align: center;
            padding: 2rem 0;
        }
        .car-icon {
            font-size: 5em;
            margin-bottom: 1rem;
        }
        .login-title {
            color: white;
            font-size: 2.5em;
            font-weight: 800;
            margin: 1rem 0;
        }
        .login-subtitle {
            color: #a0a0a0;
            font-size: 1.3em;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-header">', unsafe_allow_html=True)
        st.markdown('<div class="car-icon">🚗</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Vehicle Entry System</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">เลือกโหมดการใช้งาน</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("👤 User Mode", use_container_width=True, type="primary", key="user_btn"):
            st.session_state.user_role = "user"
            st.success("✅ เข้าสู่โหมด User")
            st.rerun()
        
        st.markdown("")
        
        if st.button("👑 Admin Mode", use_container_width=True, type="secondary", key="admin_btn"):
            st.session_state.show_password_input = True
        
        if st.session_state.show_password_input:
            st.markdown("---")
            password = st.text_input("🔒 รหัส Admin", type="password", placeholder="ใส่รหัส Admin", key="password_input")
            
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("✅ ยืนยัน", use_container_width=True, type="primary", key="confirm_btn"):
                    if password == ADMIN_PASSWORD:
                        st.session_state.user_role = "admin"
                        st.session_state.show_password_input = False
                        st.success("✅ เข้าสู่โหมด Admin")
                        st.rerun()
                    else:
                        st.error("❌ รหัสผ่านไม่ถูกต้อง")
            
            with col_cancel:
                if st.button("❌ ยกเลิก", use_container_width=True, key="cancel_btn"):
                    st.session_state.show_password_input = False
                    st.rerun()

# ==================== HEADER COMPONENT ====================
def render_header():
    """Render header with live clock"""
    load_custom_css()
    now_thailand = datetime.now(THAILAND_TZ)
    
    col_header1, col_header2 = st.columns([1, 1])
    
    with col_header1:
        st.markdown('<div class="main-header"><p class="header-title">🚗 Vehicle Entry System</p></div>', 
                   unsafe_allow_html=True)
        
        col_role, col_switch = st.columns([0.7, 0.3])
        with col_role:
            role_emoji = "👑" if st.session_state.user_role == "admin" else "👤"
            role_text = "Admin Mode" if st.session_state.user_role == "admin" else "User Mode"
            st.markdown(f"### {role_emoji} {role_text}")
        with col_switch:
            if st.button("🔄 Switch", type="secondary", use_container_width=True):
                switch_mode()
    
    with col_header2:
        clock_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                }}
                .main-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
                }}
                .datetime-box {{
                    background: rgba(255,255,255,0.15);
                    backdrop-filter: blur(10px);
                    padding: 1rem;
                    border-radius: 12px;
                    text-align: center;
                }}
                .date-text {{
                    color: white;
                    font-size: 1.2em;
                    font-weight: 600;
                    margin-bottom: 8px;
                }}
                .time-text {{
                    color: #ffd700;
                    font-size: 2em;
                    font-weight: 700;
                    font-family: 'Courier New', monospace;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
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
                    const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                    const thailandTime = new Date(utc + (3600000 * 7));
                    
                    const hours = String(thailandTime.getHours()).padStart(2, '0');
                    const minutes = String(thailandTime.getMinutes()).padStart(2, '0');
                    const seconds = String(thailandTime.getSeconds()).padStart(2, '0');
                    
                    document.getElementById('live-clock').textContent = '🕐 ' + hours + ':' + minutes + ':' + seconds;
                }}
                
                setInterval(updateClock, 1000);
                updateClock();
            </script>
        </body>
        </html>
        """
        components.html(clock_html, height=140)

# ==================== DATA LOADING ====================
@st.cache_data(ttl=10)
def load_vehicle_classes():
    """Load vehicle classes from database"""
    try:
        return pd.read_sql("SELECT * FROM vehicle_classes ORDER BY class_id", engine)
    except Exception as e:
        st.error(f"Error loading vehicle classes: {e}")
        return pd.DataFrame()

# ==================== VEHICLE ENTRY TAB (ADMIN) ====================
def render_entry_tab(df_classes):
    """Render vehicle entry tab"""
    st.markdown("### 🚗 New Vehicle Entry")
    
    with st.container(border=True):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            camera_options = [str(i) for i in range(1, 4)] + ["➕ Add New"]
            camera_selection = st.selectbox("📷 Camera ID", camera_options, key="camera_select")
            
            if camera_selection == "➕ Add New":
                camera_id = st.text_input("🆕 New Camera ID", placeholder="e.g., CAM001", key="new_camera")
            else:
                camera_id = camera_selection
        
        with col2:
            if not df_classes.empty:
                class_options = {row['class_name']: row['class_id'] for _, row in df_classes.iterrows()}
                selected_class_name = st.selectbox("🚙 Vehicle Type", list(class_options.keys()), key="vehicle_select")
            else:
                st.warning("⚠️ No vehicle classes available")
                selected_class_name = None
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if selected_class_name and not df_classes.empty:
            selected_class = df_classes[df_classes['class_name'] == selected_class_name].iloc[0]
            
            st.markdown("---")
            col_fee1, col_fee2, col_fee3 = st.columns(3)
            
            with col_fee1:
                st.metric("💵 Entry Fee", f"{selected_class['entry_fee']:.0f} ฿")
            with col_fee2:
                st.metric("🔍 X-Ray Fee", f"{selected_class['xray_fee']:.0f} ฿")
            with col_fee3:
                st.metric("💰 Total Fee", f"{selected_class['total_fee']:.0f} ฿")

        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("💾 Save Transaction", use_container_width=True, type="primary"):
                if camera_id and camera_id.strip() and camera_id != "➕ Add New":
                    if selected_class_name:
                        try:
                            selected_class = df_classes[df_classes['class_name'] == selected_class_name].iloc[0]
                            current_time_thailand = datetime.now(THAILAND_TZ)
                            
                            with engine.connect() as conn:
                                conn.execute(text("""
                                    INSERT INTO vehicle_transactions 
                                    (camera_id, class_id, applied_entry_fee, applied_xray_fee, total_applied_fee, created_at) 
                                    VALUES (:cam_id, :cid, :entry, :xray, :total, :created_at)
                                """), {
                                    "cam_id": camera_id.strip(),
                                    "cid": int(class_options[selected_class_name]),
                                    "entry": float(selected_class['entry_fee']),
                                    "xray": float(selected_class['xray_fee']),
                                    "total": float(selected_class['entry_fee']) + float(selected_class['xray_fee']),
                                    "created_at": current_time_thailand
                                })
                                conn.commit()
                            
                            st.success(f"✅ Saved successfully! Camera: {camera_id}")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.error("⚠️ Please select vehicle type")
                else:
                    st.error("⚠️ Please enter Camera ID")

# ==================== CURRENT VEHICLE TAB (USER MODE) ====================
def render_current_vehicle_tab(df_classes):
    """Render current vehicle display tab"""
    st.markdown("### 🚗 Current Vehicle")
    
    st.markdown("""
    <style>
        .vehicle-main-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 1.5rem 0;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        }
        .vehicle-title {
            text-align: center;
            color: white;
            font-size: 2em;
            font-weight: 800;
        }
        .vehicle-info-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .info-label {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        .info-value {
            color: #ffd700;
            font-size: 2em;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            word-break: break-word;
        }
        .countdown-badge {
            background: rgba(255, 255, 255, 0.15);
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 50px;
            font-size: 0.9em;
            font-weight: 600;
            display: inline-block;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        query = """
            SELECT 
                t.camera_id,
                c.class_name as vehicle_type,
                t.total_applied_fee,
                t.created_at
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            ORDER BY t.created_at DESC
            LIMIT 1
        """
        
        df_latest = pd.read_sql(text(query), engine)
        
        if not df_latest.empty:
            vehicle = df_latest.iloc[0]
            timestamp = pd.to_datetime(vehicle['created_at'])
            formatted_time = timestamp.strftime('%d/%m/%Y %H:%M:%S')
            
            st.markdown("""
            <div class="vehicle-main-card">
                <div class="vehicle-title">🚗 Latest Vehicle Entry</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown(f"""
                <div class="vehicle-info-box">
                    <div class="info-label">📷 Camera ID</div>
                    <div class="info-value">{vehicle['camera_id']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="vehicle-info-box">
                    <div class="info-label">🚙 Vehicle Type</div>
                    <div class="info-value">{vehicle['vehicle_type']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_fee = vehicle['total_applied_fee'] if vehicle['total_applied_fee'] is not None else 0.0
                
                st.markdown(f"""
                <div class="vehicle-info-box">
                    <div class="info-label">💰 Total Fee</div>
                    <div class="info-value">{total_fee:.2f} ฿</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="vehicle-info-box">
                    <div class="info-label">⏰ Timestamp</div>
                    <div class="info-value" style="font-size: 1.4em;">{formatted_time}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Auto-refresh countdown with manual refresh button
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem;">
                <span class="countdown-badge">🔄 Auto-refresh in <span id="countdown">10</span> seconds</span>
            </div>
            """, unsafe_allow_html=True)
            
            col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
            with col_r2:
                if st.button("🔄 Refresh Now", use_container_width=True, type="primary", key="manual_refresh"):
                    st.rerun()
            
        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: rgba(102, 126, 234, 0.05); 
                 border-radius: 20px; border: 2px dashed rgba(102, 126, 234, 0.3);">
                <div style="font-size: 5em; opacity: 0.5;">🚗💨</div>
                <div style="color: #667eea; font-size: 1.8em; font-weight: 700;">No Vehicles Yet</div>
                <div style="color: #999; font-size: 1.1em;">Waiting for the first vehicle to enter...</div>
                <div style="margin-top: 2rem;">
                    <span class="countdown-badge">🔄 Checking in <span id="countdown">10</span> seconds</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error loading current vehicle: {e}")
    
    # JavaScript auto-refresh timer
    auto_refresh_script = """
    <script>
        // Auto-refresh countdown
        let timeLeft = 10;
        const countdownElement = document.getElementById('countdown');
        
        const countdown = setInterval(function() {
            timeLeft--;
            if (countdownElement) {
                countdownElement.textContent = timeLeft;
            }
            
            if (timeLeft <= 0) {
                clearInterval(countdown);
                // Trigger Streamlit rerun by dispatching a custom event
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    key: 'auto_refresh_trigger',
                    value: Date.now()
                }, '*');
                
                // Fallback: reload the page if Streamlit doesn't respond
                setTimeout(function() {
                    window.location.reload();
                }, 500);
            }
        }, 1000);
    </script>
    """
    
    components.html(auto_refresh_script, height=0)


# ==================== TRANSACTION HISTORY ====================
def render_transaction_history(df_classes):
    """Render transaction history section"""
    st.markdown("---")
    st.markdown("### 📋 Transaction History")
    
    now_thailand = datetime.now(THAILAND_TZ)
    
    with st.container(border=True):
        st.markdown("#### 🔍 Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_filter = st.date_input("📅 Date", value=now_thailand.date())
        
        with col2:
            vehicle_types = ['All'] + list(df_classes['class_name'].values) if not df_classes.empty else ['All']
            selected_vehicle = st.selectbox("🚗 Vehicle Type", options=vehicle_types)
    
    try:
        query = """
            SELECT t.id, t.camera_id, c.class_name, t.total_applied_fee, t.created_at,
                   t.applied_entry_fee, t.applied_xray_fee
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.created_at) = :date_filter
        """
        
        if selected_vehicle != 'All':
            query += " AND c.class_name = :vehicle_type"
        
        query += " ORDER BY t.created_at DESC"
        
        params = {"date_filter": date_filter}
        if selected_vehicle != 'All':
            params["vehicle_type"] = selected_vehicle
        
        df_recent = pd.read_sql(text(query), engine, params=params)
    except Exception as e:
        st.error(f"Error: {e}")
        df_recent = pd.DataFrame()
    
    if not df_recent.empty:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚗 Total Entries", len(df_recent))
        with col2:
            st.metric("💰 Total Revenue", f"{df_recent['total_applied_fee'].sum():.0f} ฿")
        with col3:
            st.metric("📊 Average Fee", f"{df_recent['total_applied_fee'].mean():.0f} ฿")
    
    st.markdown("---")
    
    if not df_recent.empty:
        st.markdown(f"**Showing {len(df_recent)} transactions**")
        
        for idx, row in df_recent.iterrows():
            timestamp = pd.to_datetime(row['created_at']).strftime('%H:%M:%S')
            
            with st.expander(
                f"📷 {row['camera_id']} | {row['class_name']} | {timestamp} | {row['total_applied_fee']:.0f} ฿",
                expanded=False
            ):
                st.markdown(f"**🆔 ID:** #{row['id']}")
                st.markdown(f"**📷 Camera:** {row['camera_id']}")
                st.markdown(f"**🚗 Vehicle:** {row['class_name']}")
                st.markdown("---")
                
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    st.metric("Entry", f"{row['applied_entry_fee']:.0f} ฿")
                with col_f2:
                    st.metric("X-Ray", f"{row['applied_xray_fee']:.0f} ฿")
                with col_f3:
                    st.metric("Total", f"{row['total_applied_fee']:.0f} ฿")
                
                st.markdown(f"**🕐 Time:** {row['created_at']}")
    else:
        st.info(f"📭 No transactions found for {date_filter.strftime('%d %B %Y')}")

# ==================== MASTER DATA TAB ====================
def render_master_data_tab(df_classes):
    """Render master data management tab"""
    st.markdown("### ⚙️ Vehicle Classes Management")
    
    if not df_classes.empty:
        st.dataframe(
            df_classes,
            use_container_width=True,
            hide_index=True,
            column_config={
                "class_id": "ID",
                "class_name": "Vehicle Type",
                "entry_fee": st.column_config.NumberColumn("Entry Fee (฿)", format="%.2f"),
                "xray_fee": st.column_config.NumberColumn("X-Ray Fee (฿)", format="%.2f"),
                "total_fee": st.column_config.NumberColumn("Total Fee (฿)", format="%.2f")
            }
        )
    else:
        st.info("📭 No vehicle classes defined yet")
    
    st.markdown("---")
    st.markdown("#### ➕ Add/Edit Vehicle Class")
    
    with st.form("class_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            class_name = st.text_input("🚗 Vehicle Type", placeholder="e.g., Sedan, Truck")
            entry_fee = st.number_input("💵 Entry Fee (฿)", min_value=0.0, step=10.0, value=0.0)
        
        with col2:
            xray_fee = st.number_input("🔍 X-Ray Fee (฿)", min_value=0.0, step=10.0, value=0.0)
            total_fee = entry_fee + xray_fee
            st.metric("💰 Total Fee", f"{total_fee:.2f} ฿")
        
        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        with col_s2:
            submitted = st.form_submit_button("💾 Save Class", use_container_width=True, type="primary")
        
        if submitted and class_name.strip():
            try:
                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO vehicle_classes (class_name, entry_fee, xray_fee, total_fee)
                        VALUES (:name, :entry, :xray, :total)
                        ON CONFLICT (class_name) DO UPDATE 
                        SET entry_fee = :entry, xray_fee = :xray, total_fee = :total
                    """), {
                        "name": class_name.strip(),
                        "entry": entry_fee,
                        "xray": xray_fee,
                        "total": total_fee
                    })
                    conn.commit()
                st.success(f"✅ Saved: {class_name}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    if not df_classes.empty:
        st.markdown("---")
        st.markdown("#### 🗑️ Delete Vehicle Class")
        
        col_del1, col_del2 = st.columns([2, 1])
        with col_del1:
            class_to_delete = st.selectbox("Select class to delete", 
                                          options=df_classes['class_name'].tolist())
        with col_del2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                try:
                    with engine.connect() as conn:
                        conn.execute(text("DELETE FROM vehicle_classes WHERE class_name = :name"), 
                                   {"name": class_to_delete})
                        conn.commit()
                    st.success(f"✅ Deleted: {class_to_delete}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ==================== ANALYTICS TAB ====================
def render_analytics_tab():
    """Render analytics dashboard"""
    st.markdown("### 📊 Analytics Dashboard")
    
    now_thailand = datetime.now(THAILAND_TZ)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Start Date", value=now_thailand.date())
    with col2:
        end_date = st.date_input("📅 End Date", value=now_thailand.date())
    
    try:
        query = """
            SELECT t.*, c.class_name, t.total_applied_fee
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.created_at) BETWEEN :start_date AND :end_date
            ORDER BY t.created_at DESC
        """
        
        df_analytics = pd.read_sql(text(query), engine, 
                                   params={"start_date": start_date, "end_date": end_date})
        
        if not df_analytics.empty:
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Transactions", len(df_analytics))
            with col2:
                st.metric("💰 Total Revenue", f"{df_analytics['total_applied_fee'].sum():.0f} ฿")
            with col3:
                st.metric("📈 Avg/Transaction", f"{df_analytics['total_applied_fee'].mean():.0f} ฿")
            with col4:
                st.metric("📷 Active Cameras", df_analytics['camera_id'].nunique())
            
            st.markdown("---")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("#### 🚗 Transactions by Vehicle Type")
                vehicle_counts = df_analytics['class_name'].value_counts()
                st.bar_chart(vehicle_counts)
            
            with col_c2:
                st.markdown("#### 💰 Revenue by Vehicle Type")
                revenue_by_type = df_analytics.groupby('class_name')['total_applied_fee'].sum().sort_values(ascending=False)
                st.bar_chart(revenue_by_type)
        else:
            st.info(f"📭 No data found between {start_date} and {end_date}")
    
    except Exception as e:
        st.error(f"Error: {e}")

# ==================== MAIN APPLICATION ====================
def main():
    """Main application"""
    st.set_page_config(
        page_title="Vehicle Entry System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    check_authentication()
    init_database()
    
    if st.session_state.user_role is None:
        render_login_page()
        st.stop()
    
    df_classes = load_vehicle_classes()
    
    render_header()
    
    st.markdown("---")
    
    if st.session_state.user_role == "admin":
        tab1, tab2, tab3 = st.tabs(["📝 Entry", "⚙️ Master Data", "📊 Analytics"])
        
        with tab1:
            render_entry_tab(df_classes)
            render_transaction_history(df_classes)
        
        with tab2:
            render_master_data_tab(df_classes)
        
        with tab3:
            render_analytics_tab()
    else:
        tab1, tab2 = st.tabs(["🚗 Current Vehicle", "📊 Analytics"])
        
        with tab1:
            render_current_vehicle_tab(df_classes)
            render_transaction_history(df_classes)
        
        with tab2:
            render_analytics_tab()

if __name__ == "__main__":
    main()