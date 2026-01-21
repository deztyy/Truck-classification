import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import os
from datetime import datetime, date
from typing import Optional, Dict, Any
import pytz
import streamlit.components.v1 as components

# ==================== CONSTANTS ====================
# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@db:5432/mydb')
THAILAND_TZ = pytz.timezone('Asia/Bangkok')

# UI Configuration
CACHE_TTL_SECONDS = 5
DEFAULT_CAMERA_OPTIONS = ["1", "2", "3", "➕ Add New"]
ADD_NEW_OPTION = "➕ Add New"

# Validation Constants
MIN_FEE = 0.0
FEE_STEP = 10.0
MAX_CAMERA_ID_LENGTH = 50
MAX_TRACK_ID_LENGTH = 100

# ==================== CUSTOM CSS ====================
def load_custom_css() -> None:
    """Load custom CSS styling for the application"""
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
def get_database_engine() -> Engine:
    """
    Create and return database engine with connection validation
    
    Returns:
        Engine: SQLAlchemy database engine
        
    Raises:
        Exception: If database connection fails
    """
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
def init_database() -> None:
    """
    Initialize database tables if they don't exist
    Creates vehicle_classes and vehicle_transactions tables
    """
    try:
        with engine.connect() as conn:
            # Set timezone for this session
            conn.execute(text("SET TIME ZONE 'Asia/Bangkok'"))
            
            # Check if vehicle_classes table exists
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
                conn.commit()
                print("✅ vehicle_classes table created")
            
            # Check if vehicle_transactions table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'vehicle_transactions'
                );
            """))
            
            if not result.scalar():
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS vehicle_transactions (
                        id SERIAL PRIMARY KEY,
                        camera_id VARCHAR(50) NOT NULL,
                        track_id VARCHAR(100) NOT NULL,
                        class_id INT NOT NULL,
                        total_fee NUMERIC(10, 2),
                        time_stamp TIMESTAMPZ DEFAULT CURRENT_TIMESTAMP,
                        img_path TEXT,
                        confidence NUMERIC(5, 4)
                    );
                """))
                conn.commit()
                print("✅ vehicle_transactions table created")
                
    except Exception as e:
        st.error(f"❌ Error initializing database: {e}")
        print(f"❌ Database initialization error: {e}")

# ==================== DATA LOADING ====================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_vehicle_classes() -> pd.DataFrame:
    """
    Load vehicle classes from database with caching
    
    Returns:
        pd.DataFrame: DataFrame containing vehicle class information
    """
    try:
        query = """
            SELECT class_id, class_name, entry_fee, xray_fee, total_fee 
            FROM vehicle_classes 
            ORDER BY class_id
        """
        df = pd.read_sql(text(query), engine)
        return df
    except Exception as e:
        st.error(f"❌ Error loading vehicle classes: {e}")
        print(f"❌ Error loading vehicle classes: {e}")
        return pd.DataFrame()

# ==================== VALIDATION FUNCTIONS ====================
def validate_camera_id(camera_id: str) -> tuple[bool, str]:
    """
    Validate camera ID input
    
    Args:
        camera_id: Camera ID to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not camera_id or not camera_id.strip():
        return False, "Camera ID cannot be empty"
    
    if len(camera_id) > MAX_CAMERA_ID_LENGTH:
        return False, f"Camera ID too long (max {MAX_CAMERA_ID_LENGTH} characters)"
    
    return True, ""

def validate_fee(fee: float, fee_name: str) -> tuple[bool, str]:
    """
    Validate fee value
    
    Args:
        fee: Fee amount to validate
        fee_name: Name of the fee for error message
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if fee < MIN_FEE:
        return False, f"{fee_name} cannot be negative"
    
    return True, ""

def validate_class_name(class_name: str) -> tuple[bool, str]:
    """
    Validate vehicle class name
    
    Args:
        class_name: Class name to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not class_name or not class_name.strip():
        return False, "Vehicle type cannot be empty"
    
    if len(class_name.strip()) < 2:
        return False, "Vehicle type must be at least 2 characters"
    
    return True, ""

# ==================== TIMEZONE HELPERS ====================
def get_thailand_time() -> datetime:
    """
    Get current time in Thailand timezone
    
    Returns:
        datetime: Current datetime in Thailand timezone
    """
    return datetime.now(THAILAND_TZ)

def convert_to_thailand_tz(timestamp: pd.Timestamp) -> datetime:
    """
    Convert timestamp to Thailand timezone
    
    Args:
        timestamp: Timestamp to convert
        
    Returns:
        datetime: Converted datetime in Thailand timezone
    """
    if timestamp.tzinfo is None:
        timestamp = pytz.utc.localize(timestamp)
    return timestamp.astimezone(THAILAND_TZ)

# ==================== HEADER ====================
def render_header() -> None:
    """Render animated header with real-time clock"""
    load_custom_css()
    
    now_thailand = get_thailand_time()
    current_date = now_thailand.strftime('%d %B %Y')
    
    st.markdown(f"""
    <div class="main-header">
        <h1 class="header-title">🚗 Vehicle Entry System</h1>
        <div class="datetime-box">
            <div class="date-text">📅 {current_date}</div>
            <div class="time-text" id="clock">🕐 Loading...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Client-side clock without page refresh
    components.html(
        """
        <script>
            function updateClock() {
                const now = new Date();
                const options = { timeZone: 'Asia/Bangkok', hour12: false };
                const timeString = now.toLocaleTimeString('en-GB', options);
                
                const clockElement = window.parent.document.getElementById('clock');
                if (clockElement) {
                    clockElement.textContent = '🕐 ' + timeString;
                }
            }
            
            // Update immediately
            updateClock();
            
            // Update every second
            setInterval(updateClock, 1000);
        </script>
        """,
        height=0
    )

# ==================== ENTRY TAB ====================
def render_entry_tab(df_classes: pd.DataFrame) -> None:
    """
    Render vehicle entry form
    
    Args:
        df_classes: DataFrame containing vehicle class information
    """
    st.markdown("### 📝 New Vehicle Entry")
    
    if df_classes.empty:
        st.warning("⚠️ No vehicle classes available. Please add some in Master Data tab.")
        return
    
    # Selection section (outside form for real-time update)
    col1, col2 = st.columns(2)
    
    with col1:
        # Camera selection dropdown
        camera_selection = st.selectbox(
            "📷 Camera ID", 
            DEFAULT_CAMERA_OPTIONS, 
            key="camera_select"
        )
        
        # Show text input if "Add New" is selected
        if camera_selection == ADD_NEW_OPTION:
            camera_id = st.text_input(
                "🆕 New Camera ID", 
                placeholder="e.g., CAM-001", 
                key="new_camera",
                max_chars=MAX_CAMERA_ID_LENGTH
            )
        else:
            camera_id = camera_selection
    
    with col2:
        class_options = df_classes['class_name'].tolist()
        selected_class = st.selectbox(
            "🚗 Vehicle Type", 
            class_options, 
            key="vehicle_type_select"
        )
    
    # Display fees (updates in real-time when vehicle type changes)
    if selected_class:
        class_data = df_classes[df_classes['class_name'] == selected_class].iloc[0]
        
        st.markdown("---")
        col_fee1, col_fee2, col_fee3 = st.columns(3)
        
        with col_fee1:
            st.metric("💵 Entry Fee", f"{class_data['entry_fee']:.0f} ฿")
        with col_fee2:
            st.metric("🔍 X-Ray Fee", f"{class_data['xray_fee']:.0f} ฿")
        with col_fee3:
            st.metric("💰 Total Fee", f"{class_data['total_fee']:.0f} ฿")
    
    st.markdown("---")
    
    # Save button (outside form)
    _, col_btn2, _ = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("✅ Save Entry", use_container_width=True, type="primary", key="save_entry_btn"):
            # Validate camera ID
            is_valid, error_msg = validate_camera_id(camera_id)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
            elif camera_selection == ADD_NEW_OPTION and not camera_id.strip():
                st.error("❌ Please enter Camera ID")
            else:
                try:
                    class_data = df_classes[df_classes['class_name'] == selected_class].iloc[0]
                    current_time_thailand = get_thailand_time()
                    
                    # Generate track_id (manual entry uses timestamp)
                    track_id = f"MANUAL_{current_time_thailand.strftime('%Y%m%d%H%M%S')}"
                    
                    with engine.connect() as conn:
                        # Set timezone to Bangkok for this session
                        conn.execute(text("SET TIME ZONE 'Asia/Bangkok'"))
                        
                        conn.execute(text("""
                            INSERT INTO vehicle_transactions 
                            (camera_id, track_id, class_id, total_fee, time_stamp)
                            VALUES (:camera_id, :track_id, :class_id, :total, :time_stamp)
                        """), {
                            "camera_id": camera_id.strip(),
                            "track_id": track_id,
                            "class_id": int(class_data['class_id']),
                            "total": float(class_data['total_fee']),
                            "time_stamp": current_time_thailand
                        })
                        conn.commit()
                    
                    st.success(f"✅ Entry saved! Camera: {camera_id} | Vehicle: {selected_class} | Total: {class_data['total_fee']:.0f} ฿")
                    st.balloons()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error saving entry: {e}")
                    print(f"❌ Error saving entry: {e}")

# ==================== CURRENT VEHICLE TAB ====================
def render_current_vehicle_tab() -> None:
    """Render current vehicle display tab showing the latest entry"""
    st.markdown("### 🚗 Current Vehicle")
    
    try:
        query = """
            SELECT 
                t.camera_id,
                c.class_name as vehicle_type,
                t.total_fee,
                t.time_stamp
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            ORDER BY t.time_stamp DESC
            LIMIT 1
        """
        df_latest = pd.read_sql(text(query), engine)
        
        if not df_latest.empty:
            vehicle = df_latest.iloc[0]
            timestamp = convert_to_thailand_tz(pd.to_datetime(vehicle['time_stamp']))
            formatted_time = timestamp.strftime('%d/%m/%Y %H:%M:%S')
            total_fee = vehicle['total_fee'] if vehicle['total_fee'] is not None else 0.0
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 padding: 3rem 2rem; border-radius: 20px; margin-bottom: 2rem;
                 box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);">
                <div style="text-align: center; color: white; font-size: 2em; font-weight: 800;">
                    🚗 Latest Vehicle Entry
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                     padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);">
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9em; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                        📷 Camera ID
                    </div>
                    <div style="color: #ffd700; font-size: 2em; font-weight: 800;
                         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {vehicle['camera_id']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                     padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);
                     margin-top: 1rem;">
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9em; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                        🚙 Vehicle Type
                    </div>
                    <div style="color: #ffd700; font-size: 2em; font-weight: 800;
                         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {vehicle['vehicle_type']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                     padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);">
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9em; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                        💰 Total Fee
                    </div>
                    <div style="color: #ffd700; font-size: 2em; font-weight: 800;
                         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {total_fee:.2f} ฿
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                     padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);
                     margin-top: 1rem;">
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9em; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                        ⏰ Timestamp
                    </div>
                    <div style="color: #ffd700; font-size: 1.4em; font-weight: 800;
                         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {formatted_time}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Refresh button
            st.markdown("---")
            _, col2, _ = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Refresh", use_container_width=True, type="primary"):
                    st.rerun()
                    
        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; 
                 background: rgba(102, 126, 234, 0.05); border-radius: 20px; 
                 border: 2px dashed rgba(102, 126, 234, 0.3);">
                <div style="font-size: 5em; opacity: 0.5;">🚗💨</div>
                <div style="color: #667eea; font-size: 1.8em; font-weight: 700;">
                    No Vehicles Yet
                </div>
                <div style="color: #999; font-size: 1.1em;">
                    Waiting for the first vehicle to enter...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Error loading current vehicle: {e}")
        print(f"❌ Error loading current vehicle: {e}")

# ==================== TRANSACTION HISTORY ====================
def render_transaction_history() -> None:
    """Render transaction history with filtering and delete functionality"""
    st.markdown("---")
    st.markdown("### 📜 Transaction History")
    
    now_thailand = get_thailand_time()
    date_filter = st.date_input("📅 Select Date", value=now_thailand.date())
    
    try:
        query = """
            SELECT 
                t.id,
                t.camera_id,
                t.track_id,
                t.class_id,
                t.total_fee,
                t.time_stamp,
                t.confidence,
                c.class_name,
                c.entry_fee,
                c.xray_fee
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.time_stamp) = :date_filter
            ORDER BY t.time_stamp DESC
        """
        
        df_transactions = pd.read_sql(text(query), engine, params={"date_filter": date_filter})
        
        if not df_transactions.empty:
            # Summary metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("📊 Total Transactions", len(df_transactions))
            with col_m2:
                st.metric("💰 Total Revenue", f"{df_transactions['total_fee'].sum():.0f} ฿")
            with col_m3:
                st.metric("📈 Avg/Transaction", f"{df_transactions['total_fee'].mean():.0f} ฿")
            
            st.markdown("---")
            
            # Transaction details
            for _, row in df_transactions.iterrows():
                timestamp = convert_to_thailand_tz(pd.to_datetime(row['time_stamp']))
                time_display = timestamp.strftime('%H:%M:%S')
                conf_text = f" ({row['confidence']:.2%})" if pd.notna(row['confidence']) else ""
                
                with st.expander(
                    f"📷 {row['camera_id']} | {row['class_name']}{conf_text} | {time_display} | {row['total_fee']:.0f} ฿",
                    expanded=False
                ):
                    st.markdown(f"**🆔 ID:** #{row['id']}")
                    st.markdown(f"**📷 Camera:** {row['camera_id']}")
                    st.markdown(f"**🔖 Track ID:** {row['track_id']}")
                    st.markdown(f"**🚗 Vehicle:** {row['class_name']}")
                    if pd.notna(row['confidence']):
                        st.markdown(f"**🎯 Confidence:** {row['confidence']:.2%}")
                    st.markdown("---")
                    
                    # Fee breakdown
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        st.metric("Entry", f"{row['entry_fee']:.0f} ฿")
                    with col_f2:
                        st.metric("X-Ray", f"{row['xray_fee']:.0f} ฿")
                    with col_f3:
                        st.metric("Total", f"{row['total_fee']:.0f} ฿")
                    
                    # Full timestamp
                    full_timestamp = convert_to_thailand_tz(pd.to_datetime(row['time_stamp']))
                    st.markdown(f"**🕐 Time:** {full_timestamp.strftime('%d/%m/%Y %H:%M:%S')} (Thailand)")
                    
                    # Delete button
                    st.markdown("---")
                    _, col_d2, _ = st.columns([2, 1, 2])
                    with col_d2:
                        if st.button(f"🗑️ Delete", key=f"del_{row['id']}", type="secondary", use_container_width=True):
                            try:
                                with engine.connect() as conn:
                                    conn.execute(
                                        text("DELETE FROM vehicle_transactions WHERE id = :id"), 
                                        {"id": row['id']}
                                    )
                                    conn.commit()
                                st.success("✅ Deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting transaction: {e}")
                                print(f"❌ Error deleting transaction: {e}")
        else:
            st.info(f"📭 No transactions found for {date_filter.strftime('%d %B %Y')}")
    
    except Exception as e:
        st.error(f"❌ Error loading transactions: {e}")
        print(f"❌ Error loading transactions: {e}")

# ==================== MASTER DATA TAB ====================
def render_master_data_tab(df_classes: pd.DataFrame) -> None:
    """
    Render master data management tab for vehicle classes
    
    Args:
        df_classes: DataFrame containing current vehicle classes
    """
    st.markdown("### ⚙️ Vehicle Classes Management")
    
    if not df_classes.empty:
        st.dataframe(
            df_classes,
            use_container_width=True,
            hide_index=True,
            column_config={
                "class_id": "ID",
                "class_name": "Vehicle Type",
                "entry_fee": st.column_config.NumberColumn("Entry Fee (฿)", format="%.0f ฿"),
                "xray_fee": st.column_config.NumberColumn("X-Ray Fee (฿)", format="%.0f ฿"),
                "total_fee": st.column_config.NumberColumn("Total Fee (฿)", format="%.0f ฿")
            }
        )
    else:
        st.info("📭 No vehicle classes defined yet")
    
    #st.markdown("---")
    # st.markdown("#### ➕ Add/Edit Vehicle Class")
    
    # with st.form("class_form", clear_on_submit=True):
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         class_name = st.text_input("🚗 Vehicle Type", placeholder="e.g., Sedan, Truck")
    #         entry_fee = st.number_input(
    #             "💵 Entry Fee (฿)", 
    #             min_value=MIN_FEE, 
    #             step=FEE_STEP, 
    #             value=MIN_FEE
    #         )
        
    #     with col2:
    #         xray_fee = st.number_input(
    #             "🔍 X-Ray Fee (฿)", 
    #             min_value=MIN_FEE, 
    #             step=FEE_STEP, 
    #             value=MIN_FEE
    #         )
    #         total_fee = entry_fee + xray_fee
    #         st.metric("💰 Total Fee", f"{total_fee:.2f} ฿")
        
    #     _, col_s2, _ = st.columns([1, 1, 1])
    #     with col_s2:
    #         submitted = st.form_submit_button("💾 Save Class", use_container_width=True, type="primary")
        
    #     if submitted:
    #         # Validate class name
    #         is_valid, error_msg = validate_class_name(class_name)
            
    #         if not is_valid:
    #             st.error(f"❌ {error_msg}")
    #         else:
    #             # Validate fees
    #             entry_valid, entry_msg = validate_fee(entry_fee, "Entry fee")
    #             xray_valid, xray_msg = validate_fee(xray_fee, "X-Ray fee")
                
    #             if not entry_valid:
    #                 st.error(f"❌ {entry_msg}")
    #             elif not xray_valid:
    #                 st.error(f"❌ {xray_msg}")
    #             else:
    #                 try:
    #                     with engine.connect() as conn:
    #                         conn.execute(text("""
    #                             INSERT INTO vehicle_classes (class_name, entry_fee, xray_fee, total_fee)
    #                             VALUES (:name, :entry, :xray, :total)
    #                             ON CONFLICT (class_name) DO UPDATE 
    #                             SET entry_fee = :entry, xray_fee = :xray, total_fee = :total
    #                         """), {
    #                             "name": class_name.strip(),
    #                             "entry": entry_fee,
    #                             "xray": xray_fee,
    #                             "total": total_fee
    #                         })
    #                         conn.commit()
    #                     st.success(f"✅ Saved: {class_name}")
    #                     st.rerun()
    #                 except Exception as e:
    #                     st.error(f"❌ Error saving vehicle class: {e}")
    #                     print(f"❌ Error saving vehicle class: {e}")

# ==================== ANALYTICS TAB ====================
def render_analytics_tab() -> None:
    """Render analytics dashboard with charts and statistics"""
    st.markdown("### 📊 Analytics Dashboard")
    
    now_thailand = get_thailand_time()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Start Date", value=now_thailand.date())
    with col2:
        end_date = st.date_input("📅 End Date", value=now_thailand.date())
    
    try:
        query = """
            SELECT 
                t.id,
                t.camera_id,
                t.class_id,
                t.total_fee,
                t.time_stamp,
                c.class_name
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.time_stamp) BETWEEN :start_date AND :end_date
            ORDER BY t.time_stamp DESC
        """
        
        df_analytics = pd.read_sql(
            text(query), 
            engine, 
            params={"start_date": start_date, "end_date": end_date}
        )
        
        if not df_analytics.empty:
            st.markdown("---")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Transactions", len(df_analytics))
            with col2:
                st.metric("💰 Total Revenue", f"{df_analytics['total_fee'].sum():.0f} ฿")
            with col3:
                st.metric("📈 Avg/Transaction", f"{df_analytics['total_fee'].mean():.0f} ฿")
            with col4:
                st.metric("📷 Active Cameras", df_analytics['camera_id'].nunique())
            
            st.markdown("---")
            
            # Charts
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("#### 🚗 Transactions by Vehicle Type")
                vehicle_counts = df_analytics['class_name'].value_counts()
                st.bar_chart(vehicle_counts.to_frame("count"))
            
            with col_c2:
                st.markdown("#### 💰 Revenue by Vehicle Type")
                revenue_by_type = df_analytics.groupby('class_name')['total_fee'].sum().sort_values(ascending=False)
                st.bar_chart(revenue_by_type.to_frame("revenue"))
        else:
            st.info(f"📭 No data found between {start_date.strftime('%d %B %Y')} and {end_date.strftime('%d %B %Y')}")
    
    except Exception as e:
        st.error(f"❌ Error loading analytics: {e}")
        print(f"❌ Error loading analytics: {e}")

# ==================== MAIN APPLICATION ====================
def main() -> None:
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Vehicle Entry System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize database
    init_database()
    
    # Load vehicle classes
    df_classes = load_vehicle_classes()
    
    # Render header
    render_header()
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Entry", 
        "🚗 Current Vehicle", 
        "⚙️ Master Data", 
        "📊 Analytics"
    ])
    
    with tab1:
        render_entry_tab(df_classes)
        render_transaction_history()
    
    with tab2:
        render_current_vehicle_tab()
    
    with tab3:
        render_master_data_tab(df_classes)
    
    with tab4:
        render_analytics_tab()

if __name__ == "__main__":
    main()