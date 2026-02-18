import io
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
import streamlit as st
import streamlit.components.v1 as components
from minio import Minio
from minio.error import S3Error
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ==================== CONSTANTS ====================
# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres1234@db:5432/vehicle_db")
THAILAND_TZ = pytz.timezone("Asia/Bangkok")

# Superset Configuration
SUPERSET_BASE_URL = os.getenv("SUPERSET_BASE_URL", "http://localhost:8088")
SUPERSET_DASHBOARD_ID = os.getenv("SUPERSET_DASHBOARD_ID", "")
SUPERSET_DASHBOARD_SLUG = os.getenv("SUPERSET_DASHBOARD_SLUG", "")

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "video-frames")

# UI Configuration
CACHE_TTL_SECONDS = 5
DEFAULT_CAMERA_OPTIONS = ["1", "2", "3", "➕ Add New"]
ADD_NEW_OPTION = "➕ Add New"

# Validation Constants
MIN_FEE = 0.0
FEE_STEP = 10.0
MAX_CAMERA_ID_LENGTH = 50
MAX_TRACK_ID_LENGTH = 100

# Display Name Mapping for Vehicle Classes
CLASS_NAME_DISPLAY = {
    "car": "รถยนต์",
    "other": "รถประเภทอื่น(เช่น รถบัส รถตุ๊กตุ๊ก)",
    "other_truck": "รถบรรทุกประเภทอื่น(เช่น รถบรรทุกของเหลว)",
    "pickup_truck": "รถกระบะ",
    "truck_20_back": "รถบรรทุกที่มีตู้ขนาด 20 อยู่ด้านหลัง",
    "truck_20_front": "รถบรรทุกที่มีตู้ขนาด 20 อยู่ด้านหน้า",
    "truck_40": "รถบรรทุกที่มีตู้ขนาด 40",
    "truck_roro": "รถบรรทุกขนรถ",
    "truck_tail": "รถบรรทุกที่มีหาง",
    "motorcycle": "มอเตอร์ไซค์",
    "truck_head": "รถบรรทุกที่แต่หัว",
    "truck_20x2": "รถบรรทุกที่มีตู้ขนาด 20 อยู่ 2 ตู้",
}


# ==================== CUSTOM CSS ====================
def load_custom_css() -> None:
    """Load custom CSS styling for the application"""
    st.markdown(
        """
    <style>
        /* Global Styles */
        .stApp {
            background-color: #0e1117;
        }

        /* Header Styles */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        }

        .header-title {
            color: white;
            font-size: 2em;
            font-weight: 800;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            line-height: 1.2;
        }

        .datetime-box {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .date-text {
            color: white;
            font-weight: 600;
            line-height: 1.4;
        }

        .time-text {
            color: #ffd700;
            font-weight: 700;
            font-family: 'Courier New', monospace;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            line-height: 1.4;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header {
                padding: 1rem;
            }

            .header-title {
                font-size: 1.5em;
                text-align: center;
                margin-bottom: 1rem;
            }

            .datetime-box {
                width: 100%;
                margin-top: 1rem;
            }
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
    """,
        unsafe_allow_html=True,
    )


# ==================== HELPER FUNCTIONS ====================
def get_thailand_time() -> datetime:
    """Get current time in Thailand timezone"""
    return datetime.now(THAILAND_TZ)


def convert_to_thailand_tz(dt) -> datetime:
    """Convert datetime to Thailand timezone"""
    if dt.tzinfo is None:
        # If no timezone info, assume it's already in Bangkok time (from database)
        dt = THAILAND_TZ.localize(dt)
        return dt
    return dt.astimezone(THAILAND_TZ)


def translate_class_name(class_name: str) -> str:
    """
    Translate class name from database to display name

    Args:
        class_name: Original class name from database

    Returns:
        Translated display name
    """
    if pd.isna(class_name):
        return class_name

    # Try to get translated name, if not found return original with warning indicator
    translated = CLASS_NAME_DISPLAY.get(class_name.lower(), None)
    if translated is None:
        print(f"⚠️ Warning: No translation found for '{class_name}'")
        return f"{class_name} ⚠️"
    return translated


def check_system_status() -> dict:
    """Check database and MinIO connection status"""
    status = {
        "database": False,
        "minio": False,
        "database_msg": "",
        "minio_msg": "",
        "buckets": [],
    }

    # Check database
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            status["database"] = True
            status["database_msg"] = "Connected"
    except Exception as e:
        status["database_msg"] = f"Error: {str(e)[:50]}"

    # Check MinIO
    if minio_client:
        try:
            # List all buckets
            buckets = minio_client.list_buckets()
            status["buckets"] = [bucket.name for bucket in buckets]
            status["minio"] = True
            status["minio_msg"] = f"Connected ({len(status['buckets'])} buckets)"
        except Exception as e:
            status["minio_msg"] = f"Error: {str(e)[:50]}"
    else:
        status["minio_msg"] = "Not initialized"

    return status


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


# ==================== MINIO CONNECTION ====================
@st.cache_resource
def get_minio_client() -> Minio:
    """
    Create and return MinIO client with connection validation

    Returns:
        Minio: MinIO client instance

    Raises:
        Exception: If MinIO connection fails
    """
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        # Test connection by checking if bucket exists
        client.bucket_exists(MINIO_BUCKET_NAME)
        print("✅ MinIO connected successfully!")
        return client
    except Exception as e:
        print(f"⚠️ MinIO connection failed: {e}")
        # Return None instead of stopping the app
        return None


minio_client = get_minio_client()


# ==================== MINIO HELPER FUNCTIONS ====================
def get_image_from_minio(img_path: str) -> Optional[Image.Image]:
    """
    Retrieve image from MinIO storage

    Args:
    img_path: Path to image in MinIO (format: 'bucket-name/object-key' or just 'object-key')

    Returns:
        PIL.Image or None if image not found
    """
    if not minio_client or not img_path:
        return None

    try:
        # Parse bucket and object key from img_path
        # Format can be: "process-frames/output_xxx.jpg" or just "output_xxx.jpg"
        if "/" in img_path:
            parts = img_path.split("/", 1)
            bucket_name = parts[0]
            object_key = parts[1] if len(parts) > 1 else parts[0]
        else:
            bucket_name = MINIO_BUCKET_NAME
            object_key = img_path

        print(f"🔍 Trying to load image from bucket: {bucket_name}, key: {object_key}")

        # Get object from MinIO
        response = minio_client.get_object(bucket_name, object_key)
        # Read image data
        image_data = response.read()
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        print(f"✅ Successfully loaded image: {img_path}")
        return image
    except S3Error as e:
        print(f"❌ MinIO S3 error loading {img_path}: {e.code} - {e.message}")
        return None
    except Exception as e:
        print(f"❌ Error loading image {img_path}: {type(e).__name__} - {str(e)}")
        return None
    finally:
        if "response" in locals():
            response.close()
            response.release_conn()


# ==================== DATABASE INITIALIZATION ====================
def init_database() -> None:
    """
    Initialize database - check tables and data exist
    Note: Tables should be created by init-db.sql on first PostgreSQL startup
    """
    try:
        with engine.connect() as conn:
            # Set timezone for this session to Asia/Bangkok
            conn.execute(text("SET TIME ZONE 'Asia/Bangkok'"))
            
            # Set default timezone for the database (affects all connections)
            try:
                conn.execute(text("ALTER DATABASE vehicle_db SET timezone TO 'Asia/Bangkok'"))
                conn.commit()
                print("✅ Database timezone set to Asia/Bangkok")
            except Exception as e:
                print(f"⚠️ Could not set database timezone: {e}")
                # Continue anyway - session timezone is already set

            # Check if tables exist
            result = conn.execute(
                text("""
                SELECT
                    (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'vehicle_classes') as classes_exists,
                    (SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'vehicle_transactions') as transactions_exists;
            """)
            )
            row = result.fetchone()

            if row[0] == 0 or row[1] == 0:
                st.warning(
                    "⚠️ Database tables not found. Please ensure init-db.sql was executed."
                )
                print("⚠️ Database tables missing - check init-db.sql execution")
            else:
                # Check if vehicle_classes has data
                result = conn.execute(text("SELECT COUNT(*) FROM vehicle_classes"))
                count = result.scalar()

                if count == 0:
                    st.warning("⚠️ No vehicle classes found in database")
                    print("⚠️ vehicle_classes table is empty")
                else:
                    print(f"✅ Database initialized - {count} vehicle classes found")

    except Exception as e:
        st.error(f"❌ Error checking database: {e}")
        print(f"❌ Database initialization error: {e}")


# ==================== SUPERSET EMBED ====================
def build_superset_dashboard_url() -> str:
    """Build Superset dashboard URL from environment variables."""
    if SUPERSET_DASHBOARD_SLUG:
        return f"{SUPERSET_BASE_URL}/superset/dashboard/p/{SUPERSET_DASHBOARD_SLUG}/"
    if SUPERSET_DASHBOARD_ID:
        return f"{SUPERSET_BASE_URL}/superset/dashboard/{SUPERSET_DASHBOARD_ID}/?standalone=1"
    return ""


def render_superset_tab() -> None:
    """Render Superset dashboard in an embedded iframe."""
    st.markdown("### 📊 Superset Dashboards")
    st.caption("เปิดกราฟจาก Apache Superset ในหน้าแอพนี้")

    default_url = build_superset_dashboard_url()
    
    if default_url:
        st.link_button("🔗 Open Superset Dashboard", default_url, use_container_width=True)
    
    dashboard_url = st.text_input(
        "Superset Dashboard URL",
        value=default_url,
        help="ใส่ลิงก์แดชบอร์ด เช่น http://localhost:8088/superset/dashboard/<slug-or-id>/?standalone=1",
    )

    if not dashboard_url:
        st.info(
            "ยังไม่ได้ตั้งค่า URL ของแดชบอร์ด Superset. ตั้งค่า SUPERSET_DASHBOARD_SLUG หรือ SUPERSET_DASHBOARD_ID หรือวาง URL เอง"
        )
        return

    if dashboard_url != default_url:
        st.link_button("🔗 Open Custom URL", dashboard_url, use_container_width=True)
    
    components.iframe(dashboard_url, height=900, scrolling=True)


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


# ==================== HEADER ====================
def render_header() -> None:
    """Render animated header with real-time clock"""
    load_custom_css()

    now_thailand = get_thailand_time()
    current_date = now_thailand.strftime("%d %B %Y")

    st.markdown(
        f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                <h1 class="header-title" style="margin: 0; text-align: left;">🚗 Vehicle Entry System</h1>
            </div>
            <div class="datetime-box" style="flex: 0 0 auto; min-width: 280px; margin-top: 0;">
                <div style="display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;">
                    <div style="text-align: left;">
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                            Date
                        </div>
                        <div class="date-text" style="font-size: 1em; margin: 0;">📅 {current_date}</div>
                    </div>
                    <div style="width: 1px; height: 40px; background: rgba(255,255,255,0.2);"></div>
                    <div style="text-align: left;">
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                            Time
                        </div>
                        <div class="time-text" id="clock" style="font-size: 1.1em; margin: 0;">🕐 Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

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
        height=0,
    )


# ==================== IMAGE CLEANUP ====================
def cleanup_old_images() -> None:
    """
    Note: This function is deprecated when using MinIO.
    MinIO handles object lifecycle through bucket policies.
    Images are stored in MinIO and referenced by path in database.
    """
    # No-op when using MinIO - object lifecycle managed by MinIO policies
    pass


# ==================== CURRENT VEHICLE TAB ====================
def render_current_vehicle_tab() -> None:
    """Render current vehicle display tab showing the latest entry"""
    st.markdown("### 🚗 Current Vehicle")

    try:
        # Run cleanup first
        cleanup_old_images()

        query = """
            SELECT
                t.camera_id,
                c.class_name as vehicle_type,
                t.total_fee,
                t.time_stamp,
                t.img_path
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            ORDER BY t.time_stamp DESC
            LIMIT 1
        """
        df_latest = pd.read_sql(text(query), engine)

        if not df_latest.empty:
            vehicle = df_latest.iloc[0]
            timestamp = convert_to_thailand_tz(pd.to_datetime(vehicle["time_stamp"]))
            formatted_time = timestamp.strftime("%d/%m/%Y %H:%M:%S")
            total_fee = (
                vehicle["total_fee"] if vehicle["total_fee"] is not None else 0.0
            )

            st.markdown(
                """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 padding: 3rem 2rem; border-radius: 20px; margin-bottom: 2rem;
                 box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);">
                <div style="text-align: center; color: white; font-size: 2em; font-weight: 800;">
                    🚗 Latest Vehicle Entry
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown(
                    f"""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                     padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);">
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9em; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                        📷 Camera ID
                    </div>
                    <div style="color: #ffd700; font-size: 2em; font-weight: 800;
                         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {vehicle["camera_id"]}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
                     padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);
                     margin-top: 1rem;">
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9em; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                        🚙 Vehicle Type
                    </div>
                    <div style="color: #ffd700; font-size: 2em; font-weight: 800;
                         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {vehicle["vehicle_type"]}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
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
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
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
                """,
                    unsafe_allow_html=True,
                )

            # Display image if available
            if (
                "img_path" in vehicle
                and vehicle["img_path"]
                and vehicle["img_path"] != ""
            ):
                st.markdown("---")
                st.markdown("### 📸 Vehicle Image")

                # Try to get image from MinIO
                image = get_image_from_minio(vehicle["img_path"])
                if image:
                    st.image(image, use_container_width=True)
                else:
                    st.warning("⚠️ Image not available in MinIO storage")
                    with st.expander("🔍 Debug Info"):
                        st.code(f"Path: {vehicle['img_path']}", language="text")
                        st.info(
                            "Image should be in format: 'bucket-name/object-key' (e.g., 'process-frames/output_123.jpg')"
                        )

            # Refresh button
            st.markdown("---")
            _, col2, _ = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Refresh", use_container_width=True, type="primary"):
                    st.rerun()

        else:
            st.markdown(
                """
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
            """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"❌ Error loading current vehicle: {e}")
        print(f"❌ Error loading current vehicle: {e}")


# ==================== TRANSACTION HISTORY ====================
# Replace the incomplete render_transaction_history() function

def render_transaction_history() -> None:
    """Render transaction history for today only with filters"""
    st.markdown("---")
    st.markdown("### 📜 Transaction History (Today)")

    now_thailand = get_thailand_time()
    today = now_thailand.date()

    # Display current date
    st.info(f"📅 Showing transactions for: {today.strftime('%d %B %Y')}")

    try:
        # Get all vehicle classes from master data
        query_classes = """
            SELECT class_name
            FROM vehicle_classes
            ORDER BY class_id
        """
        df_classes = pd.read_sql(text(query_classes), engine)
        all_vehicle_types = df_classes["class_name"].tolist()

        # Get all transactions for today
        query = """
            SELECT
                t.id,
                t.camera_id,
                t.track_id,
                t.class_id,
                t.total_fee,
                t.time_stamp,
                t.confidence,
                t.img_path,
                c.class_name,
                c.entry_fee,
                c.xray_fee
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.time_stamp) = :today
            ORDER BY t.time_stamp DESC
        """

        df_all = pd.read_sql(text(query), engine, params={"today": today})

        if not df_all.empty:
            # Convert timestamps to Bangkok timezone
            df_all["time_bangkok"] = pd.to_datetime(
                df_all["time_stamp"]
            ).dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')

            # Filter options
            st.markdown("#### 🔍 Filters")

            # Search box for Track ID
            search_track = st.text_input(
                "🔎 Search Track ID",
                placeholder="Enter track ID to search...",
                key="search_track",
            )

            col_f1, col_f2, col_f3 = st.columns(3)

            with col_f1:
                all_cameras = sorted(df_all["camera_id"].unique().tolist())
                camera_options = ["All Cameras"] + all_cameras
                selected_camera = st.selectbox(
                    "📷 Select Camera",
                    options=camera_options,
                    index=0,
                    key="camera_filter",
                )

            with col_f2:
                translated_types = [
                    translate_class_name(vt) for vt in all_vehicle_types
                ]
                vehicle_type_options = ["All Types"] + translated_types
                selected_vehicle_type = st.selectbox(
                    "🚗 Select Vehicle Type",
                    options=vehicle_type_options,
                    index=0,
                    key="vehicle_type_filter",
                )

           # ...existing code...
            with col_f3:
                time_filter_type = st.selectbox(
                    "⏰ Time Filter",
                    options=["All Day", "Time Period"],
                    index=0,
                    key="time_filter_type",
                )

            # Apply filters
            if selected_camera == "All Cameras":
                selected_cameras = all_cameras
            else:
                selected_cameras = [selected_camera]

            if selected_vehicle_type == "All Types":
                selected_vehicle_types = all_vehicle_types
            else:
                selected_vehicle_types = [
                    vt
                    for vt in all_vehicle_types
                    if translate_class_name(vt) == selected_vehicle_type
                ]

            df_transactions = df_all[
                (df_all["camera_id"].isin(selected_cameras))
                & (df_all["class_name"].isin(selected_vehicle_types))
            ]

            # Apply Track ID search
            if search_track and search_track.strip():
                df_transactions = df_transactions[
                    df_transactions["track_id"].str.contains(
                        search_track.strip(), case=False, na=False
                    )
                ]

            # Apply time filter
            if time_filter_type == "Time Period":
                st.markdown("#### ⏰ Select Time Period")
                
                col_t1, col_t2 = st.columns(2)
                
                with col_t1:
                    st.markdown("**Start Time**")
                    start_hour = st.number_input(
                        "Hour",
                        min_value=0,
                        max_value=23,
                        value=0,
                        key="start_hour",
                    )
                    start_minute = st.number_input(
                        "Minute",
                        min_value=0,
                        max_value=59,
                        value=0,
                        key="start_minute",
                    )
                
                with col_t2:
                    st.markdown("**End Time**")
                    end_hour = st.number_input(
                        "Hour",
                        min_value=0,
                        max_value=23,
                        value=23,
                        key="end_hour",
                    )
                    end_minute = st.number_input(
                        "Minute",
                        min_value=0,
                        max_value=59,
                        value=59,
                        key="end_minute",
                    )

                # Validate time range
                start_time_minutes = start_hour * 60 + start_minute
                end_time_minutes = end_hour * 60 + end_minute
                
                if start_time_minutes > end_time_minutes:
                    st.error("⚠️ Start time must be before end time!")
                else:
                    st.info(f"📊 Filtering from {start_hour:02d}:{start_minute:02d} to {end_hour:02d}:{end_minute:02d}")
                    
                    # Extract hour and minute for filtering
                    df_transactions["hour"] = df_transactions["time_bangkok"].dt.hour
                    df_transactions["minute"] = df_transactions["time_bangkok"].dt.minute
                    df_transactions["time_minutes"] = df_transactions["hour"] * 60 + df_transactions["minute"]
                    
                    # Filter by time period
                    df_transactions = df_transactions[
                        (df_transactions["time_minutes"] >= start_time_minutes)
                        & (df_transactions["time_minutes"] <= end_time_minutes)
                    ]
                    
                    # Clean up temporary columns
                    df_transactions = df_transactions.drop(columns=["hour", "minute", "time_minutes"])

            st.markdown("---")
            # ...existing code...

            # Summary metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("📊 Total Vehicles", len(df_transactions))
            with col_m2:
                total_revenue = df_transactions["total_fee"].sum()
                st.metric("💰 Total Revenue", f"{total_revenue:.2f} ฿")
            with col_m3:
                avg_confidence = df_transactions["confidence"].mean()
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}%")

            st.markdown("---")

            # Display transactions table
            st.markdown(f"#### 📋 Records Found: {len(df_transactions)}")
            
            # Prepare display dataframe
            for idx, (_, row) in enumerate(df_transactions.iterrows(), 1):
                with st.expander(
                    f"🚗 Transaction #{idx} | {row['camera_id']} | {row['time_bangkok'].strftime('%H:%M:%S')} | {translate_class_name(row['class_name'])}",
                    expanded=False
                ):
                    col_img, col_info = st.columns([1, 1], gap="large")
                    
                    # Image column
                    with col_img:
                        st.markdown("**📸 Vehicle Image**")
                        if row["img_path"] and row["img_path"] != "":
                            image = get_image_from_minio(row["img_path"])
                            if image:
                                st.image(image, use_container_width=True)
                            else:
                                st.warning("⚠️ Image not available")
                        else:
                            st.info("📭 No image recorded")
                    
                    # Info column
                    with col_info:
                        st.markdown("**📋 Transaction Details**")
                        st.write(f"**⏰ Time:** {row['time_bangkok'].strftime('%H:%M:%S')}")
                        st.write(f"**📷 Camera ID:** {row['camera_id']}")
                        st.write(f"**🔖 Track ID:** {row['track_id']}")
                        st.write(f"**🚗 Vehicle Type:** {translate_class_name(row['class_name'])}")
                        st.write(f"**💰 Total Fee:** {row['total_fee']:.2f} ฿")
                        st.write(f"**🎯 Confidence:** {row['confidence']:.2f}%")
                        st.divider()
                        st.write(f"**Transaction ID:** `{row['id']}`")
        else:
            st.markdown(
                """
            <div style="text-align: center; padding: 4rem 2rem;
                 background: rgba(102, 126, 234, 0.05); border-radius: 20px;
                 border: 2px dashed rgba(102, 126, 234, 0.3);">
                <div style="font-size: 5em; opacity: 0.5;">📜📭</div>
                <div style="color: #667eea; font-size: 1.8em; font-weight: 700;">
                    No Transactions Yet
                </div>
                <div style="color: #999; font-size: 1.1em;">
                    No vehicle transactions recorded for today.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

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
        # แปลงชื่อสำหรับแสดงผล
        df_display = df_classes.copy()
        df_display["class_name"] = df_display["class_name"].apply(translate_class_name)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "class_id": "ID",
                "class_name": "Vehicle Type",
                "entry_fee": st.column_config.NumberColumn(
                    "Entry Fee (฿)", format="%.0f ฿"
                ),
                "xray_fee": st.column_config.NumberColumn(
                    "X-Ray Fee (฿)", format="%.0f ฿"
                ),
                "total_fee": st.column_config.NumberColumn(
                    "Total Fee (฿)", format="%.0f ฿"
                ),
            },
        )
    else:
        st.info("📭 No vehicle classes defined yet")


# ==================== DASHBOARD TAB ====================
def render_dashboard_tab() -> None:
    """Render dashboard with overview statistics"""
    st.markdown("### 📊 Dashboard Overview")

    now_thailand = get_thailand_time()
    today = now_thailand.date()

    try:
        # Get today's data
        query_today = """
            SELECT
                t.id,
                t.camera_id,
                t.class_id,
                t.total_fee,
                t.time_stamp,
                c.class_name
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.time_stamp) = :today
        """
        df_today = pd.read_sql(text(query_today), engine, params={"today": today})

        # Get this month's data
        first_day_month = today.replace(day=1)
        query_month = """
            SELECT
                t.id,
                t.total_fee,
                c.class_name
            FROM vehicle_transactions t
            JOIN vehicle_classes c ON t.class_id = c.class_id
            WHERE DATE(t.time_stamp) >= :first_day
        """
        df_month = pd.read_sql(
            text(query_month), engine, params={"first_day": first_day_month}
        )

        # Display metrics
        st.markdown("#### 📅 Today's Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🚗 Total Vehicles", len(df_today))
        with col2:
            st.metric(
                "💰 Total Revenue",
                f"{df_today['total_fee'].sum():.0f} ฿" if not df_today.empty else "0 ฿",
            )
        with col3:
            st.metric(
                "📷 Cameras",
                df_today["camera_id"].nunique() if not df_today.empty else 0,
            )
        with col4:
            latest_time = (
                pd.to_datetime(df_today["time_stamp"].max())
                if not df_today.empty
                else None
            )
            if latest_time:
                formatted_time = convert_to_thailand_tz(latest_time).strftime(
                    "%H:%M:%S"
                )
                st.metric("🕐 Last Entry", formatted_time)
            else:
                st.metric("🕐 Last Entry", "N/A")

        st.markdown("---")

        # This month summary
        st.markdown("#### 📆 This Month's Summary")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric("🚗 Total Vehicles", len(df_month))
        with col_m2:
            st.metric(
                "💰 Total Revenue",
                f"{df_month['total_fee'].sum():.0f} ฿" if not df_month.empty else "0 ฿",
            )

        if not df_today.empty:
            st.markdown("---")
            st.markdown("#### 🚗 Today's Vehicle Distribution")

            # Prepare data for chart
            df_display = df_today.copy()
            df_display["class_name"] = df_display["class_name"].apply(
                translate_class_name
            )

            col_c1, col_c2 = st.columns(2)

            with col_c1:
                # Pie chart would be nice but streamlit doesn't have it, use bar chart
                vehicle_counts = df_display["class_name"].value_counts()
                st.bar_chart(vehicle_counts)

            with col_c2:
                # Top 5 vehicle types today
                st.markdown("**Top 5 Vehicle Types Today:**")
                for idx, (vtype, count) in enumerate(vehicle_counts.head(5).items(), 1):
                    st.write(f"{idx}. {vtype}: **{count}** คัน")
        else:
            st.info("📭 No data for today yet")

        # ==================== ANALYTICS SECTION ====================
        st.markdown("---")
        st.markdown("### 📊 Analytics")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("📅 Start Date", value=now_thailand.date())
        with col2:
            end_date = st.date_input("📅 End Date", value=now_thailand.date())

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
            text(query), engine, params={"start_date": start_date, "end_date": end_date}
        )

        if not df_analytics.empty:
            st.markdown("---")

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📊 Total Transactions", len(df_analytics))
            with col2:
                st.metric(
                    "💰 Total Revenue", f"{df_analytics['total_fee'].sum():.0f} ฿"
                )
            with col3:
                st.metric("📷 Cameras", df_analytics["camera_id"].nunique())

            st.markdown("---")

            # Charts
            col_c1, col_c2 = st.columns(2)

            # แปลงชื่อสำหรับแสดงใน chart
            df_analytics_display = df_analytics.copy()
            df_analytics_display["class_name"] = df_analytics_display[
                "class_name"
            ].apply(translate_class_name)

            with col_c1:
                st.markdown("#### 🚗 Transactions by Vehicle Type")
                vehicle_counts = df_analytics_display["class_name"].value_counts()
                st.bar_chart(vehicle_counts.to_frame("count"))

            with col_c2:
                st.markdown("#### 💰 Revenue by Vehicle Type")
                revenue_by_type = (
                    df_analytics_display.groupby("class_name")["total_fee"]
                    .sum()
                    .sort_values(ascending=False)
                )
                st.bar_chart(revenue_by_type.to_frame("revenue"))
        else:
            st.info(
                f"📭 No data found between {start_date.strftime('%d %B %Y')} and {end_date.strftime('%d %B %Y')}"
            )

    except Exception as e:
        st.error(f"❌ Error loading dashboard: {e}")
        print(f"❌ Error loading dashboard: {e}")


# ==================== MAIN APPLICATION ====================
def main() -> None:
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Vehicle Entry System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize database
    init_database()

    # Show connection status in sidebar
    with st.sidebar:
        st.markdown("### 🔌 System Status")
        status = check_system_status()

        # Database status
        if status["database"]:
            st.success(f"✅ PostgreSQL: {status['database_msg']}")
        else:
            st.error(f"❌ PostgreSQL: {status['database_msg']}")

        # MinIO status
        if status["minio"]:
            st.success(f"✅ MinIO: {status['minio_msg']}")
            if status["buckets"]:
                with st.expander("📦 Available Buckets"):
                    for bucket in status["buckets"]:
                        st.text(f"• {bucket}")
        else:
            st.warning(f"⚠️ MinIO: {status['minio_msg']}")

        st.markdown("---")
        st.markdown("**Database URL:**")
        st.code(
            DATABASE_URL.replace(os.getenv("POSTGRES_PASSWORD", "password"), "****"),
            language="text",
        )
        st.markdown("**MinIO Endpoint:**")
        st.code(MINIO_ENDPOINT, language="text")

    # Load vehicle classes
    df_classes = load_vehicle_classes()

    # Render header
    render_header()

    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠 Dashboard", "📜 History", "⚙️ Master Data", "📊 Superset"]
    )

    with tab1:
        render_dashboard_tab()

    with tab2:
        render_transaction_history()

    with tab3:
        render_master_data_tab(df_classes)

    with tab4:
        render_superset_tab()


if __name__ == "__main__":
    main()