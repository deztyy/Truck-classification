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
HISTORY_PAGE_SIZE = 10

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
        /* ========== MODERN GLOBAL STYLES ========== */
        .stApp {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
            background-attachment: fixed;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
        }

        /* ========== HEADER STYLES ========== */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            padding: 2rem 2.5rem;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.15);
            animation: fadeInDown 0.6s ease;
        }

        .header-title {
            color: white;
            font-size: 2.5em;
            font-weight: 900;
            margin: 0;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
            line-height: 1.2;
            letter-spacing: -0.5px;
        }

        .datetime-box {
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(20px);
            padding: 1rem 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }

        .datetime-box:hover {
            background: rgba(255,255,255,0.15);
            border-color: rgba(255,255,255,0.35);
            transform: translateY(-2px);
        }

        .date-text {
            color: rgba(255,255,255,0.95);
            font-weight: 600;
            line-height: 1.6;
            font-size: 0.95rem;
        }

        .time-text {
            color: #ffd700;
            font-weight: 800;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            text-shadow: 0 2px 8px rgba(255,215,0,0.4);
            line-height: 1.6;
            font-size: 1.1rem;
            letter-spacing: 1px;
        }

        /* ========== NAVIGATION (RADIO) STYLES ========== */
        div[data-testid="stRadio"] {
            background: rgba(255,255,255,0.04);
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            margin: 1.5rem 0;
        }

        div[data-testid="stRadio"] > label {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        div[data-testid="stRadio"] input[type="radio"] {
            accent-color: #667eea;
        }

        /* ========== RESPONSIVE DESIGN ========== */
        @media (max-width: 768px) {
            .main-header {
                padding: 1.5rem;
            }

            .header-title {
                font-size: 1.6em;
                text-align: center;
                margin-bottom: 1.5rem;
            }

            .datetime-box {
                width: 100%;
                margin-top: 1rem;
            }
        }

        /* ========== CARD & EXPANDER STYLES ========== */
        div[data-testid="stExpander"] {
            background: rgba(102,126,234,0.08) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(102,126,234,0.25) !important;
            transition: all 0.3s ease;
        }

        div[data-testid="stExpander"]:hover {
            background: rgba(102,126,234,0.12) !important;
            border-color: rgba(102,126,234,0.4) !important;
            box-shadow: 0 8px 24px rgba(102,126,234,0.15);
        }

        /* ========== METRIC STYLES ========== */
        div[data-testid="stMetricContainer"] {
            background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(240,147,251,0.08) 100%) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            padding: 1.5rem !important;
            box-shadow: 0 4px 15px rgba(102,126,234,0.1);
            transition: all 0.3s ease;
        }

        div[data-testid="stMetricContainer"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(102,126,234,0.2);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.8em;
            font-weight: 800;
            color: #ffffff;
        }

        div[data-testid="stMetricLabel"] {
            color: rgba(255,255,255,0.8);
            font-weight: 600;
        }

        /* ========== BUTTON STYLES ========== */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.75rem 1.5rem !important;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            font-size: 0.95rem !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
        }

        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 8px 30px rgba(102,126,234,0.5) !important;
        }

        .stButton > button:active {
            transform: translateY(-1px) !important;
        }

        /* ========== FORM CONTAINER ========== */
        .form-container {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(102,126,234,0.05) 100%);
            padding: 2rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(102,126,234,0.08);
        }

        /* ========== INPUT & SELECT STYLES ========== */
        input, select, textarea {
            background: rgba(255,255,255,0.07) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 10px !important;
            color: white !important;
            transition: all 0.3s ease !important;
        }

        input:focus, select:focus, textarea:focus {
            background: rgba(255,255,255,0.1) !important;
            border-color: rgba(102,126,234,0.5) !important;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1) !important;
        }

        /* ========== HISTORY PANEL STYLES ========== */
        .history-panel {
            background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(240,147,251,0.1) 100%);
            border: 1px solid rgba(102,126,234,0.3);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 24px rgba(102,126,234,0.12);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .history-panel:hover {
            border-color: rgba(102,126,234,0.5);
            box-shadow: 0 12px 32px rgba(102,126,234,0.2);
        }

        .history-panel-title {
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            letter-spacing: -0.3px;
        }

        .history-panel-sub {
            color: rgba(255,255,255,0.7);
            font-size: 0.9rem;
            margin: 0;
            font-weight: 500;
        }

        .history-page-badge {
            background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(240,147,251,0.1) 100%);
            border: 2px solid rgba(102,126,234,0.3);
            border-radius: 14px;
            padding: 1rem 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .history-page-badge:hover {
            border-color: rgba(102,126,234,0.5);
            background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(240,147,251,0.15) 100%);
        }

        .history-page-text {
            color: #ffd700;
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.3px;
        }

        .history-page-sub {
            color: rgba(255,255,255,0.75);
            font-size: 0.85rem;
            margin-top: 0.3rem;
            font-weight: 500;
        }

        /* ========== ANIMATIONS ========== */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* ========== DIVIDER STYLES ========== */
        hr {
            border: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, rgba(255,255,255,0.2), transparent);
            margin: 2rem 0;
        }

        /* ========== TEXT STYLES ========== */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: -0.3px !important;
        }

        /* ========== SCROLLBAR STYLES ========== */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #667eea, #764ba2);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #764ba2, #667eea);
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
                <h1 class="header-title" style="margin: 0; text-align: left;">Vehicle Entry System</h1>
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
    st.markdown("### Current Vehicle")

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
                    Latest Vehicle Entry
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
                <div style="font-size: 5em; opacity: 0.5;"></div>
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
            timestamps = pd.to_datetime(df_all["time_stamp"], errors="coerce")
            if timestamps.dt.tz is None:
                # If DB returns naive timestamps, treat them as Bangkok local time
                df_all["time_bangkok"] = timestamps.dt.tz_localize(THAILAND_TZ)
            else:
                # If DB returns timezone-aware timestamps, convert to Bangkok
                df_all["time_bangkok"] = timestamps.dt.tz_convert(THAILAND_TZ)

            # Filter options
            st.markdown(
                """
                <div class="history-panel">
                    <div class="history-panel-title">🔍 Filters</div>
                    <p class="history-panel-sub">เลือกเงื่อนไขเพื่อค้นหา transaction ที่ต้องการ</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
                    "Select Vehicle Type",
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
            start_hour = None
            start_minute = None
            end_hour = None
            end_minute = None
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

                    # Extract hour and minute for filtering (avoid mutating slice)
                    time_minutes = (
                        df_transactions["time_bangkok"].dt.hour * 60
                        + df_transactions["time_bangkok"].dt.minute
                    )

                    # Filter by time period
                    df_transactions = df_transactions[
                        (time_minutes >= start_time_minutes)
                        & (time_minutes <= end_time_minutes)
                    ]

            # Keep current page when using Next/Previous; reset only when filters change
            current_filter_state = (
                selected_camera,
                selected_vehicle_type,
                search_track.strip() if search_track else "",
                time_filter_type,
                start_hour if time_filter_type == "Time Period" else None,
                start_minute if time_filter_type == "Time Period" else None,
                end_hour if time_filter_type == "Time Period" else None,
                end_minute if time_filter_type == "Time Period" else None,
            )

            if st.session_state.get("history_filter_state") != current_filter_state:
                st.session_state["history_page"] = 1
                st.session_state["history_filter_state"] = current_filter_state

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
            st.markdown(
                f"""
                <div class="history-panel">
                    <div class="history-panel-title">📋 Records Found: {len(df_transactions)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if df_transactions.empty:
                st.info("📭 No transactions match the selected filters")
                return

            # Pagination (10 records per page)
            total_records = len(df_transactions)
            total_pages = (total_records + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE

            if "history_page" not in st.session_state:
                st.session_state["history_page"] = 1
            if st.session_state["history_page"] > total_pages:
                st.session_state["history_page"] = total_pages
            if st.session_state["history_page"] < 1:
                st.session_state["history_page"] = 1

            def go_previous_page() -> None:
                st.session_state["history_page"] = max(1, st.session_state["history_page"] - 1)

            def go_next_page() -> None:
                st.session_state["history_page"] = min(total_pages, st.session_state["history_page"] + 1)

            page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
            with page_col1:
                st.button(
                    "⬅️ Previous",
                    disabled=st.session_state["history_page"] <= 1,
                    key="history_prev_btn",
                    use_container_width=True,
                    on_click=go_previous_page,
                )
            with page_col2:
                current_page = st.session_state["history_page"]
                st.markdown(
                    f"""
                    <div class="history-page-badge">
                        <p class="history-page-text">📄 Page {current_page} of {total_pages}</p>
                        <p class="history-page-sub">{HISTORY_PAGE_SIZE} records per page</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with page_col3:
                st.button(
                    "Next ➡️",
                    disabled=st.session_state["history_page"] >= total_pages,
                    key="history_next_btn",
                    use_container_width=True,
                    on_click=go_next_page,
                )

            start_idx = (current_page - 1) * HISTORY_PAGE_SIZE
            end_idx = start_idx + HISTORY_PAGE_SIZE
            df_page = df_transactions.iloc[start_idx:end_idx]

            st.caption(
                f"Showing records {start_idx + 1}-{min(end_idx, total_records)} of {total_records}"
            )
            
            # Prepare display dataframe
            for _, row in df_page.iterrows():
                with st.expander(
                    f"Transaction #{row['id']} | {row['camera_id']} | {row['time_bangkok'].strftime('%H:%M:%S')} | {translate_class_name(row['class_name'])}",
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
                        st.write(f"**Vehicle Type:** {translate_class_name(row['class_name'])}")
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
    
    st.markdown("""
    <style>
        .master-data-container {
            background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(240,147,251,0.08) 100%);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(102,126,234,0.25);
            margin-bottom: 2rem;
        }
        
        .master-data-card {
            background: rgba(255,255,255,0.05);
            border-radius: 14px;
            padding: 1.25rem;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }
        
        .master-data-card:hover {
            background: rgba(255,255,255,0.08);
            border-color: rgba(102,126,234,0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(102,126,234,0.1);
        }
        
        .fee-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.9rem;
            margin: 0.5rem 0.25rem 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    if not df_classes.empty:
        # Show total classes count
        col1, col2 = st.columns([3, 1])
        with col2:
            st.metric("Total Classes", len(df_classes), delta="Active")
        
        st.markdown("---")

        # แปลงชื่อสำหรับแสดงผล
        df_display = df_classes.copy()
        df_display["class_name"] = df_display["class_name"].apply(translate_class_name)

        # Display as cards instead of dataframe for better visuals
        for idx, row in df_display.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="master-data-card">
                    <div style="font-weight: 700; color: white; font-size: 1.05rem; margin-bottom: 0.5rem;">
                        {row['class_name']}
                    </div>
                    <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                        Class ID: <code>{row['class_id']}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="master-data-card">
                    <div style="font-weight: 700; color: #ffd700; font-size: 1.1rem;">฿ {row['entry_fee']:.0f}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.75rem;">Entry Fee</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="master-data-card">
                    <div style="font-weight: 700; color: #ffd700; font-size: 1.1rem;">฿ {row['xray_fee']:.0f}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.75rem;">X-Ray Fee</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="master-data-card" style="background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(240,147,251,0.1) 100%);">
                    <div style="font-weight: 800; color: #ffffff; font-size: 1.15rem;">฿ {row['total_fee']:.0f}</div>
                    <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; font-weight: 600;">Total</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show summary stats
        st.markdown("---")
        st.markdown("#### 📊 Summary Statistics")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            total_entry = df_display['entry_fee'].sum()
            st.metric("Total Entry Fees", f"฿ {total_entry:.0f}")
        
        with col_s2:
            total_xray = df_display['xray_fee'].sum()
            st.metric("Total X-Ray Fees", f"฿ {total_xray:.0f}")
        
        with col_s3:
            total_all = df_display['total_fee'].sum()
            st.metric("Total Revenue", f"฿ {total_all:.0f}")
    else:
        st.info("📭 No vehicle classes defined yet")


# ==================== DASHBOARD TAB ====================
def render_dashboard_tab() -> None:
    """Render dashboard with overview statistics"""
    st.markdown("### 📊 Dashboard Overview")
    
    # Add custom styles for dashboard
    st.markdown("""
    <style>
        .dashboard-section {
            background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(240,147,251,0.08) 100%);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(102,126,234,0.25);
            margin-bottom: 2rem;
        }
        
        .section-title {
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0 0 1rem 0;
            letter-spacing: -0.3px;
        }
    </style>
    """, unsafe_allow_html=True)

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

        # Display metrics in a nice section
        st.markdown('<div class="dashboard-section"><div class="section-title">📅 Today\'s Summary</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Vehicles", len(df_today))
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
        
        st.markdown('</div>', unsafe_allow_html=True)

        # This month summary
        st.markdown('<div class="dashboard-section"><div class="section-title">📆 This Month\'s Summary</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric("Total Vehicles", len(df_month))
        with col_m2:
            st.metric(
                "💰 Total Revenue",
                f"{df_month['total_fee'].sum():.0f} ฿" if not df_month.empty else "0 ฿",
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

        if not df_today.empty:
            st.markdown('<div class="dashboard-section"><div class="section-title">Today\'s Vehicle Distribution</div>', unsafe_allow_html=True)

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
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📭 No data for today yet")

        # ==================== ANALYTICS SECTION ====================
        st.markdown('<div class="dashboard-section"><div class="section-title">📊 Analytics</div>', unsafe_allow_html=True)

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

            st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)

            # Charts
            col_c1, col_c2 = st.columns(2)

            # แปลงชื่อสำหรับแสดงใน chart
            df_analytics_display = df_analytics.copy()
            df_analytics_display["class_name"] = df_analytics_display[
                "class_name"
            ].apply(translate_class_name)

            with col_c1:
                st.markdown("#### Transactions by Vehicle Type")
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
            
            st.markdown('</div>', unsafe_allow_html=True)
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

    # ========== STYLIZED TAB NAVIGATION ==========
    navigation_options = ["🏠 Dashboard", "📜 History", "⚙️ Master Data", "📊 Superset"]
    current_active = st.session_state.get("active_section", navigation_options[0])
    
    # Create custom styled navigation
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4, gap="medium")
    nav_cols = [nav_col1, nav_col2, nav_col3, nav_col4]
    
    st.markdown(
        """
        <style>
            .nav-button {
                display: inline-block;
                padding: 0.75rem 1.5rem;
                border-radius: 12px 12px 0 0;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.3s ease;
                border: none;
                font-size: 0.95rem;
                width: 100%;
                text-align: center;
            }
            
            .nav-button-active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                border-bottom: 3px solid #ffd700;
            }
            
            .nav-button-inactive {
                background: rgba(255,255,255,0.05);
                color: rgba(255,255,255,0.7);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .nav-button-inactive:hover {
                background: rgba(255,255,255,0.08);
                color: rgba(255,255,255,0.9);
            }
            
            .nav-container {
                display: flex;
                gap: 0.5rem;
                margin-bottom: 1rem;
                padding: 0 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    for idx, option in enumerate(navigation_options):
        with nav_cols[idx]:
            is_active = current_active == option
            button_class = "nav-button nav-button-active" if is_active else "nav-button nav-button-inactive"
            
            if st.button(option, key=f"nav_{idx}", use_container_width=True):
                st.session_state["active_section"] = option
                st.rerun()
    
    active_section = st.session_state.get("active_section", navigation_options[0])

    if active_section == "🏠 Dashboard":
        render_dashboard_tab()
    elif active_section == "📜 History":
        render_transaction_history()
    elif active_section == "⚙️ Master Data":
        render_master_data_tab(df_classes)
    elif active_section == "📊 Superset":
        render_superset_tab()


if __name__ == "__main__":
    main()