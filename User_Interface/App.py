"""
Truck Classification System
Developed by: Aik Pick Tle
Year: 2026
Description: ระบบจัดประเภทรถบรรทุกพร้อมแสดงสถิติและประวัติ
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import pytz
from streamlit.components.v1 import html as st_html
import psycopg2
from psycopg2.extras import RealDictCursor
# ฟังก์ชันเชื่อมต่อ Database (ดึงค่าจาก secrets.toml หรือ st.secrets)
def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["DB_HOST"],
        database=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        port=st.secrets["DB_PORT"]
    )

# ฟังก์ชันดึงข้อมูลประเภทรถจากฐานข้อมูล
def get_vehicle_types():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT class_id, class_name FROM vehicle_classes ORDER BY class_id;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {row[0]: row[1] for row in rows}

# ฟังก์ชันดึงข้อมูลสรุปรายวัน (สำหรับ Chart)
def get_daily_summary(selected_date, gate_id=None):
    conn = get_db_connection()
    # Query เพื่อนับจำนวนรถและยอดรวมแยกตามประเภท (class_id 0-11)
    # หมายเหตุ: ใน SQL ของคุณไม่มี Column Gate แนะนำให้กรองด้วย camera_id หรือเพิ่ม column gate
    query = """
        SELECT class_id, COUNT(*), SUM(total_applied_fee) 
        FROM toll_transactions 
        WHERE DATE(created_at) = %s 
    """
    params = [selected_date]
    if gate_id:
        query += " AND camera_id = %s "
        params.append(f"0{gate_id}") # สมมติ camera_id คือ '01', '02'
    
    query += " GROUP BY class_id"
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df
# ===========================
# Configuration
# ===========================
st.set_page_config(
    page_title="Truck Classification",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================
# Session State Initialization
# ===========================
if 'selected_gate' not in st.session_state:
    st.session_state.selected_gate = 1

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'summary'

if 'selected_date' not in st.session_state:
    st.session_state.selected_date = date.today()

# ===========================
# Constants
# ===========================
COLORS = {
    'background': '#d9d9d9',
    'button': '#c9c9c9',
    'button_back': '#808080',
    'button_active': '#ff4444',
    'truck_log': '#9e9e9e',
    'truck_log_header': '#b8b8b8',
    'chart_bg': '#c9c9c9',
    'footer': '#808080',
    'text_dark': '#404040',
    'text_black': '#000000',
}

VEHICLE_TYPES = {
    0: 'car', 1: 'other', 2: 'other_truck',
    3: 'pickup_truck', 4: 'truck_20_back', 5: 'truck_20_front',
    6: 'truck_20×2', 7: 'truck_40', 8: 'truck_roro',
    9: 'truck_tail', 10: 'motorcycle', 11: 'truck_head'
}

# ===========================
# Data (Mock Data - Replace with DB later)
# ===========================
GATE_DATA = {
    1: {
        'camera_id': '01',
        'truck_type': 'truck_20×2',
        'entry_fee': 100,
        'xray_fee': 500,
        'time': '12:43:06',
        'total': 600
    },
    2: {
        'camera_id': '02',
        'truck_type': 'truck_40',
        'entry_fee': 150,
        'xray_fee': 600,
        'time': '13:15:22',
        'total': 750
    },
    3: {
        'camera_id': '03',
        'truck_type': 'pickup_truck',
        'entry_fee': 80,
        'xray_fee': 400,
        'time': '14:28:45',
        'total': 480
    }
}

GATE_CHART_DATA = {
    1: {
        'values': [10000, 5000, 12000, 2000, 8000, 10000, 11000, 13000, 6000, 12000, 4000, 9000],
        'count': 100,
        'amount': 99900
    },
    2: {
        'values': [8000, 7000, 9000, 3000, 11000, 8500, 9500, 10000, 7500, 9000, 5000, 8000],
        'count': 95,
        'amount': 85500
    },
    3: {
        'values': [7000, 6000, 10000, 4000, 9000, 7500, 10500, 11000, 8000, 10500, 6000, 9500],
        'count': 88,
        'amount': 78000
    }
}

HISTORY_DATA = [
    {'gate': 1, 'camera_id': 1, 'type': 'truck_20×2', 'entry_fee': 100, 'xray_fee': 500, 'time': '12:43:06', 'date': '6/1/68', 'amount': 600},
    {'gate': 1, 'camera_id': 1, 'type': 'truck_20×2', 'entry_fee': 100, 'xray_fee': 500, 'time': '12:43:06', 'date': '7/1/68', 'amount': 600},
    {'gate': 1, 'camera_id': 1, 'type': 'truck_20×2', 'entry_fee': 100, 'xray_fee': 500, 'time': '12:43:06', 'date': '8/1/68', 'amount': 600},
    {'gate': 2, 'camera_id': 2, 'type': 'truck_40', 'entry_fee': 150, 'xray_fee': 600, 'time': '13:15:22', 'date': '6/1/68', 'amount': 750},
    {'gate': 3, 'camera_id': 3, 'type': 'pickup_truck', 'entry_fee': 80, 'xray_fee': 400, 'time': '14:28:45', 'date': '6/1/68', 'amount': 480}
]

# ===========================
# Styling Functions
# ===========================
def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown(f"""
    <style>
        .main {{
            background-color: {COLORS['background']};
        }}
        
        /* Button Styles */
        .stButton>button {{
            width: 100%;
            border-radius: 10px;
            height: 60px;
            font-size: 18px;
            font-weight: bold;
            background-color: {COLORS['button']};
            color: {COLORS['text_black']} !important;
            border: none;
        }}
        .stButton>button:hover,
        .stButton>button:active,
        .stButton>button:focus {{
            color: {COLORS['text_black']} !important;
        }}
        
        /* Back Button */
        button[kind="secondary"][data-testid="baseButton-secondary"] {{
            background-color: {COLORS['button_back']} !important;
            color: white !important;
            font-size: 24px !important;
            height: 50px !important;
            border-radius: 10px !important;
        }}
        
        /* Truck Log */
        .truck-log {{
            background-color: {COLORS['truck_log']};
            padding: 20px;
            border-radius: 15px;
            color: black;
            margin-bottom: 20px;
        }}
        .truck-log-header {{
            background-color: {COLORS['truck_log_header']};
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 15px;
            font-size: 24px;
            font-weight: bold;
        }}
        .info-item {{
            font-size: 18px;
            margin: 10px 0;
        }}
        .info-label {{
            font-weight: bold;
        }}
        
        /* Gate Buttons */
        div[data-testid="column"] button[kind="secondary"] {{
            border-radius: 50% !important;
            height: 50px !important;
            width: 50px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            border: none !important;
            padding: 0 !important;
            background-color: {COLORS['truck_log_header']} !important;
            color: black !important;
        }}
        div[data-testid="column"] button[kind="primary"] {{
            background-color: {COLORS['button_active']} !important;
            color: white !important;
            border-radius: 50% !important;
            height: 50px !important;
            width: 50px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            border: none !important;
        }}
        
        /* Footer */
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: {COLORS['footer']};
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            z-index: 999;
        }}
        .footer a {{
            color: #ffcc00;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Date Input */
        .stDateInput > div > div > input {{
            background-color: #3a3a3a !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 15px 20px !important;
            font-size: 18px !important;
            text-align: center !important;
        }}
    </style>
    """, unsafe_allow_html=True)

def render_footer():
    """Render footer"""
    st.markdown("""
    <div class="footer">
        Developed by <strong>Aik Pick Tle</strong> | © 2026 Truck Classification System | 
        Contact: <a href="mailto:AikPickTle@example.com">AikPickTle@example.com</a>
    </div>
    """, unsafe_allow_html=True)

def render_header():
    """Render header"""
    st.markdown("""
    <div style="display: flex; align-items: center; padding: 10px 20px; background-color: #d9d9d9;">
        <h1 style="margin: 0; font-size: 20px; font-weight: bold; color: #000000;">TRUCK CLASSIFICATION</h1>
    </div>
    """, unsafe_allow_html=True)

# ===========================
# Component Functions
# ===========================
def render_truck_log(gate_num):
    """Render truck log section"""
    data = GATE_DATA[gate_num]
    st.markdown(f"""
    <div class="truck-log">
        <div class="truck-log-header">Current Vehicle</div>
        <div class="info-item"><span class="info-label">Camera ID:</span> {data['camera_id']}</div>
        <div class="info-item"><span class="info-label">Type:</span> <span style="color: {COLORS['text_dark']};">{data['truck_type']}</span></div>
        <div class="info-item"><span class="info-label">Entry fee:</span> {data['entry_fee']}</div>
        <div class="info-item"><span class="info-label">Xray-fee:</span> {data['xray_fee']}</div>
        <div class="info-item"><span class="info-label">Time:</span> {data['time']}</div>
        <div class="info-item"><span class="info-label">Amount:</span> {data['total']}</div>
    </div>
    """, unsafe_allow_html=True)

def render_navigation_buttons():
    """Render History and Summary buttons"""
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    
    # History Button
    if st.button("  History", use_container_width=True, key="nav_history"):
        st.session_state.current_page = 'history'
        st.rerun()
    
    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # Summary Button
    if st.button(" Summary", use_container_width=True, key="nav_summary"):
        st.session_state.current_page = 'summary_all'
        st.rerun()

def render_realtime_clock():
    """Render real-time clock using HTML/JavaScript"""
    clock_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; 
                margin-bottom: 20px; background-color: {COLORS['background']}; 
                padding: 10px 20px; border-radius: 10px;">
        <div style="width: 100%;">
            <h3 id="realtime-clock" style="font-family: 'Source Sans Pro', sans-serif; 
                                           font-weight: 600; color: rgb(0, 0, 0); margin: 0;">
                Summary today: <span id="date-display">--/--/----</span> time: <span id="time-display">--:--:--</span>
            </h3>
        </div>
    </div>
    
    <script>
    function updateClock() {{
        const now = new Date();
        const thailandTime = new Date(now.toLocaleString("en-US", {{timeZone: "Asia/Bangkok"}}));
        
        const day = String(thailandTime.getDate()).padStart(2, '0');
        const month = String(thailandTime.getMonth() + 1).padStart(2, '0');
        const year = thailandTime.getFullYear();
        
        const hours = String(thailandTime.getHours()).padStart(2, '0');
        const minutes = String(thailandTime.getMinutes()).padStart(2, '0');
        const seconds = String(thailandTime.getSeconds()).padStart(2, '0');
        
        const dateStr = day + '/' + month + '/' + year;
        const timeStr = hours + ':' + minutes + ':' + seconds;
        
        const dateEl = document.getElementById('date-display');
        const timeEl = document.getElementById('time-display');
        if (dateEl) dateEl.textContent = dateStr;
        if (timeEl) timeEl.textContent = timeStr;
    }}
    
    updateClock();
    setInterval(updateClock, 1000);
    </script>
    """
    st_html(clock_html, height=70)

def render_gate_buttons(key_prefix="gate"):
    """Render gate selection buttons with GATE label"""
    col_label, col1, col2, col3 = st.columns([1, 1, 1, 1])
    
    with col_label:
        st.markdown(
            '<div style="display: flex; align-items: center; justify-content: center; height: 50px;">'
            '<span style="font-weight: bold; font-size: 20px;">GATE</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col1:
        gate1_type = "primary" if st.session_state.selected_gate == 1 else "secondary"
        if st.button("1", key=f"{key_prefix}1", use_container_width=True, type=gate1_type):
            st.session_state.selected_gate = 1
            st.rerun()
    
    with col2:
        gate2_type = "primary" if st.session_state.selected_gate == 2 else "secondary"
        if st.button("2", key=f"{key_prefix}2", use_container_width=True, type=gate2_type):
            st.session_state.selected_gate = 2
            st.rerun()
    
    with col3:
        gate3_type = "primary" if st.session_state.selected_gate == 3 else "secondary"
        if st.button("3", key=f"{key_prefix}3", use_container_width=True, type=gate3_type):
            st.session_state.selected_gate = 3
            st.rerun()

def render_bar_chart(values, max_range=15000):
    """Render bar chart using Plotly"""
    categories = list(range(12))
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color='black',
            text=[f'{v:,.0f}' if v > 0 else '' for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        plot_bgcolor=COLORS['chart_bg'],
        paper_bgcolor=COLORS['chart_bg'],
        xaxis=dict(
            title='ประเภทรถ (Vehicle Type)',
            tickmode='linear',
            tick0=0,
            dtick=1,
            gridcolor='#b0b0b0'
        ),
        yaxis=dict(
            title='ยอดเงิน (Amount)',
            gridcolor='#b0b0b0',
            range=[0, max_range]
        ),
        margin=dict(l=50, r=50, t=20, b=50),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_vehicle_type_legend():
    """Render vehicle type legend"""
    st.markdown("---")
    st.markdown(f"<h4 style='color: {COLORS['text_dark']};'>Type</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    legend_groups = [
        [(0, 'car'), (1, 'other'), (2, 'other_truck')],
        [(3, 'pickup_truck'), (4, 'truck_20_back'), (5, 'truck_20_front')],
        [(6, 'truck_20×2'), (7, 'truck_40'), (8, 'truck_roro')],
        [(9, 'truck_tail'), (10, 'motorcycle'), (11, 'truck_head')]
    ]
    
    for col, group in zip([col1, col2, col3, col4], legend_groups):
        with col:
            for num, name in group:
                st.markdown(
                    f"<span style='color: {COLORS['text_black']};'><strong>{num}:</strong> {name}</span>",
                    unsafe_allow_html=True
                )

def render_stats(count, amount):
    """Render count and amount statistics"""
    col_count, col_amount = st.columns(2)
    with col_count:
        st.markdown(f"<h3 style='color: {COLORS['text_black']};'>Count: {count}</h3>", unsafe_allow_html=True)
    with col_amount:
        st.markdown(f"<h3 style='color: {COLORS['text_black']};'>Amount: {amount:,}</h3>", unsafe_allow_html=True)

# ===========================
# Page Rendering Functions
# ===========================
def render_summary_page():
    """Render main summary page"""
    render_realtime_clock()
    render_gate_buttons()
    
    # Get data for selected gate
    data = GATE_CHART_DATA[st.session_state.selected_gate]
    
    # Render chart
    render_bar_chart(data['values'])
    
    # Render stats
    render_stats(data['count'], data['amount'])
    
    # Render legend
    render_vehicle_type_legend()

def render_history_page():
    """Render history page"""
    st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)
    
    # Header row
    cols = st.columns([0.6, 1.5, 0.2, 0.8, 0.5, 0.5, 0.5, 0.2, 1.5])
    
    # Back button
    with cols[0]:
        if st.button("⬅", key="back_btn", help="Back to Summary", use_container_width=True):
            st.session_state.current_page = 'summary'
            st.rerun()
    
    # History title
    with cols[1]:
        st.markdown('<div style="display: flex; align-items: center; height: 60px;">'
                   '<h1 style="margin: 0; font-size: 38px; line-height: 60px;">🕐 History</h1>'
                   '</div>', unsafe_allow_html=True)
    
    # Gate label
    with cols[3]:
        st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 60px;">'
                   '<span style="font-size: 18px; font-weight: bold;">GATE</span>'
                   '</div>', unsafe_allow_html=True)
    
    # Gate buttons
    for i, col in enumerate(cols[4:7], 1):
        with col:
            gate_type = "primary" if st.session_state.selected_gate == i else "secondary"
            if st.button(str(i), key=f"gate{i}_hist", use_container_width=True, type=gate_type):
                st.session_state.selected_gate = i
                st.rerun()
    
    # Date picker
    with cols[8]:
        selected_date = st.date_input(
            "Select Date",
            value=st.session_state.selected_date,
            format="MM/DD/YYYY",
            label_visibility="collapsed",
            key="history_date_picker"
        )
        
        if selected_date != st.session_state.selected_date:
            st.session_state.selected_date = selected_date
            st.rerun()
    
    # History records
    st.markdown("---")
    history_data = [r for r in HISTORY_DATA if r['gate'] == st.session_state.selected_gate]
    
    if history_data:
        st.markdown(f'<div style="margin: 20px 0; color: #888; font-size: 16px;">'
                   f'Showing records for: {selected_date.strftime("%B %d, %Y")} | '
                   f'Gate {st.session_state.selected_gate}</div>', unsafe_allow_html=True)
        
        # Build history cards HTML
        cards_html = ""
        for record in history_data:
            cards_html += f"""
            <div style="background-color: #b8b8b8; padding: 20px; border-radius: 15px; margin-bottom: 15px;">
                <div style="display: flex; align-items: center;">
                    <div style="flex: 0 0 200px; background-color: #808080; padding: 20px; 
                               border-radius: 10px; text-align: center; color: white; 
                               font-size: 48px; font-weight: bold; margin-right: 20px;">
                        GATE<br>{record['gate']}
                    </div>
                    <div style="flex: 0 0 350px; background-color: #404040; padding: 10px; 
                               border-radius: 10px; margin-right: 20px; display: flex; 
                               align-items: center; justify-content: center; height: 200px;">
                        <div style="color: white; text-align: center;">
                            <div style="font-size: 60px;">▶</div>
                            <div style="font-size: 12px;">Video Placeholder</div>
                        </div>
                    </div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <p style="margin: 5px 0; font-size: 18px;"><strong>Camera ID:</strong> {record['camera_id']}</p>
                                <p style="margin: 5px 0; font-size: 18px;"><strong>Type:</strong> {record['type']}</p>
                                <p style="margin: 5px 0; font-size: 18px;"><strong>Entry fee:</strong> {record['entry_fee']}</p>
                                <p style="margin: 5px 0; font-size: 18px;"><strong>Xray fee:</strong> {record['xray_fee']}</p>
                            </div>
                            <div style="text-align: right;">
                                <p style="margin: 5px 0; font-size: 18px;"><strong>Time:</strong> {record['time']}</p>
                                <p style="margin: 5px 0; font-size: 18px;"><strong>Date:</strong> {record['date']}</p>
                                <p style="margin: 5px 0; font-size: 18px;"><strong>Amount:</strong> {record['amount']}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        # Scrollable container
        full_html = f"""
        <style>
            .scrollable-history {{
                max-height: 600px;
                overflow-y: auto;
                overflow-x: hidden;
                padding-right: 10px;
            }}
            .scrollable-history::-webkit-scrollbar {{
                width: 10px;
            }}
            .scrollable-history::-webkit-scrollbar-track {{
                background: #d9d9d9;
                border-radius: 5px;
            }}
            .scrollable-history::-webkit-scrollbar-thumb {{
                background: #808080;
                border-radius: 5px;
            }}
        </style>
        <div class="scrollable-history">{cards_html}</div>
        """
        
        st_html(full_html, height=650, scrolling=False)
    else:
        st.markdown('<div style="text-align: center; padding: 50px; color: #888; font-size: 20px;">'
                   'No records found for this gate</div>', unsafe_allow_html=True)

def render_summary_all_page():
    """Render summary all gates page"""
    st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)
    
    col_back, col_title = st.columns([0.8, 9])
    
    with col_back:
        if st.button("⬅", key="back_btn_summary", help="Back to Main", use_container_width=True):
            st.session_state.current_page = 'summary'
            st.rerun()
    
    with col_title:
        st.markdown('<div style="display: flex; align-items: center; height: 60px;">'
                   '<h1 style="margin: 0; font-size: 38px; line-height: 60px;">📋 Summary All</h1>'
                   '</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
    
    # Combine data from all gates
    all_values = [sum(GATE_CHART_DATA[g]['values'][i] for g in [1, 2, 3]) for i in range(12)]
    total_count = sum(GATE_CHART_DATA[g]['count'] for g in [1, 2, 3])
    total_amount = sum(GATE_CHART_DATA[g]['amount'] for g in [1, 2, 3])
    
    # Render chart
    render_bar_chart(all_values, max_range=35000)
    
    # Render stats
    st.markdown("---")
    render_stats(total_count, total_amount)
    
    # Render legend
    render_vehicle_type_legend()

# ===========================
# Main Application
# ===========================
def main():
    """Main application logic"""
    # Apply styles
    apply_custom_css()
    render_footer()
    render_header()
    
    # Layout
    left_col, right_col = st.columns([1, 3])
    
    # Left column - Truck Log & Navigation
    with left_col:
        render_truck_log(st.session_state.selected_gate)
        render_navigation_buttons()
    
    # Right column - Main content
    with right_col:
        if st.session_state.current_page == 'history':
            render_history_page()
        elif st.session_state.current_page == 'summary_all':
            render_summary_all_page()
        else:
            render_summary_page()

# Run the app
if __name__ == "__main__":
    main()