import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
from streamlit.components.v1 import html

# Page configuration
st.set_page_config(page_title="Truck Classification", layout="wide")

# Initialize session state for selected gate
if 'selected_gate' not in st.session_state:
    st.session_state.selected_gate = 1

# Initialize session state for page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'summary'  # Default to summary page

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #d9d9d9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        background-color: #c9c9c9;
        color: black;
        border: none;
    }
    /* Back button specific style */
    button[kind="secondary"][data-testid="baseButton-secondary"] {
        background-color: #808080 !important;
        color: white !important;
        font-size: 24px !important;
        height: 50px !important;
        border-radius: 10px !important;
    }
    .truck-log {
        background-color: #9e9e9e;
        padding: 20px;
        border-radius: 15px;
        color: black;
        margin-bottom: 20px;
    }
    .truck-log-header {
        background-color: #b8b8b8;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
        font-size: 24px;
        font-weight: bold;
    }
    .info-item {
        font-size: 18px;
        margin: 10px 0;
    }
    .info-label {
        font-weight: bold;
    }
    /* Gate button styles */
    div[data-testid="column"] button[kind="secondary"] {
        border-radius: 50% !important;
        height: 50px !important;
        width: 50px !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with home icon
col_home, col_title = st.columns([0.5, 9.5])
with col_home:
    st.markdown('<div style="font-size: 40px; margin-top: -10px;"></div>', unsafe_allow_html=True)
with col_title:
    st.markdown('<h1 style="margin-top: 0;">TRUCK CLASSIFICATION</h1>', unsafe_allow_html=True)

# Main layout
left_col, right_col = st.columns([1, 3])

with left_col:
    # Get current gate data for truck log
    gate_data = {
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
    
    current_truck_data = gate_data[st.session_state.selected_gate]
    
    # Truck Log Section
    st.markdown(f"""
    <div class="truck-log">
        <div class="truck-log-header">TRUCK LOG</div>
        <div class="info-item"><span class="info-label">Camera ID:</span> {current_truck_data['camera_id']}</div>
        <div class="info-item"><span class="info-label">Type:</span> {current_truck_data['truck_type']}</div>
        <div class="info-item"><span class="info-label">Entry fee:</span> {current_truck_data['entry_fee']}</div>
        <div class="info-item"><span class="info-label">Xray-fee:</span> {current_truck_data['xray_fee']}</div>
        <div class="info-item"><span class="info-label">Time:</span> {current_truck_data['time']}</div>
        <div class="info-item"><span class="info-label">Amount:</span> {current_truck_data['total']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # History Button
    if st.button("History", use_container_width=True):
        st.session_state.current_page = 'history'
        st.rerun()
    
    st.write("")
    
    # Summary Button
    if st.button("Summary", use_container_width=True):
        st.session_state.current_page = 'summary_all'
        st.rerun()

with right_col:
    # Check which page to display
    if st.session_state.current_page == 'history':
        # HISTORY PAGE
        # Add CSS for better styling
        st.markdown("""
        <style>
            /* Date input styling */
            .stDateInput > div > div > input {
                background-color: #3a3a3a !important;
                color: white !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 15px 20px !important;
                font-size: 18px !important;
                font-weight: normal !important;
                text-align: center !important;
            }
            /* Back button styling */
            button[data-testid="baseButton-secondary"]:has-text("⬅") {
                background-color: #2a2a2a !important;
                color: white !important;
                font-size: 24px !important;
                height: 60px !important;
                width: 60px !important;
                border-radius: 10px !important;
                border: none !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Single row header with all elements aligned
        st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)
        
        col_back, col_history, col_space1, col_gate_section, col_space2, col_date = st.columns([0.8, 2, 0.3, 4, 0.3, 2])
        
        with col_back:
            if st.button("⬅", key="back_btn", help="Back to Summary", use_container_width=True):
                st.session_state.current_page = 'summary'
                st.rerun()
        
        with col_history:
            st.markdown('''
                <div style="display: flex; align-items: center; height: 60px;">
                    <h1 style="margin: 0; font-size: 38px; line-height: 60px;">🕐 History</h1>
                </div>
            ''', unsafe_allow_html=True)
        
        with col_gate_section:
            # Create sub-columns for GATE label and buttons
            inner_cols = st.columns([1, 1, 1, 1])
            
            with inner_cols[0]:
                st.markdown('''
                    <div style="display: flex; align-items: center; justify-content: center; height: 60px;">
                        <span style="font-size: 18px; font-weight: bold;">GATE</span>
                    </div>
                ''', unsafe_allow_html=True)
            
            # Gate buttons
            with inner_cols[1]:
                gate1_type = "primary" if st.session_state.selected_gate == 1 else "secondary"
                if st.button("1", key="gate1_hist", use_container_width=True, type=gate1_type):
                    st.session_state.selected_gate = 1
                    st.rerun()
            
            with inner_cols[2]:
                gate2_type = "primary" if st.session_state.selected_gate == 2 else "secondary"
                if st.button("2", key="gate2_hist", use_container_width=True, type=gate2_type):
                    st.session_state.selected_gate = 2
                    st.rerun()
            
            with inner_cols[3]:
                gate3_type = "primary" if st.session_state.selected_gate == 3 else "secondary"
                if st.button("3", key="gate3_hist", use_container_width=True, type=gate3_type):
                    st.session_state.selected_gate = 3
                    st.rerun()
        
        with col_date:
            # Date picker
            from datetime import datetime, date
            if 'selected_date' not in st.session_state:
                st.session_state.selected_date = date.today()
            
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
        
        # Additional CSS for gate buttons
        st.markdown("""
        <style>
            div[data-testid="column"] button[kind="primary"] {
                background-color: #ff4444 !important;
                color: white !important;
                border-radius: 50% !important;
                height: 50px !important;
                width: 50px !important;
                font-size: 20px !important;
                font-weight: bold !important;
                border: none !important;
            }
            div[data-testid="column"] button[kind="secondary"] {
                background-color: #b8b8b8 !important;
                color: black !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # History records - filter by selected date if needed
        all_history_data = [
            {
                'gate': 1,
                'camera_id': 1,
                'type': 'truck_20×2',
                'entry_fee': 100,
                'xray_fee': 500,
                'time': '12:43:06',
                'date': '6/1/68',
                'amount': 600,
                'video': 'truck_video.mp4'
            },
            {
                'gate': 1,
                'camera_id': 1,
                'type': 'truck_20×2',
                'entry_fee': 100,
                'xray_fee': 500,
                'time': '12:43:06',
                'date': '7/1/68',
                'amount': 600,
                'video': 'truck_video.mp4'
            },
            {
                'gate': 1,
                'camera_id': 1,
                'type': 'truck_20×2',
                'entry_fee': 100,
                'xray_fee': 500,
                'time': '12:43:06',
                'date': '8/1/68',
                'amount': 600,
                'video': 'truck_video.mp4'
            },
            {
                'gate': 2,
                'camera_id': 2,
                'type': 'truck_40',
                'entry_fee': 150,
                'xray_fee': 600,
                'time': '13:15:22',
                'date': '6/1/68',
                'amount': 750,
                'video': 'truck_video.mp4'
            },
            {
                'gate': 3,
                'camera_id': 3,
                'type': 'pickup_truck',
                'entry_fee': 80,
                'xray_fee': 400,
                'time': '14:28:45',
                'date': '6/1/68',
                'amount': 480,
                'video': 'truck_video.mp4'
            }
        ]
        
        # Filter by selected gate
        history_data = [record for record in all_history_data if record['gate'] == st.session_state.selected_gate]
        
        # Display selected date info
        st.markdown(f'<div style="margin: 20px 0; color: #888; font-size: 16px;">Showing records for: {selected_date.strftime("%B %d, %Y")} | Gate {st.session_state.selected_gate}</div>', unsafe_allow_html=True)
        
        # Display history cards
        if len(history_data) == 0:
            st.markdown('<div style="text-align: center; padding: 50px; color: #888; font-size: 20px;">No records found for this gate</div>', unsafe_allow_html=True)
        else:
            # Create scrollable container with HTML component
            from streamlit.components.v1 import html as st_html
            
            # Build HTML for all cards
            cards_html = ""
            for record in history_data:
                cards_html += f"""
                <div style="background-color: #b8b8b8; padding: 20px; border-radius: 15px; margin-bottom: 15px; font-family: 'Source Sans Pro', sans-serif;">
                    <div style="display: flex; align-items: center;">
                        <div style="flex: 0 0 200px; background-color: #808080; padding: 20px; border-radius: 10px; text-align: center; color: white; font-size: 48px; font-weight: bold; margin-right: 20px; font-family: 'Source Sans Pro', sans-serif;">
                            GATE<br>{record['gate']}
                        </div>
                        <div style="flex: 0 0 350px; background-color: #404040; padding: 10px; border-radius: 10px; margin-right: 20px; display: flex; align-items: center; justify-content: center; height: 200px;">
                            <div style="color: white; text-align: center; font-family: 'Source Sans Pro', sans-serif;">
                                <div style="font-size: 60px;">▶</div>
                                <div style="font-size: 12px; margin-top: -10px;">Video Placeholder</div>
                            </div>
                        </div>
                        <div style="flex: 1;">
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <p style="margin: 5px 0; font-size: 18px; color: black; font-family: 'Source Sans Pro', sans-serif;"><strong>Camera ID:</strong> {record['camera_id']}</p>
                                    <p style="margin: 5px 0; font-size: 18px; color: black; font-family: 'Source Sans Pro', sans-serif;"><strong>Type:</strong> {record['type']}</p>
                                    <p style="margin: 5px 0; font-size: 18px; color: black; font-family: 'Source Sans Pro', sans-serif;"><strong>Entry fee:</strong> {record['entry_fee']}</p>
                                    <p style="margin: 5px 0; font-size: 18px; color: black; font-family: 'Source Sans Pro', sans-serif;"><strong>Xray fee:</strong> {record['xray_fee']}</p>
                                </div>
                                <div style="text-align: right;">
                                    <p style="margin: 5px 0; font-size: 18px; color: black; font-family: 'Source Sans Pro', sans-serif;"><strong>Time:</strong> {record['time']}</p>
                                    <p style="margin: 5px 0; font-size: 18px; color: black; font-family: 'Source Sans Pro', sans-serif;"><strong>Date:</strong> {record['date']}</p>
                                    <p style="margin: 5px 0; font-size: 18px; color: black; font-family: 'Source Sans Pro', sans-serif;"><strong>Amount:</strong> {record['amount']}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """
            
            # Complete HTML with scrollable container
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
                
                .scrollable-history::-webkit-scrollbar-thumb:hover {{
                    background: #606060;
                }}
            </style>
            <div class="scrollable-history">
                {cards_html}
            </div>
            """
            
            # Render with HTML component
            st_html(full_html, height=650, scrolling=False)
    
    elif st.session_state.current_page == 'summary_all':
        # SUMMARY ALL PAGE (รวมทุก Gate)
        # Header with back button
        st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)
        
        col_back, col_title = st.columns([0.8, 9])
        
        with col_back:
            if st.button("⬅", key="back_btn_summary", help="Back to Main", use_container_width=True):
                st.session_state.current_page = 'summary'
                st.rerun()
        
        with col_title:
            st.markdown('''
                <div style="display: flex; align-items: center; height: 60px;">
                    <h1 style="margin: 0; font-size: 38px; line-height: 60px;">📋 Summary All</h1>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
        
        # Combined data from all gates
        all_gates_data = {
            'values': [
                10000 + 8000 + 7000,  # Type 0: sum from all gates
                5000 + 7000 + 6000,   # Type 1
                12000 + 9000 + 10000, # Type 2
                2000 + 3000 + 4000,   # Type 3
                8000 + 11000 + 9000,  # Type 4
                10000 + 8500 + 7500,  # Type 5
                11000 + 9500 + 10500, # Type 6
                13000 + 10000 + 11000,# Type 7
                6000 + 7500 + 8000,   # Type 8
                12000 + 9000 + 10500, # Type 9
                4000 + 5000 + 6000,   # Type 10
                9000 + 8000 + 9500    # Type 11
            ],
            'count': 100 + 95 + 88,  # Total from all gates
            'amount': 99900 + 85500 + 78000  # Total amount from all gates
        }
        
        # Bar chart data
        categories = list(range(12))
        values = all_gates_data['values']
        
        # Create bar chart using Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color='black',
                text=values,
                textposition='none'
            )
        ])
        
        fig.update_layout(
            plot_bgcolor='#c9c9c9',
            paper_bgcolor='#c9c9c9',
            xaxis=dict(
                title='',
                tickmode='linear',
                tick0=0,
                dtick=1,
                gridcolor='#b0b0b0'
            ),
            yaxis=dict(
                title='',
                gridcolor='#b0b0b0',
                range=[0, 35000]  # Adjusted range for combined data
            ),
            margin=dict(l=50, r=50, t=20, b=50),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Count and Amount
        st.markdown("---")
        col_count, col_amount = st.columns(2)
        with col_count:
            st.markdown(f"### Count: {all_gates_data['count']}")
        with col_amount:
            st.markdown(f"### Amount: {all_gates_data['amount']:,}")
        
        # Vehicle type legend
        st.markdown("---")
        st.markdown("#### Type")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**0:** car")
            st.markdown("**1:** other")
            st.markdown("**2:** other_truck")
        
        with col2:
            st.markdown("**3:** pickup_truck")
            st.markdown("**4:** truck_20_back")
            st.markdown("**5:** truck_20_front")
        
        with col3:
            st.markdown("**6:** truck_20×2")
            st.markdown("**7:** truck_40")
            st.markdown("**8:** truck_roro")
        
        with col4:
            st.markdown("**9:** truck_tail")
            st.markdown("**10:** motorcycle")
            st.markdown("**11:** truck_head")
    
    else:
        # SUMMARY PAGE (original content)
        # Summary header with gate buttons
        col_summary, col_gates = st.columns([2, 1])
        with col_summary:
            # Create HTML component with real-time clock using JavaScript
            html_code = """
            <div id="clock-container">
                <h3 id="realtime-clock" style="font-family: 'Source Sans Pro', sans-serif; font-weight: 600; color: rgb(49, 51, 63); margin: 0;">
                    Summary today: <span id="date-display"></span> time: <span id="time-display"></span>
                </h3>
            </div>
            
            <script>
            function updateClock() {
                const now = new Date();
                
                // Convert to Thailand timezone (UTC+7)
                const thailandTime = new Date(now.toLocaleString("en-US", {timeZone: "Asia/Bangkok"}));
                
                const day = String(thailandTime.getDate()).padStart(2, '0');
                const month = String(thailandTime.getMonth() + 1).padStart(2, '0');
                const year = thailandTime.getFullYear();
                
                const hours = String(thailandTime.getHours()).padStart(2, '0');
                const minutes = String(thailandTime.getMinutes()).padStart(2, '0');
                const seconds = String(thailandTime.getSeconds()).padStart(2, '0');
                
                const dateStr = day + '/' + month + '/' + year;
                const timeStr = hours + ':' + minutes + ':' + seconds;
                
                document.getElementById('date-display').textContent = dateStr;
                document.getElementById('time-display').textContent = timeStr;
            }
            
            // Update clock immediately and then every second
            updateClock();
            setInterval(updateClock, 1000);
            </script>
            """
            html(html_code, height=50)
        
        with col_gates:
            st.markdown('<div style="text-align: right; margin-bottom: 10px;"><span style="font-weight: bold; margin-right: 10px;">GATE</span></div>', unsafe_allow_html=True)
            gate_col1, gate_col2, gate_col3 = st.columns(3)
            
            # Gate button 1
            with gate_col1:
                gate1_type = "primary" if st.session_state.selected_gate == 1 else "secondary"
                if st.button("1", key="gate1", use_container_width=True, type=gate1_type):
                    st.session_state.selected_gate = 1
                    st.rerun()
            
            # Gate button 2
            with gate_col2:
                gate2_type = "primary" if st.session_state.selected_gate == 2 else "secondary"
                if st.button("2", key="gate2", use_container_width=True, type=gate2_type):
                    st.session_state.selected_gate = 2
                    st.rerun()
            
            # Gate button 3
            with gate_col3:
                gate3_type = "primary" if st.session_state.selected_gate == 3 else "secondary"
                if st.button("3", key="gate3", use_container_width=True, type=gate3_type):
                    st.session_state.selected_gate = 3
                    st.rerun()
        
        # Additional CSS for primary buttons (selected gate)
        st.markdown("""
        <style>
            div[data-testid="column"] button[kind="primary"] {
                background-color: #ff4444 !important;
                color: white !important;
                border-radius: 50% !important;
                height: 50px !important;
                width: 50px !important;
                font-size: 20px !important;
                font-weight: bold !important;
                border: none !important;
            }
            div[data-testid="column"] button[kind="secondary"] {
                background-color: #b8b8b8 !important;
                color: black !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Data for each gate (ข้อมูลต่างกันสำหรับแต่ละเกท)
        gate_chart_data = {
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
        
        current_gate_data = gate_chart_data[st.session_state.selected_gate]
        
        # Bar chart data
        categories = list(range(12))
        values = current_gate_data['values']
        
        # Create bar chart using Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color='black',
                text=values,
                textposition='none'
            )
        ])
        
        fig.update_layout(
            plot_bgcolor='#c9c9c9',
            paper_bgcolor='#c9c9c9',
            xaxis=dict(
                title='',
                tickmode='linear',
                tick0=0,
                dtick=1,
                gridcolor='#b0b0b0'
            ),
            yaxis=dict(
                title='',
                gridcolor='#b0b0b0',
                range=[0, 15000]
            ),
            margin=dict(l=50, r=50, t=20, b=50),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Count and Amount
        col_count, col_amount = st.columns(2)
        with col_count:
            st.markdown(f"### Count: {current_gate_data['count']}")
        with col_amount:
            st.markdown(f"### Amount: {current_gate_data['amount']:,}")
        
        # Vehicle type legend
        st.markdown("---")
        st.markdown("#### Type")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**0:** car")
            st.markdown("**1:** other")
            st.markdown("**2:** other_truck")
        
        with col2:
            st.markdown("**3:** pickup_truck")
            st.markdown("**4:** truck_20_back")
            st.markdown("**5:** truck_20_front")
        
        with col3:
            st.markdown("**6:** truck_20×2")
            st.markdown("**7:** truck_40")
            st.markdown("**8:** truck_roro")
        
        with col4:
            st.markdown("**9:** truck_tail")
            st.markdown("**10:** motorcycle")
            st.markdown("**11:** truck_head")
            
        # End of summary page
        st.write("")
# import streamlit as st
# import plotly.express as px
# import pandas as pd

# # 1. ตั้งค่าหน้าจอ
# st.set_page_config(layout="wide", page_title="Truck Classification")

# # 2. Custom CSS เพื่อทำให้ UI เหมือนในแบบ (เช่น กล่องสีเทา, ขอบมน)
# st.markdown("""
#     <style>
#     .main { background-color: #f0f2f6; }
#     .stSidebar { background-color: #d1d1d1; }
#     .truck-card {
#         background-color: #707070;
#         padding: 15px;
#         border-radius: 10px;
#         color: white;
#         margin-bottom: 10px;
#         display: flex;
#         align-items: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # 3. ส่วน Sidebar (เหมือนในรูป 2 และ 3)
# with st.sidebar:
#     st.image("https://cdn-icons-png.flaticon.com/512/1995/1995471.png", width=40) # ไอคอนจำลอง
#     st.title("TRUCK LOG")
#     st.write("**Camera ID:** 01")
#     st.write("**Type:** truck_20x2")
#     st.write("**Entry fee:** 100")
#     st.write("**Xray-fee:** 500")
#     st.write("**Time:** 12:43:06")
#     st.subheader("Amount: 600")
#     st.divider()
#     st.button("📜 History", use_container_width=True)
#     st.button("📊 Summary", use_container_width=True)

# # 4. ส่วนเนื้อหาหลัก
# st.header("🏠 TRUCK CLASSIFICATION")

# # ส่วนเลือก Gate แบบ Radio แนวนอน (เหมือนปุ่ม 1 2 3 ด้านบน)
# gate = st.radio("GATE", [1, 2, 3], horizontal=True)

# tab1, tab2 = st.tabs(["History List", "Summary Chart"])

# with tab1:
#     # จำลองการสร้าง Card แบบในรูปที่ 1
#     for i in range(3):
#         st.markdown(f"""
#         <div class="truck-card">
#             <div style="flex: 1;"> <img src="https://via.placeholder.com/150x80" width="100%"> </div>
#             <div style="flex: 2; padding-left: 20px;">
#                 <b>GATE {gate}</b><br>
#                 Type: truck_20x2<br>
#                 Date: {i+6}/1/68
#             </div>
#             <div style="flex: 1; text-align: right;">
#                 Amount: 600
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

# with tab2:
#     st.subheader("📊 Summary Today")
#     # ข้อมูลจำลองสำหรับกราฟแท่ง
#     data = pd.DataFrame({
#         'Type': [f'Type {i}' for i in range(12)],
#         'Count': [10, 5, 18, 2, 12, 15, 18, 20, 10, 18, 4, 15]
#     })
#     fig = px.bar(data, x='Type', y='Count', color_discrete_sequence=['black'])
#     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(230,230,230,1)')
#     st.plotly_chart(fig, use_container_width=True)