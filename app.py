import os
import random
import io
import sqlite3
import hashlib
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import types
import pypdf

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="HealthPulse AI | Clinical Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Database Initialization & Management (SQLite)
# ---------------------------------------------------------
DB_FILE = "healthpulse.db"

def init_db():
    """Initializes SQLite database tables for auth, profile, vitals, history, and reports."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')

    # Patient Profile Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            blood_type TEXT,
            allergies TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Vital Signs Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            heart_rate INTEGER,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            glucose INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Medical History Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            condition TEXT,
            category TEXT,
            status TEXT,
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Reports Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            upload_date TEXT,
            content TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# Authentication & Database Helper Functions
# ---------------------------------------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
        user_id = c.lastrowid
        # Seed default profile
        c.execute("INSERT INTO profiles (user_id, name, age, blood_type, allergies) VALUES (?, ?, ?, ?, ?)",
                  (user_id, username, 28, "O+", "None"))
        
        # Seed default sample vitals for initial render
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14, -1, -1)]
        sample_vitals = [
            (user_id, dates[0], 72, 120, 80, 95), (user_id, dates[1], 75, 122, 82, 98),
            (user_id, dates[2], 71, 118, 78, 92), (user_id, dates[3], 78, 124, 83, 105),
            (user_id, dates[4], 82, 128, 85, 110), (user_id, dates[5], 74, 121, 80, 96),
            (user_id, dates[6], 73, 119, 79, 94), (user_id, dates[7], 70, 117, 77, 91),
            (user_id, dates[8], 68, 120, 78, 95), (user_id, dates[9], 76, 125, 82, 102),
            (user_id, dates[10], 79, 122, 80, 99), (user_id, dates[11], 72, 118, 78, 93),
            (user_id, dates[12], 71, 120, 79, 96), (user_id, dates[13], 73, 121, 81, 97),
            (user_id, dates[14], 75, 119, 78, 95)
        ]
        c.executemany("INSERT INTO vitals (user_id, date, heart_rate, systolic_bp, diastolic_bp, glucose) VALUES (?, ?, ?, ?, ?, ?)", sample_vitals)

        conn.commit()
        conn.close()
        return True, "Account created successfully! Please log in."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[1] == hash_password(password):
        return row[0]
    return None

def load_user_data(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Load Profile
    c.execute("SELECT name, age, blood_type, allergies FROM profiles WHERE user_id = ?", (user_id,))
    prof = c.fetchone()
    st.session_state.patient_data = {
        "name": prof[0] if prof else "User",
        "age": prof[1] if prof else 25,
        "blood_type": prof[2] if prof else "O+",
        "allergies": prof[3] if prof else "None"
    }

    # Load Vitals
    df_vitals = pd.read_sql_query(
        "SELECT date AS Date, heart_rate AS `Heart Rate (BPM)`, systolic_bp AS `Systolic BP (mmHg)`, "
        "diastolic_bp AS `Diastolic BP (mmHg)`, glucose AS `Blood Glucose (mg/dL)` FROM vitals WHERE user_id = ? ORDER BY id ASC",
        conn, params=(user_id,)
    )
    st.session_state.vitals_data = df_vitals

    # Load History
    c.execute("SELECT date, condition, category, status, notes FROM history WHERE user_id = ? ORDER BY id DESC", (user_id,))
    hist_rows = c.fetchall()
    st.session_state.history_logs = [
        {"date": r[0], "condition": r[1], "category": r[2], "status": r[3], "notes": r[4]} for r in hist_rows
    ]

    # Load Reports
    c.execute("SELECT filename, upload_date, content FROM reports WHERE user_id = ? ORDER BY id DESC", (user_id,))
    rep_rows = c.fetchall()
    st.session_state.uploaded_reports = [
        {"filename": r[0], "upload_date": r[1], "content": r[2]} for r in rep_rows
    ]

    conn.close()

def save_profile_db(user_id, name, age, blood_type, allergies):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE profiles SET name = ?, age = ?, blood_type = ?, allergies = ? WHERE user_id = ?",
              (name, age, blood_type, allergies, user_id))
    conn.commit()
    conn.close()

def add_vital_db(user_id, log_date, bpm, sys_bp, dia_bp, glucose):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO vitals (user_id, date, heart_rate, systolic_bp, diastolic_bp, glucose) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, str(log_date), bpm, sys_bp, dia_bp, glucose))
    conn.commit()
    conn.close()

def add_history_db(user_id, rec_date, condition, category, status, notes):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO history (user_id, date, condition, category, status, notes) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, str(rec_date), condition, category, status, notes))
    conn.commit()
    conn.close()

def add_report_db(user_id, filename, upload_date, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO reports (user_id, filename, upload_date, content) VALUES (?, ?, ?, ?)",
              (user_id, filename, str(upload_date), content))
    conn.commit()
    conn.close()

def clear_logs_and_reports_db(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM reports WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# Custom Glassmorphism CSS Design System
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Background Canvas */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%) !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
    }

    /* Glass Header Banner */
    .glass-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .glass-header h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
    }

    .glass-header p {
        color: #94a3b8 !important;
        margin-top: 6px !important;
        margin-bottom: 0 !important;
        font-size: 1.05rem;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar Health Tip Card */
    .glass-tip-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 16px;
        margin-top: 20px;
    }

    /* Custom Input Fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        backdrop-filter: blur(8px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Glass Metric Stat Blocks */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 18px 20px !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4) !important;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-ongoing {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-resolved {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-periodic {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Glass Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.4) 0%, rgba(99, 102, 241, 0.4) 100%) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State & Auth Gate
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Render Authentication Interface if not logged in
if not st.session_state.authenticated:
    st.markdown("""
    <div class="glass-header" style="text-align: center; max-width: 500px; margin: 40px auto 20px auto;">
        <h1>🩺 HealthPulse AI</h1>
        <p>Clinical Insights & Health Portal Login</p>
    </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2, auth_col3 = st.columns([1, 1.2, 1])
    with auth_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            login_user_input = st.text_input("Username", key="login_user")
            login_pass_input = st.text_input("Password", type="password", key="login_pass")
            if st.button("Log In", use_container_width=True):
                uid = login_user(login_user_input, login_pass_input)
                if uid:
                    st.session_state.authenticated = True
                    st.session_state.user_id = uid
                    load_user_data(uid)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_register:
            reg_user_input = st.text_input("Choose Username", key="reg_user")
            reg_pass_input = st.text_input("Choose Password", type="password", key="reg_pass")
            if st.button("Register Account", use_container_width=True):
                if reg_user_input and reg_pass_input:
                    success, msg = register_user(reg_user_input, reg_pass_input)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.error("Please enter both username and password.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# Load state if authenticated and state variables missing
if "patient_data" not in st.session_state:
    load_user_data(st.session_state.user_id)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def generate_pdf_summary(patient_data, vitals_data, history_logs, uploaded_reports):
    """Generates a downloadable clinical summary PDF using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    # Custom Palette
    primary_color = colors.HexColor("#1e1b4b")
    accent_color = colors.HexColor("#0284c7")
    text_dark = colors.HexColor("#0f172a")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=primary_color,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    )
    
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=accent_color,
        spaceBefore=12,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9,
        textColor=text_dark,
        leading=12
    )

    elements = []

    # Title Banner
    elements.append(Paragraph("HealthPulse AI — Clinical Summary Report", title_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Confidential Medical Record", sub_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=12))

    # Patient Details Table
    elements.append(Paragraph("1. Patient Demographic Profile", heading_style))
    patient_table_data = [
        [Paragraph("<b>Full Name:</b>", body_style), Paragraph(patient_data['name'], body_style),
         Paragraph("<b>Age:</b>", body_style), Paragraph(str(patient_data['age']), body_style)],
        [Paragraph("<b>Blood Group:</b>", body_style), Paragraph(patient_data['blood_type'], body_style),
         Paragraph("<b>Known Allergies:</b>", body_style), Paragraph(patient_data['allergies'], body_style)]
    ]
    t_patient = Table(patient_table_data, colWidths=[110, 160, 110, 160])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_patient)

    # Latest Vitals Table
    elements.append(Paragraph("2. Recent Vital Signs Analytics", heading_style))
    latest_vitals = vitals_data.tail(3)
    vitals_table_data = [["Date", "Heart Rate (BPM)", "Systolic BP", "Diastolic BP", "Glucose (mg/dL)"]]
    for _, row in latest_vitals.iterrows():
        vitals_table_data.append([
            str(row['Date']), str(row['Heart Rate (BPM)']), str(row['Systolic BP (mmHg)']),
            str(row['Diastolic BP (mmHg)']), str(row['Blood Glucose (mg/dL)'])
        ])
    t_vitals = Table(vitals_table_data, colWidths=[100, 110, 110, 110, 110])
    t_vitals.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e0f2fe")),
        ('TEXTCOLOR', (0, 0), (-1, 0), accent_color),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_vitals)

    # Medical History Logs
    elements.append(Paragraph("3. Logged Medical History Events", heading_style))
    if history_logs:
        hist_table_data = [["Date", "Condition", "Category", "Status", "Clinical Notes"]]
        for log in history_logs:
            hist_table_data.append([
                Paragraph(log['date'], body_style),
                Paragraph(log['condition'], body_style),
                Paragraph(log['category'], body_style),
                Paragraph(log['status'], body_style),
                Paragraph(log['notes'], body_style)
            ])
        t_hist = Table(hist_table_data, colWidths=[70, 110, 90, 70, 200])
        t_hist.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(t_hist)
    else:
        elements.append(Paragraph("No active medical history logs recorded.", body_style))

    # Uploaded Reports Summary
    elements.append(Paragraph("4. Attached Medical Reports & Lab Files", heading_style))
    if uploaded_reports:
        for rep in uploaded_reports:
            elements.append(Paragraph(f"• <b>{rep['filename']}</b> (Uploaded: {rep['upload_date']})", body_style))
            elements.append(Paragraph(f"<i>Excerpt:</i> {rep['content'][:300]}...", body_style))
            elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("No uploaded medical reports on file.", body_style))

    # Medical Disclaimer Footer
    elements.append(Spacer(1, 15))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=8))
    disclaimer_text = (
        "<b>Disclaimer:</b> This document is automatically generated by HealthPulse AI for patient reference and "
        "physician communication. It does not constitute a formal diagnosis or medical prescription."
    )
    elements.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor("#64748b"))))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Healthcare Facilities Coordinates
if "facilities_map" not in st.session_state:
    st.session_state.facilities_map = pd.DataFrame({
        'lat': [12.9345, 12.9279, 12.9401, 12.9312],
        'lon': [77.6265, 77.6271, 77.6200, 77.6350],
        'name': ['City Care Hospital', 'Apollo Pharmacy 24/7', 'Greenview Urgent Care', 'Metro Diagnostics'],
        'type': ['Hospital', 'Pharmacy', 'Clinic', 'Diagnostics']
    })

# Daily Health Tips Bank
HEALTH_TIPS = [
    "💧 **Hydration:** Aim for 2.5L of water daily to maintain cognitive focus and blood volume equilibrium.",
    "🧘 **Stress Reduction:** Take 5 deep diaphragmatic breaths every 2 hours to activate parasympathetic recovery.",
    "🥗 **Nutritional Balance:** Incorporate green leafy veg with lunch to stabilize glycemic variability.",
    "😴 **Sleep Hygiene:** Limit blue-spectrum light 60 minutes before rest to preserve natural melatonin secretion."
]

if "daily_tip" not in st.session_state:
    st.session_state.daily_tip = random.choice(HEALTH_TIPS)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am **HealthPulse AI**. How can I assist you with your health logs, uploaded lab reports, or symptoms today?"}
    ]

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="glass-header">
    <h1>🩺 HealthPulse AI</h1>
    <p>Clinical Insights, Vital Analytics & Patient Portal</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 Patient Profile")
    
    with st.container():
        updated_name = st.text_input("Full Name", value=st.session_state.patient_data["name"])
        updated_age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.patient_data["age"])
        blood_opts = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
        blood_idx = blood_opts.index(st.session_state.patient_data["blood_type"]) if st.session_state.patient_data["blood_type"] in blood_opts else 4
        updated_blood = st.selectbox("Blood Group", blood_opts, index=blood_idx)
        updated_allergies = st.text_area("Known Allergies", value=st.session_state.patient_data["allergies"])

        if st.button("Save Profile"):
            st.session_state.patient_data = {
                "name": updated_name,
                "age": updated_age,
                "blood_type": updated_blood,
                "allergies": updated_allergies
            }
            save_profile_db(st.session_state.user_id, updated_name, updated_age, updated_blood, updated_allergies)
            st.success("Profile updated successfully!")

    # PDF Download Export Button
    st.markdown("---")
    st.markdown("### 📥 Export Clinical Report")
    pdf_buffer = generate_pdf_summary(
        st.session_state.patient_data,
        st.session_state.vitals_data,
        st.session_state.history_logs,
        st.session_state.uploaded_reports
    )
    st.download_button(
        label="📄 Download Doctor PDF Summary",
        data=pdf_buffer,
        file_name=f"HealthPulse_Summary_{st.session_state.patient_data['name'].replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # Daily Health Tip Card
    st.markdown(f"""
    <div class="glass-tip-card">
        <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 0.95rem;">💡 Daily Health Tip</h4>
        <p style="margin: 0; color: #cbd5e1; font-size: 0.88rem; line-height: 1.4;">{st.session_state.daily_tip}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔒 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.rerun()

# ---------------------------------------------------------
# Key Metrics Summary Bar
# ---------------------------------------------------------
latest_bpm = st.session_state.vitals_data["Heart Rate (BPM)"].iloc[-1] if not st.session_state.vitals_data.empty else 0
latest_sys = st.session_state.vitals_data["Systolic BP (mmHg)"].iloc[-1] if not st.session_state.vitals_data.empty else 0
latest_dia = st.session_state.vitals_data["Diastolic BP (mmHg)"].iloc[-1] if not st.session_state.vitals_data.empty else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Patient Name", st.session_state.patient_data["name"])
m2.metric("Latest Heart Rate", f"{latest_bpm} BPM", delta="-2 BPM" if latest_bpm < 77 else "+3 BPM")
m3.metric("Latest Blood Pressure", f"{latest_sys}/{latest_dia} mmHg")
m4.metric("Active Records", f"{len(st.session_state.history_logs)} Logs")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab_vitals, tab_history, tab_map, tab_ai = st.tabs([
    "📊 Vital Analytics", 
    "📜 Medical Records & Reports", 
    "🏥 Nearby Healthcare Map", 
    "💬 Clinical AI Assistant"
])

# =========================================================
# TAB 1: PLOTLY VITAL TRACKING ANALYTICS
# =========================================================
with tab_vitals:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_v1, col_v2 = st.columns([1, 3])
    
    with col_v1:
        st.subheader("⚙️ Analytics Controls")
        metric_choice = st.selectbox(
            "Select Vital Metric",
            ["Heart Rate (BPM)", "Blood Pressure (mmHg)", "Blood Glucose (mg/dL)"]
        )
        
        st.markdown("---")
        st.markdown("**➕ Add New Reading**")
        with st.form("log_vital_form", clear_on_submit=True):
            log_date = st.date_input("Date", value=date.today())
            bpm_val = st.number_input("Heart Rate (BPM)", value=72, min_value=40, max_value=200)
            sys_val = st.number_input("Systolic BP", value=120, min_value=70, max_value=220)
            dia_val = st.number_input("Diastolic BP", value=80, min_value=40, max_value=140)
            glu_val = st.number_input("Glucose (mg/dL)", value=95, min_value=50, max_value=400)
            
            if st.form_submit_button("Log Reading"):
                add_vital_db(st.session_state.user_id, log_date, bpm_val, sys_val, dia_val, glu_val)
                load_user_data(st.session_state.user_id)
                st.success("Vital recorded!")
                st.rerun()

    with col_v2:
        st.subheader(f"📈 Trend Analysis: {metric_choice}")
        df = st.session_state.vitals_data

        fig = go.Figure()

        if not df.empty:
            if metric_choice == "Heart Rate (BPM)":
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["Heart Rate (BPM)"],
                    mode='lines+markers', name='Heart Rate',
                    line=dict(color='#38bdf8', width=3, shape='spline'),
                    marker=dict(size=8, color='#38bdf8'),
                    fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.08)'
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(52, 211, 153, 0.5)", annotation_text="Target Rest Rate (70 BPM)")

            elif metric_choice == "Blood Pressure (mmHg)":
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["Systolic BP (mmHg)"],
                    mode='lines+markers', name='Systolic',
                    line=dict(color='#f43f5e', width=3, shape='spline'),
                    marker=dict(size=8, color='#f43f5e')
                ))
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["Diastolic BP (mmHg)"],
                    mode='lines+markers', name='Diastolic',
                    line=dict(color='#818cf8', width=3, shape='spline'),
                    marker=dict(size=8, color='#818cf8')
                ))
                fig.add_hline(y=120, line_dash="dash", line_color="rgba(244, 63, 94, 0.4)", annotation_text="Normal Systolic Limit (120)")

            else:
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["Blood Glucose (mg/dL)"],
                    mode='lines+markers', name='Glucose',
                    line=dict(color='#fbbf24', width=3, shape='spline'),
                    marker=dict(size=8, color='#fbbf24'),
                    fill='tozeroy', fillcolor='rgba(251, 191, 36, 0.08)'
                ))
                fig.add_hline(y=100, line_dash="dash", line_color="rgba(251, 191, 36, 0.5)", annotation_text="Fasting Limit (100 mg/dL)")

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickfont=dict(color='#94a3b8'), zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickfont=dict(color='#94a3b8'), zeroline=False)
        )

        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2: MEDICAL RECORDS & REPORT UPLOAD
# =========================================================
with tab_history:
    col_input, col_view = st.columns([1, 1.2])

    with col_input:
        # Event Logger
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("➕ Log Medical Event")
        with st.form("add_event_form", clear_on_submit=True):
            rec_date = st.date_input("Event Date", value=date.today())
            rec_condition = st.text_input("Medical Condition / Diagnosis")
            rec_category = st.selectbox("Category", ["Infectious Disease", "Allergy", "Chronic", "Injury", "General Checkup"])
            rec_status = st.selectbox("Status", ["Ongoing", "Resolved", "Periodic"])
            rec_notes = st.text_area("Clinical Notes & Symptoms")

            if st.form_submit_button("Record Event"):
                if rec_condition:
                    add_history_db(st.session_state.user_id, rec_date, rec_condition, rec_category, rec_status, rec_notes)
                    load_user_data(st.session_state.user_id)
                    st.success("Record committed to history!")
                    st.rerun()
                else:
                    st.error("Please provide a condition title.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Upload Medical Report Section (RAG Document Ingestion)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📄 Upload Medical Report (PDF/Text)")
        st.caption("Upload lab reports, prescriptions, or discharge summaries to feed into your AI Assistant.")

        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

        if uploaded_file is not None:
            extracted_text = ""
            if uploaded_file.type == "application/pdf":
                try:
                    pdf_reader = pypdf.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        extracted_text += page.extract_text() + "\n"
                except Exception as ex:
                    st.error(f"Failed to parse PDF: {ex}")
            else:
                extracted_text = str(uploaded_file.read(), "utf-8")

            if extracted_text.strip():
                if st.button("Process & Save Report"):
                    add_report_db(st.session_state.user_id, uploaded_file.name, date.today(), extracted_text.strip())
                    load_user_data(st.session_state.user_id)
                    st.success(f"Successfully processed '{uploaded_file.name}'!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_view:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📋 Patient Timeline & Attached Reports")
        
        # Display Logs
        st.markdown("#### 🗓️ Clinical Events")
        if not st.session_state.history_logs:
            st.info("No medical records logged yet.")
        else:
            for item in st.session_state.history_logs:
                status_class = "badge-ongoing" if item['status'] == 'Ongoing' else ("badge-resolved" if item['status'] == 'Resolved' else "badge-periodic")
                
                with st.expander(f"🗓️ {item['date']} - {item['condition']}"):
                    st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <span class="badge {status_class}">{item['status']}</span>
                            <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">Category: <b>{item['category']}</b></span>
                        </div>
                        <p style="color: #cbd5e1; font-size: 0.95rem;">{item['notes']}</p>
                    """, unsafe_allow_html=True)

        # Display Attached Reports
        st.markdown("---")
        st.markdown("#### 📑 Uploaded Lab Reports")
        if not st.session_state.uploaded_reports:
            st.info("No reports uploaded yet.")
        else:
            for idx, rep in enumerate(st.session_state.uploaded_reports):
                with st.expander(f"📄 {rep['filename']} ({rep['upload_date']})"):
                    st.text_area(f"Extracted Content #{idx+1}", value=rep['content'], height=150, disabled=True)

        if (st.session_state.history_logs or st.session_state.uploaded_reports) and st.button("Clear All Logs & Reports"):
            clear_logs_and_reports_db(st.session_state.user_id)
            load_user_data(st.session_state.user_id)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3: HEALTHCARE LOCATOR MAP
# =========================================================
with tab_map:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🗺️ Nearby Hospitals & Medical Emergency Centers")
    st.caption("Locate verified clinics, pharmacies, and emergency rooms in your local vicinity.")

    map_df = st.session_state.facilities_map
    
    # Interactive Map Rendering
    st.map(map_df, latitude='lat', longitude='lon', zoom=13)
    
    st.markdown("#### 🏥 Directory Overview")
    cols = st.columns(len(map_df))
    for idx, row in map_df.iterrows():
        with cols[idx]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px;">
                <h5 style="color: #38bdf8; margin: 0 0 6px 0;">{row['name']}</h5>
                <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">Type: <b>{row['type']}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 4: CLINICAL AI ASSISTANT (INTEGRATED GEMINI LLM)
# =========================================================
with tab_ai:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💬 AI Symptom & Record Assistant")
    
    # Render chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your medical history, lab reports, or symptoms..."):
        # Append user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve API key securely
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            ai_reply = (
                "⚠️ **API Key Missing**: Please set `GEMINI_API_KEY` in `.streamlit/secrets.toml` "
                "to enable dynamic AI responses."
            )
        else:
            try:
                # Initialize Google GenAI client
                client = genai.Client(api_key=api_key)

                # Fetch patient context dynamically from session state
                p_name = st.session_state.patient_data['name']
                p_age = st.session_state.patient_data['age']
                p_allergies = st.session_state.patient_data['allergies']
                
                latest_sys = st.session_state.vitals_data["Systolic BP (mmHg)"].iloc[-1] if not st.session_state.vitals_data.empty else "N/A"
                latest_dia = st.session_state.vitals_data["Diastolic BP (mmHg)"].iloc[-1] if not st.session_state.vitals_data.empty else "N/A"
                latest_bpm = st.session_state.vitals_data["Heart Rate (BPM)"].iloc[-1] if not st.session_state.vitals_data.empty else "N/A"

                # Format uploaded medical reports for system prompt context injection
                reports_context = ""
                if st.session_state.uploaded_reports:
                    reports_context = "\nAttached Uploaded Medical Reports:\n"
                    for rep in st.session_state.uploaded_reports:
                        reports_context += f"--- Report: {rep['filename']} ({rep['upload_date']}) ---\n{rep['content'][:1500]}\n\n"

                # System instruction containing live patient profile + uploaded report context
                system_instruction = (
                    f"You are HealthPulse AI, a clinical medical assistant.\n"
                    f"Patient Profile Context:\n"
                    f"- Name: {p_name}, Age: {p_age}\n"
                    f"- Known Allergies: {p_allergies}\n"
                    f"- Recent Vitals: BP {latest_sys}/{latest_dia} mmHg, Heart Rate {latest_bpm} BPM\n"
                    f"{reports_context}\n"
                    f"Guidelines:\n"
                    f"1. Provide clear, empathetic, and evidence-based health guidance.\n"
                    f"2. Reference details from the attached medical reports whenever relevant to the user's question.\n"
                    f"3. ALWAYS check for potential interactions or alerts related to the patient's known allergies ({p_allergies}).\n"
                    f"4. Include appropriate medical disclaimers advising consults with a licensed physician."
                )

                # Request dynamic response from Gemini
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3
                    )
                )
                ai_reply = response.text

            except Exception as e:
                ai_reply = f"❌ Error communicating with AI model: `{str(e)}`"

        # Display AI response
        st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})
        with st.chat_message("assistant"):
            st.markdown(ai_reply)

    st.markdown('</div>', unsafe_allow_html=True)