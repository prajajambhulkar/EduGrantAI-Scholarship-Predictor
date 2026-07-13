import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(page_title="EduGrant AI", layout="wide", initial_sidebar_state="expanded")

# Professional Blue Theme CSS
st.markdown("""
    <style>
        .main-title { font-size:38px; font-weight:800; color:#1E3A8A; text-align:center; margin-top:-10px; }
        .subtitle { font-size:16px; color:#4B5563; text-align:center; margin-bottom:30px; }
        .section-header { font-size:22px; font-weight:700; color:#1E40AF; border-left: 5px solid #3B82F6; padding-left: 10px; margin-bottom: 20px; margin-top: 20px; }
        .metric-card { background-color:#FFFFFF; border:1px solid #E2E8F0; padding:15px; border-radius:10px; text-align:center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
        .metric-num { font-size:24px; font-weight:800; color:#1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ML MODELS ---
@st.cache_resource
def load_artifacts():
    try:
        with open('model.pkl', 'rb') as f:
            m = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            s = pickle.load(f)
        return m, s
    except:
        return None, None

model, scaler = load_artifacts()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏠 EduGrant AI")
page = st.sidebar.radio("Navigation Pages", ["🏠 Home", "🎓 Scholarship Eligibility", "📄 Required Documents", "💼 Career Guidance", "📚 Skill Recommendations", "ℹ About"])

# --- PAGE: HOME ---
if page == "🏠 Home":
    st.markdown('<div class="main-title">EduGrant AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Scholarship Eligibility & Education Assistance Platform</div>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1523050854058-8df90110c9f1?q=80&w=1200", use_container_width=True)

# --- PAGE: ELIGIBILITY ENGINE ---
elif page == "🎓 Scholarship Eligibility":
    st.markdown('<div class="main-title">Scholarship Eligibility Profiler</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">USER INPUTS</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input("Student Name", "John Doe")
        age = st.number_input("Age", 17, 35, 20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        state = st.selectbox("State", ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Texas", "California"])
        category = st.selectbox("Category", ["General", "OBC", "SC", "ST"])

    with col2:
        income = st.number_input("Annual Family Income", min_value=0, value=250000)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 8.5, step=0.1)
        p_10 = st.number_input("10th Percentage", 0.0, 100.0, 85.0)
        p_12 = st.number_input("12th Percentage", 0.0, 100.0, 80.0)

    with col3:
        course = st.selectbox("Current Course", ["B.Tech", "B.Sc", "B.Com", "Arts", "Medicine"])
        year = st.selectbox("Current Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
        minority = st.selectbox("Minority Status", ["No", "Yes"])
        disability = st.selectbox("Disability Status", ["No", "Yes"])
        location = st.selectbox("Rural or Urban", ["Urban", "Rural"])

    if st.button("Predict Scholarship Eligibility", type="primary"):
        st.markdown('<div class="section-header">DASHBOARD & AI OUTPUT</div>', unsafe_allow_html=True)
        
        # Format variables directly for the scaled model array
        min_val = 1 if minority == "Yes" else 0
        dis_val = 1 if disability == "Yes" else 0
        yr_val = int(year[0])
        
        if model and scaler:
            features = np.array([[age, income, cgpa, p_10, p_12, yr_val, min_val, dis_val]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0][1]
        else:
            # Fallback rules if artifacts are missing on disk
            prediction = 1 if (cgpa >= 7.5 and income <= 600000) else 0
            prob = 0.88 if prediction == 1 else 0.34

        status = "Eligible" if prediction == 1 else "Not Eligible"
        match_score = int(prob * 100)

        # Dashboard grid layouts
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f'<div class="metric-card"><div style="font-size:12px; color:#64748B;">Eligibility Meter</div><div class="metric-num" style="color:{"green" if prediction==1 else "red"};">{status}</div></div>', unsafe_allow_html=True)
        with d2:
            st.markdown(f'<div class="metric-card"><div style="font-size:12px; color:#64748B;">Scholarship Match Score</div><div class="metric-num">{match_score}%</div></div>', unsafe_allow_html=True)
        with d3:
            st.markdown(f'<div class="metric-card"><div style="font-size:12px; color:#64748B;">Income Analysis</div><div class="metric-num" style="font-size:18px;">{"Low Income Tier" if income <= 600000 else "Standard Tier"}</div></div>', unsafe_allow_html=True)
        with d4:
            st.markdown(f'<div class="metric-card"><div style="font-size:12px; color:#64748B;">CGPA Analysis</div><div class="metric-num" style="font-size:18px;">{"High Merit" if cgpa >= 8.0 else "Standard Merit"}</div></div>', unsafe_allow_html=True)

        # Gauge Chart Components
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = match_score, title = {'text': "Eligibility Probability Gauge"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1E3A8A"}, 'steps': [{'range': [0, 50], 'color': "#fee2e2"}, {'range': [50, 100], 'color': "#dcfce7"}]}
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">RECOMMENDATIONS</div>', unsafe_allow_html=True)
        if prediction == 1:
            st.success(f"### 🎉 Congratulations {name}!")
            st.markdown("""
            * **Recommended Government Scholarships:** National Merit Scholarship, State Post-Matric Allocation scheme.
            * **Recommended Private Scholarships:** Corporate Excellence Technical Grant Foundation.
            * **Required Documents:** Certified Grade Sheets, Authorized Income Transcripts, National Identity Card.
            * **Application Tips:** Ensure all tax-registry files are digitally signed prior to submission cycles.
            """)
        else:
            st.error("### ⚠️ Application Parameters Not Fully Met")
            st.markdown("""
            * **Reasons:** The combination of family asset metrics and CGPA thresholds is outside the active allocation criteria rules.
            * **How to Improve Eligibility:** Raise standard evaluation scores past the 8.5 benchmark line or target niche localization tracks.
            * **Alternative Scholarships:** Work-Study Assistance Programs, Low-Interest Educational Financing models.
            """)
            
        # Download data card proxy
        report_txt = f"EduGrant AI Diagnostic Audit Report\nDate: {datetime.now().strftime('%Y-%m-%d')}\nStudent: {name}\nStatus: {status}\nMatch Score: {match_score}%"
        st.download_button("Generate Report File", data=report_txt, file_name=f"{name}_evaluation.txt", mime="text/plain")

elif page == "📄 Required Documents":
    st.markdown("<div class='main-title'>Required Documents Vault</div>", unsafe_allow_html=True)
    st.write("1. Income Declaration Asset Certificate\n2. Primary and Higher Secondary Marksheets\n3. Active Institution Registration Transcripts")

elif page == "💼 Career Guidance":
    st.markdown("<div class='main-header'>AI Career Matrix Recommendations</div>", unsafe_allow_html=True)
    st.write("* Suggested Professional Trajectory: Technical Infrastructure Engineer / Data Systems Analyst")

elif page == "📚 Skill Recommendations":
    st.markdown("<div class='main-header'>Skill Gap Matrix Analysis & Free Portals</div>", unsafe_allow_html=True)
    st.write("* Gap Items: Enterprise Automation Scripting, Structured Query Language Management\n* Free Platforms: Kaggle Core Tracks, FreeCodeCamp Micro-degrees")

elif page == "ℹ About":
    st.markdown("<div class='main-header'>About Platform Architecture</div>", unsafe_allow_html=True)
    st.write("EduGrant AI optimizes modern academic fund provisioning loops through machine learning metrics workflows.")
