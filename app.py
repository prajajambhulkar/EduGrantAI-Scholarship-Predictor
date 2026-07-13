import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fpdf import FPDF

# Load the trained model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: model.pkl or scaler.pkl not found. Please run the training steps first.")
    st.stop()

# Set page config
st.set_page_config(page_title="EduGrant AI", page_icon="🎓", layout="wide")
st.title("EduGrant AI")
st.write("Intelligent Scholarship Eligibility & Education Assistance Platform")
