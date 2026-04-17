import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from streamlit_lottie import st_lottie
import requests

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Student Financial Profiler", layout="wide")

# --- LOAD ASSETS ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Use this verified URL for a spinning data/tech sphere
lottie_sphere = load_lottieurl("https://lottie.host/8636f333-3330-4e33-875c-352b2f67644d/I9XGZl2Y1O.json")
# --- LOAD MODELS & ENCODERS ---
# Note: Ensure these files are in your 'models/' folder
@st.cache_resource
def load_artifacts():
    oe = joblib.load('models/ordinal_encoder.joblib')
    mms = joblib.load('models/scaler.joblib')
    kp = joblib.load('models/kproto_model.joblib')
    rf = joblib.load('models/risk_rf_model.joblib')
    nn = tf.keras.models.load_model('models/spend_nn_model.h5')
    return oe, mms, kp, rf, nn

# --- MAIN APP ---
def main():
    # 1. LANDING PAGE
    if 'survey_started' not in st.session_state:
        st.session_state.survey_started = False
    if lottie_sphere:
        st_lottie(lottie_sphere, height=400, key="initial_sphere")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=200) # Fallback if link fails
    if not st.session_state.survey_started:
        st.markdown("<h1 style='text-align: center;'>Student Financial Behavior & Risk Profiler</h1>", unsafe_allow_html=True)
        st_lottie(lottie_sphere, height=400, key="initial_sphere")
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Begin Financial Analysis", use_container_width=True):
                st.session_state.survey_started = True
                st.rerun()

    # 2. SURVEY & INPUT PAGE
    else:
        st.sidebar.header("Personal Details")
        name = st.sidebar.text_input("Full Name")
        age = st.sidebar.number_input("Age", min_value=16, max_value=30, value=20)
        gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

        st.header("Financial Habits Survey")
        
        # --- INPUT FORM ---
        with st.form("survey_form"):
            # Categorical Inputs [cite: 13, 15-17, 19]
            place = st.selectbox("Place you grew up in?", ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
            
            # Importance Grid [cite: 31, 35-40]
            st.write("How important are these factors when purchasing?")
            col_a, col_b = st.columns(2)
            price_imp = col_a.select_slider("Price/Cost", options=["Not important", "Slightly important", "Very important"])
            brand_imp = col_b.select_slider("Brand Reputation", options=["Not important", "Slightly important", "Very important"])
            peer_imp = col_a.select_slider("Peer Recommendation", options=["Not important", "Slightly important", "Very important"])
            util_imp = col_b.select_slider("Long-term Utility", options=["Not important", "Slightly important", "Very important"])

            # Behavioral Scales [cite: 20-25, 66-74, 90-97]
            unplanned = st.slider("Frequency of unplanned purchases (1-5)", 1, 5)
            confidence = st.slider("Financial Management Confidence (1-5)", 1, 5)
            peer_infl = st.slider("Peer pressure influence (1-5)", 1, 5)

            # Tracking & Graph [cite: 79-89, 137, 213-217]
            track = st.selectbox("How do you track expenses?", [
                "I check my bank balance occasionally.",
                "I review my history within payment apps (e.g., UPI, Paytm).",
                "I do not keep the track",
                "I use a dedicated expense-tracking app or spreadsheet."
            ])
            
            graph = st.selectbox("Which graph represents your monthly spending?", [
                "Uniform Daily Expenses", 
                "Irregular and Random Spending", 
                "Spend a lot once and then low spending for rest", 
                "Steady Weekdays with High Weekends"
            ])

            # Budget Categories [cite: 98-113]
            st.write("Select major budget categories:")
            b_food = st.checkbox("Food & Dining")
            b_travel = st.checkbox("Travel")
            b_fashion = st.checkbox("Fashion")
            b_sub = st.checkbox("Subscriptions")
            b_fun = st.checkbox("Fun & Entertainment")

            submit = st.form_submit_button("Analyze My Profile")

        if submit:
            # --- PREPROCESSING LOGIC ---
            # Replicating your Ordinal and Scaling logic [cite: 7]
            # (Note: You'll need to add your specific transformation code here 
            # based on the objects loaded in load_artifacts)
            
            with st.spinner('Calculating your financial DNA...'):
                # 1. Run Neural Network for Spend
                # 2. Feed Spend + Inputs into K-Proto for Cluster
                # 3. Feed all into Random Forest for Risk Score
                
                st.success(f"Analysis Complete for {name}!")
                
                # --- DASHBOARD DISPLAY ---
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Predicted Monthly Spend", "₹XXXX")
                res_col2.metric("Behavior Cluster", "Medium Spender")
                res_col3.metric("Risk Score", "Low Risk", delta="-5% Improvement")
                
                # Warning System [cite: 8]
                if "High Risk" in "Risk Level":
                    st.warning("⚠️ Warning: Your current habits suggest a high financial risk profile.")

if __name__ == "__main__":
    main()
