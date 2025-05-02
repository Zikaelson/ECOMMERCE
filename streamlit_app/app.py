# streamlit_app/app.py

import streamlit as st
import pandas as pd
import mlflow

# Set page config
st.set_page_config(page_title="💸 Predict Customer Spend", layout="centered")

# Set tracking URI to the local mlruns path inside Docker
mlflow.set_tracking_uri("file:/app/mlruns")

# Load MLflow production model
model = mlflow.sklearn.load_model("models:/ecommerce_best_model/Production")

# UI Title
st.title("🛍️ Ecommerce Customer Spend Predictor")
st.caption("Enter customer data to estimate their yearly spending.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    session_length = st.number_input("Avg. Session Length (min)", min_value=0.0, value=34.5, step=0.1)
    website_time = st.number_input("Time on Website (min)", min_value=0.0, value=15.1, step=0.1)

with col2:
    app_time = st.number_input("Time on App (min)", min_value=0.0, value=12.3, step=0.1)
    membership_length = st.number_input("Length of Membership (years)", min_value=0.0, value=5.2, step=0.1)

# Predict button
if st.button("🎯 Predict Yearly Spend"):
    input_data = pd.DataFrame([{
        "Avg. Session Length": session_length,
        "Time on App": app_time,
        "Time on Website": website_time,
        "Length of Membership": membership_length
    }])

    prediction = model.predict(input_data)[0]
    
    st.success(f"💰 Estimated Yearly Amount Spent: **${prediction:,.2f}**")

    st.caption("This estimate is based on the customer’s digital behavior and membership history.")

