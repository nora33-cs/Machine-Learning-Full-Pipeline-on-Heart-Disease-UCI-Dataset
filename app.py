import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# --- Load trained model safely ---
# Get absolute path to this project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to the model file
model_path = os.path.join(BASE_DIR, "models", "final_model.pkl")

print("App running from:", os.getcwd())
print("Looking for model at:", model_path)

# Try loading the model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Could not find the trained model at: " + model_path)
    st.stop()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details to check the probability of heart disease.")

# Sidebar input form
st.sidebar.header("Patient Data Input")

def user_input_features():
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
    cp_4 = st.sidebar.selectbox("Chest Pain Type (cp_4.0)", [0, 1])
    thal_7 = st.sidebar.selectbox("Thal (thal_7.0)", [0, 1])
    age = st.sidebar.slider("Age", 20, 90, 50)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    slope_2 = st.sidebar.selectbox("Slope (slope_2.0)", [0, 1])

    # Make sure DataFrame matches training features exactly
    data = {
        "ca": ca,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "cp_4.0": cp_4,
        "thal_7.0": thal_7,
        "age": age,
        "chol": chol,
        "trestbps": trestbps,
        "exang": exang,
        "slope_2.0": slope_2,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Show inputs
st.subheader("🔹 Patient Data Entered")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]
st.subheader("🔮 Prediction Result")
if prediction == 1:
    st.error(f"⚠️ The model predicts **Heart Disease Risk** with probability {prediction_proba:.2f}")
else:
    st.success(f"✅ The model predicts **No Heart Disease Risk** with probability {1-prediction_proba:.2f}")

# Bonus: show dataset trends if available
st.subheader("📊 Example Heart Disease Trend (from dataset)")
try:
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "heart_clean_encoded.csv"))
    st.bar_chart(df["target"].value_counts())
except:
    st.info("Dataset not found for visualization.")
