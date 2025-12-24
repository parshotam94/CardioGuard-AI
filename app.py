import streamlit as st
import pickle
import numpy as np

# 1. Load the saved model and scaler
# Note: Ensure these paths are correct for your folder structure
try:
    model = pickle.load(open('pkl_files/grid_pramas.pkl', 'rb'))
    scaler = pickle.load(open('pkl_files/scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or Scaler files not found. Please check the 'pkl_files' folder.")

# 2. Page Configuration
st.set_page_config(page_title="Heart Risk Triage", page_icon="❤️")
st.title("❤️ Heart Attack Risk Triage System")
st.markdown("""
This application uses a **Support Vector Machine (SVM)** to classify patient risk.
Please fill in the clinical data below.
""")

# --- DOCUMENTATION SECTION ---
with st.expander("ℹ️ Click here to understand the input values"):
    st.write("""
    - **CP (Chest Pain):** 0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic.
    - **Resting BP:** Normal is usually < 120 mm Hg.
    - **Cholesterol:** Normal is usually < 200 mg/dl.
    - **Oldpeak:** ST depression induced by exercise. Higher = Higher Risk.
    - **CA:** Number of major vessels (0-3) colored by fluoroscopy.
    - **Thal:** 1: Normal, 2: Fixed defect, 3: Reversable defect.
    """)

# 3. Layout with Columns
col1, col2 , col3, col4= st.columns(4)

with col1:
    st.header("Patient Bio")
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")

with col2:
    st.header("Patient Bio")
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "True" if x == 1 else "False")

with col3:
    st.header("Clinical Tests")
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col4:
    st.header("Clinical Tests")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("ST Slope", [0, 1, 2], help="Slope of peak exercise ST segment")
    ca = st.selectbox("Major Vessels (CA)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (Thal)", [1, 2, 3])
    

st.divider()

# 4. Prediction Logic
if st.button("Generate Triage Report", use_container_width=True):
    # Prepare data for model
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    features_scaled = scaler.transform(features)
    
    # Run Prediction
    prediction = model.predict(features_scaled)
    
    # Calculate Confidence using Distance from Hyperplane
    distance = model.decision_function(features_scaled)[0]
    
    st.subheader("Triage Assessment")
    
    if prediction[0] == 1:
        st.error("🚨 **RESULT: HIGH RISK**")
        st.write(f"The patient is deep in the high-risk zone (Distance: {distance:.2f}). Immediate cardiovascular screening is recommended.")
    else:
        st.success("✅ **RESULT: LOW RISK**")
        st.write(f"The patient is currently in the low-risk zone (Safety Margin: {abs(distance):.2f}).")

    # Try to show probability if enabled
    try:
        probs = model.predict_proba(features_scaled)
        conf_score = probs[0][1] * 100
        st.write(f"**Calculated Probability:** {conf_score:.1f}%")
        st.progress(int(conf_score))
    except:
        st.caption("Probability scores not enabled. Triage based on Hyperplane Distance.")