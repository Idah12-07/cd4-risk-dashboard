
import streamlit as st
import pandas as pd
import joblib

st.markdown("<h1 style='text-align: center; color: teal;'>🧠 CD4 Risk Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 16px;'>
Predict immunological suppression in HIV patients using proxy features — even when CD4 lab results are missing.
</p>
""", unsafe_allow_html=True)

model = joblib.load('cd4_risk_classifier.pkl')

st.sidebar.header("📋 Patient Details")
age = st.sidebar.number_input("🎂 Age at reporting", min_value=0, max_value=120)
sex = st.sidebar.selectbox("⚧️ Sex", ["Male", "Female"])
regimen = st.sidebar.text_input("💊 Current Regimen")
who_stage = st.sidebar.selectbox("🩺 WHO Stage", [1, 2, 3, 4])
vl_result = st.sidebar.number_input("🧪 Last VL Result", min_value=0)
tb_status = st.sidebar.selectbox("🦠 Active in TB", ["Yes", "No"])
months_rx = st.sidebar.number_input("📆 Months of Prescription", min_value=0)
cd4_missing = st.sidebar.selectbox("❓ CD4 Missing", ["Yes", "No"])

input_df = pd.DataFrame({
    'Age at reporting': [age],
    'Sex': [1 if sex == "Male" else 0],
    'Current Regimen': [hash(regimen) % 1000],
    'Last WHO Stage': [who_stage],
    'Last VL Result': [vl_result],
    'Active in TB': [1 if tb_status == "Yes" else 0],
    'Months of Prescription': [months_rx],
    'CD4_Missing': [1 if cd4_missing == "Yes" else 0],
    'First Regimen': [0],  # Placeholder
    'Differentiated care model': [0]  # Placeholder
})

if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.markdown("<h3 style='color: red;'>⚠️ Immunological Risk: Suppressed (CD4 < 200)</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>✅ Immunological Risk: Safe</h3>", unsafe_allow_html=True)

with st.expander("ℹ️ About This Model"):
    st.markdown("""
    - **Model Type**: Random Forest Classifier  
    - **Accuracy**: 83.6%  
    - **Trained on**: ART patient data with proxy features  
    - **Balanced using**: SMOTE resampling  
    - **Purpose**: Flag suppressed patients when CD4 is missing  
    """)
