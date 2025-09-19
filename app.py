
import streamlit as st
import pandas as pd
import joblib
model = joblib.load('cd4_risk_classifier.pkl')

st.markdown("<h1 style='text-align: center; color: teal;'>🧠 CD4 Risk Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 16px;'>
Predict immunological suppression in HIV patients using proxy features — even when CD4 lab results are missing.
</p>
""", unsafe_allow_html=True)

st.sidebar.header("📋 Patient Details")
age = st.sidebar.number_input("🎂 Age at reporting", min_value=0, max_value=120)
sex = st.sidebar.selectbox("⚧️ Sex", ["Male", "Female"])
first_regimen = st.sidebar.selectbox("🧬 First Regimen", ["TDF/3TC/EFV", "AZT/3TC/NVP", "ABC/3TC/EFV", "Other"])
regimen = st.sidebar.selectbox("💊 Current Regimen", ["TDF/3TC/DTG", "ABC/3TC/DTG", "AZT/3TC/ATV/r", "Other"])
care_model = st.sidebar.selectbox("🏥 Differentiated Care Model", ["Facility Fast Track", "Community ART Group", "Home Delivery", "Standard Facility", "Other"])
who_stage = st.sidebar.selectbox("🩺 WHO Stage", [1, 2, 3, 4])
vl_result = st.sidebar.number_input("🧪 Last VL Result", min_value=0)
tb_status = st.sidebar.selectbox("🦠 Active in TB", ["Yes", "No"])
months_rx = st.sidebar.number_input("📆 Months of Prescription", min_value=0)
cd4_missing = st.sidebar.selectbox("❓ CD4 Missing", ["Yes", "No"])

regimen_map = {"TDF/3TC/DTG": 0, "AZT/3TC/ATV/r": 1, "ABC/3TC/DTG": 2, "Other": 3}
first_regimen_map = {"TDF/3TC/EFV": 0, "AZT/3TC/NVP": 1, "ABC/3TC/EFV": 2, "Other": 3}
care_model_map = {"Facility Fast Track": 0, "Community ART Group": 1, "Home Delivery": 2, "Standard Facility": 3, "Other": 4}

# Sidebar already defines: first_regimen, regimen, care_model, age, sex, who_stage, vl_result, tb_status, months_rx, cd4_missing

first_regimen_map = {
    "TDF/3TC/EFV": 0, "AZT/3TC/NVP": 1, "ABC/3TC/EFV": 2, "Other": 3
}
regimen_map = {
    "TDF/3TC/DTG": 0, "ABC/3TC/DTG": 1, "AZT/3TC/ATV/r": 2, "Other": 3
}
care_model_map = {
    "Facility Fast Track": 0, "Community ART Group": 1,
    "Home Delivery": 2, "Standard Facility": 3, "Other": 4
}

input_df = pd.DataFrame({
    'Age at reporting': [age],
    'Sex': [1 if sex == "Male" else 0],
    'First Regimen': [first_regimen_map[first_regimen]],
    'Current Regimen': [regimen_map[regimen]],
    'Last WHO Stage': [who_stage],
    'Last VL Result': [vl_result],
    'Active in TB': [1 if tb_status == "Yes" else 0],
    'Differentiated care model': [care_model_map[care_model]],
    'Months of Prescription': [months_rx],
    'CD4_Missing': [1 if cd4_missing == "Yes" else 0]
})
# Let the user tune the risk threshold
threshold = st.sidebar.slider(
    "Risk threshold",
    0.0, 1.0, 0.5, 0.01
)

if st.button("🔍 Predict Risk"):
    # 1. Get the probability of “Suppressed” (CD4<200)
    proba = model.predict_proba(input_df)[0][1]
    if proba >= threshold:
        st.markdown(
            "<h3 style='color: red;'>⚠️ Immunological Risk: Suppressed (CD4 < 200)</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='color: green;'>✅ Immunological Risk: Safe</h3>",
            unsafe_allow_html=True
        )

    # 3. Always show the probability
    st.write(f"🔍 Suppression Risk Probability: {proba:.2f}")

with st.expander("ℹ️ About This Model"):
    st.markdown("""
    - **Model Type**: Random Forest Classifier  
    - **Accuracy**: 83.6%  
    - **Trained on**: ART patient data with proxy features  
    - **Balanced using**: SMOTE resampling  
    - **Purpose**: Flag suppressed patients when CD4 is missing  
    """)
