
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

# 🧪 Demo Mode Profiles
demo_profiles = {
    "None": {},
    "Low Risk": {
        "Age": 28, "Sex": "Female", "WHO Stage": 1, "VL": 200, "TB": "No",
        "Months Rx": 6, "CD4 Missing": "No", "First Regimen": "TDF/3TC/EFV",
        "Current Regimen": "TDF/3TC/DTG", "Care Model": "Facility Fast Track"
    },
    "High Risk": {
        "Age": 42, "Sex": "Male", "WHO Stage": 4, "VL": 100000, "TB": "Yes",
        "Months Rx": 1, "CD4 Missing": "Yes", "First Regimen": "ABC/3TC/EFV",
        "Current Regimen": "AZT/3TC/ATV/r", "Care Model": "Home Delivery"
    }
}

selected_demo = st.sidebar.selectbox("🧪 Load Demo Profile", list(demo_profiles.keys()))
demo = demo_profiles[selected_demo]

st.sidebar.header("📋 Patient Details")
age = st.sidebar.number_input("🎂 Age at reporting", min_value=0, max_value=120, value=demo.get("Age", 30))
sex = st.sidebar.selectbox("⚧️ Sex", ["Male", "Female"], index=["Male", "Female"].index(demo.get("Sex", "Male")))
first_regimen = st.sidebar.selectbox("🧬 First Regimen", ["TDF/3TC/EFV", "AZT/3TC/NVP", "ABC/3TC/EFV", "Other"], index=["TDF/3TC/EFV", "AZT/3TC/NVP", "ABC/3TC/EFV", "Other"].index(demo.get("First Regimen", "TDF/3TC/EFV")))
regimen = st.sidebar.selectbox("💊 Current Regimen", ["TDF/3TC/DTG", "ABC/3TC/DTG", "AZT/3TC/ATV/r", "Other"], index=["TDF/3TC/DTG", "ABC/3TC/DTG", "AZT/3TC/ATV/r", "Other"].index(demo.get("Current Regimen", "TDF/3TC/DTG")))
care_model = st.sidebar.selectbox("🏥 Differentiated Care Model", ["Facility Fast Track", "Community ART Group", "Home Delivery", "Standard Facility", "Other"], index=["Facility Fast Track", "Community ART Group", "Home Delivery", "Standard Facility", "Other"].index(demo.get("Care Model", "Facility Fast Track")))
who_stage = st.sidebar.selectbox("🩺 WHO Stage", [1, 2, 3, 4], index=[1, 2, 3, 4].index(demo.get("WHO Stage", 1)))
vl_result = st.sidebar.number_input("🧪 Last VL Result", min_value=0, value=demo.get("VL", 0))
tb_status = st.sidebar.selectbox("🦠 Active in TB", ["Yes", "No"], index=["Yes", "No"].index(demo.get("TB", "No")))
months_rx = st.sidebar.number_input("📆 Months of Prescription", min_value=0, value=demo.get("Months Rx", 1))
cd4_missing = st.sidebar.selectbox("❓ CD4 Missing", ["Yes", "No"], index=["Yes", "No"].index(demo.get("CD4 Missing", "No")))


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
with st.expander("ℹ️ What does this prediction mean?"):
    st.markdown("""
    - **Suppressed (CD4 < 200)**: Indicates a weakened immune system. Patients are at high risk for opportunistic infections.
    - **Safe**: Suggests CD4 is likely above 200, but clinical judgment is still needed.
    - This prediction is based on proxy features like WHO Stage, TB status, VL, and ART regimen.
    - Use the **risk threshold slider** to adjust sensitivity.
    """)

with st.expander("ℹ️ About This Model"):
    st.markdown("""
    - **Model Type**: Random Forest Classifier  
    - **Accuracy**: 83.6%  
    - **Trained on**: ART patient data with proxy features  
    - **Balanced using**: SMOTE resampling  
    - **Purpose**: Flag suppressed patients when CD4 is missing  
    """)
