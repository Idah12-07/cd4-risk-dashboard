# cd4-risk-dashboard
This Streamlit dashboard predicts the likelihood of immunological suppression (CD4 < 200) in ART patients using proxy clinical features. It is designed to support clinicians and stakeholders in identifying high-risk cases when CD4 testing is delayed or unavailable.

🔍 Model Overview
Type: Random Forest Classifier
Target: Binary classification — Suppressed vs Safe
Training Data: De-identified ART patient records

Features Used:
Age, Sex,WHO Stage, TB Status
Viral Load Result
ART Regimen (First & Current)
Differentiated Care Model
Months of Prescription
CD4 Missing Indicator

📌 Notes
This tool is intended for supportive decision-making, not standalone diagnosis.
Threshold slider allows clinicians to adjust sensitivity based on context.
Future versions may include calibrated probabilities and feature importance visualization.
