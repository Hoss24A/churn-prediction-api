import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to churn, with AI-generated explanations.")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior         = st.selectbox("Senior Citizen", [0, 1])
        partner        = st.selectbox("Partner", ["Yes", "No"])
        dependents     = st.selectbox("Dependents", ["Yes", "No"])
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        monthly        = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total          = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)

    with col2:
        internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment        = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        phone          = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines    = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bak     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_sup       = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
        "DeviceProtection": device_prot, "TechSupport": tech_sup,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    API_URL = "https://churn-prediction-api-8lgw.onrender.com"

    with st.spinner("Predicting..."):
        res = requests.post(f"{API_URL}/predict", json=payload)

    if res.status_code == 200:
        result = res.json()
        prob = result['churn_probability']
        color = "🔴" if prob > 0.5 else "🟢"
        st.subheader(f"{color} Prediction: {result['prediction']}")
        st.metric("Churn Probability", f"{prob:.1%}")

        st.subheader("Top Factors Driving This Prediction")
        for factor in result['top_factors']:
            direction = "↑ increases" if factor['shap_value'] > 0 else "↓ decreases"
            st.write(f"**{factor['feature']}** — {direction} churn risk (SHAP: {factor['shap_value']:.3f})")
    else:
        st.error("API error. Is the server running?")