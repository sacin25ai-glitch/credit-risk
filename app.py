import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Risk Assessment System")

age = st.number_input("Age", 18, 100, 30)
job = st.selectbox("Job", [0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)
duration = st.number_input("Duration", min_value=1, value=12)

if st.button("Predict"):

    data = pd.DataFrame({
        "Age": [age],
        "Job": [job],
        "Credit amount": [credit_amount],
        "Duration": [duration]
    })

    st.write("Input Data")
    st.dataframe(data)

    try:
        scaled = scaler.transform(data)
        prediction = model.predict(scaled)

        if prediction[0] == 1:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")

    except Exception as e:
        st.error(e)