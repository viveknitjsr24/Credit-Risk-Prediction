import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_model.pkl")
encoder = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

st.title("Credit Risk Predictor")
st.write("Enter applicant details to predict the credit risk")

age = st.number_input("Age", min_value=18, max_value=85, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
checking_accounts = st.selectbox("Checking Accounts", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value = 0, value = 1000)
duration = st.number_input("Duration (in months)", min_value = 1, value = 12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoder["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoder["Housing"].transform([housing])[0]],
    "Saving accounts": [encoder["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoder["Checking account"].transform([checking_accounts])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The Predicted Credit Risk is : **GOOD**")
    else:
        st.error("The Predicted Credit Risk is : **BAD**")
