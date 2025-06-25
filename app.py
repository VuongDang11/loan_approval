import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    return joblib.load("model\decision_model.pkl") 
    ### PICKLE Library for loading the model
model = load_model()

st.title("🏦 Loan Approval Prediction App")

# Input fields for each feature

# Sidebar input
with st.sidebar:
    st.header("Applicant Information")
    person_age = st.slider("Age", 18, 90, 30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    person_emp_exp = st.slider("Employment Experience (Years)", 0, 60, 5)
    person_home_ownership = st.selectbox("Home Ownership", ["Rent", "Mortgage", "Own", "Other"])

    st.header("Loan Information")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, value=10000, step=500)
    # Define mapping
    loan_intent_map = {
      "Business": 0,
      "Education": 1,
      "Home Improvement": 2,
      "Medical": 3,
      "Personal": 4,
      "Debt Consolidation": 5
    }
    loan_intent_display = st.selectbox("Loan Intent", list(loan_intent_map.keys()))
    loan_intent = loan_intent_map[loan_intent_display]

    loan_int_rate = st.number_input("Loan Interest Rate (%)", 0.0, 20.0, 10.0, step=0.10)

    st.header("Credit History")
    cb_person_cred_hist_length = st.slider("Credit History Length (Years)", 0, 50, 5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1, value=700)
    default = st.selectbox("Previous Loan Defaults on File", ["No", "Yes"])
    previous_loan_defaults_on_file = 1 if default == "Yes" else 0

# Create input DataFrame for prediction
if st.button("Predict Your Loan Status"):
    input_data = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }])

    # Map categorical features to numeric codes 
    # This is necessary for the model to understand the input (cannot be strings)
    ownership_map = {"Rent": 0, "Mortgage": 1, "Own": 2, "Other": 3}

    

    input_data['person_home_ownership'] = input_data['person_home_ownership'].map(ownership_map)

    st.subheader("📊 Prediction Result:")
    st.write(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]  # Get the prediction result at index 0


    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")