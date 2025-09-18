import streamlit as st
import pandas as pd
import pickle

# Load saved model, scaler, and column order
with open("xgb_churn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("columns.pkl", "rb") as file:
    columns = pickle.load(file)

st.title("🔮 Customer Churn Prediction App")
st.write("Enter customer details to predict whether they will churn or not.")

# Input fields for user
seats = st.number_input("Seats", min_value=1, value=10)
mrr_amount = st.number_input("Monthly Recurring Revenue (MRR)", min_value=0.0, value=1000.0)
arr_amount = st.number_input("Annual Recurring Revenue (ARR)", min_value=0.0, value=12000.0)
usage_count = st.number_input("Usage Count", min_value=0, value=50)
usage_duration_secs = st.number_input("Usage Duration (secs)", min_value=0, value=10000)
error_count = st.number_input("Error Count", min_value=0, value=2)
support_tickets_count = st.number_input("Support Tickets Count", min_value=0, value=1)
subscription_duration = st.number_input("Subscription Duration (days)", min_value=0, value=365)

industry = st.selectbox("Industry", ["Software", "Finance", "Healthcare", "Education", "Other"])
country = st.selectbox("Country", ["USA", "India", "UK", "Germany", "Other"])
plan_tier = st.selectbox("Plan Tier", ["Basic", "Standard", "Premium"])
billing_frequency = st.selectbox("Billing Frequency", ["Monthly", "Quarterly", "Yearly"])
is_trial = st.selectbox("Is Trial?", [0, 1])
upgrade_flag = st.selectbox("Upgrade Flag", [0, 1])
downgrade_flag = st.selectbox("Downgrade Flag", [0, 1])
auto_renew_flag = st.selectbox("Auto Renew Flag", [0, 1])

# Convert categorical selections to encoded values
industry_map = {v: i for i, v in enumerate(["Software", "Finance", "Healthcare", "Education", "Other"])}
country_map = {v: i for i, v in enumerate(["USA", "India", "UK", "Germany", "Other"])}
plan_tier_map = {v: i for i, v in enumerate(["Basic", "Standard", "Premium"])}
billing_map = {v: i for i, v in enumerate(["Monthly", "Quarterly", "Yearly"])}

# Create input dataframe
input_data = pd.DataFrame([[
    seats, mrr_amount, arr_amount, usage_count, usage_duration_secs, error_count,
    support_tickets_count, subscription_duration,
    industry_map[industry], country_map[country], plan_tier_map[plan_tier], billing_map[billing_frequency],
    is_trial, upgrade_flag, downgrade_flag, auto_renew_flag
]], columns=columns)

# Scale numeric features
num_cols = ['seats','mrr_amount','arr_amount','usage_count','usage_duration_secs','error_count','support_tickets_count']
input_data[num_cols] = scaler.transform(input_data[num_cols])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is likely to churn. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is unlikely to churn. (Probability: {probability:.2f})")
