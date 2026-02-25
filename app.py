import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open("best_xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer details below:")

# ---- INPUT SECTION ---- #

age = st.number_input("Age", 0, 100, 30)
tenure_months = st.number_input("Tenure (Months)", 0, 120, 12)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")
avg_data_gb_month = st.number_input("Avg Data Usage (GB/Month)")
avg_voice_mins_month = st.number_input("Avg Voice Minutes/Month")
sms_count_month = st.number_input("SMS Count/Month")
overage_charges = st.number_input("Overage Charges")
is_family_plan = st.selectbox("Family Plan", [0,1])
is_multi_service = st.selectbox("Multi Service", [0,1])
network_issues_3m = st.number_input("Network Issues (3M)")
dropped_call_rate = st.number_input("Dropped Call Rate")
avg_data_speed_mbps = st.number_input("Avg Data Speed (Mbps)")
num_complaints_3m = st.number_input("Complaints (3M)")
num_complaints_12m = st.number_input("Complaints (12M)")
call_center_interactions_3m = st.number_input("Call Center Interactions (3M)")
last_complaint_resolution_days = st.number_input("Last Complaint Resolution Days")
app_logins_30d = st.number_input("App Logins (30D)")
selfcare_transactions_30d = st.number_input("Selfcare Transactions (30D)")
auto_pay_enrolled = st.selectbox("Auto Pay Enrolled", [0,1])
late_payment_flag_3m = st.selectbox("Late Payment (3M)", [0,1])
avg_payment_delay_days = st.number_input("Avg Payment Delay Days")
arpu = st.number_input("ARPU")
nps_score = st.number_input("NPS Score")
service_rating_last_6m = st.number_input("Service Rating (6M)")
received_competitor_offer_flag = st.selectbox("Received Competitor Offer", [0,1])
retention_offer_accepted_flag = st.selectbox("Retention Offer Accepted", [0,1])
service_count = st.number_input("Service Count")
high_data_user = st.selectbox("High Data User", [0,1])
high_voice_user = st.selectbox("High Voice User", [0,1])
engagement_score = st.number_input("Engagement Score")
avg_monthly_charges = st.number_input("Avg Monthly Charges")
bill_increase_ratio = st.number_input("Bill Increase Ratio")
bill_increase_pct = st.number_input("Bill Increase %")
bill_shock_flag = st.selectbox("Bill Shock Flag", [0,1])
high_bill_customer_flag = st.selectbox("High Bill Customer", [0,1])
billing_stress_score = st.number_input("Billing Stress Score")
services_per_dollar = st.number_input("Services Per Dollar")
overcharging_flag = st.selectbox("Overcharging Flag", [0,1])
usage_intensity_index = st.number_input("Usage Intensity Index")
revenue_quality_score = st.number_input("Revenue Quality Score")

# ----- CREATE DATAFRAME (ORDER MUST MATCH TRAINING) ----- #

input_data = pd.DataFrame([[
    age, tenure_months, monthly_charges, total_charges,
    avg_data_gb_month, avg_voice_mins_month, sms_count_month,
    overage_charges, is_family_plan, is_multi_service,
    network_issues_3m, dropped_call_rate, avg_data_speed_mbps,
    num_complaints_3m, num_complaints_12m,
    call_center_interactions_3m, last_complaint_resolution_days,
    app_logins_30d, selfcare_transactions_30d, auto_pay_enrolled,
    late_payment_flag_3m, avg_payment_delay_days, arpu, nps_score,
    service_rating_last_6m, received_competitor_offer_flag,
    retention_offer_accepted_flag, service_count, high_data_user,
    high_voice_user, engagement_score, avg_monthly_charges,
    bill_increase_ratio, bill_increase_pct, bill_shock_flag,
    high_bill_customer_flag, billing_stress_score,
    services_per_dollar, overcharging_flag,
    usage_intensity_index, revenue_quality_score
]])

if st.button("Predict Churn"):

    # Scale input
    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ Customer Likely to Churn")
    else:
        st.success(f"✅ Customer Not Likely to Churn")

    st.write(f"Churn Probability: {round(probability*100,2)}%")