import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent
MODEL = joblib.load(BASE / "models" / "random_forest_model.joblib")

def floor_parts(s):
    s = str(s).strip()
    first, total = s.split(" out of ", 1)
    mapping = {"ground": 0, "lower basement": -1, "upper basement": -2}
    try:
        floor = mapping[first.lower()] if first.lower() in mapping else int(first)
    except:
        floor = np.nan
    try:
        total_floors = int(total)
    except:
        total_floors = np.nan
    if pd.notna(floor) and pd.notna(total_floors) and floor > total_floors:
        floor, total_floors = np.nan, np.nan
    return floor, total_floors

st.set_page_config(page_title="House Rent Predictor", page_icon="🏠")
st.title("🏠 House Rent Prediction")
st.write("Machine-learning demo using the Random Forest Regressor.")

col1, col2 = st.columns(2)
with col1:
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    size = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1000)
    floor = st.selectbox("Floor", ["Ground out of 2", "1 out of 2", "1 out of 3", "2 out of 3", "3 out of 5"])
    area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
    city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"])
with col2:
    furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
    tenant = st.selectbox("Tenant Preferred", ["Bachelors", "Family", "Bachelors/Family"])
    bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    posted_month = st.slider("Posted Month", 1, 12, 5)
    posted_day = st.slider("Posted Day of Week (Mon=0)", 0, 6, 2)

if st.button("Predict Rent"):
    fnum, tfloor = floor_parts(floor)
    row = pd.DataFrame([{
        "BHK": bhk, "Size": size, "Floor_Number": fnum, "Total_Floors": tfloor,
        "Area Type": area_type, "City": city,
        "Furnishing Status": furnishing, "Tenant Preferred": tenant,
        "Bathroom": bathroom, "Posted_Month": posted_month,
        "Posted_DayOfWeek": posted_day
    }])
    prediction = MODEL.predict(row)[0]
    st.success(f"Predicted Monthly Rent: ₹{prediction:,.0f}")
