import streamlit as st
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# Declare and import module
model = joblib.load("../model_exports/best_model.pkl")

# Page Composing
st.title("Student Exam Scores Predictor")

study_hours = st.slider("Study Hours Per Day", 0.0, 12.0, 2.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)", 1.0, 10.0, 5.0)
sleep_hours = st.slider("Sleep hours per night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-Time Job? [Yes/No]", ["No", "Yes"])

# Transfor for label encoded value in model
ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_data = [[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]]
    prediction = model.predict(input_data)[0] #[0] Takes the output values
    # Constrain prediction values
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score {prediction:.2f}")
