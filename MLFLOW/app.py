import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from datetime import datetime

st.set_page_config(page_title="HBA1C Prediction", layout="centered")

df_patients = pd.read_csv("../Hba1cData/dim_patients_final_rev01.csv")
patient_ids = df_patients['patient_id'].tolist()

# Set the MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/healthcaremlflow.mlflow")

# Load the model from MLflow model registry
model_name = "ElasticnetHealthcareModel"
model_version = 3
model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.sklearn.load_model(model_uri)

def predict_hba1c(patient_id, visited_date, sugar):
    # Prepare input data
    visited_date = pd.to_datetime(visited_date)
    data = {
        'patient_id': [patient_id],
        'sugar': [sugar],
        'year': [visited_date.year],
        'month': [visited_date.month],
        'day': [visited_date.day]
    }
    input_df = pd.DataFrame(data)
    
    # Make prediction
    prediction = loaded_model.predict(input_df)
    return prediction[0]

# Streamlit interface
st.title("HBA1C Prediction")
st.write("Select Patient ID, Visited Date, and Sugar value to predict HBA1C.")

st.markdown(
    """
    <div style="background-color: #FF9798; color: white; padding: 10px; border-radius: 5px;">
        Choose sugar levels between 50 and 600. HBA1C levels are influenced by sugar values: higher sugar typically results in higher HBA1C. 
        This machine learning project uses synthetic data and is not a definitive method to determine HBA1C levels. For accurate results, 
        please select a date within 2024. The dataset contains dates from 2023 to April 2024. Patient names are fictional. Users can only select 
        from existing patient IDs, and there is no correlation between User ID and sugar levels.
    </div>
    """, unsafe_allow_html=True
)

patient_id = st.selectbox("Patient ID", patient_ids)
visited_date = st.date_input("Visited Date", min_value=datetime(2023, 1, 1), max_value=datetime(2024, 4, 30))
sugar = st.number_input("Sugar", min_value=50.0, max_value=600.0, value=100.0)

if st.button("Predict HBA1C"):
    prediction = predict_hba1c(patient_id, visited_date, sugar)
    st.write(f"Predicted HBA1C: {prediction}")
