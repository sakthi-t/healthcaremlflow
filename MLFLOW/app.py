import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from google.cloud import bigquery
import os
from datetime import datetime

st.set_page_config(page_title="hba1c Prediction", layout="centered")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'biquerykey.json'

# Initialize BigQuery client
client = bigquery.Client()

# Load patient IDs for the dropdown
dim_patients_table = """SELECT patient_id FROM `bigqueryimdb.healthcare.dim_patients`;"""
df_patients = client.query(dim_patients_table).to_dataframe()
patient_ids = df_patients['patient_id'].tolist()

# Set the MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/healthcaremlflow.mlflow")

# Load the model from MLflow model registry
model_name = "ElasticnetHealthcareModel"
model_version = 1
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
st.title("hba1c Prediction")
st.write("Select Patient ID, Visited Date, and Sugar value to predict hba1c.")

patient_id = st.selectbox("Patient ID", patient_ids)
visited_date = st.date_input("Visited Date")
sugar = st.number_input("Sugar", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Predict hba1c"):
    prediction = predict_hba1c(patient_id, visited_date, sugar)
    st.write(f"Predicted hba1c: {prediction}")
