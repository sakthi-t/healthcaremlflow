import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from google.cloud import bigquery
import os
from sklearn.model_selection import train_test_split

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'biquerykey.json'

# Initialize BigQuery client
client = bigquery.Client()

# Load data
dim_patients_table = """SELECT * FROM `bigqueryimdb.healthcare.dim_patients`;"""
fact_visits_table = """SELECT * FROM `bigqueryimdb.healthcare.fact_visits`;"""

df_patients = client.query(dim_patients_table).to_dataframe()
df_visits = client.query(fact_visits_table).to_dataframe()

# Prepare data
data = df_visits[['patient_id', 'visited_date', 'sugar', 'hba1c']]
data['visited_date'] = pd.to_datetime(data['visited_date'])
data['year'] = data['visited_date'].dt.year
data['month'] = data['visited_date'].dt.month
data['day'] = data['visited_date'].dt.day
data = data.drop(columns=['visited_date'])

# Define features and target
X = data.drop(columns=['hba1c'])
y = data['hba1c']

# Split the data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)

# Load the model
model_uri = "runs:/6ce4f95bce3f422683c08682cf8f4a3b/model"  # Replace with your actual run ID
loaded_model = mlflow.sklearn.load_model(model_uri)

# Select a sample from the test set
sample_input = test_x.iloc[0:1]

# Make prediction
prediction = loaded_model.predict(sample_input)

print(f"Sample input:\n{sample_input}")
print(f"Prediction: {prediction[0]}")
