import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from google.cloud import bigquery
import os
from urllib.request import urlretrieve  # Using urllib for download
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Load the model from MLflow
# model_uri = "models:/ElasticnetHealthcareModel/1"  # Adjust this path according to your model version
model_uri = "https://dagshub.com/sakthi-t/healthcaremlflow.mlflow/#/models/ElasticnetHealthcareModel/versions/1"

# Download the model from the provided URL
model_filename = "downloaded_model.pkl"  # Specify a filename
urlretrieve(model_uri, model_filename)

try:
  # Load the downloaded model (handle potential errors)
  loaded_model = mlflow.sklearn.load_model(model_filename)
except (AttributeError, FileNotFoundError) as e:
  # Handle potential errors during model loading (e.g., corrupted file)
  loaded_model = None

if loaded_model is not None:  
  

    # Select a sample from the test set
    sample_input = test_x.iloc[0:1]

    # Make prediction
    prediction = loaded_model.predict(sample_input)

    print(f"Sample input:\n{sample_input}")
    print(f"Prediction: {prediction[0]}")

    # Evaluate the model on the test set and log metrics
    predicted_qualities = loaded_model.predict(test_x)

    rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))
    mae = mean_absolute_error(test_y, predicted_qualities)
    r2 = r2_score(test_y, predicted_qualities)

    print(f"Metrics on the test set:")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

else:
   print("Model loading failed. Skipping prediction and evaluation.")
