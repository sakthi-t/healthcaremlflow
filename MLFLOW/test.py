import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import os
from google.cloud import bigquery

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'biquerykey.json'
client = bigquery.Client()


dim_patients_table = """SELECT * FROM `bigqueryimdb.healthcare.dim_patients`;"""
fact_visits_table = """SELECT * FROM `bigqueryimdb.healthcare.fact_visits`;"""

df_patients = client.query(dim_patients_table).to_dataframe()
df_visits = client.query(fact_visits_table).to_dataframe()

print(df_patients.head())
print(df_visits.head())
