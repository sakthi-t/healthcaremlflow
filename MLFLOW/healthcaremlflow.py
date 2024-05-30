import logging
import sys
import warnings
from urllib.parse import urlparse
import dagshub

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


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    dim_patients_table = """SELECT * FROM `bigqueryimdb.healthcare.dim_patients`;"""
    fact_visits_table = """SELECT * FROM `bigqueryimdb.healthcare.fact_visits`;"""

    df_patients = client.query(dim_patients_table).to_dataframe()
    df_visits = client.query(fact_visits_table).to_dataframe()

    try:
        data = df_visits[['patient_id', 'visited_date', 'sugar', 'hba1c']]
    except Exception as e:
        logger.exception(
            "Unable to download the data from BigQuery! %s", e
        )
        sys.exit(1)

    # Convert visited_date to datetime 
    data['visited_date'] = pd.to_datetime(data['visited_date'])

    # Extracting year, month, and day from visited_date
    data['year'] = data['visited_date'].dt.year
    data['month'] = data['visited_date'].dt.month
    data['day'] = data['visited_date'].dt.day

    # Dropping the original visited_date column
    data = data.drop(columns=['visited_date'])

    # Defining features and target
    X = data.drop(columns=['hba1c'])
    y = data['hba1c']

    # Splitting the data
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7

    
    # Initialize DagsHub
    dagshub.init(repo_owner='sakthi-t', repo_name='healthcaremlflow', mlflow=True)

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/healthcaremlflow.mlflow")

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # predictions = lr.predict(train_x)
        # signature = infer_signature(train_x, predictions)

        remote_server_uri = "https://dagshub.com/sakthi-t/healthcaremlflow.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        # if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        #    mlflow.sklearn.log_model(
        #        lr, "model", registered_model_name="ElasticnetHealthcareModel", signature=signature
        #    )
        # else:
        #    mlflow.sklearn.log_model(lr, "model", signature=signature)

        
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetHealthcareModel")
        else:
            mlflow.sklearn.log_model(lr, "model")


# set MLFLOW_TRACKING_URI=https://dagshub.com/sakthi-t/healthcaremlflow.mlflow
# set MLFLOW_TRACKING_USERNAME=sakthi-t