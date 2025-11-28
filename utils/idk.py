import os
import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet


def load_model(run_id):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.local"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    MLFLOW_TRACKING_URI = "http://mlflow-server.local" # This is the URL of the MLflow service
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.sklearn.load_model(logged_model)

    return loaded_model