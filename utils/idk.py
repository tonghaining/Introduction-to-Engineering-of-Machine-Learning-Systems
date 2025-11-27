import os
import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet


def load_model(experiment_name, run_id):
    # Set an environmental variable named "MLFLOW_S3_ENDPOINT_URL" so that MLflow client knows where to save artifacts.
    # The MinIO storage service can be accessed via http://mlflow-minio.local
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.local"
    # Configure the credentials needed for accessing the MinIO storage service.
    # "AWS_ACCESS_KEY_ID" has been configured in a ComfigMap and "AWS_SECRET_ACCESS_KEY" in a Secret in your K8s cluster when you set up the MLOps platform
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    MLFLOW_TRACKING_URI = "http://mlflow-server.local" # This is the URL of the MLflow service
    #MLFLOW_EXPERIMENT_NAME = "mlflow-minio-test"
    MLFLOW_EXPERIMENT_NAME = experiment_name

    # add rng string to model name to avoid name conflicts
    #YOUR_MODEL_NAME="REPLACE_WITH_YOUR_MODEL_NAME"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    #run_id = '9878008178af4b77be72a93a3ca1d67f'
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.sklearn.load_model(logged_model)

    return loaded_model