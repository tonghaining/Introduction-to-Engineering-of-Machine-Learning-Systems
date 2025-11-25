import os
import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
# import random forest
from sklearn.ensemble import RandomForestRegressor

# Set an environmental variable named "MLFLOW_S3_ENDPOINT_URL" so that MLflow client knows where to save artifacts.
# The MinIO storage service can be accessed via http://mlflow-minio.local
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.local"

# Configure the credentials needed for accessing the MinIO storage service.
# "AWS_ACCESS_KEY_ID" has been configured in a ComfigMap and "AWS_SECRET_ACCESS_KEY" in a Secret in your K8s cluster when you set up the MLOps platform
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://mlflow-server.local" # This is the URL of the MLflow service
#MLFLOW_EXPERIMENT_NAME = "YOUR EXPERIMENT NAME HERE" # Remember that others on the same MLflow service may see these names so be nice.
#YOUR_MODEL_NAME="REPLACE_WITH_YOUR_MODEL_NAME"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse

def retrain_model(X_train, y_train, X_test, y_test, experiment_name, model_name):
    # MLFLOW_EXPERIMENT_NAME = "testink teemu eksperiment" # Remember that others on the same MLflow service may see these names so be nice.
    # YOUR_MODEL_NAME="forest teemu test"
    MLFLOW_EXPERIMENT_NAME = experiment_name
    YOUR_MODEL_NAME=model_name

    def run_mlflow_example():
        np.random.seed(40)
        # Just use hard-coded hyperparameters
        alpha = 0.5
        l1_ratio = 0.5

        logger.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")

        # Configure the MLflow client to connect to the MLflow service
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        logger.info(f"Using MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            print("MLflow run_id:", run.info.run_id) # Each MLflow Run has a unique identifier called run_id

            #model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)

            logger.info("Fitting model...")

            model.fit(X_train, y_train)

            logger.info("Finished fitting")

            predicted_qualities = model.predict(X_test)

            rmse = eval_metrics(y_test, predicted_qualities)

            logger.info("Elasticnet model (alpha=%f, l1_ratio=%f):" %
                        (alpha, l1_ratio))
            logger.info("  RMSE: %s" % rmse)


            logger.info("Logging parameters to MLflow")
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)

            logger.info("Logging trained model")
            artifact_name = "model"
            mlflow.sklearn.log_model(
                model, artifact_name, registered_model_name=YOUR_MODEL_NAME)
            print("The S3 URI of the logged model:", mlflow.get_artifact_uri(artifact_path=artifact_name))
    
    run_mlflow_example()