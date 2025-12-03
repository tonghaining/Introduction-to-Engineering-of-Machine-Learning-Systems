#%%
import requests
import time
#%%
# Add model s3 path to manifest first
!kubectl apply -f bonus/wine-model.yaml
# %%
# 'wine-model-workshop' is from the manifest
# let's check its running now
#!kubectl get isvc wine-model-workshop -n kserve-inference
# waiting version
!kubectl get isvc wine-model-workshop -n kserve-inference -w
#%%
# check pods
!kubectl get pods -n kserve-inference -l serving.kserve.io/inferenceservice=wine-model-workshop
#%%
# Send some requests
from utils import send_requests

# Send requests
# TODO import data from data folder
single_input = [7.8, 0.58, 0.02, 2, 0.073, 9, 18, 0.9968, 3.36, 0.57, 9.5]
model_name = "wine-quality"

headers = {}
headers["Host"] = f"{model_name}.kserve-inference.example.com"
url = f"http://kserve-gateway.local:30200/v1/models/{model_name}:predict"

def send_requests(count=30, input_length=1):
    """
    Send requests to a inference service for predicting wine quality score in every 0.5s
    Args; 
        count: Number of requests
        input_length: Number of inputs in each request
    """
    for _ in range(count):
        req_data={"instances": [single_input for _ in range(input_length)]}
        requests.post(url, json=req_data, headers=headers)
        time.sleep(0.5)

#%%
# Send 30 requests in every 0.5s, each request has one input(i.e., data point)
send_requests(count=30, input_length=1)
#%%
send_requests(count=30, input_length=100000)