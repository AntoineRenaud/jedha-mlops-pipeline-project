import mlflow
import os
import requests
from mlflow.exceptions import MlflowException

import mlflow

# Set the tracking URI for MLflow
mlflow.set_tracking_uri(os.environ['APP_URI'])


def set_production_model(model_name):
    """
    Fetch the latest version of a model from MLflow and associate it with the 'production' alias.
    
    Parameters:
    model_name (str): The name of the model to update.
    """
    try:
        # Initialize the MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Fetch all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        # Sort versions by version number in descending order
        latest_version = max(versions, key=lambda v: int(v.version))
        latest_version_number = latest_version.version
        
        print(f"Latest version of model '{model_name}' is {latest_version_number}.")
        
        # Assign the 'production' alias to the latest version
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=latest_version_number
        )
        print(f"Version {latest_version_number} of model '{model_name}' is now associated with the 'production' alias.")
        return True
    
    except MlflowException as e:
        print(f"MLflowException occurred: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    return False

def reload_api_model():
        """
    Sends a GET request to the reload_model API endpoint and handles the response.
"""
api_endpoint = "https://mlops-api-a1b16d875978.herokuapp.com/reload_model"
try:
    response = requests.get(os.environ['API_ENDPOINT_RELOADMODEL'])
    
    # Check if the request was successful
    if response.status_code == 200:
        print("Model reload request successful.")
        print(f"Response: {response.json()}")  # Assuming the API returns a JSON response
    else:
        print(f"Failed to reload model. Status code: {response.status_code}")
        print(f"Error response: {response.text}")

except requests.RequestException as e:
    print(f"An error occurred while making the API request: {str(e)}")
        
        
# Replace 'YourModelName' with the actual model name
model_name = "fraud-detection"
success = set_production_model(model_name)
if success:
    reload_api_model()

