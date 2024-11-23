import mlflow

import uvicorn
import pandas as pd
import numpy as np
import joblib
import boto3
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from pydantic import BaseModel
from fastapi import FastAPI

# Loading model from MLFlow server.
mlflow.set_tracking_uri(os.environ["APP_URI"])
model_name = 'fraud_detection'
model_alias = 'Production'
model_uri = f"models:/{model_name}/{model_alias}"
try:
    # mlflow.pyfunc.get_model_dependencies(model_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

app = FastAPI(
    title='FraudDetection API',
    description= """
    This API provides the model for the Fraud Detection project
    """,
    version="1.0"
)

class TransactionRecord(BaseModel):
    merchant: str
    category: str
    amt: float
    first: str
    last :str
    gender :str
    street: str
    city: str
    state: str
    lat: float
    long: float
    city_pop: int
    job: str
    merch_lat: float
    merch_long: float
    trans_year: int
    trans_month: int
    trans_day: int
    trans_hour: int
    trans_minutes: int
    trans_seconds: int
    dob_year: int
    dob_month: int
    dob_day: int
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    'merchant': 'fraud_Hoppe-Parisian',
                    'category': 'kids_pets',
                    'amt': 	111.84,
                    'first': 'Jose',
                    'last': 'Vasquez',
                    'gender':'M',
                    'street': '572 Davis Mountains',
                    'city': 'Lake Jackson',
                    'state':'TX',
                    'lat': 29.0393,
                    'long': -95.4401,
                    'city_pop': 28739,
                    'job': 'Futures trader',
                    'merch_lat': 29.661049,
                    'merch_long': -96.186633,
                    'trans_year': 2020,
                    'trans_month': 12,
                    'trans_day': 31,
                    'trans_hour': 23,
                    'trans_minutes': 59,
                    'trans_seconds': 9,
                    'dob_year': 1999,
                    'dob_month': 12,
                    'dob_day': 27
                }
            ]
        }

@app.get("/", tags=['Test Endpoint'])
async def index(): 
    """
    Return a test message.
    """
    message = "Hello world! This is the default endpoint. It means that the API server is up and everything works."

    return message

@app.post('/predict', tags=['Predict Endpoint'])
async def predict_fraud(userform: TransactionRecord):
    
    """
    Return a fraud prediction. 
    """
    new_data = pd.DataFrame([userform.model_dump()])
    
    # Predict on a Pandas DataFrame.
    prediction = model.predict(new_data)
    
    return {"prediction": prediction.tolist()[0]}


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000) # Here you define your web server to run the `app` variable (which contains FastAPI instance), with a specific host IP (0.0.0.0) and port (4000)