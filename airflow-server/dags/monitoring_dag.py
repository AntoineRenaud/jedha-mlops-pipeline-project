import os
import datetime
import pandas as pd
import boto3

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import (BranchPythonOperator,
                                               PythonOperator)

from sqlalchemy import create_engine

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Load environment variables from Airflow Variables
AWS_ACCESS_KEY_ID = Variable.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = Variable.get('AWS_SECRET_ACCESS_KEY')
DATASET_BUCKET = Variable.get('DATASET_BUCKET')
EVIDENTLY_TOKEN = Variable.get('EVIDENTLY_TOKEN')
PROJECT_ID = Variable.get('PROJECT_ID')

RDS_USER = Variable.get("USER")
RDS_PASSWORD = Variable.get("PASSWORD")
RDS_HOST = Variable.get("DATABASE_URL")
RDS_PORT = Variable.get("PORT")
RDS_SCHEMA = Variable.get("DATABASE")

# # Define default arguments for Airflow DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2024, 11, 26),
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=-1)
}

# Create an SQLAlchemy engine to connect to PostgreSQL
engine = create_engine(f"postgresql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_SCHEMA}")

# Function to fetch data from PostgreSQL
def load_data():
    try:
        query = "SELECT * FROM fraud_database LIMIT 15;"
        df_production = pd.read_sql_query(query, con=engine)
    except Exception as e:
        print(f"Error fetching data from SQL: {e}")
        df_production = pd.DataFrame()
        
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    response = s3_client.get_object(Bucket=DATASET_BUCKET, Key="fraud-detection/cleaned-dataset.csv")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    
    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        df_reference = pd.read_csv(response.get("Body"))
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")
        df_reference = pd.DataFrame()    
        
    # Preprocess data
    df_production = df_production.drop(['id', 'is_fraud'], axis=1)
    df_reference = df_reference.sample(15).drop(['unix_time', 'is_fraud'], axis=1)
    df_production = df_production[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'trans_year']]
    df_reference = df_reference[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'trans_year']]
    
    return df_production, df_reference

# Define the drift test
def detect_data_drift():
    
    df_reference, df_production = load_data()
    data_drift_report = Report(metrics=[
        DataDriftPreset()
    ])

    data_drift_report.run(reference_data=df_reference, current_data=df_production)
    
    report = data_drift_report.as_dict()

    if report["metrics"][0]["result"]["dataset_drift"]:
        return "data_drift_detected"
    else:
        return "no_data_drift_detected"

def _data_drift_detected(**context):
    """Produces a report on evidently cloud
    """

    ws = CloudWorkspace(token=EVIDENTLY_TOKEN, url="https://app.evidently.cloud")

    project = ws.get_project(PROJECT_ID)
    
    current_data, reference_data = load_data()

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None)
    ws.add_report(project.id, data_drift_report, include_data=True)


with DAG(dag_id="monitoring_dag", default_args=default_args, schedule_interval="0 0 * * *", catchup=False) as dag:
    
    detect_data_drift = BranchPythonOperator(task_id="detect_data_drift", python_callable=detect_data_drift)
    data_drift_detected = PythonOperator(task_id="data_drift_detected", python_callable=_data_drift_detected)
    no_data_drift_detected = DummyOperator(task_id="no_data_drift_detected")
    end = DummyOperator(task_id="end")

    detect_data_drift >> [data_drift_detected, no_data_drift_detected] >> end
