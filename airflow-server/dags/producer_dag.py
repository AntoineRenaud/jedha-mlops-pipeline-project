from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, BigInteger, String
from sqlalchemy.orm import sessionmaker

# Get credentials for RDS connection
user = Variable.get("USER")
password = Variable.get("PASSWORD")
host = Variable.get("DATABASE_URL")
port = Variable.get("PORT")
schema = Variable.get("DATABASE")

# Define transaction processing
def request_transaction():
    transaction_api_url = Variable.get('TRANSACTION_API_URL',default_var=None)
    response = requests.get(transaction_api_url)
    return response.text

def process_data(ti):
    raw_data = ti.xcom_pull(task_ids='fetch_transaction_data')

    df = pd.read_json(eval(raw_data), orient='split')

    # Process your data (e.g., add extracted date components)
    df['trans_date_trans_time'] = pd.to_datetime(df['current_time'])
    df['trans_year'] = df['trans_date_trans_time'].dt.year
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    df['trans_day'] = df['trans_date_trans_time'].dt.day
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_minutes'] = df['trans_date_trans_time'].dt.minute
    df['trans_seconds'] = df['trans_date_trans_time'].dt.second

    df.drop(['current_time', 'trans_date_trans_time', 'is_fraud'], axis=1, inplace=True)
    df['dob'] = pd.to_datetime(df['dob'])
    df['dob_year'] = df['dob'].dt.year
    df['dob_month'] = df['dob'].dt.month
    df['dob_day'] = df['dob'].dt.day
    df.drop('dob', axis=1, inplace=True)
    
    # Convert columns to string as needed
    columns_to_convert = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job']
    df[columns_to_convert] = df[columns_to_convert].astype(str)

    ti.xcom_push(key='processed_data', value=df.to_dict(orient='records'))

def predict_fraud(ti):
    processed_data = ti.xcom_pull(task_ids='process_transaction_data', key='processed_data')
    data_for_prediction = processed_data[0]
    prediction_api_url = Variable.get('PREDICTION_API_URL',default_var=None)
    
    response = requests.post(prediction_api_url, json=data_for_prediction)
    prediction = response.json().get('prediction')

    # Add prediction back to the data
    data_for_prediction['is_fraud'] = prediction
    ti.xcom_push(key='data_with_prediction', value=data_for_prediction)

def save_to_db(ti):
    record = ti.xcom_pull(task_ids='predict_fraud', key='data_with_prediction')

    Base = declarative_base()

    class Transaction(Base):
        __tablename__ = "fraud_database"
        id = Column(Integer, primary_key=True, autoincrement=True)
        cc_num = Column(BigInteger)
        merchant = Column(String)
        category = Column(String)
        amt = Column(Float)
        first = Column(String)
        last = Column(String)
        gender = Column(String)
        street = Column(String)
        city = Column(String)
        state = Column(String)
        zip = Column(Integer)
        lat = Column(Float)
        long = Column(Float)
        city_pop = Column(Integer)
        job = Column(String)
        trans_num = Column(String)
        merch_lat = Column(Float)
        merch_long = Column(Float)
        trans_year = Column(Integer)
        trans_month = Column(Integer)
        trans_day = Column(Integer)
        trans_hour = Column(Integer)
        trans_minutes = Column(Integer)
        trans_seconds = Column(Integer)
        dob_year = Column(Integer)
        dob_month = Column(Integer)
        dob_day = Column(Integer)
        is_fraud = Column(Integer)
    
    engine = create_engine(f"postgresql://{user}:{password}"
                    f"@{host}:{port}/{schema}")
     
    inspector = inspect(engine) 

    # Check if the table exists
    if not inspector.has_table("fraud_database"):
        Base.metadata.create_all(engine)  

    Session = sessionmaker(bind=engine)
    session = Session()
    transaction_record = Transaction(**record)
    session.add(transaction_record)
    session.commit()
    session.close()

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 11, 25),
    'retries': 1,
}

with DAG(
    'fraud_detection_dag',
    default_args=default_args,
    schedule_interval='*/5 * * * *',
    catchup=False,
) as dag:
    
    fetch_transaction_data = PythonOperator(
        task_id='fetch_transaction_data',
        python_callable=request_transaction
    )
    
    process_transaction_data = PythonOperator(
        task_id='process_transaction_data',
        python_callable=process_data
    )
    
    predict_fraud = PythonOperator(
        task_id='predict_fraud',
        python_callable=predict_fraud
    )
    
    save_data_to_db = PythonOperator(
        task_id='save_data_to_db',
        python_callable=save_to_db
    )

    # Set task dependencies
    fetch_transaction_data >> process_transaction_data >> predict_fraud >> save_data_to_db
