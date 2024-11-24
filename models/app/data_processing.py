import os

import pandas as pd

import boto3
import io

def processing():
    dateset_url = 'https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv'

    print("Downloading dataset...")    
    df = pd.read_csv(dateset_url, index_col=0)
    print("...done.")    

    print("Cleaning dataset...")
    #Date column extraction
    df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'])
    df['trans_year']=df['trans_date_trans_time'].dt.year
    df['trans_month']=df['trans_date_trans_time'].dt.month
    df['trans_day']=df['trans_date_trans_time'].dt.day
    df['trans_hour']=df['trans_date_trans_time'].dt.hour
    df['trans_minutes']=df['trans_date_trans_time'].dt.minute
    df['trans_seconds']=df['trans_date_trans_time'].dt.second
    df = df.drop('trans_date_trans_time', axis=1)

    #Date of birth column extraction
    df['dob']=pd.to_datetime(df['dob'])
    df['dob_year']=df['dob'].dt.year
    df['dob_month']=df['dob'].dt.month
    df['dob_day']=df['dob'].dt.day
    df = df.drop('dob', axis=1)  

    #Convert 
    def convert_to_string(dataset, columns):
        for column in columns:
            dataset[column] = dataset[column].astype(str)
                
    columns_to_convert = ['merchant','category','first','last','gender','street','city','state','job']  
    convert_to_string(df, columns_to_convert)

    print("...done.") 
    
    print("Exporting cleaned dataset...")
    
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    DATASET_BUCKET = os.getenv('DATASET_BUCKET')
    
    s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    
    
    with io.StringIO() as csv_buffer:
        df.to_csv(csv_buffer, index=False)

        response = s3_client.put_object(
            Bucket=DATASET_BUCKET, Key="fraud-detection/cleaned-dataset.csv", Body=csv_buffer.getvalue()
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")
            
            
if __name__=="__main__":
    processing()