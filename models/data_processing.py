import os

import pandas as pd

def processing():
    dateset_url = 'https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv'
    filepath = './src/fraudTest.csv'

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

    if not os.path.isdir('src'):
        os.mkdir('src')
    df.to_csv('./src/cleaned_dataset.csv', index=False)
    print("...done.")