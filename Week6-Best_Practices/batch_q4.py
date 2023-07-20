#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os


def get_input_path(year, month) -> str:
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month)-> str:
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename)-> pd.DataFrame:
    """Read data"""
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None and S3_ENDPOINT_URL != "": 
        options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

        df = pd.read_parquet(filename, storage_options=options)
        return df
    else:
        df = pd.read_parquet(filename)
        return df


def prepare_data(df: pd.DataFrame, categorical: list) -> pd.DataFrame:
    """Prepare data"""
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def write_data(filename, df) -> None:
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if S3_ENDPOINT_URL is not None and S3_ENDPOINT_URL != "":
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
    else:
        df.to_parquet(filename, engine='pyarrow', index=False)
        
        

def main():
    
    
    year = int(sys.argv[1])
    month = int(sys.argv[2])

     
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('./homework/model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file)
    df = prepare_data(df, categorical)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    print('predicted duration sum:', df_result['predicted_duration'].sum())

    print(output_file)
    write_data(output_file, df_result)
    
    
if __name__ == '__main__':
    
    INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"

    main()