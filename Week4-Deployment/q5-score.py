import pickle
import pandas as pd
import argparse

 
# Read and Preprocess Data

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



# Load the trained and saved model along with the feature engineering objects

def load_model():
    with open('./homework/model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model
 


# Predict on Validation Data

def predict(df):

    dv, model = load_model()

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred



# Predict Ride data for selected month and year and find the mean prediction for that data

def ride_prediction( year: int , month: int ):    
    
    # Input Data URL
    input = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04}-{month:02}.parquet'

    print(f'Reading Data for the month:{month:02} of the year:{year:04} to predict the mean riding time\n')
    df = read_data(input)
    print(f'Predicting...\n')
    pred_value = predict(df)
    print(f'The predicted mean riding time for the month:{month:02} of the year:{year:04} is {round(pred_value.mean(),2)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mean ride duration prediction')
    parser.add_argument('year', type=int, help='Enter year(from 2022):')
    parser.add_argument('month', type=int,  help='Enter month(1 to 12):')
    args = parser.parse_args()

    year = args.year
    month = args.month

    ride_prediction(year, month)


