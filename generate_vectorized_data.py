import json
import pandas as pd
import numpy as np
import csv


def load_data(json_data):
    """
    Load the given JSON data into a pandas DataFrame.
    """
    return pd.json_normalize(json_data)

def flatten_positions(dataframe):
    """
    Flatten nested JSON positions into individual columns for each 'start', 'traded', and drop name
    """
    positions = dataframe['positions'].explode()
    positions_df = pd.json_normalize(positions)
    return positions_df



def vectorize_data(dataframe, columns=None):
    """
    Convert the selected columns of the DataFrame to a NumPy array.
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=np.number).columns
    return dataframe[columns].values

def add_feature_ratios(df):
    """
    Add a new feature that captures the ratio between 'start' and 'end'.
    Can be useful in identifying anomalies based on the relationship between start and traded values.
    """
    df['start_end_ratio'] = df['start'] / (df['traded'] + 1e-6)  # Adding small epsilon to avoid division by zero
    return df



if __name__ == "__main__":
    data = None

    
    with open("training_data/prepped_data.json", "r") as file:
        data = json.load(file)

    with open("training_data/vectorized_data.csv", "w") as file:
        writer = csv.writer(file)
        for row in data:

    
            df = load_data(row)

            df = flatten_positions(df)
            
            arr = vectorize_data(df).flatten()

            writer.writerow(arr)
           


