import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import argparse 

def load_data(file_path, directory=False):
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file
    :return: Pandas DataFrame containing the data
    """
    if not directory:
        return pd.read_csv(file_path)

    if directory:
        #Trains in batches with ensemble voting 
        # TODO
        pass 

def min_max_normalize_data(array, scaler_file_path):
    """
    Normalize numerical columns using Min-Max scaling.
    Default is to normalize all numerical columns.
    """
    array_reshaped = array.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(array_reshaped).flatten()

    
    with open(scaler_file_path, 'wb') as file:
        pickle.dump(scaler, file)
    

    return normalized_array

# Function to standardize the data
def standardize_data(array):
    """
    Standardize numerical columns to have mean=0 and variance=1.
    Default is to standardize all numerical columns.
    """

    array_reshaped = array.reshape(-1, 1)
    scaler = StandardScaler()
    standardized_array = scaler.fit_transform(array_reshaped).flatten()
    return standardized_array


#need to perform hyper-parameter tuning 
def train_one_class_svm(data):
    """
    Train One-Class SVM model.
    :param data: Preprocessed data
    :return: Trained One-Class SVM model
    """
    model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    model.fit(data)
    return model

def train_isolation_forest(data):
    """
    Train Isolation Forest model.
    :param data: Preprocessed data
    :return: Trained Isolation Forest model
    """
    model = IsolationForest(contamination=0.1)
    #contamination is what proportion of the data the model should expect to be anomalous during testing 
    #this has no effect during training
    model.fit(data)
    return model

def train_lof(data, n_neighbors=20, contamination=0.05):
    """
    Train the Local Outlier Factor (LOF) model for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data, where each row is a 1100-dimensional vector.
    - n_neighbors (int): The number of neighbors to use for LOF. Default is 20.
    - contamination (float): The proportion of outliers in the data. Default is 0.05.
    
    Returns:
    - lof_model (LocalOutlierFactor): The trained LOF model.
    """
    lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_model.fit(data)
    return lof_model

def save_model(model, filepath):
    """
    Save the model to the specified file path using pickle
    """
    with open(filepath, "wb") as file:
            pickle.dump(model, file)

"""
This will train a specified model, by first transforming the data into a large 
1D array then normalizing - so all data is scaled the same, and then training 
on this normalised array. 

"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--one_svm', action='store_true')
    parser.add_argument('-i', '--isolation_forest', action='store_true')
    parser.add_argument('-l', '--local_outlier', action='store_true')

    args = parser.parse_args()

    if not args.one_svm and not args.isolation_forest and not args.local_outlier:
        parser.error("You must specify at least one of -o, -i or -l")

    data = None

    df = pd.read_csv("training_data/vectorized_data.csv", header=None)

    training_data = df.values

    #flatten our data set into one large 1D array 
    flattened_training_data = training_data.flatten()

    #normalize our entire dataset 
    normalized_data = min_max_normalize_data(flattened_training_data, "models/scaler.pkl")

    #reshape the 1d array back to its original shape
    reshaped_data = normalized_data.reshape(training_data.shape)

    if args.isolation_forest:
        #Train the model
        model = train_isolation_forest(reshaped_data)

        # model = train_one_class_svm(reshaped_data)

        save_model(model, "models/isolation_forest/model.pkl")

    if args.one_svm:

        #Train the model

        model = train_one_class_svm(reshaped_data)

        save_model(model, "models/one_svm/model.pkl")
        

    if args.local_outlier:
        #Train the model

        model = train_lof(reshaped_data)

        save_model(model, "models/LOF/model.pkl")
