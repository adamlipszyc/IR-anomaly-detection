import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def load_data(file_path):
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file
    :return: Pandas DataFrame containing the data
    """
    return pd.read_csv(file_path)

def min_max_normalize_data(array):
    """
    Normalize numerical columns using Min-Max scaling.
    Default is to normalize all numerical columns.
    """
    array_reshaped = array.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(array_reshaped).flatten()

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
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
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



"""
This will train a specified model, by first transforming the data into a large 
1D array then normalizing - so all data is scaled the same, and then training 
on this normalised array. 

"""

if __name__ == "__main__":
    data = None

    df = pd.read_csv("training_data/vectorized_data.csv", header=None)

    training_data = df.values

    #flatten our data set into one large 1D array 
    flattened_training_Data = training_data.flatten()

    #normalize our entire dataset 
    normalized_data = min_max_normalize_data(flattened_training_Data)

    #reshape the 1d array back to its original shape
    reshaped_data = normalized_data.reshape(training_data.shape)

    #Train the model
    model = train_one_class_svm(reshaped_data)

    with open("models/one_svm.pkl", "rb") as file:
        pickle.dump(model, file)

    # with open("training_data/vectorized_data.csv", "r") as file:
    #     reader = csv.reader(file)
    #     for row in reader:
            
    #         training_entry = np.array(row, dtype=float)

    #         print(training_entry)

    #         normalized_data = min_max_normalize_data(training_entry)

    #         # standardized_and_normalized_data = standardize_data(normalized_data)
    #         print(normalized_data)
    #         break
