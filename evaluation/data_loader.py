import numpy as np
import pandas as pd 
import logging 
import glob 
import os 
from log.utils import catch_and_log
from .config import CLASS_SIZE, TEST_DATA_DIR

class DataLoader:

    def __init__(self, split: int, fifty_fifty: bool):
        self.logger = logging.getLogger(self.__class__.__name__)
        test_file_dir = os.path.join(TEST_DATA_DIR, f"split_{split}")
        self.test_file_path = os.path.join(test_file_dir, "test_50_50.csv" if fifty_fifty else "test_95_5.csv")

    @catch_and_log(Exception, "Loading labeled test data")
    def load_labeled_test_data(self):
        """
        Loads test data with labels from a CSV file. Assumes last column is the label.
        
        Sets self.X_test and self.y_test accordingly.
        
        Args:
            file_path (str): Path to a CSV file with test data and labels.
        """
        self.logger.info(f"Loading test data with labels from: {self.test_file_path}")

        df = pd.read_csv(self.test_file_path, header=None)
        
        if df.shape[1] < 2:
            raise ValueError("Test file must contain at least 2 columns (features + label).")

        X_test = df.iloc[:, :-1].values.astype(float)  # all columns except last
        y_test = df.iloc[:, -1].values.astype(int)     # last column = label (0 or 1)

        self.logger.info(f"Loaded test data shape: {X_test.shape}, Labels: {np.bincount(y_test)}")

        return X_test, y_test



    @catch_and_log(Exception, "Loading anomalous data")
    def load_anomalous_data(self, file_paths, directory=False, num_samples=None):
        """
        Loads the data from a CSV file and creates a label vector Y where all entries are 1.
        Assumes all rows in the file are anomalous examples.
        """

        if directory:
            file_paths = glob.glob(os.path.join(file_paths[0], "*.csv"))
            if len(file_paths) == 0:
                raise RuntimeError("No CSV files found in directory.")

        anomalous_data = None
        for file_path in file_paths:
            data = pd.read_csv(file_path, header=None)
            
            X = data.values  # All columns

            np.random.shuffle(X)
            
            if anomalous_data is not None:
                anomalous_data = np.vstack((anomalous_data, X))
            else:
                anomalous_data = X
        
        # If num_samples is set and smaller than total, randomly select subset
        if num_samples is not None and num_samples < len(anomalous_data):
            indices = np.random.choice(len(anomalous_data), size=num_samples, replace=False)
            anomalous_data = anomalous_data[indices]


        Y = np.ones(len(anomalous_data)) # Label 1 for each row
        return anomalous_data, Y

    @catch_and_log(Exception, "Loading good data")
    def load_good_data(self, file_paths, ensemble, directory=False, used_indices={}):
        """
        Loads the data from multiple CSV files and creates a label vector Y where all entries are 0.
        Assumes all rows in the file are good examples.
        """

        if directory:
            file_paths = glob.glob(os.path.join(file_paths[0], "*.csv"))
            if len(file_paths) == 0:
                raise RuntimeError("No CSV files found in directory.")

        good_data = None
        for file_path in file_paths:
            data = pd.read_csv(file_path, header=None)

            if file_path in used_indices:
                data = data[~data.index.isin(used_indices[file_path])]
            
            X = data.values  # All columns

            np.random.shuffle(X)

            if good_data is not None:
                good_data = np.vstack((good_data, X))
            else:
                good_data = X
        
        if ensemble:
            indices = np.random.choice(len(good_data), size=CLASS_SIZE, replace=False)
            good_data = good_data[indices]

        Y = np.zeros(len(good_data))         # Label 0 for each row
        return good_data, Y