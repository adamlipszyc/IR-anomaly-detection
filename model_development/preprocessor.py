import numpy as np
import os 
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
from log.utils import catch_and_log

class Preprocessor:
    def __init__(self):
        """
        Initialize the preprocessor.
        """
        self.scaler = None
        self.logger = logging.getLogger(self.__class__.__name__)


    @catch_and_log(Exception, "Scaling data")
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Flattens, scales, and reshapes data using the selected scaler.
        Returns preprocessed data with the same shape.
        """

        #flatten our data set into one large 1D array 
        flattened_training_data = data.flatten()
        
        #Normalize the data 
        array_reshaped = flattened_training_data.reshape(-1, 1)
        self.scaler = MinMaxScaler()
        normalized_array = self.scaler.fit_transform(array_reshaped).flatten()

        
        #reshape the 1d array back to its original shape
        reshaped_data = normalized_array.reshape(data.shape)
    

        return reshaped_data


    @catch_and_log(Exception, "Saving scaler")
    def save(self, scaler_file_path: str) -> None:
        """
        Saves the fitted scaler to a file using pickle.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted yet.")
        
        os.makedirs(os.path.dirname(scaler_file_path), exist_ok=True)
        
        with open(scaler_file_path, 'wb') as file:
            pickle.dump(self.scaler, file)
            
        self.logger.info("Scaler saved: %s", scaler_file_path)
