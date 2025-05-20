import pickle 
import logging 
import os 
import numpy as np
from log.utils import catch_and_log


class ModelHandler:
    def __init__(self, model_file_path, scaler_file_path):
        self.model_path = model_file_path
        self.logger = logging.getLogger(self.__class__.__name__)

        #Load the trained model
        self.model = self.load_trained_model()

        #Load the corresponding scaler
        self.scaler = self.load_scaler(scaler_file_path)



    # Function to load a trained model using pickle
    @catch_and_log(Exception, "Loading trained model")
    def load_trained_model(self):
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
        
        self.logger.info("Model found: %s", self.model_path)
        return model

    @catch_and_log(Exception, "Loading scaler")
    def load_scaler(self, scaler_file_path):
        with open(scaler_file_path, 'rb') as file:
            scaler = pickle.load(file)
        
        self.logger.info("Scaler found: %s", scaler_file_path)
        return scaler

    @catch_and_log(Exception, "Getting scaler path")
    def get_scaler_path(self, file_path="") -> str:
        """
        Given a model path like 'models/one_svm/batch_model_....pkl',
        return the corresponding scaler path with 'scaler_' prefixed to the filename.
        """
        path = file_path if file_path else self.model_path
        dir_path, filename = os.path.split(path)
        scaler_filename = f"scaler_{filename}"
        scaler_path = os.path.join(dir_path, scaler_filename)
        self.logger.info("Scaler path found: %s", scaler_path)
        return scaler_path
    
    @catch_and_log(Exception, "Carrying out prediction")
    def predict(self, X_test):
        """
        Take the already trained anomaly detection model and predict anomalies on the test set.
        Output: 1 for anomaly, 0 for normal.
        """
        
        # Predict anomalies in the test data
        y_pred = self.model.predict(X_test)
        y_pred = np.where(y_pred == 1, 0.0, 1.0)  # OneClassSVM returns 1 for normal and -1 for anomaly. We need to map to 0 and 1.
        
        return y_pred
    
    @catch_and_log(Exception, "Preparing the data")
    def prepare_data(self, X_test: np.ndarray):
        #Reshape the test data to prepare for scaling
        flattened_x_test = X_test.flatten()
        reshaped = flattened_x_test.reshape(-1, 1)

        X_test_scaled = self.scaler.transform(reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        return X_test_scaled

