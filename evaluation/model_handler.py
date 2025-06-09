import pickle 
import logging 
import os 
import numpy as np
from log.utils import catch_and_log

from model_development.models.autoencoder import Autoencoder
from model_development.models.anoGAN import AnoGAN
from model_development.models.CNN_anoGAN import CNN_AnoGAN
from model_development.models.CNN_supervised_1d import CNN1DSupervisedAnomalyDetector
from model_development.models.CNN_supervised_2d import CNN2DAnomalyDetector
from model_development.models.lstm import LSTMAnomalyDetector
from model_development.models.model import BaseModel
from model_development.models.IF import IsolationForestModel
from model_development.models.lof import LOFModel
from model_development.models.osvm import OneSVMModel
from model_development.config import base_models

MODEL_REGISTRY = {
    "autoencoder": Autoencoder,
    "anogan": AnoGAN,
    "cnn_anogan": CNN_AnoGAN,
    "cnn_supervised_1d": CNN1DSupervisedAnomalyDetector,
    "cnn_supervised_2d": CNN2DAnomalyDetector,
    "isolation_forest": IsolationForestModel,
    "LOF": LOFModel,
    "one_svm": OneSVMModel,
    "lstm": LSTMAnomalyDetector
}


class ModelHandler:
    def __init__(self, model_file_path, model_name, scaler_file_path):
        self.model_path = model_file_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name

        #Load the trained model
        self.model = self.load_trained_model()

        #Load the corresponding scaler
        self.scaler = self.load_scaler(scaler_file_path)




    # Function to load a trained model using pickle
    @catch_and_log(Exception, "Loading trained model")
    def load_trained_model(self):

        model_class: BaseModel = MODEL_REGISTRY.get(self.model_name)
        if model_class is not None:

            self.logger.info(f"Using model class: {model_class.__name__}")
            model = model_class.load(self.model_path)

            self.logger.info("Model successfully loaded from: %s", self.model_path)

            return model

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
    def predict(self, X_test, threshold: bool = False):
        """
        Take the already trained anomaly detection model and predict anomalies on the test set.
        Output: 1 for anomaly, 0 for normal.
        """
        if threshold:
        # Predict anomalies in the test data
            y_scores, y_pred = self.model.predict(X_test, threshold)
        else:
            y_pred = self.model.predict(X_test)
            y_scores = -self.model.decision_function(X_test)
            y_pred = np.where(y_pred == 1, 0.0, 1.0)  # Base models return 1 for normal and -1 for anomaly. We need to map to 0 and 1.
        
        return y_pred, y_scores
    
    @catch_and_log(Exception, "Preparing the data")
    def prepare_data(self, X_test: np.ndarray):
        #Reshape the test data to prepare for scaling
        flattened_x_test = X_test.flatten()
        reshaped = flattened_x_test.reshape(-1, 1)

        X_test_scaled = self.scaler.transform(reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        return X_test_scaled

