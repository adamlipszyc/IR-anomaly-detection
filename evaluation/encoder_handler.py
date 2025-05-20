import pickle 
import logging 
import os 
import numpy as np
from log.utils import catch_and_log
from model_development.models.pca import PCAencoder
from model_development.models.autoencoder import Autoencoder
from model_development.models.encoder import Encoder

ENCODER_REGISTRY = {
    "pca": PCAencoder,
    "autoencoder": Autoencoder,
}

class EncoderHandler:
    def __init__(self, encoder_file_path, encoder_name: str = None):
        self.encoder_path = encoder_file_path
        self.logger = logging.getLogger(self.__class__.__name__)

        self.encoder_name = encoder_name

        #Load the trained model
        self.encoder: Encoder = self.load_trained_encoder()



    # Function to load a trained model using pickle
    @catch_and_log(Exception, "Loading trained model")
    def load_trained_encoder(self):


        encoder_class: Encoder = ENCODER_REGISTRY.get(self.encoder_name)
        if encoder_class is None:
            raise ValueError(f"Unknown encoder type '{self.encoder_name}'")

        self.logger.info(f"Using encoder class: {encoder_class.__name__}")
        encoder = encoder_class.load(self.encoder_path)

        self.logger.info("Encoder successfully loaded from: %s", self.encoder_path)

        return encoder
    
    @catch_and_log(Exception, "Encoding data")
    def encode(self, X_test):
        """
        Take the already trained encoder model and encode the test set.
        """
        
        # Predict anomalies in the test data
        X_encoded = self.encoder.encode(X_test)

        return X_encoded
    
   

