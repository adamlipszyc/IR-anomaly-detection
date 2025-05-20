import os
import pickle
import json
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import logging
import numpy as np
from log.utils import catch_and_log
from .models.autoencoder import Autoencoder
from .models.pca import PCAencoder
from .models.encoder import Encoder

class EncoderTrainer:
    def __init__(self, encoder_name: str, encoding_dim: int, stats: dict):
        self.encoder_name = encoder_name
        self.encoding_dim = encoding_dim
        self.stats = stats
        self.logger = logging.getLogger(self.__class__.__name__)

    @catch_and_log(Exception, "Training model")
    def train(self, X: np.ndarray):
        """
        Trains a model based on type.
        """
        encoder = None
        encoded_data = None
       
        if self.encoder_name == "autoencoder":
            self.logger.info("Training autoencoder")
            input_dim = X.shape[1]
            encoder = Autoencoder(input_dim=input_dim, encoding_dim=self.encoding_dim)
        elif self.encoder_name == "pca":
            self.logger.info("Fitting PCA")
            encoder = PCAencoder(n_components=self.encoding_dim)

        encoder.fit(X)
        
        self.logger.info("Encoding data for base model")
        encoded_data = encoder.encode(X)

        self.logger.info("Encoder Trained")
        return encoder, encoded_data
    
    @catch_and_log(Exception, "Saving model")
    def save(self, encoder: Encoder, encoder_path: str, num_rows: int, train_indices: dict = None) -> None:
        """
        Saves the trained model and optionally the indices used to train it.
        """
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

        
        encoder.save(encoder_path)
        self.logger.info("Saved encoder: %s | Trained on %s rows", encoder_path, num_rows)


        # if train_indices:
        #     indices_path = filepath[:-4] + "_indices.json" #replace .pkl 
        #     with open(indices_path, "w") as file:
        #         json.dump(train_indices, file)
            
        #     self.logger.info("Saved model indices: %s", indices_path)

    def run(self, X: np.ndarray, encoder_path: str, train_indices: dict = None):
        """
        Complete model pipeline: train and save.
        """
        encoder, encoded_data = self.train(X)
        self.save(encoder, encoder_path, len(X), train_indices)

        
        task = f"{self.encoder_name} -{self.encoding_dim} dimensions build"
        self.stats[task] = "Success"

        return encoded_data
