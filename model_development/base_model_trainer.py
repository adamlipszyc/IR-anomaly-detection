import os
import pickle
import json
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import logging
import numpy as np
from log.utils import catch_and_log
from .complex_models.autoencoder import Autoencoder
from .complex_models.pca import PCAencoder
from .complex_models.encoder import Encoder

class BaseModelTrainer:
    def __init__(self, model_type: str, stats: dict, encoder: str = None, encoding_dim: int = None):
        self.model_type = model_type
        self.encoder = encoder
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
        if self.encoder is not None and self.encoding_dim is not None:
            if self.encoder_type == "auto-encoder":
                self.logger.info("Training autoencoder")
                input_dim = X.shape[1]
                encoder = Autoencoder(input_dim=input_dim, encoding_dim=self.encoding_dim)
            elif self.encoder_type == "pca":
                self.logger.info("Fitting PCA")
                encoder = PCAencoder(n_components=self.encoding_dim)

            encoder.fit(X)
            
            self.logger.info("Encoding data for base model")
            encoded_data = encoder.encode(X)

        self.logger.info("Training model")
        if self.model_type == "one_svm":
            model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
        elif self.model_type == "isolation_forest":
            model = IsolationForest(contamination=0.1)
            #contamination is what proportion of the data the model should expect to be anomalous during testing 
            #this has no effect during training
        elif self.model_type == "LOF":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if encoded_data is not None:
            model.fit(encoded_data)
        else:
            model.fit(X)

        self.logger.info("Model Trained")
        return model, encoder
    
    @catch_and_log(Exception, "Saving model")
    def save(self, model, encoder: Encoder, model_path: str, encoder_path: str, num_rows: int, train_indices: dict = None) -> None:
        """
        Saves the trained model and optionally the indices used to train it.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        
        if encoder is not None:
            encoder.save(encoder_path)
            self.logger.info("Saved encoder: %s | Trained on %s rows", encoder_path, num_rows)

        self.logger.info("Saved model: %s | Trained on %s rows", model_path, num_rows)

        # if train_indices:
        #     indices_path = filepath[:-4] + "_indices.json" #replace .pkl 
        #     with open(indices_path, "w") as file:
        #         json.dump(train_indices, file)
            
        #     self.logger.info("Saved model indices: %s", indices_path)

    def run(self, X: np.ndarray, model_path: str, encoder_path: str, train_indices: dict = None) -> None:
        """
        Complete model pipeline: train and save.
        """
        model, encoder = self.train(X)
        self.save(model, encoder, model_path, encoder_path, len(X), train_indices)

        task = f"{self.model_type} model build"
        if encoder is not None:
            task = f"Hybrid {self.encoder}-{self.model_type} model build"
        self.stats[task] = "Success"
