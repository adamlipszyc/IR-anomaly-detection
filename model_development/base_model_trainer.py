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

class BaseModelTrainer:
    def __init__(self, model_type: str, model_args: dict, stats: dict):
        self.model_type = model_type
        self.model_args: dict = model_args
        self.stats = stats
        self.logger = logging.getLogger(self.__class__.__name__)

    @catch_and_log(Exception, "Training model")
    def train(self, X: np.ndarray):
        """
        Trains a model based on type.
        """

        self.logger.info("Training model")
        if self.model_type == "one_svm":
            model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
        elif self.model_type == "isolation_forest":
            model = IsolationForest(contamination=0.05)
            #contamination is what proportion of the data the model should expect to be anomalous during testing 
            #this has no effect during training
        elif self.model_type == "LOF":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        elif self.model_type == "autoencoder":
            model = Autoencoder(**self.model_args)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model.fit(X)

        self.logger.info("Model Trained")
        return model
    
    @catch_and_log(Exception, "Saving model")
    def save(self, model, model_path: str, num_rows: int, train_indices: dict = None) -> None:
        """
        Saves the trained model and optionally the indices used to train it.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if self.model_type == "autoencoder":
            model.save(model_path)

        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        

        self.logger.info("Saved model: %s | Trained on %s rows", model_path, num_rows)

        # if train_indices:
        #     indices_path = filepath[:-4] + "_indices.json" #replace .pkl 
        #     with open(indices_path, "w") as file:
        #         json.dump(train_indices, file)
            
        #     self.logger.info("Saved model indices: %s", indices_path)
    
    def extract_encoder_info(self, path):
        """
        Extracts encoder name and dimension from a path of the form:
        'models/hybrid/{encoder_name}_{encoder_dim}/...'

        Returns:
            (encoder_name: str, encoder_dim: int)
        Raises:
            ValueError if the path does not match the expected format.
        """
        parts = os.path.normpath(path).split(os.sep)

        try:
            encoder_part = parts[2]  # index 2 corresponds to {encoder_name}_{encoder_dim}
            
            return encoder_part
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid path format for extracting encoder info: {path}") from e

    def run(self, X: np.ndarray, model_path: str, train_indices: dict = None) -> None:
        """
        Complete model pipeline: train and save.
        """
        model = self.train(X)
        self.save(model, model_path, len(X), train_indices)
        
        task = f"{self.model_type} model build"

        if "hybrid" in model_path:
            encoder = self.extract_encoder_info(model_path)
            task = f"{self.model_type} model build (using encoder {encoder})"

        self.stats[task] = "Success"
