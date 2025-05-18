import os
import pickle
import json
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import logging
import numpy as np
from log.utils import catch_and_log

class BaseModelTrainer:
    def __init__(self, model_type: str, stats: dict):
        self.model_type = model_type
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
            model = IsolationForest(contamination=0.1)
            #contamination is what proportion of the data the model should expect to be anomalous during testing 
            #this has no effect during training
        elif self.model_type == "LOF":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        model.fit(X)
        self.logger.info("Model Trained")
        return model
    
    @catch_and_log(Exception, "Saving model")
    def save(self, model, filepath: str, num_rows: int, train_indices: dict = None) -> None:
        """
        Saves the trained model and optionally the indices used to train it.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as file:
            pickle.dump(model, file)
        self.logger.info("Saved model: %s | Trained on %s rows", filepath, num_rows)

        # if train_indices:
        #     indices_path = filepath[:-4] + "_indices.json" #replace .pkl 
        #     with open(indices_path, "w") as file:
        #         json.dump(train_indices, file)
            
        #     self.logger.info("Saved model indices: %s", indices_path)

    def run(self, X: np.ndarray, model_path: str, train_indices: dict = None) -> None:
        """
        Complete model pipeline: train and save.
        """
        model = self.train(X)
        self.save(model, model_path, len(X), train_indices)
        self.stats[f"{self.model_type} model build"] = "Success"
