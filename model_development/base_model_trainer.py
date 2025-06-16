import os

import logging
import numpy as np
from log.utils import catch_and_log
from .models.autoencoder import Autoencoder
from .models.IF import IsolationForestModel
from .models.lof import LOFModel
from .models.osvm import OneSVMModel
from .models.anoGAN import AnoGAN
from .models.CNN_anoGAN import CNN_AnoGAN
from .models.CNN_supervised_2d import CNN2DAnomalyDetector
from .models.lstm import LSTMAnomalyDetector

from .config import base_models


class BaseModelTrainer:
    def __init__(self, model_type: str, model_args: dict, stats: dict):
        self.model_type = model_type
        self.model_args: dict = model_args
        self.stats = stats
        self.logger = logging.getLogger(self.__class__.__name__)

    @catch_and_log(Exception, "Training model")
    def train(self, X: np.ndarray, y: np.ndarray = None):
        """
        Trains a model based on type.
        """

        self.logger.info("Training model")

        if y is not None:
            if self.model_type == "cnn_supervised_2d":
                model = CNN2DAnomalyDetector(**self.model_args)
            elif self.model_type == "lstm":
                model = LSTMAnomalyDetector(**self.model_args)
            else:
                raise ValueError(f"Unknown supervised model type: {self.model_type}")
            
            model.fit(X, y_train=y)
            self.logger.info("Supervised model trained")
            return model

        if self.model_type == "one_svm":
            model = OneSVMModel(**self.model_args)
        elif self.model_type == "isolation_forest":
            model = IsolationForestModel(**self.model_args)
            #contamination is what proportion of the data the model should expect to be anomalous during testing 
            #this has no effect during training
        elif self.model_type == "LOF":
            model = LOFModel(**self.model_args)
        elif self.model_type == "autoencoder":
            model = Autoencoder(**self.model_args)
        elif self.model_type == "anogan":
            model = AnoGAN(**self.model_args)
        elif self.model_type == "cnn_anogan":
            model = CNN_AnoGAN(**self.model_args)
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

       
        model.save(model_path, num_rows)
        

        self.logger.info("Saved model: %s | Trained on %s rows", model_path, num_rows)

    
    def extract_encoder_info(self, path):
        """
        Extracts encoder name and dimension from a path of the form:
        'models/hybrid/{encoder_name}/...'

        Returns:
            (encoder_name: str)
        Raises:
            ValueError if the path does not match the expected format.
        """
        parts = os.path.normpath(path).split(os.sep)

        try:
            encoder_part = parts[2]  # index 2 corresponds to {encoder_name}_{encoder_dim}
            
            return encoder_part
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid path format for extracting encoder info: {path}") from e

    def run(self, X: np.ndarray, model_path: str, y = None, train_indices: dict = None, return_model: bool = False):
        """
        Complete model pipeline: train and save.
        """
        model = self.train(X, y)
        self.save(model, model_path, len(X), train_indices)
        
        task = f"{self.model_type} model build"

        if "hybrid" in model_path:
            encoder = self.extract_encoder_info(model_path)
            task = f"{self.model_type} model build (using encoder {encoder})"

        self.stats[task] = "Success"

        if return_model:
            return model
