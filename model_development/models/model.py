import os
import pickle 
import logging
import numpy as np
from abc import ABC, abstractmethod
from log.utils import catch_and_log

class BaseModel(ABC):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None


    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train=None) -> None:
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray, threshold: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str, num_rows: int):
        """
        Saves the trained model and optionally the indices used to train it.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(self.model, file)

        self.logger.info("Saved model: %s | Trained on %s rows", path, num_rows)

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        
        instance = cls(model)

        return instance
