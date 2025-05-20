import numpy as np
from abc import ABC, abstractmethod

class Encoder(ABC):

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train = None) -> None: 
        pass

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray: 
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        pass

