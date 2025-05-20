
import logging
import numpy as np
import pickle
from sklearn.decomposition import PCA
from .encoder import Encoder
from log.utils import catch_and_log


class PCAencoder(Encoder):

    def __init__(self, n_components):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encoder = None
        self.n_components = n_components

    @catch_and_log(Exception, "Fitting PCA encoder")
    def fit(self, X_train: np.ndarray, y_train = None) -> None:
        """
        Fits the PCA model to the input data
        """
        pca = PCA(n_components=self.n_components)
        pca.fit(X_train)
        self.encoder = pca


    @catch_and_log(Exception, "Encoding data using PCA")
    def encode(self, X: np.ndarray) -> np.ndarray:
        transformed_data = self.encoder.transform(X)
        return transformed_data
    

    @catch_and_log(Exception, "Saving PCA encoder")
    def save(self, path: str):
        """
        Saves the PCA encoder to the specified file path
        """
        if self.encoder is None:
            raise ValueError("PCA encoder has not been fitted yet and cannot be saved.")

        state = {
            "pca_model": self.encoder,
            "n_components": self.n_components
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        self.logger.info(f"PCA encoder saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """
        Loads the PCA encoder from the specified file path using pickle.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(n_components=state["n_components"])
        instance.encoder = state["pca_model"]

        instance.logger = logging.getLogger(cls.__name__)
        instance.logger.info(f"PCA encoder loaded from {path}")

        return instance



