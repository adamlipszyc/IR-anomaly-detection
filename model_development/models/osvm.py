import os 
import numpy as np 
import pickle
from .model import BaseModel
from sklearn.svm import OneClassSVM

class OneSVMModel(BaseModel):
    def __init__(self, kernel="rbf", nu=0.05, gamma="scale", threshold: float = None):
        super().__init__()
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.threshold = threshold
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X_train: np.ndarray, y_train=None) -> None:
        self.model.fit(X_train)

    def predict(self, X_test: np.ndarray, threshold: bool = False) -> np.ndarray:
        scores = -self.model.decision_function(X_test)
        thresholded_scores = None
        if threshold:
            if self.threshold is None:
                raise ValueError("Threshold not set")
            thresholded_scores = (scores >= self.threshold).astype(int)
        return scores, thresholded_scores


    def save(self, path: str, num_rows: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump({
                "model": self.model,
                "kernel": self.kernel,
                "nu": self.nu,
                "gamma": self.gamma,
                "threshold": self.threshold
            }, file)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as file:
            state = pickle.load(file)
        instance = cls(
            kernel=state["kernel"],
            nu=state["nu"],
            gamma=state["gamma"],
            threshold=state["threshold"]
        )
        instance.model = state["model"]
        return instance
