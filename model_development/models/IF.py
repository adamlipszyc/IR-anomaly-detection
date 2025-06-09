import os 
import numpy as np 
import pickle
from .model import BaseModel
from sklearn.ensemble import IsolationForest
class IsolationForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_samples=1.0, contamination=0.05, threshold: float = None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.threshold = threshold
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination
        )

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
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "contamination": self.contamination,
                "threshold": self.threshold
            }, file)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as file:
            state = pickle.load(file)
        instance = cls(
            n_estimators=state["n_estimators"],
            max_samples=state["max_samples"],
            contamination=state["contamination"],
            threshold=state["threshold"]
        )
        instance.model = state["model"]
        return instance
