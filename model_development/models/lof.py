
import os 
import numpy as np 
import pickle
from .model import BaseModel
from sklearn.neighbors import LocalOutlierFactor

class LOFModel(BaseModel):
    def __init__(self, n_neighbors=20, metric="euclidean", contamination=0.05, threshold: float = None):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.contamination = contamination
        self.threshold = threshold
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            metric=metric,
            contamination=contamination,
            novelty=True
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
                "n_neighbors": self.n_neighbors,
                "metric": self.metric,
                "contamination": self.contamination,
                "threshold": self.threshold
            }, file)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as file:
            state = pickle.load(file)
        instance = cls(
            n_neighbors=state["n_neighbors"],
            metric=state["metric"],
            contamination=state["contamination"],
            threshold=state["threshold"]
        )
        instance.model = state["model"]
        return instance
