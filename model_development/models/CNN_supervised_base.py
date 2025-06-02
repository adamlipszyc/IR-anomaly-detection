import torch
import torch.nn as nn
import os
import numpy as np
import pickle
from .model import BaseModel
from abc import abstractmethod

class BaseCNNAnomalyDetector(nn.Module, BaseModel):
    def __init__(self):
        super().__init__()
        self.threshold = None
        self.kernel_size = None
        self.epochs = None 
        self.lr = None

    @abstractmethod
    def forward(self, X):
        pass

    def fit(self, X_train: np.ndarray, y_train=None):
        """Train the model using binary cross-entropy loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        # Convert input data to PyTorch tensors and move to GPU
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

    def predict(self, X_test: np.ndarray, threshold: bool = False) -> np.ndarray:
        """Return anomaly scores, or binary labels if threshold is True."""
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Disable gradient tracking for inference
        with torch.no_grad():
            scores = self.forward(X_tensor).cpu().numpy()

        # Optionally return binary anomaly decisions based on threshold
        if threshold:
            if self.threshold is None:
                raise ValueError("Threshold must be added for thresholding")
            return (scores >= self.threshold).astype(int)
        return scores

    def save(self, path: str, num_rows: int):
        """Save model weights and configuration using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'state_dict': self.state_dict(),
            'threshold': self.threshold,
            'kernel_size': self.kernel_size,
            'lr': self.lr,
            'num_epochs': self.epochs
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

        self.logger.info("Saved CNN-supervised model to %s | Trained on %d rows", path, num_rows)

    @classmethod
    def load(cls, path: str):
        """Load model weights and configuration from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        model = cls(
            threshold=state['threshold'],
            kernel_size=state['kernel_size'],
            num_epochs=state['num_epochs'],
            lr=state['lr']
        )
        model.load_state_dict(state['state_dict'])
        model.eval()
        return model