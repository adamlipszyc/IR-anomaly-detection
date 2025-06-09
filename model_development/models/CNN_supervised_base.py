import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import pickle
import csv
from .model import BaseModel
from abc import abstractmethod

class BaseCNNAnomalyDetector(nn.Module, BaseModel):
    def __init__(self):
        super().__init__()
        self.threshold = None
        self.kernel_size = None
        self.epochs = None 
        self.lr = None
        self.out_channels = None 
        self.activation = None
        self.fc1_size = None
        self.loss_history = []

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
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(loader):.4f}")

            self.loss_history.append(epoch_loss / len(loader))


    def predict(self, X_test: np.ndarray, threshold: bool = False) -> np.ndarray:
        """Return anomaly scores, or binary labels if threshold is True."""
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Disable gradient tracking for inference
        with torch.no_grad():
            logits = self.forward(X_tensor)             # Raw outputs from model
            probs = torch.sigmoid(logits)               # Convert to [0, 1] range
            scores = probs.cpu().numpy()                # Move to CPU, convert to NumPy

        thresholded_scores = None
        # Optionally return binary anomaly decisions based on threshold
        if threshold:
            if self.threshold is None:
                raise ValueError("Threshold must be added for thresholding")
            print(self.threshold)
            thresholded_scores = (scores >= self.threshold).astype(int)
        return scores, thresholded_scores

    def save(self, path: str, num_rows: int):
        """Save model weights and configuration using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'state_dict': self.state_dict(),
            'threshold': self.threshold,
            'kernel_size': self.kernel_size,
            'lr': self.lr,
            'num_epochs': self.epochs,
            'out_channels': self.out_channels,
            'activation': self.activation,
            'fc1_size': self.fc1_size
        }

        with open(os.path.join(os.path.dirname(path), "losses.csv"), "w") as f:
            writer = csv.writer(f)
            print(self.loss_history)
            writer.writerow(self.loss_history) 

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
            lr=state['lr'],
            out_channels=state['out_channels'],
            activation=state['activation'],
            fc1_size=state['fc1_size']
        )
        model.load_state_dict(state['state_dict'])
        model.eval()
        return model