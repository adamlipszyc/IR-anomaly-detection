import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
from log.utils import catch_and_log
from .model import BaseModel
from .encoder import Encoder

class AutoencoderBase(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] = None, activation="leaky_relu",encoding_dim: int = 32):
        """
        hidden_dims: e.g., [512, 256, 128] 
        encoding_dim: dimensions of the bottleneck
        """
        super().__init__()

        # Choose activation
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        dims = hidden_dims.copy() if hidden_dims is not None else []
        dims.append(encoding_dim)

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(self.activation_fn)
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        reversed_dims = list(reversed(dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            decoder_layers.append(self.activation_fn)
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = self.decoder(x)
        return x



class Autoencoder(BaseModel, Encoder):
    def __init__(
        self,
        input_dim: int = 1100,
        encoding_dim: int = 32,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 20,
        hidden_dims: List[int] = None,
        activation: str = "leaky_relu",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        self.model = AutoencoderBase(input_dim, hidden_dims, activation, encoding_dim).to(device)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.loss_history: List[float] = []
        self.model_path = model_path
        self.threshold = threshold
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.activation = activation
        self.threshold = threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Autoencoder created with : lr = {lr}, batch_size = {batch_size}, num_epochs = {num_epochs}, encoding_dim: {encoding_dim}")

    @catch_and_log(Exception, "Training autoencoder")
    def fit(self, X_train: np.ndarray, y_train = None) -> None:
        """
        Train the autoencoder on the given NumPy array.

        Args:
            X_train (np.ndarray): Training data of shape (N, input_dim)

        """
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_train_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in loader:
                inputs = batch[0].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)

            epoch_loss /= len(dataset)
            self.loss_history.append(epoch_loss)
            print(f"Epoch {epoch+1}/{self.num_epochs} | Loss: {epoch_loss:.6f}")

        if self.model_path:
            self.save(self.model_path)


    @catch_and_log(Exception, "Encoding data")
    def encode(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            encoded = self.model.encoder(inputs)
            return encoded.cpu().numpy()


    @catch_and_log(Exception, "Reconstructing data")
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            return outputs.cpu().numpy()

    @catch_and_log(Exception, "Predicting")
    def predict(self, X: np.ndarray, threshold: bool = False) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            errors = F.mse_loss(outputs, inputs, reduction='none')
            per_sample_errors = errors.mean(dim=1)
            result = per_sample_errors.cpu().numpy()
            thresholded_scores = None
            if threshold:
                if self.threshold is None:
                    raise ValueError("Threshold must be added for thresholding")
                thresholded_scores = (result > self.threshold).astype(int)
            return result, thresholded_scores

    @catch_and_log(Exception, "Saving model")
    def save(self, path: str, num_rows: int):
        """
        Safely save the model weights (state_dict) to a .pkl file.
        """
        state = {
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "threshold": self.threshold,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation
        }
        torch.save(state, path)
        self.logger.info("Saved auto-encoder model to %s | Trained on %d rows", path, num_rows)

    @classmethod
    def load(cls, path: str):
        """
        Load the model from a .pkl file.
        """
        state = torch.load(path)
        input_dim = state["input_dim"]
        encoding_dim = state["encoding_dim"]
        threshold = state["threshold"]
        lr = state["lr"]
        batch_size = state["batch_size"]
        num_epochs = state["num_epochs"]
        hidden_dims = state["hidden_dims"]
        activation = state["activation"]

        # Create a new instance of the model
        instance = cls(input_dim=input_dim, encoding_dim=encoding_dim, hidden_dims=hidden_dims , lr=lr, batch_size=batch_size, num_epochs=num_epochs, activation=activation, threshold=threshold)
        
        # Load the saved weights into the model
        instance.model.load_state_dict(state["model_state"])
        
        print(f"Model loaded from {path}")
        return instance

