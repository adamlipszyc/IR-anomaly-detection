
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import logging
from typing import List, Optional, Tuple

from log.utils import catch_and_log
from .model import BaseModel

class LSTMAutoencoderBase(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, encoding_dim: int):
        super().__init__()

        self.seq_len = 550
        self.feature_dim = 2
        self.encoder = nn.LSTM(self.feature_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.bottleneck = nn.Linear(hidden_dim, encoding_dim)
        self.decoder_input = nn.Linear(encoding_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, self.feature_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # Reshape flat input (N, 1100) to (N, 550, 2)
        x = x.view(-1, self.seq_len, self.feature_dim)

        # Encode
        enc_out, _ = self.encoder(x)
        bottleneck = self.bottleneck(enc_out[:, -1, :])  # Use last hidden state

        # Repeat bottleneck across sequence length
        repeat_bottleneck = bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decode
        dec_input = self.decoder_input(repeat_bottleneck)
        dec_out, _ = self.decoder(dec_input)

        # Flatten back to (N, 1100)
        return dec_out.view(-1, self.seq_len * self.feature_dim)



class LSTMAutoencoder(BaseModel):
    def __init__(
        self,
        input_dim: int = 1100,  # Always 1100 for raw input
        hidden_dim: int = 64,
        encoding_dim: int = 32,
        num_layers: int = 1,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: Optional[str] = None,
        threshold: Optional[float] = None
    ):
        self.seq_len = 550
        self.feature_dim = 2
        self.model = LSTMAutoencoderBase(hidden_dim, num_layers, encoding_dim).to(device)
        
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.model_path = model_path
        self.threshold = threshold
        self.loss_history: List[float] = []

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"LSTM-AE initialized for reshaped input (550, 2) from flat 1100-dim vectors")


    @catch_and_log(Exception, "Training LSTM autoencoder")
    def fit(self, X_train: np.ndarray, y_train=None) -> None:
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


    @catch_and_log(Exception, "Predicting with LSTM autoencoder")
    def predict(self, X_test: np.ndarray, threshold: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            errors = F.mse_loss(outputs, inputs, reduction='none')
            per_sample_errors = errors.view(errors.size(0), -1).mean(dim=1)
            result = per_sample_errors.cpu().numpy()

            if threshold:
                if self.threshold is None:
                    raise ValueError("Threshold must be specified for thresholding")
                thresholded_scores = (result > self.threshold).astype(int)
                return result, thresholded_scores

            return result, None

    @catch_and_log(Exception, "Saving LSTM model")
    def save(self, path: str, num_rows: int):
        state = {
            "model_state": self.model.state_dict(),
            "hidden_dim": self.hidden_dim,
            "encoding_dim": self.encoding_dim,
            "num_layers": self.num_layers,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "threshold": self.threshold,
        }
        torch.save(state, path)
        self.logger.info("Saved LSTM-AE model to %s | Trained on %d rows", path, num_rows)

    @classmethod
    def load(cls, path: str):
        state = torch.load(path)
        instance = cls(
            hidden_dim=state["hidden_dim"],
            encoding_dim=state["encoding_dim"],
            num_layers=state["num_layers"],
            lr=state["lr"],
            batch_size=state["batch_size"],
            num_epochs=state["num_epochs"],
            threshold=state["threshold"]
        )
        instance.model.load_state_dict(state["model_state"])
        print(f"Model loaded from {path}")
        return instance
