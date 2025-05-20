import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
from log.utils import catch_and_log
from .model import BaseModel
from .encoder import Encoder

class AutoencoderBase(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x



class Autoencoder(BaseModel, Encoder):
    def __init__(
        self,
        input_dim: int = 1100,
        encoding_dim: int = 32,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: Optional[str] = None
    ):
        self.model = AutoencoderBase(input_dim, encoding_dim).to(device)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.loss_history: List[float] = []
        self.model_path = model_path

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
            self.save_pickle(self.model_path)


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
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            errors = F.mse_loss(outputs, inputs, reduction='none')
            per_sample_errors = errors.mean(dim=1)
            return per_sample_errors.cpu().numpy()

    @catch_and_log(Exception, "Saving model")
    def save(self, path: str):
        """
        Safely save the model weights (state_dict) to a .pkl file.
        """
        state = {
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load the model from a .pkl file.
        """
        state = torch.load(path)
        input_dim = state["input_dim"]
        encoding_dim = state["encoding_dim"]

        # Create a new instance of the model
        instance = cls(input_dim=input_dim, encoding_dim=encoding_dim)
        
        # Load the saved weights into the model
        instance.model.load_state_dict(state["model_state"])
        
        print(f"Model loaded from {path}")
        return instance


# def build_autoencoder(input_dim, encoding_dim=32):
#     """
#     Build and compile an autoencoder model for anomaly detection.
    
#     Args:
#     - input_dim (int): The dimension of input data.
#     - encoding_dim (int): The dimension of the encoded representation.
    
#     Returns:
#     - autoencoder (keras.Model): The autoencoder model.
#     """
#     input_layer = Input(shape=(input_dim,))
#     encoded = Dense(encoding_dim, activation='relu')(input_layer)
#     decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
#     autoencoder = Model(input_layer, decoded)
#     autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    
#     return autoencoder

# def train_autoencoder(data, encoding_dim=32, epochs=50, batch_size=256):
#     """
#     Train an autoencoder on the data.
    
#     Args:
#     - data (np.ndarray): The data to train the autoencoder.
#     - encoding_dim (int): Dimension of the encoded representation.
#     - epochs (int): Number of epochs for training.
#     - batch_size (int): Batch size for training.
    
#     Returns:
#     - autoencoder (keras.Model): The trained autoencoder model.
#     """
#     input_dim = data.shape[1]
#     autoencoder = build_autoencoder(input_dim, encoding_dim)
#     autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)
#     return autoencoder

# def compute_reconstruction_error(autoencoder, data):
#     """
#     Compute the reconstruction error for each data point.
    
#     Args:
#     - autoencoder (keras.Model): The trained autoencoder.
#     - data (np.ndarray): The data for which to calculate the reconstruction error.
    
#     Returns:
#     - reconstruction_error (np.ndarray): The reconstruction error for each data point.
#     """
#     reconstructed = autoencoder.predict(data)
#     reconstruction_error = np.mean(np.square(data - reconstructed), axis=1)
#     return reconstruction_error

def plot_reconstruction_error(reconstruction_error, threshold=None):
    """
    Plot the reconstruction error and optionally display an anomaly detection threshold.
    
    Args:
    - reconstruction_error (np.ndarray): The reconstruction error for each data point.
    - threshold (float): Optional threshold value for anomaly detection.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_error, bins=50, alpha=0.7)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
        plt.legend()
    
    plt.show()

def detect_anomalies(reconstruction_error, threshold):
    """
    Detect anomalies based on the reconstruction error threshold.
    
    Args:
    - reconstruction_error (np.ndarray): The reconstruction error for each data point.
    - threshold (float): The threshold above which data points are considered anomalies.
    
    Returns:
    - anomalies (np.ndarray): An array with -1 for anomalies and 1 for normal data points.
    """
    anomalies = np.where(reconstruction_error > threshold, -1, 1)
    return anomalies

# def main():
#     # Load the data
#     file_path = 'training_data/original_data/vectorized_data.csv'  # Replace with your actual data file path
#     data = load_data(file_path)
    
#     # Standardize the data (important for neural networks)
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
    
#     # Train the autoencoder
#     autoencoder = train_autoencoder(data_scaled, encoding_dim=32, epochs=50, batch_size=256)
    
#     # Compute reconstruction error on the training data
#     reconstruction_error = compute_reconstruction_error(autoencoder, data_scaled)
    
#     # Plot the reconstruction error
#     plot_reconstruction_error(reconstruction_error)
    
#     # Set a threshold based on the distribution of reconstruction errors
#     threshold = np.percentile(reconstruction_error, 95)  # You can adjust this percentile
#     print(f"Anomaly detection threshold set at: {threshold}")
    
#     # Detect anomalies
#     anomalies = detect_anomalies(reconstruction_error, threshold)
#     print(f"Detected anomalies: {np.sum(anomalies == -1)} anomalies detected.")
    
#     # Optionally: Evaluate the results if ground truth is available
#     # For demonstration purposes, we will assume that ground truth labels are available in a separate file.
#     # ground_truth = load_data('ground_truth_labels.csv')  # 1 for normal, -1 for anomaly
#     # roc_score = roc_auc_score(ground_truth, reconstruction_error)
#     # print(f"ROC AUC Score: {roc_score}")

# if __name__ == "__main__":
#     main()
