import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .model import BaseModel

class LSTMAnomalyDetector(nn.Module, BaseModel):
    """
    An LSTM-based model for supervised anomaly detection in time series data.
    This model uses an LSTM (or stacked LSTMs) to learn temporal patterns from sequences 
    (e.g., interest rate swap trading behavior) and outputs an anomaly probability for each sequence.
    Inherits from BaseModel and implements fit, predict, save, and load methods.
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.3, learning_rate=0.001):
        super().__init__()  # initialize BaseModel if needed
        # Store configuration
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define LSTM layer (or layers)
        # batch_first=True means input tensors are of shape (batch, seq_len, input_size)
        # If num_layers > 1, PyTorch LSTM applies dropout between LSTM layers (not on the last layer output).
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                             batch_first=True, bidirectional=bidirectional, 
                             dropout=dropout if num_layers > 1 else 0.0)
        # Determine the output dimension from LSTM to the fully connected layer.
        # If bidirectional, LSTM has two directions, effectively doubling the hidden state size.
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        # Define a fully connected layer that maps the LSTM final hidden state to a single anomaly score.
        self.fc = nn.Linear(lstm_out_dim, 1)
        # Define a sigmoid activation for output to obtain a probability (0 to 1).
        # (Alternatively, we could omit this and use nn.BCEWithLogitsLoss for stability.)
        self.sigmoid = nn.Sigmoid()

        # Define the optimizer (Adam is a good default for LSTMs) and loss function (binary cross-entropy).
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification

        # Move model to GPU if available for faster training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, x):
        """
        Forward pass through the LSTM and fully connected layer.
        :param x: Input tensor of shape (batch, seq_len, input_size).
        :return: Output tensor of shape (batch, 1) with anomaly probabilities.
        """
        # x is expected to be a float tensor of shape (batch_size, sequence_length, num_features)
        # Pass through LSTM. We only need the final hidden state for classification.
        # out_seq shape: (batch, seq_len, num_directions*hidden_size) [if batch_first=True]
        # h_n shape: (num_layers * num_directions, batch, hidden_size) -> final hidden state for each layer
        # c_n shape: (num_layers * num_directions, batch, hidden_size) -> final cell state for each layer (not used here)
        out_seq, (h_n, c_n) = self.lstm(x)
        # h_n contains the final hidden state for each layer and direction.
        # For classification, use the final hidden state from the last LSTM layer.
        if self.bidirectional:
            # h_n is of shape (num_layers*2, batch, hidden_size).
            # The last two entries of h_n correspond to the last layer's forward and backward hidden states.
            forward_hidden = h_n[-2]  # shape (batch, hidden_size), last layer forward direction
            backward_hidden = h_n[-1]  # shape (batch, hidden_size), last layer backward direction
            # Concatenate forward and backward hidden states to get full context.
            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # shape (batch, 2*hidden_size)
        else:
            # h_n is of shape (num_layers, batch, hidden_size). Take the last layer (index -1).
            final_hidden = h_n[-1]  # shape (batch, hidden_size)
        # Pass the final hidden state through the fully connected layer to get logits.
        logits = self.fc(final_hidden)  # shape (batch, 1)
        # Apply sigmoid to get anomaly probability between 0 and 1.
        prob = self.sigmoid(logits)    # shape (batch, 1)
        return prob  # This is the anomaly score (probability of being anomalous).

    def fit(self, X_train, y_train, epochs=20, batch_size=32, verbose=True):
        """
        Train the LSTM model on the given training data.
        :param X_train: Training features, shape (N, seq_len, input_size), e.g., N sequences of length 550.
        :param y_train: Training labels, shape (N,) or (N,1), with 1 for anomaly and 0 for normal.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param verbose: If True, print loss per epoch.
        :return: self
        """
        # Set model to training mode
        self.train()
        # Move model to GPU if available for faster training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        # Prepare training data as TensorDataset for batching
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        # If labels are one-dimensional, unsqueeze to shape (N,1) for compatibility
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(1, epochs+1):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                # Move data to the same device as model
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                # Forward pass: compute model output and loss
                outputs = self.forward(batch_X)          # anomaly probabilities
                loss = self.criterion(outputs, batch_y)  # binary cross-entropy loss
                # Backward pass and optimization
                self.optimizer.zero_grad()  # reset gradients from previous step
                loss.backward()             # compute gradients
                self.optimizer.step()       # update parameters
                epoch_loss += loss.item()
            # Optionally print the average loss for this epoch
            if verbose:
                avg_loss = epoch_loss / len(train_loader)
                print(f"Epoch {epoch}/{epochs} - Training loss: {avg_loss:.4f}")
        return self  # enable method chaining if desired

    def predict(self, X, threshold=None):
        """
        Predict anomaly scores or labels for the given input sequences.
        :param X: Input features, shape (M, seq_len, input_size) for M sequences.
        :param threshold: Optional threshold in [0,1] for classifying anomalies.
                          If provided, outputs binary labels (1 = anomaly if score >= threshold, else 0).
                          If None, outputs the anomaly score/probability for each sequence.
        :return: Numpy array of anomaly scores (M,1) or labels (M,) depending on threshold.
        """
        self.eval()  # set model to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        # Convert input to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = self.forward(X_tensor)  # shape (M, 1) tensor of anomaly probabilities
        scores = scores.cpu().numpy()  # move to CPU and convert to numpy for output
        if threshold is not None:
            # Apply threshold to get binary labels (anomaly if score >= threshold)
            return (scores >= threshold).astype(int).flatten()
        else:
            return scores.flatten()  # return the anomaly probability for each sequence

    def save(self, filepath):
        """
        Save the model parameters to the given file path.
        """
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """
        Load model parameters from the given file path.
        """
        # Load the state dict into the model instance.
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.eval()  # set to evaluation mode after loading (good practice)
