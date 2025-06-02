import torch
import torch.nn as nn
import numpy as np
import logging
from .CNN_supervised_base import BaseCNNAnomalyDetector


class CNN1DSupervisedAnomalyDetector(BaseCNNAnomalyDetector):
    def __init__(self, kernel_size = 5, num_epochs=10, lr=0.001, threshold = None):
        super().__init__()
        # super(CNNAnomalyDetector, self).__init__()
        # Define CNN layers
        input_channels = 1
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=kernel_size, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # Optional third conv layer (commented out or can be enabled as needed)
        # self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Compute size after conv/pooling layers to define linear layer input:
        # After conv1 (stride = 2) + pool: length = seq_len/4 (if padding keeps length same for conv)
        # After conv2 + pool: length = seq_len/8
        pooled_len = 1100 // 8  # 1100//8 = 137 (assuming seq_len divisible by 2 twice)
        conv2_channels = 64
        # If using conv3 + pool:
        # pooled_len = seq_len // 8 (approx 68), conv3_channels = 128
        # Define fully connected layers
        self.fc1 = nn.Linear(conv2_channels * pooled_len, 64)   # hidden dense layer
        self.fc_out = nn.Linear(64, 1)                          # output layer (single score)
        # Sigmoid to output probability between 0 and 1
        self.activation = nn.Sigmoid()
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.seq_length = 1100
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = num_epochs 
        self.lr = lr 
    
    def forward(self, x):
        # Convert input from (batch_size, seq_len, channels) → (batch_size, channels, seq_len)
        x = x.view(-1, 1, self.seq_length)    #  (batch, 1, 1100)
        # Apply first conv + ReLU + pool
        x = self.pool(self.relu(self.conv1(x)))
        # Apply second conv + ReLU + pool
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten the conv feature maps to a 1D vector per example
        x = x.view(x.size(0), -1)
        # Dense hidden layer with ReLU activation
        x = self.relu(self.fc1(x))
        # Final output: sigmoid to get anomaly score
        return self.activation(self.fc_out(x))
