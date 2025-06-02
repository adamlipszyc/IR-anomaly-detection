import torch
import torch.nn as nn
import numpy as np
import logging
from .CNN_supervised_base import BaseCNNAnomalyDetector

class CNN2DAnomalyDetector(BaseCNNAnomalyDetector):
    def __init__(self, kernel_size= 5, seq_length=550, num_epochs=10, lr=0.001, threshold=None):
        super().__init__()
        self.seq_length = seq_length
        self.input_channels = 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(kernel_size, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.relu = nn.ReLU()

        # pooled_len = (seq_length - 4) // 2  # after conv1 + pool
        # pooled_len = (pooled_len - 4) // 2  # after conv2 + pool
        flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc_out = nn.Linear(64, 1)
        self.activation = nn.Sigmoid()
        self.kernel_size = kernel_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = num_epochs 
        self.lr = lr
        self.threshold = threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)    

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.seq_length, 2)  # simulate a single input
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.view(-1, 1, self.seq_length, 2)  # (batch, 1, 550, 2)
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 32, H, 1)
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 64, H2, 1)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.activation(self.fc_out(x))
