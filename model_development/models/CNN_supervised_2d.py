import torch
import torch.nn as nn
import numpy as np
import logging
from .CNN_supervised_base import BaseCNNAnomalyDetector

class CNN2DAnomalyDetector(BaseCNNAnomalyDetector):
    def __init__(self, kernel_size=5, seq_length=550, num_epochs=10, lr=0.001, fc1_size=64, out_channels=[32, 64], activation="relu", threshold=None):
        super().__init__()
        self.seq_length = seq_length
        self.input_channels = 1
        self.out_channels = out_channels.copy()

        # Choose activation
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []


        first_out = out_channels[0]
        out_channels = out_channels[1:]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=first_out, kernel_size=(kernel_size, 2))
        layers.append(self.conv1)
        layers.append(self.activation_fn)
        layers.append(nn.MaxPool2d(kernel_size=(2,1)))

        in_channels = first_out
        for out_channel in out_channels:
            layers.append(nn.Conv2d(in_channels, out_channel, kernel_size=(kernel_size, 1)))
            layers.append(self.activation_fn)
            layers.append(nn.MaxPool2d(kernel_size=(2,1)))
            in_channels = out_channel

        self.feature_extractor = nn.Sequential(*layers)

        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1))
        # self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        # self.relu = nn.ReLU()

        flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(flattened_size, fc1_size)    
        self.fc_out = nn.Linear(fc1_size, 1)
        
        self.kernel_size = kernel_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = num_epochs 
        self.lr = lr
        self.threshold = threshold
        self.fc1_size = fc1_size 
        self.activation = activation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)    

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.seq_length, 2)  # simulate a single input
            x = self.feature_extractor(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.view(-1, 1, self.seq_length, 2)  # (batch, 1, 550, 2)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.activation_fn(self.fc1(x))
        return self.fc_out(x)
