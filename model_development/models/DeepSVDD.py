import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ========== Step 2: Define the Encoder ==========

class Encoder(nn.Module):
    def __init__(self, input_dim=1100, encoding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )

    def forward(self, x):
        return self.net(x)

# ========== Step 3: Initialize Center ==========

def initialize_center(model, dataloader):
    model.eval()
    with torch.no_grad():
        all_outputs = []
        for (x,) in dataloader:
            outputs = model(x)
            all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, dim=0)
        return all_outputs.mean(dim=0)

# ========== Step 4: Train Deep SVDD ==========

def train_deep_svdd(model, dataloader, center, epochs=30, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x,) in dataloader:
            optimizer.zero_grad()
            encoded = model(x)
            loss = torch.mean(torch.sum((encoded - center) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# ========== Step 5: Scoring ==========

def compute_scores(model, data_tensor, center):
    model.eval()
    with torch.no_grad():
        encoded = model(data_tensor)
        distances = torch.sum((encoded - center) ** 2, dim=1)
    return distances

