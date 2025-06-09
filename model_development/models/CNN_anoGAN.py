import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import os
import pickle
import logging
import time
from tqdm import tqdm
from .model import BaseModel
from log.utils import catch_and_log
import math

class Conv2dGenerator(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=[256, 128, 64, 32], target_shape=(550, 2)):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.target_shape = target_shape  # e.g. (550, 2)
        self.output_dim = target_shape[0] * target_shape[1]

        num_upsample_layers = len(hidden_dims) - 1
        upsample_factor = 2 ** num_upsample_layers

        # Choose a safe init height slightly larger, so we can interpolate back
        self.init_height = math.ceil(target_shape[0] / upsample_factor)
        self.init_width = target_shape[1]
        self.init_shape = (self.init_height, self.init_width)

        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.init_height * self.init_width)

        layers = []
        in_channels = hidden_dims[0]
        for out_channels in hidden_dims[1:]:
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), output_padding=(0, 0)
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels

        layers.append(nn.ConvTranspose2d(
            in_channels, 1,
            kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)
        ))

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.hidden_dims[0], self.init_height, self.init_width)
        x = self.net(x)
        x = F.interpolate(x, size=self.target_shape, mode='bilinear', align_corners=True)
        return x.view(z.size(0), -1)  # Flatten to (batch, 1100)



class Conv2dDiscriminator(nn.Module):
    def __init__(self, input_shape=(550, 2), hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        self.input_shape = input_shape  # (550, 2)
        self.input_dim = input_shape[0] * input_shape[1]  # 1100

        layers = []
        in_channels = 1
        for idx, out_channels in enumerate(hidden_dims):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=(5, 1), stride=(2, 1), padding=(1, 0)))
            if idx > 0:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Calculate flattened feature size using dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            out = self.features(dummy)
            self.flattened_size = out.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flattened_size, 1)

    def forward(self, x):
        # x is (B, 1100)
        x = x.view(x.size(0), 1, *self.input_shape)  # (B, 1, 550, 2)
        feat = self.features(x)
        flat = feat.view(x.size(0), -1)
        logit = self.classifier(flat)
        return feat, logit.view(-1)


class CNN_AnoGAN(BaseModel):
    def __init__(self, latent_dim=128, hidden_dims=[256, 128, 64, 32], input_dim=1100, lr = 0.0001, n_steps=100, lambda_weight=0.1, threshold = None, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = Conv2dGenerator(latent_dim, hidden_dims=hidden_dims).to(self.device)
        self.D = Conv2dDiscriminator(hidden_dims=hidden_dims[::-1]).to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.threshold = threshold
        self.history = {"D_loss": [], "G_loss": []}
        self.n_steps = n_steps 
        self.lambda_weight = lambda_weight
        self.lr = lr 
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

    @catch_and_log(Exception, "Fitting CNN-AnoGAN model")
    def fit(self, X, epochs=20, batch_size=64, verbose=True):
        self.G.train()
        self.D.train()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.unsqueeze(1)  # (N, 1, 1100)
        dataloader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            batch_count = 0
            for real in dataloader:
                batch_count += 1
                real = real.to(self.device)
                bs = real.size(0)
                valid = torch.ones(bs, device=self.device)
                fake = torch.zeros(bs, device=self.device)

                # Train Discriminator
                self.opt_D.zero_grad()
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_data = self.G(z).detach()
                _, real_logits = self.D(real)
                _, fake_logits = self.D(fake_data)
                d_loss = self.bce_loss(real_logits, valid) + self.bce_loss(fake_logits, fake)
                d_loss.backward()
                self.opt_D.step()

                # Train Generator
                self.opt_G.zero_grad()

                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_data = self.G(z)
                _, fake_logits = self.D(fake_data)
                g_loss = self.bce_loss(fake_logits, valid)
                g_loss.backward()
                self.opt_G.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()


            # Logging at epoch level
            avg_d_loss = epoch_d_loss / batch_count
            avg_g_loss = epoch_g_loss / batch_count
            self.history["D_loss"].append(avg_d_loss)
            self.history["G_loss"].append(avg_g_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")

    def _optimize_z(self, x, steps, lambda_weight, lr=1e-2):
        self.G.eval()
        self.D.eval() 
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1100)
        z = torch.randn(1, self.latent_dim, requires_grad=True, device=self.device)
        optimizer = optim.Adam([z], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            gen = self.G(z)
            feat_x, _ = self.D(x)
            feat_g, _ = self.D(gen)
            res_loss = torch.sum(torch.abs(x - gen))
            feat_loss = torch.sum(torch.abs(feat_x - feat_g))
            loss = (1 - lambda_weight) * res_loss + lambda_weight * feat_loss
            loss.backward()
            optimizer.step()
            if loss.item() < 1e-4:
                break

        z.detach()
        return loss.item()

    @catch_and_log(Exception, "Predicting CNN-AnoGAN on new data")
    def predict(self, X: np.ndarray, threshold: bool = False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        anomaly_scores = []
        for x in tqdm(X, desc="Predicting anomaly scores"):
            score = self._optimize_z(x, self.n_steps, self.lambda_weight)
            anomaly_scores.append(score)
        anomaly_scores = np.array(anomaly_scores)

        thresholded_scores = None
        if threshold:
            if self.threshold is None:
                raise ValueError("Threshold must be added for thresholding")
            thresholded_scores = (anomaly_scores > self.threshold).astype(int)

        return anomaly_scores, thresholded_scores

   
    @catch_and_log(Exception, "Saving CNN_AnoGAN model")
    def save(self, path: str, num_rows: int):
        """Save model weights and config to a file."""
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'latent_dim': self.latent_dim,
            'threshold': self.threshold,
            'lr': self.lr,
            'lambda_weight': self.lambda_weight,
            'n_steps': self.n_steps,
            'hidden_dims': self.hidden_dims
        }

        history = pd.DataFrame(self.history)
        history.to_csv(os.path.join(os.path.dirname(path), "loss_history.csv"), index=False)


        with open(path, 'wb') as f:
            pickle.dump(state, f)
        self.logger.info("Saved CNN-AnoGAN model to %s | Trained on %d rows", path, num_rows)

    @classmethod
    def load(cls, path: str):
        """Load model weights and config from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        model = cls(latent_dim=state['latent_dim'], lr=state["lr"], lambda_weight=state["lambda_weight"], n_steps=state["n_steps"], hidden_dims=state["hidden_dims"] ,threshold=state['threshold'])
        model.G.load_state_dict(state['G'])
        model.D.load_state_dict(state['D'])
        model.G.to(model.device)
        model.D.to(model.device)
        return model