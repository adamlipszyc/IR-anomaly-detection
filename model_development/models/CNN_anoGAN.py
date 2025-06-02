import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import logging
from .model import BaseModel
from log.utils import catch_and_log

class Conv1dGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_length=1100):
        super().__init__()
        self.init_channels = 128
        self.init_length = 16
        self.output_length = output_length

        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_length)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=8, stride=4, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 128, 16)
        x = self.net(x)
        x = F.interpolate(x, size=self.output_length, mode="linear", align_corners=True)
        return x
        # return x[:, :, :self.output_length]  # crop to exact length


class Conv1dDiscriminator(nn.Module):
    def __init__(self, input_dim=1100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        # Compute flattened feature size using a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)
            # print(f"Flattened size: {self.flattened_size}") 

        self.classifier = nn.Linear(self.flattened_size, 1)

    def forward(self, x):
        feat = self.features(x)
        flat = feat.view(x.size(0), -1)
        # print("Flattened shape before classifier:", flat.shape)
        out = self.classifier(flat)
        return feat, out.view(-1)


class CNN_AnoGAN(BaseModel):
    def __init__(self, latent_dim=100, input_dim=1100, lr = 0.0002,threshold = None, device=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.G = Conv1dGenerator(latent_dim, input_dim).to(self.device)
        self.D = Conv1dDiscriminator(input_dim).to(self.device)
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.threshold = threshold

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
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_data = self.G(z).detach()
                _, real_logits = self.D(real)
                _, fake_logits = self.D(fake_data)
                d_loss = self.bce_loss(real_logits, valid) + self.bce_loss(fake_logits, fake)
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

                # Train Generator
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_data = self.G(z)
                _, fake_logits = self.D(fake_data)
                g_loss = self.bce_loss(fake_logits, valid)
                self.opt_G.zero_grad()
                g_loss.backward()
                self.opt_G.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()


            # Logging at epoch level
            avg_d_loss = epoch_d_loss / batch_count
            avg_g_loss = epoch_g_loss / batch_count
            if verbose:
                print(f"Epoch {epoch}/{epochs} - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")

    def _optimize_z(self, x, steps=500, lr=1e-2, lambda_=0.1):
        self.G.eval()
        self.D.eval()
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1100)
        z = torch.randn(1, self.latent_dim, requires_grad=True, device=self.device)
        optimizer = optim.Adam([z], lr=lr)

        for _ in range(steps):
            gen = self.G(z)
            feat_x, _ = self.D(x)
            feat_g, _ = self.D(gen)
            res_loss = torch.sum(torch.abs(x - gen))
            feat_loss = torch.sum(torch.abs(feat_x - feat_g))
            loss = (1 - lambda_) * res_loss + lambda_ * feat_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    @catch_and_log(Exception, "Predicting CNN-AnoGAN on new data")
    def predict(self, X: np.ndarray, threshold: bool = False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        anomaly_scores = []
        for x in X:
            score = self._optimize_z(x)
            anomaly_scores.append(score)
        anomaly_scores = np.array(anomaly_scores)

        if threshold:
            if self.threshold is None:
                raise ValueError("Threshold must be added for thresholding")
            return (anomaly_scores > self.threshold).astype(int)

        return anomaly_scores

    def save(self, path: str, num_rows: int):
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'threshold': self.threshold,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        self.logger.info("Saved CNN-AnoGAN model to %s | Trained on %d rows", path, num_rows)

    @classmethod
    def load(cls, path: str):
        """Load model weights and config from a file"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        model = cls(latent_dim=state['latent_dim'], input_dim=state['input_dim'], threshold=state['threshold'])
        model.G.load_state_dict(state['G'])
        model.D.load_state_dict(state['D'])
        model.G.to(model.device)
        model.D.to(model.device)
        return model
