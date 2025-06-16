import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os 
import pickle
import numpy as np
import logging
from .model import BaseModel
from log.utils import catch_and_log

# Generator network: MLP mapping latent vector z to 1100-d output
class Generator(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=[], output_dim=1100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        # Define a series of linear layers with increasing dimensionality
        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        # Final output layer to desired output_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


# Discriminator network: MLP mapping 1100-d input to probability of being real.
# Also yields an intermediate feature representation for feature matching.
class Discriminator(nn.Module):
    def __init__(self, input_dim=1100, hidden_dims=[]):
        super(Discriminator, self).__init__()

        layers = []
        in_dim = input_dim
        # First layer (no BatchNorm on first layer as per DCGAN practice)
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        in_dim = hidden_dims[0]
        # Subsequent hidden layers with BatchNorm
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h
        # We will not add the final output layer here in the sequential model,
        # because we need to grab the features from the last hidden layer.
        self.feature_extractor = nn.Sequential(*layers)
        # Final output layer to a single logit (no Sigmoid here; we'll use BCEWithLogitsLoss)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x):
        # Get intermediate feature representation
        features = self.feature_extractor(x)
        # Compute the discriminator output (logit)
        logit = self.classifier(features)
        # Return both for use in feature matching if needed
        return features, logit.view(-1)  # flatten logit to 1D for convenience
        

# AnoGAN model class integrating generator and discriminator
class AnoGAN(BaseModel): 
    def __init__(self, latent_dim=128, input_dim=1100, hidden_dims = [], lr=0.0001, beta1=0.5, beta2=0.999, n_steps=100, lambda_weight = 0.1, threshold = None, device=None):
        super().__init__()
        # Model components
        self.G = Generator(latent_dim=latent_dim, hidden_dims=hidden_dims, output_dim=input_dim)
        self.D = Discriminator(input_dim=input_dim, hidden_dims=hidden_dims)
        # Save latent_dim for use in inference
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        # Choose device (CPU or CUDA)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device)
        self.D.to(self.device)
        # Optimizers for G and D
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))
        # Loss function (binary cross-entropy with logits for stability)
        self.bce_loss = nn.BCEWithLogitsLoss()
        # For logging
        self.history = {"D_loss": [], "G_loss": []}
        self.threshold = threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.n_steps = n_steps 
        self.lambda_weight = lambda_weight
        self.lr = lr 
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

    @catch_and_log(Exception, "Fitting the AnoGAN model")
    def fit(self, X, epochs=20, batch_size=32, verbose=True):
        """
        Train the GAN on the given data X (only normal data for AnoGAN).
        X can be a NumPy array or torch Tensor of shape (N, input_dim).
        """
        self.G.train()
        self.D.train()
        # Convert X to tensor if it's a numpy array
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        # No labels (unsupervised), create DataLoader for batching
        dataset = torch.utils.data.TensorDataset(X)  # single tensor dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(1, epochs+1):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            batch_count = 0
            for batch in data_loader:
                batch_count += 1
                real_batch = batch[0].to(self.device)  # batch is tuple (data,)
                bs = real_batch.size(0)
                # Train Discriminator:
                self.opt_D.zero_grad()
                # Real data loss:
                real_feat, real_logit = self.D(real_batch)
                # Real labels = 1
                real_labels = torch.ones(bs, device=self.device)
                loss_real = self.bce_loss(real_logit, real_labels)
                # Fake data loss:
                # Sample random latent vectors
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_data = self.G(z).detach()  # .detach() to avoid generator gradient in D update
                fake_feat, fake_logit = self.D(fake_data)
                # Fake labels = 0
                fake_labels = torch.zeros(bs, device=self.device)
                loss_fake = self.bce_loss(fake_logit, fake_labels)
                # Total D loss
                d_loss = loss_real + loss_fake
                d_loss.backward()
                self.opt_D.step()
                
                # Train Generator:
                self.opt_G.zero_grad()
                # Generate new fake data (since G was updated by D's .detach, we need fresh generation)
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_data = self.G(z)
                _, fake_logit = self.D(fake_data)  # get logit for new fake
                # Generator tries to fool D: labels = 1 for these fakes
                gen_labels = torch.ones(bs, device=self.device)
                g_loss = self.bce_loss(fake_logit, gen_labels)
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
                print(f"Epoch {epoch}/{epochs} - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
    
    def _optimize_z(self, x, n_steps=100, lr=1e-2, lambda_weight=0.1):
        """
        Given a single input sample x (tensor), optimize a latent vector z to minimize
        the AnoGAN mapping loss. Returns the optimized z and the corresponding losses.
        """
        # Ensure model in eval mode for inference
        self.G.eval()
        self.D.eval()
        # Initialize z from normal distribution
        z = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
        # Use an optimizer for z (we can use Adam or SGD; here Adam for faster convergence)
        z_optimizer = optim.Adam([z], lr=lr)
        # Optimize for n_steps
        for step in range(n_steps):
            z_optimizer.zero_grad()
            # Generate from z
            x_gen = self.G(z)
            # Compute residual loss (L1 norm between x and x_gen)
            residual_loss = torch.sum(torch.abs(x_gen - x))
            # Compute discrimination (feature matching) loss (L1 between D features)
            real_feat, _ = self.D(x)
            gen_feat, _ = self.D(x_gen)
            discrimination_loss = torch.sum(torch.abs(real_feat - gen_feat))
            # Combined loss
            total_loss = (1 - lambda_weight) * residual_loss + lambda_weight * discrimination_loss
            total_loss.backward()
            z_optimizer.step()
        # After optimization, compute final losses
        final_residual = residual_loss.item()
        final_discrimination = discrimination_loss.item()
        final_total = total_loss.item()
        return z.detach(), final_total, final_residual, final_discrimination, x_gen.detach()
    
    @catch_and_log(Exception, "Predicting model")
    def predict(self, X: np.ndarray, threshold: bool = False):
        """
        Compute the anomaly score for each sample in X.
        If return_stats=True, also return the reconstructed output G(z) and individual loss components.
        """
        self.logger.info("Starting AnoGAN prediction")
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        X = X.to(self.device)
        self.G.eval()
        self.D.eval()
        anomaly_scores = []
        outputs = []        # reconstructed outputs
        residual_losses = []
        feature_losses = []
        # Iterate over samples (optimize each one)
        for i in range(X.size(0)):
            x = X[i:i+1]  # single sample
            _, total_loss, res_loss, feat_loss, x_gen = self._optimize_z(x, n_steps=self.n_steps, lr=1e-2, lambda_weight=self.lambda_weight)
            anomaly_scores.append(total_loss)
            outputs.append(x_gen.cpu().numpy())
            residual_losses.append(res_loss)
            feature_losses.append(feat_loss)
        anomaly_scores = np.array(anomaly_scores)

        thresholded_scores = None
        if threshold:
            if self.threshold is None:
                raise ValueError("Threshold must be added for thresholding")
            thresholded_scores = (anomaly_scores > self.threshold).astype(int)

        return anomaly_scores, thresholded_scores
    
    @catch_and_log(Exception, "Saving AnoGAN model")
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
        self.logger.info("Saved MLP-AnoGAN model to %s | Trained on %d rows", path, num_rows)

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
