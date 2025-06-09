import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import logging
from .model import BaseModel
from log.utils import catch_and_log



class BaseAnoGAN():

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.G = None 
        self.D = None
        self.opt_D = None 
        self.opt_G = None
    
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