import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from ae.decoder import Decoder
from vae.encoder import Encoder

class VariationalAutoencoder(pl.LightningModule):

    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class=Encoder,
        decoder_class=Decoder,
        num_input_channels: int = 1,
        width: int = 28,
        height: int = 28,
        beta: float = 1.0,
        beta_warmup_epochs: int = 10,
        use_cyclical_annealing: bool = True,
        cycle_length: int = 100,
        max_epochs: int = 100,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.encoder = encoder_class(
            num_input_channels, base_channel_size, latent_dim
        )
        self.decoder = decoder_class(
            num_input_channels, base_channel_size, latent_dim
        )

        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.use_cyclical_annealing = use_cyclical_annealing
        self.cycle_length = cycle_length
        self.max_epochs = max_epochs
        
        self.example_input_array = torch.zeros(
            2, num_input_channels, width, height
        )

    def get_current_beta(self):
        """
        Get beta value with warmup schedule to prevent posterior collapse.
        
        Returns:
            beta: Current beta value
        """
        epoch = self.current_epoch
        if self.beta_warmup_epochs == 0:
            return self.beta
        
        if epoch < self.beta_warmup_epochs:
            # Linear warmup from 0 to beta
            return self.beta * (epoch / self.beta_warmup_epochs)
        elif self.use_cyclical_annealing:
            # Cyclical annealing to prevent posterior collapse
            cycle_position = (epoch - self.beta_warmup_epochs) % self.cycle_length
            cycle_progress = cycle_position / self.cycle_length
            return self.beta * (0.5 + 0.5 * np.cos(np.pi * cycle_progress))
        else:
            # Constant beta after warmup
            return self.beta

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization(mu, torch.exp(0.5 * logvar))
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
    def _get_vae_loss(self, batch):
        x, _ = batch
        x_hat_logits, mu, logvar = self.forward(x)

        # print(f"x_hat min/max: {x_hat_logits.min().item()}/{x_hat_logits.max().item()}")
        # print(f"x min/max: {x.min().item()}/{x.max().item()}")

        # Reconstruction loss (same as AE)
        recon_loss = F.binary_cross_entropy_with_logits(
            x_hat_logits, x, reduction='none'
        )
        recon_loss = recon_loss.sum(dim=[1,2,3]).mean(dim=0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        ).mean()

        # Get current beta value with scheduling
        current_beta = self.get_current_beta()
        
        loss = recon_loss + current_beta * kl_loss
        return loss, recon_loss, kl_loss, current_beta

    def training_step(self, batch, batch_idx):
        loss, recon, kl, current_beta = self._get_vae_loss(batch)
        self.log_dict({
            "train_loss": loss,
            "train_recon": recon,
            "train_kl": kl,
            "current_beta": current_beta
        })
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, kl, current_beta = self._get_vae_loss(batch)
        self.log_dict({
            "val_loss": loss,
            "val_recon": recon,
            "val_kl": kl
        })

    def test_step(self, batch, batch_idx):
        loss, recon, kl, current_beta = self._get_vae_loss(batch)
        self.log_dict({
            "test_loss": loss,
            "test_recon": recon,
            "test_kl": kl
        })
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # ReduceLROnPlateau scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        return {
            'optimizer': optimizer,
            'gradient_clip_val': 1.0,  # PyTorch Lightning handles gradient clipping
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }