"""
Latent Rectified Flow model for efficient image generation.
Combines VAE with Rectified Flow in latent space.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Literal
import time

from ldm.unet_latent import UNetLatent
from .flow_scheduler import RectifiedFlowScheduler
from vae.model import VariationalAutoencoder


class LatentRectifiedFlow(pl.LightningModule):
    """
    Latent Rectified Flow model operating in VAE latent space.

    Combines:
    1. Pre-trained VAE for encoding/decoding
    2. Rectified Flow for latent space generation
    3. Velocity prediction in low-dimensional space

    This significantly reduces computational cost compared to pixel-space
    while maintaining generation quality.

    Args:
        vae_checkpoint: Path to pre-trained VAE checkpoint
        vae_model: Pre-trained VAE model (alternative to checkpoint)
        latent_dim: Dimension of VAE latent space
        model_channels: Base number of UNet channels
        channel_mult: Channel multipliers for UNet
        num_res_blocks: Number of residual blocks per level
        num_timesteps: Number of timesteps for training
        lr: Learning rate
        num_inference_steps: Default number of steps for sampling
        freeze_vae: Whether to freeze VAE weights during training
        dropout: Dropout probability
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        vae_checkpoint: Optional[str] = None,
        vae_model: Optional[VariationalAutoencoder] = None,
        latent_dim: int = 128,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        num_timesteps: int = 1000,
        lr: float = 2e-4,
        num_inference_steps: int = 50,
        freeze_vae: bool = True,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae_model'])

        # Load or use provided VAE
        if vae_model is not None:
            self.vae = vae_model
        elif vae_checkpoint is not None:
            self.vae = VariationalAutoencoder.load_from_checkpoint(vae_checkpoint)
        else:
            raise ValueError("Must provide either vae_checkpoint or vae_model")

        # Freeze VAE if requested
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

        # UNet model for velocity prediction in latent space
        self.model = UNetLatent(
            in_channels=latent_dim,
            model_channels=model_channels,
            out_channels=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            dropout=dropout,
            num_heads=num_heads
        )

        # Rectified Flow scheduler
        self.scheduler = RectifiedFlowScheduler(num_timesteps=num_timesteps)

        self.lr = lr
        self.latent_dim = latent_dim

    def forward(self, z, t):
        """Predict velocity in latent space"""
        return self.model(z, t)

    @torch.no_grad()
    def encode_to_latent(self, x):
        """Encode images to latent space using VAE encoder"""
        self.vae.eval()
        mu, logvar = self.vae.encoder(x)
        # Use mean for encoding (deterministic)
        return mu

    @torch.no_grad()
    def decode_from_latent(self, z):
        """Decode latent vectors to images using VAE decoder"""
        self.vae.eval()
        return self.vae.decoder(z)

    def training_step(self, batch, batch_idx):
        """
        Training step: predict velocity in latent space.

        Loss: E[||v(z_t, t) - (z_1 - z_0)||^2]
        """
        x, _ = batch

        # Encode to latent space
        with torch.no_grad():
            z_1 = self.encode_to_latent(x)

        # Sample random timesteps
        batch_size = z_1.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        # Get interpolated latents and true velocity
        z_t, z_0, true_velocity = self.scheduler.add_flow(z_1, t)

        # Predict velocity
        predicted_velocity = self.model(z_t, t)

        # Compute loss
        loss = nn.functional.mse_loss(predicted_velocity, true_velocity)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, _ = batch

        # Encode to latent space
        with torch.no_grad():
            z_1 = self.encode_to_latent(x)

        batch_size = z_1.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        z_t, z_0, true_velocity = self.scheduler.add_flow(z_1, t)
        predicted_velocity = self.model(z_t, t)

        loss = nn.functional.mse_loss(predicted_velocity, true_velocity)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        method: Literal['euler', 'heun'] = 'euler',
        num_inference_steps: Optional[int] = None,
        return_all_steps: bool = False,
        return_latents: bool = False
    ):
        """
        Generate samples using Rectified Flow in latent space.

        Args:
            num_samples: Number of samples to generate
            method: Integration method ('euler' or 'heun')
            num_inference_steps: Number of ODE integration steps
            return_all_steps: If True, return images at all timesteps
            return_latents: If True, also return latent vectors

        Returns:
            Generated images, optionally with intermediate steps and latents
        """
        self.model.eval()
        self.vae.eval()

        if num_inference_steps is None:
            num_inference_steps = self.hparams.num_inference_steps

        # Start from random noise in latent space
        z_0 = torch.randn(num_samples, self.latent_dim, device=self.device)

        # Integrate ODE in latent space
        if method == 'euler':
            z_1, z_steps = self.scheduler.sample_euler(
                self.model, z_0, num_inference_steps
            )
        elif method == 'heun':
            z_1, z_steps = self.scheduler.sample_heun(
                self.model, z_0, num_inference_steps
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Decode to image space
        if return_all_steps:
            x_steps = [torch.sigmoid(self.decode_from_latent(z_step)) for z_step in z_steps]
            x_1 = x_steps[-1]

            if return_latents:
                return x_1, x_steps, z_1, z_steps
            return x_1, x_steps
        else:
            x_1 = torch.sigmoid(self.decode_from_latent(z_1))

            if return_latents:
                return x_1, z_1
            return x_1

    def configure_optimizers(self):
        """Configure Adam optimizer"""
        # Only optimize flow model parameters (VAE is frozen)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    @torch.no_grad()
    def compare_sampling_speed(
        self,
        num_samples: int = 16,
        methods: list = ['euler', 'heun'],
        step_counts: list = [10, 25, 50, 100]
    ):
        """
        Compare sampling speed across different methods and step counts.

        Args:
            num_samples: Number of samples to generate
            methods: List of integration methods to compare
            step_counts: List of step counts to test

        Returns:
            Dictionary with timing results
        """
        self.eval()
        results = {}

        for method in methods:
            for steps in step_counts:
                start_time = time.time()
                _ = self.sample(
                    num_samples=num_samples,
                    method=method,
                    num_inference_steps=steps
                )
                elapsed = time.time() - start_time

                key = f'lrf_{method}_{steps}'
                results[key] = {
                    'time': elapsed,
                    'steps': steps,
                    'time_per_sample': elapsed / num_samples,
                    'method': method
                }

        return results
