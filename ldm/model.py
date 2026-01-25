"""
Latent Diffusion Model (LDM) training module with PyTorch Lightning.
Combines a pre-trained VAE with a diffusion model operating in latent space.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Literal
import time

from .unet_latent import UNetLatent
from ddpm.noise_scheduler import DDPMScheduler, DDIMScheduler
from vae.model import VariationalAutoencoder


class LatentDiffusionModel(pl.LightningModule):
    """
    Latent Diffusion Model (LDM) that operates in VAE latent space.

    This model trains a diffusion model on the latent representations from a
    pre-trained VAE, significantly reducing computational cost while maintaining
    generation quality.

    Args:
        vae_checkpoint: Path to pre-trained VAE checkpoint
        latent_dim: Dimension of VAE latent space
        model_channels: Base number of UNet channels
        channel_mult: Channel multipliers for UNet
        num_res_blocks: Number of residual blocks per level
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Type of noise schedule
        lr: Learning rate
        num_inference_steps: Number of steps for DDIM sampling
        freeze_vae: Whether to freeze VAE weights during training
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
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
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

        # UNet model for noise prediction in latent space
        self.model = UNetLatent(
            in_channels=latent_dim,
            model_channels=model_channels,
            out_channels=latent_dim,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            dropout=dropout,
            num_heads=num_heads
        )

        # DDPM scheduler for training
        self.scheduler = DDPMScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule
        )

        # DDIM scheduler for fast inference
        self.ddim_scheduler = None

        self.lr = lr
        self.latent_dim = latent_dim

    def forward(self, z, t):
        """Predict noise in latent space"""
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
        """Training step: predict noise in latent space"""
        x, _ = batch

        # Encode to latent space
        with torch.no_grad():
            z = self.encode_to_latent(x)

        # Sample random timesteps
        batch_size = z.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        # Add noise to latent vectors
        noise = torch.randn_like(z)
        z_t, _ = self.scheduler.add_noise(z, t, noise)

        # Predict noise
        predicted_noise = self.model(z_t, t)

        # Compute loss (simple MSE)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, _ = batch

        # Encode to latent space
        with torch.no_grad():
            z = self.encode_to_latent(x)

        batch_size = z.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        noise = torch.randn_like(z)
        z_t, _ = self.scheduler.add_noise(z, t, noise)

        predicted_noise = self.model(z_t, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        method: Literal['ddpm', 'ddim'] = 'ddpm',
        num_inference_steps: Optional[int] = None,
        return_all_steps: bool = False,
        return_latents: bool = False
    ):
        """
        Generate samples using DDPM or DDIM in latent space.

        Args:
            num_samples: Number of samples to generate
            method: Sampling method ('ddpm' or 'ddim')
            num_inference_steps: Number of inference steps (for DDIM)
            return_all_steps: If True, return images at all timesteps
            return_latents: If True, also return latent vectors

        Returns:
            Generated images, optionally with intermediate steps and latents
        """
        self.model.eval()
        self.vae.eval()

        # Start from random noise in latent space
        z = torch.randn(num_samples, self.latent_dim, device=self.device)

        # Sample in latent space
        if method == 'ddpm':
            z_final = self._sample_ddpm(z, return_all_steps)
        elif method == 'ddim':
            if num_inference_steps is None:
                num_inference_steps = self.hparams.num_inference_steps
            z_final = self._sample_ddim(z, num_inference_steps, return_all_steps)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Decode to image space
        if return_all_steps:
            z_final, z_steps = z_final
            x_steps = [torch.sigmoid(self.decode_from_latent(z_step)) for z_step in z_steps]
            x_final = x_steps[-1]

            if return_latents:
                return x_final, x_steps, z_final, z_steps
            return x_final, x_steps
        else:
            x_final = torch.sigmoid(self.decode_from_latent(z_final))

            if return_latents:
                return x_final, z_final
            return x_final

    def _sample_ddpm(self, z, return_all_steps=False):
        """Sample using DDPM in latent space"""
        self.scheduler.to(self.device)

        all_steps = [z] if return_all_steps else None

        # Reverse diffusion process in latent space
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(z, t_batch)

            # Remove noise
            z = self.scheduler.sample_prev_timestep(z, t_batch, predicted_noise)

            if return_all_steps:
                all_steps.append(z)

        if return_all_steps:
            return z, all_steps
        return z

    def _sample_ddim(self, z, num_inference_steps, return_all_steps=False):
        """Sample using DDIM in latent space"""
        # Create DDIM scheduler if needed
        if self.ddim_scheduler is None or self.ddim_scheduler.num_inference_steps != num_inference_steps:
            self.ddim_scheduler = DDIMScheduler(
                self.scheduler,
                num_inference_steps=num_inference_steps,
                eta=0.0
            )

        self.ddim_scheduler.to(self.device)

        all_steps = [z] if return_all_steps else None

        # Reverse diffusion with DDIM in latent space
        timesteps = self.ddim_scheduler.timesteps.flip(0)
        for i, t in enumerate(timesteps):
            t_idx = len(timesteps) - 1 - i
            t_batch = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(z, t_batch)

            # DDIM step
            z = self.ddim_scheduler.sample_prev_timestep(z, t_idx, predicted_noise)

            if return_all_steps:
                all_steps.append(z)

        if return_all_steps:
            return z, all_steps
        return z

    def configure_optimizers(self):
        """Configure Adam optimizer"""
        # Only optimize diffusion model parameters (VAE is frozen)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    @torch.no_grad()
    def compare_sampling_speed(
        self,
        num_samples: int = 16,
        methods: list = ['ddpm', 'ddim'],
        ddim_steps: int = 50
    ):
        """
        Compare sampling speed between different methods.

        Args:
            num_samples: Number of samples to generate
            methods: List of methods to compare
            ddim_steps: Number of steps for DDIM

        Returns:
            Dictionary with timing results
        """
        self.eval()
        results = {}

        for method in methods:
            if method == 'ddpm':
                start_time = time.time()
                _ = self.sample(num_samples, method='ddpm')
                elapsed = time.time() - start_time
                results['ldm_ddpm'] = {
                    'time': elapsed,
                    'steps': self.scheduler.num_timesteps,
                    'time_per_sample': elapsed / num_samples
                }
            elif method == 'ddim':
                start_time = time.time()
                _ = self.sample(num_samples, method='ddim', num_inference_steps=ddim_steps)
                elapsed = time.time() - start_time
                results['ldm_ddim'] = {
                    'time': elapsed,
                    'steps': ddim_steps,
                    'time_per_sample': elapsed / num_samples
                }

        return results
