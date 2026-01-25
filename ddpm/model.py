"""
DDPM training module with PyTorch Lightning.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Literal

from .unet import UNet
from .noise_scheduler import DDPMScheduler, DDIMScheduler


class DDPM(pl.LightningModule):
    """
    Denoising Diffusion Probabilistic Model (DDPM) with PyTorch Lightning.

    Args:
        in_channels: Number of input channels
        model_channels: Base number of UNet channels
        channel_mult: Channel multipliers for UNet
        num_res_blocks: Number of residual blocks per level
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Type of noise schedule
        lr: Learning rate
        num_inference_steps: Number of steps for DDIM sampling
    """

    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        lr: float = 2e-4,
        num_inference_steps: int = 50,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()

        # UNet model for noise prediction
        self.model = UNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
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
        self.ddim_scheduler = None  # Will be created when needed

        self.lr = lr

    def forward(self, x, t):
        """Predict noise"""
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        """Training step: predict noise added to images"""
        x, _ = batch  # We don't need labels for unconditional generation

        # Sample random timesteps for each image
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        # Add noise to images
        noise = torch.randn_like(x)
        x_t, _ = self.scheduler.add_noise(x, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Compute loss (simple MSE)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, _ = batch

        batch_size = x.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        noise = torch.randn_like(x)
        x_t, _ = self.scheduler.add_noise(x, t, noise)

        predicted_noise = self.model(x_t, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        img_shape: tuple = (1, 28, 28),
        method: Literal['ddpm', 'ddim'] = 'ddpm',
        num_inference_steps: Optional[int] = None,
        return_all_steps: bool = False
    ):
        """
        Generate samples using DDPM or DDIM.

        Args:
            num_samples: Number of samples to generate
            img_shape: Shape of images (C, H, W)
            method: Sampling method ('ddpm' or 'ddim')
            num_inference_steps: Number of inference steps (for DDIM)
            return_all_steps: If True, return images at all timesteps

        Returns:
            Generated images, optionally with intermediate steps
        """
        self.model.eval()

        # Start from random noise
        x = torch.randn(num_samples, *img_shape, device=self.device)

        if method == 'ddpm':
            return self._sample_ddpm(x, return_all_steps)
        elif method == 'ddim':
            if num_inference_steps is None:
                num_inference_steps = self.hparams.num_inference_steps
            return self._sample_ddim(x, num_inference_steps, return_all_steps)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _sample_ddpm(self, x, return_all_steps=False):
        """Sample using DDPM (all timesteps)"""
        # Move scheduler to device
        self.scheduler.to(self.device)

        all_steps = [x] if return_all_steps else None

        # Reverse diffusion process
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # Remove noise
            x = self.scheduler.sample_prev_timestep(x, t_batch, predicted_noise)

            if return_all_steps:
                all_steps.append(x)

        if return_all_steps:
            return x, all_steps
        return x

    def _sample_ddim(self, x, num_inference_steps, return_all_steps=False):
        """Sample using DDIM (fewer timesteps)"""
        # Create DDIM scheduler if needed
        if self.ddim_scheduler is None or self.ddim_scheduler.num_inference_steps != num_inference_steps:
            self.ddim_scheduler = DDIMScheduler(
                self.scheduler,
                num_inference_steps=num_inference_steps,
                eta=0.0  # Deterministic
            )

        # Move scheduler to device
        self.ddim_scheduler.to(self.device)

        all_steps = [x] if return_all_steps else None

        # Reverse diffusion with DDIM
        timesteps = self.ddim_scheduler.timesteps.flip(0)  # Reverse order
        for i, t in enumerate(timesteps):
            t_idx = len(timesteps) - 1 - i
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # DDIM step
            x = self.ddim_scheduler.sample_prev_timestep(x, t_idx, predicted_noise)

            if return_all_steps:
                all_steps.append(x)

        if return_all_steps:
            return x, all_steps
        return x

    def configure_optimizers(self):
        """Configure Adam optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific arguments to parser"""
        parser = parent_parser.add_argument_group("DDPM")
        parser.add_argument("--model_channels", type=int, default=64)
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--num_timesteps", type=int, default=1000)
        parser.add_argument("--beta_start", type=float, default=1e-4)
        parser.add_argument("--beta_end", type=float, default=0.02)
        parser.add_argument("--beta_schedule", type=str, default='linear')
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--num_inference_steps", type=int, default=50)
        return parent_parser
