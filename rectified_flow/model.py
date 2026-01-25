"""
Rectified Flow model for image generation in pixel space.
Uses velocity prediction instead of noise prediction.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Literal
import time

from ddpm.unet import UNet
from .flow_scheduler import RectifiedFlowScheduler


class RectifiedFlow(pl.LightningModule):
    """
    Rectified Flow model operating in pixel space.

    Key differences from DDPM:
    1. Predicts velocity v(x,t) instead of noise
    2. Uses straight-line interpolation: x_t = t*x_1 + (1-t)*x_0
    3. Training loss: MSE between predicted and true velocity (x_1 - x_0)
    4. Sampling via ODE integration (Euler or Heun's method)

    Args:
        in_channels: Number of input channels
        model_channels: Base number of UNet channels
        channel_mult: Channel multipliers for UNet
        num_res_blocks: Number of residual blocks per level
        num_timesteps: Number of timesteps for training
        lr: Learning rate
        num_inference_steps: Default number of steps for sampling
        dropout: Dropout probability
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        num_timesteps: int = 1000,
        lr: float = 2e-4,
        num_inference_steps: int = 50,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()

        # UNet model for velocity prediction
        self.model = UNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            dropout=dropout,
            num_heads=num_heads
        )

        # Rectified Flow scheduler
        self.scheduler = RectifiedFlowScheduler(num_timesteps=num_timesteps)

        self.lr = lr
        self.in_channels = in_channels

    def forward(self, x, t):
        """Predict velocity v(x, t)"""
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        """
        Training step: predict velocity of straight-line flow.

        Loss: E[||v(x_t, t) - (x_1 - x_0)||^2]
        """
        x_1, _ = batch  # x_1 is the data (clean images)

        # Sample random timesteps
        batch_size = x_1.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        # Get interpolated samples and true velocity
        # x_t = t * x_1 + (1-t) * x_0
        x_t, x_0, true_velocity = self.scheduler.add_flow(x_1, t)

        # Predict velocity
        predicted_velocity = self.model(x_t, t)

        # Compute loss (MSE between predicted and true velocity)
        loss = nn.functional.mse_loss(predicted_velocity, true_velocity)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x_1, _ = batch

        batch_size = x_1.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        x_t, x_0, true_velocity = self.scheduler.add_flow(x_1, t)
        predicted_velocity = self.model(x_t, t)

        loss = nn.functional.mse_loss(predicted_velocity, true_velocity)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        img_shape: tuple = (1, 28, 28),
        method: Literal['euler', 'heun'] = 'euler',
        num_inference_steps: Optional[int] = None,
        return_all_steps: bool = False
    ):
        """
        Generate samples using Rectified Flow.

        Args:
            num_samples: Number of samples to generate
            img_shape: Shape of images (C, H, W)
            method: Integration method ('euler' or 'heun')
            num_inference_steps: Number of ODE integration steps
            return_all_steps: If True, return images at all timesteps

        Returns:
            Generated images, optionally with intermediate steps
        """
        self.model.eval()

        if num_inference_steps is None:
            num_inference_steps = self.hparams.num_inference_steps

        # Start from random noise (x_0 ~ N(0, I))
        x_0 = torch.randn(num_samples, *img_shape, device=self.device)

        # Integrate ODE from t=0 to t=1
        if method == 'euler':
            x_1, all_steps = self.scheduler.sample_euler(
                self.model, x_0, num_inference_steps
            )
        elif method == 'heun':
            x_1, all_steps = self.scheduler.sample_heun(
                self.model, x_0, num_inference_steps
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        if return_all_steps:
            return x_1, all_steps
        return x_1

    def configure_optimizers(self):
        """Configure Adam optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

                key = f'rf_{method}_{steps}'
                results[key] = {
                    'time': elapsed,
                    'steps': steps,
                    'time_per_sample': elapsed / num_samples,
                    'method': method
                }

        return results

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model-specific arguments to parser"""
        parser = parent_parser.add_argument_group("RectifiedFlow")
        parser.add_argument("--model_channels", type=int, default=64)
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--num_timesteps", type=int, default=1000)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--num_inference_steps", type=int, default=50)
        return parent_parser
