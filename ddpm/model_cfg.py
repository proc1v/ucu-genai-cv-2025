"""
DDPM with Classifier-Free Guidance (CFG) training module.
Supports both input concatenation and cross-attention conditioning methods.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Literal

from .unet_cfg_concat import UNetCFGConcat
from .unet_cfg_crossattn import UNetCFGCrossAttn
from .noise_scheduler import DDPMScheduler, DDIMScheduler


class DDPMCFG(pl.LightningModule):
    """
    Denoising Diffusion Probabilistic Model with Classifier-Free Guidance.

    Implements CFG training where the model learns both conditional and unconditional
    generation. During inference, outputs are interpolated based on guidance scale:
        output = uncond_output + guidance_scale * (cond_output - uncond_output)

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
        num_classes: Number of classes for conditioning
        class_emb_dim: Dimension of class embeddings
        conditioning_type: Type of conditioning ('concat' or 'cross_attn')
        cfg_dropout: Probability of dropping conditioning (for CFG training)
        cfg_scale: Default classifier-free guidance scale for sampling
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
        num_heads: int = 4,
        num_classes: int = 10,
        class_emb_dim: int = 128,
        conditioning_type: Literal['concat', 'cross_attn'] = 'cross_attn',
        cfg_dropout: float = 0.1,
        cfg_scale: float = 3.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # Select UNet architecture based on conditioning type
        if conditioning_type == 'concat':
            self.model = UNetCFGConcat(
                in_channels=in_channels,
                model_channels=model_channels,
                out_channels=in_channels,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult,
                dropout=dropout,
                num_heads=num_heads,
                num_classes=num_classes,
                class_emb_dim=class_emb_dim
            )
        elif conditioning_type == 'cross_attn':
            self.model = UNetCFGCrossAttn(
                in_channels=in_channels,
                model_channels=model_channels,
                out_channels=in_channels,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult,
                dropout=dropout,
                num_heads=num_heads,
                num_classes=num_classes,
                class_emb_dim=class_emb_dim
            )
        else:
            raise ValueError(f"Unknown conditioning_type: {conditioning_type}")

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
        self.num_classes = num_classes
        self.cfg_dropout = cfg_dropout
        self.cfg_scale = cfg_scale

    def forward(self, x, t, class_labels=None):
        """Predict noise with optional class conditioning"""
        return self.model(x, t, class_labels)

    def training_step(self, batch, batch_idx):
        """
        Training step with classifier-free guidance.
        Randomly drops conditioning with probability cfg_dropout.
        """
        x, labels = batch

        batch_size = x.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        # Add noise to images
        noise = torch.randn_like(x)
        x_t, _ = self.scheduler.add_noise(x, t, noise)

        # Classifier-free guidance: randomly drop conditioning
        # Replace some labels with unconditional token (num_classes)
        cfg_mask = torch.rand(batch_size, device=self.device) < self.cfg_dropout
        class_labels = labels.clone()
        class_labels[cfg_mask] = self.num_classes  # Unconditional token

        # Predict noise
        predicted_noise = self.model(x_t, t, class_labels)

        # Compute loss (simple MSE)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, labels = batch

        batch_size = x.shape[0]
        t = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device, dtype=torch.long
        )

        noise = torch.randn_like(x)
        x_t, _ = self.scheduler.add_noise(x, t, noise)

        # Conditional prediction
        predicted_noise = self.model(x_t, t, labels)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        class_labels: Optional[torch.Tensor] = None,
        img_shape: tuple = (1, 28, 28),
        method: Literal['ddpm', 'ddim'] = 'ddim',
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        return_all_steps: bool = False
    ):
        """
        Generate samples using classifier-free guidance.

        Args:
            num_samples: Number of samples to generate
            class_labels: Class labels [num_samples] (None for random)
            img_shape: Shape of images (C, H, W)
            method: Sampling method ('ddpm' or 'ddim')
            num_inference_steps: Number of inference steps (for DDIM)
            cfg_scale: Guidance scale (None uses default from hparams)
            return_all_steps: If True, return images at all timesteps

        Returns:
            Generated images, optionally with intermediate steps
        """
        self.model.eval()

        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # Generate or validate class labels
        if class_labels is None:
            class_labels = torch.randint(
                0, self.num_classes, (num_samples,),
                device=self.device, dtype=torch.long
            )
        else:
            class_labels = class_labels.to(self.device)
            assert len(class_labels) == num_samples

        # Start from random noise
        x = torch.randn(num_samples, *img_shape, device=self.device)

        if method == 'ddpm':
            return self._sample_ddpm_cfg(x, class_labels, cfg_scale, return_all_steps)
        elif method == 'ddim':
            if num_inference_steps is None:
                num_inference_steps = self.hparams.num_inference_steps
            return self._sample_ddim_cfg(
                x, class_labels, cfg_scale, num_inference_steps, return_all_steps
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _sample_ddpm_cfg(self, x, class_labels, cfg_scale, return_all_steps=False):
        """Sample using DDPM with classifier-free guidance"""
        self.scheduler.to(self.device)

        all_steps = [x] if return_all_steps else None
        batch_size = x.shape[0]

        # Create unconditional labels
        uncond_labels = torch.full_like(class_labels, self.num_classes)

        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise with and without conditioning
            if cfg_scale > 0:
                # Conditional prediction
                cond_noise = self.model(x, t_batch, class_labels)

                # Unconditional prediction
                uncond_noise = self.model(x, t_batch, uncond_labels)

                # Classifier-free guidance
                predicted_noise = uncond_noise + cfg_scale * (cond_noise - uncond_noise)
            else:
                # No guidance, just conditional
                predicted_noise = self.model(x, t_batch, class_labels)

            # Remove noise
            x = self.scheduler.sample_prev_timestep(x, t_batch, predicted_noise)

            if return_all_steps:
                all_steps.append(x)

        if return_all_steps:
            return x, all_steps
        return x

    def _sample_ddim_cfg(self, x, class_labels, cfg_scale, num_inference_steps, return_all_steps=False):
        """Sample using DDIM with classifier-free guidance"""
        # Create DDIM scheduler if needed
        if self.ddim_scheduler is None or self.ddim_scheduler.num_inference_steps != num_inference_steps:
            self.ddim_scheduler = DDIMScheduler(
                self.scheduler,
                num_inference_steps=num_inference_steps,
                eta=0.0
            )

        self.ddim_scheduler.to(self.device)

        all_steps = [x] if return_all_steps else None
        batch_size = x.shape[0]

        # Create unconditional labels
        uncond_labels = torch.full_like(class_labels, self.num_classes)

        timesteps = self.ddim_scheduler.timesteps.flip(0)
        for i, t in enumerate(timesteps):
            t_idx = len(timesteps) - 1 - i
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise with and without conditioning
            if cfg_scale > 0:
                # Conditional prediction
                cond_noise = self.model(x, t_batch, class_labels)

                # Unconditional prediction
                uncond_noise = self.model(x, t_batch, uncond_labels)

                # Classifier-free guidance
                predicted_noise = uncond_noise + cfg_scale * (cond_noise - uncond_noise)
            else:
                # No guidance, just conditional
                predicted_noise = self.model(x, t_batch, class_labels)

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
        parser = parent_parser.add_argument_group("DDPM-CFG")
        parser.add_argument("--model_channels", type=int, default=64)
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--num_timesteps", type=int, default=1000)
        parser.add_argument("--beta_start", type=float, default=1e-4)
        parser.add_argument("--beta_end", type=float, default=0.02)
        parser.add_argument("--beta_schedule", type=str, default='linear')
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--num_inference_steps", type=int, default=50)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--class_emb_dim", type=int, default=128)
        parser.add_argument("--conditioning_type", type=str, default='cross_attn',
                          choices=['concat', 'cross_attn'])
        parser.add_argument("--cfg_dropout", type=float, default=0.1)
        parser.add_argument("--cfg_scale", type=float, default=3.0)
        return parent_parser
