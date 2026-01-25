"""
DDPM and DDIM noise schedulers for diffusion models.
Implements forward diffusion (adding noise) and reverse diffusion (denoising).
"""

import torch
import torch.nn as nn
import numpy as np


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) scheduler.

    Implements the forward and reverse diffusion process from:
    "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

    Args:
        num_timesteps: Number of diffusion steps (T)
        beta_start: Starting value of beta schedule
        beta_end: Ending value of beta schedule
        beta_schedule: Type of schedule ('linear', 'quadratic', 'cosine')
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear'
    ):
        self.num_timesteps = num_timesteps

        # Create beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-compute useful values for diffusion process
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # Calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for reverse diffusion q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance: q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Adds noise to clean images according to the schedule.

        Args:
            x_0: Clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional noise tensor (will be generated if None)

        Returns:
            x_t: Noisy images at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get schedule values for the given timesteps
        # Ensure t is on CPU for indexing, then move result to x_0's device
        t_cpu = t.cpu() if t.is_cuda else t
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t_cpu].to(x_0.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(x_0.device)

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(x_0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

        return x_t, noise

    def sample_prev_timestep(self, x_t, t, predicted_noise, clip_denoised=True):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        Sample x_{t-1} given x_t and the predicted noise.

        Args:
            x_t: Noisy images at timestep t [B, C, H, W]
            t: Current timesteps [B]
            predicted_noise: Noise predicted by the model [B, C, H, W]
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            x_{t-1}: Denoised images at timestep t-1
        """
        # Get schedule values
        # Ensure t is on CPU for indexing, then move result to x_t's device
        t_cpu = t.cpu() if t.is_cuda else t
        beta_t = self.betas[t_cpu].to(x_t.device)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(x_t.device)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_cpu].to(x_t.device)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_cpu].to(x_t.device)

        # Reshape for broadcasting
        while len(beta_t.shape) < len(x_t.shape):
            beta_t = beta_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
            sqrt_recip_alpha_t = sqrt_recip_alpha_t.unsqueeze(-1)
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)

        # Predict x_0 from x_t and noise
        # x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t

        # Clip predicted x_0 to prevent numerical instability
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Get alpha values for computing mean
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t_cpu].to(x_t.device)
        alpha_cumprod_t = self.alphas_cumprod[t_cpu].to(x_t.device)
        alpha_t = self.alphas[t_cpu].to(x_t.device)

        while len(alpha_cumprod_prev_t.shape) < len(x_t.shape):
            alpha_cumprod_prev_t = alpha_cumprod_prev_t.unsqueeze(-1)
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1)
            alpha_t = alpha_t.unsqueeze(-1)

        # Compute mean of q(x_{t-1} | x_t, x_0) using the predicted (and possibly clipped) x_0
        # Formula from DDPM paper:
        # mean = (sqrt(alpha_bar_{t-1}) * beta_t) / (1 - alpha_bar_t) * pred_x0
        #      + (sqrt(alpha_t) * (1 - alpha_bar_{t-1})) / (1 - alpha_bar_t) * x_t
        posterior_mean_coef1 = (torch.sqrt(alpha_cumprod_prev_t) * beta_t) / (1.0 - alpha_cumprod_t)
        posterior_mean_coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t)

        model_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x_t

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_cpu].to(x_t.device)
            while len(posterior_variance_t.shape) < len(x_t.shape):
                posterior_variance_t = posterior_variance_t.unsqueeze(-1)

            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def to(self, device):
        """Move all tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class DDIMScheduler:
    """
    Denoising Diffusion Implicit Model (DDIM) scheduler for faster sampling.

    Implements deterministic sampling from:
    "Denoising Diffusion Implicit Models" (Song et al., 2020)

    Args:
        ddpm_scheduler: Base DDPM scheduler
        num_inference_steps: Number of steps for inference (can be << num_timesteps)
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
    """

    def __init__(
        self,
        ddpm_scheduler: DDPMScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0
    ):
        self.ddpm_scheduler = ddpm_scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Create subset of timesteps for inference
        # Use uniform spacing across the original timesteps
        step_ratio = ddpm_scheduler.num_timesteps // num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio).astype(np.int64)
        self.timesteps = torch.from_numpy(self.timesteps)

    def sample_prev_timestep(self, x_t, t_idx, predicted_noise):
        """
        DDIM reverse process: deterministic/semi-stochastic sampling.

        Args:
            x_t: Noisy images at current timestep [B, C, H, W]
            t_idx: Index in the inference timesteps (not actual timestep)
            predicted_noise: Noise predicted by model [B, C, H, W]

        Returns:
            x_{t-1}: Denoised images at previous timestep
        """
        # Get actual timesteps
        t = self.timesteps[t_idx].item()
        prev_t = self.timesteps[t_idx - 1].item() if t_idx > 0 else -1

        # Get alpha values
        alpha_prod_t = self.ddpm_scheduler.alphas_cumprod[t].to(x_t.device)
        if prev_t >= 0:
            alpha_prod_t_prev = self.ddpm_scheduler.alphas_cumprod[prev_t].to(x_t.device)
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t).to(x_t.device)

        # Reshape for broadcasting
        while len(alpha_prod_t.shape) < len(x_t.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)

        # Predict x_0 from x_t and noise
        pred_x0 = (x_t - torch.sqrt(1 - alpha_prod_t) * predicted_noise) / torch.sqrt(alpha_prod_t)

        # Clip predicted x_0 to valid range
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Compute variance
        variance = self._get_variance(t, prev_t).to(x_t.device)
        while len(variance.shape) < len(x_t.shape):
            variance = variance.unsqueeze(-1)

        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_prod_t_prev - variance) * predicted_noise

        # DDIM sampling equation
        x_prev = torch.sqrt(alpha_prod_t_prev) * pred_x0 + pred_dir

        # Add noise if eta > 0 (stochastic)
        if self.eta > 0:
            noise = torch.randn_like(x_t)
            x_prev += torch.sqrt(variance) * noise

        return x_prev

    def _get_variance(self, t, prev_t):
        """Compute variance for DDIM sampling"""
        if prev_t < 0:
            return torch.tensor(0.0)

        alpha_prod_t = self.ddpm_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.ddpm_scheduler.alphas_cumprod[prev_t]

        variance = (
            self.eta ** 2
            * (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )

        return variance

    def to(self, device):
        """Move tensors to device"""
        self.timesteps = self.timesteps.to(device)
        self.ddpm_scheduler.to(device)
        return self
