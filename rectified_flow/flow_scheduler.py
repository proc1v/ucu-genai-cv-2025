"""
Rectified Flow scheduler for straight-path ODE sampling.
Implements forward flow (interpolation) and reverse flow (sampling) based on velocity prediction.
"""

import torch
import numpy as np


class RectifiedFlowScheduler:
    """
    Rectified Flow scheduler implementing straight-line interpolation between data and noise.

    Unlike DDPM which uses a complex noise schedule, Rectified Flow learns to transform
    data to noise (and vice versa) along straight paths in probability space.

    Key equations:
    - Forward: X_t = t * X_1 + (1-t) * X_0, where t ∈ [0,1]
    - Target velocity: v = X_1 - X_0
    - Training loss: E[||v(X_t, t) - (X_1 - X_0)||^2]
    - Sampling ODE: dZ_t/dt = v(Z_t, t)

    Args:
        num_timesteps: Number of discrete timesteps for training (typically 1000)
    """

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps

    def add_flow(self, x_1, t, x_0=None):
        """
        Forward flow: linear interpolation from x_0 (noise) to x_1 (data).

        X_t = t * X_1 + (1-t) * X_0

        Args:
            x_1: Clean data samples [B, ...]
            t: Timesteps in [0, num_timesteps-1] [B]
            x_0: Noise samples [B, ...] (generated if None)

        Returns:
            x_t: Interpolated samples at timestep t
            x_0: Noise samples used
            velocity: True velocity (x_1 - x_0)
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        # Normalize t to [0, 1]
        t_normalized = t.float() / (self.num_timesteps - 1)

        # Reshape t for broadcasting
        t_normalized = t_normalized.view(-1, *([1] * (len(x_1.shape) - 1)))

        # Linear interpolation: x_t = t * x_1 + (1-t) * x_0
        x_t = t_normalized * x_1 + (1 - t_normalized) * x_0

        # True velocity is the constant direction from x_0 to x_1
        velocity = x_1 - x_0

        return x_t, x_0, velocity

    @torch.no_grad()
    def sample_euler(self, model, x_0, num_steps: int):
        """
        Sample from the model using Euler integration of the ODE.

        Solves: dZ_t/dt = v(Z_t, t) from t=0 to t=1

        Args:
            model: Neural network that predicts velocity v(x, t)
            x_0: Initial noise samples [B, ...]
            num_steps: Number of integration steps

        Returns:
            x_1: Final samples (should be data-like)
            all_steps: List of intermediate samples (if return_all_steps=True)
        """
        x = x_0.clone()
        dt = 1.0 / num_steps

        all_steps = [x.clone()]

        for i in range(num_steps):
            # Current time in [0, 1]
            t_continuous = i / num_steps

            # Convert to discrete timestep for model
            t = torch.full((x.shape[0],),
                          int(t_continuous * (self.num_timesteps - 1)),
                          device=x.device, dtype=torch.long)

            # Predict velocity
            velocity = model(x, t)

            # Euler step: x_{t+dt} = x_t + dt * v(x_t, t)
            x = x + dt * velocity
            all_steps.append(x.clone())

        return x, all_steps

    @torch.no_grad()
    def sample_heun(self, model, x_0, num_steps: int):
        """
        Sample using second-order Heun's method for better accuracy.

        Args:
            model: Neural network that predicts velocity v(x, t)
            x_0: Initial noise samples [B, ...]
            num_steps: Number of integration steps

        Returns:
            x_1: Final samples
            all_steps: List of intermediate samples
        """
        x = x_0.clone()
        dt = 1.0 / num_steps

        all_steps = [x.clone()]

        for i in range(num_steps):
            # Current time
            t_continuous = i / num_steps
            t = torch.full((x.shape[0],),
                          int(t_continuous * (self.num_timesteps - 1)),
                          device=x.device, dtype=torch.long)

            # First prediction (Euler step)
            v1 = model(x, t)
            x_temp = x + dt * v1

            # Second prediction at next timestep
            t_next_continuous = min((i + 1) / num_steps, 1.0)
            t_next = torch.full((x.shape[0],),
                               int(t_next_continuous * (self.num_timesteps - 1)),
                               device=x.device, dtype=torch.long)

            # Clip to avoid going beyond bounds
            t_next = torch.clamp(t_next, 0, self.num_timesteps - 1)

            v2 = model(x_temp, t_next)

            # Heun's method: average of two velocities
            x = x + dt * (v1 + v2) / 2
            all_steps.append(x.clone())

        return x, all_steps

    def to(self, device):
        """Move tensors to device (no tensors to move for this scheduler)"""
        return self
