"""
UNet-like architecture for noise prediction in latent space.
Since latent representations are 1D vectors, we use MLP-based architecture
with residual connections and time conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal time embeddings for diffusion timesteps.
    Similar to positional encodings in Transformers.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with time conditioning for latent space.
    """

    def __init__(self, dim, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim)
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.SiLU()
        )

    def forward(self, x, time_emb):
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = x + time_emb

        # MLP with residual connection
        return x + self.mlp(h)


class UNetLatent(nn.Module):
    """
    MLP-based UNet-like architecture for noise prediction in latent space.

    Since latent representations are 1D vectors (not 2D images), we use
    an MLP architecture with residual connections and time conditioning.

    Args:
        in_channels: Dimension of input latent vector
        model_channels: Base number of hidden dimensions
        out_channels: Dimension of output latent vector (same as in_channels)
        num_res_blocks: Number of residual blocks per level
        channel_mult: Multipliers for hidden dimensions at each level
        dropout: Dropout probability
        num_heads: Number of attention heads (not used in MLP version)
    """

    def __init__(
        self,
        in_channels=128,
        model_channels=256,
        out_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        dropout=0.1,
        num_heads=4  # Not used but kept for compatibility
    ):
        super().__init__()

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial projection
        self.proj_in = nn.Linear(in_channels, model_channels)

        # Encoder (downsampling in feature space)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_projections = nn.ModuleList()

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # Projection at the beginning of each level (except first)
            if i > 0:
                prev_ch = model_channels * channel_mult[i - 1]
                self.encoder_projections.append(nn.Linear(prev_ch, out_ch))
            else:
                self.encoder_projections.append(nn.Identity())

            # Residual blocks at this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualMLPBlock(out_ch, time_emb_dim, dropout))
            self.encoder_blocks.append(blocks)

        # Middle blocks
        mid_ch = model_channels * channel_mult[-1]
        self.middle_blocks = nn.ModuleList([
            ResidualMLPBlock(mid_ch, time_emb_dim, dropout),
            ResidualMLPBlock(mid_ch, time_emb_dim, dropout),
        ])

        # Decoder (upsampling in feature space)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_skip_projections = nn.ModuleList()

        reversed_mult = list(reversed(channel_mult))
        for i, mult in enumerate(reversed_mult):
            out_ch = model_channels * mult

            # Projection for concatenated skip connection
            # First decoder level: middle (mid_ch) + skip from last encoder (mid_ch)
            # Other levels: current (out_ch) + skip from corresponding encoder (out_ch)
            if i == 0:
                skip_proj = nn.Linear(mid_ch + mid_ch, mid_ch)
            else:
                skip_proj = nn.Linear(out_ch + out_ch, out_ch)
            self.decoder_skip_projections.append(skip_proj)

            # Residual blocks at this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualMLPBlock(out_ch, time_emb_dim, dropout))
            self.decoder_blocks.append(blocks)

            # Projection to next decoder level (if not last)
            if i < len(reversed_mult) - 1:
                next_mult = reversed_mult[i + 1]
                next_ch = model_channels * next_mult
                self.decoder_skip_projections.append(nn.Linear(out_ch, next_ch))

        # Output projection
        self.proj_out = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels)
        )

    def forward(self, x, timesteps):
        """
        Args:
            x: Input latent tensor [B, latent_dim]
            timesteps: Timestep tensor [B]

        Returns:
            Predicted noise [B, latent_dim]
        """
        # Time embedding
        t_emb = self.time_mlp(timesteps)

        # Initial projection
        h = self.proj_in(x)

        # Encoder with skip connections
        skips = []
        for proj, blocks in zip(self.encoder_projections, self.encoder_blocks):
            # Project to new dimension
            h = proj(h)
            # Apply residual blocks and save skip connections
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)

        # Middle
        for block in self.middle_blocks:
            h = block(h, t_emb)

        # Decoder with skip connections
        proj_idx = 0
        for i, blocks in enumerate(self.decoder_blocks):
            # Get skip connection and concatenate
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            # Project concatenated features
            h = self.decoder_skip_projections[proj_idx](h)
            proj_idx += 1

            # Apply residual blocks
            for block in blocks:
                h = block(h, t_emb)

            # Project to next level (if not last)
            if i < len(self.decoder_blocks) - 1:
                h = self.decoder_skip_projections[proj_idx](h)
                proj_idx += 1

        # Output projection
        return self.proj_out(h)
