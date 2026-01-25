"""
UNet with Classifier-Free Guidance using Input Concatenation.
Conditions the model by concatenating class embeddings as additional input channels.
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


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing global dependencies.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        h = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        h = h.reshape(B, C, H, W)

        return x + self.proj(h)


class UNetCFGConcat(nn.Module):
    """
    Time-conditional UNet with Classifier-Free Guidance via Input Concatenation.

    The class label is embedded and spatially tiled, then concatenated with the
    noisy input as additional channels. This provides a simple conditioning mechanism.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        model_channels: Base number of channels (will be multiplied by channel_mult)
        out_channels: Number of output channels (same as in_channels)
        num_res_blocks: Number of residual blocks per resolution level
        channel_mult: Channel multipliers for each resolution level
        dropout: Dropout probability
        num_heads: Number of attention heads
        num_classes: Number of classes for conditioning (10 for MNIST)
        class_emb_dim: Dimension of class embeddings
    """

    def __init__(
        self,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        dropout=0.1,
        num_heads=4,
        num_classes=10,
        class_emb_dim=64
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim

        # Class embedding layer (num_classes + 1 for unconditional)
        self.class_embedding = nn.Embedding(num_classes + 1, class_emb_dim)

        # Projection to convert class embedding to spatial feature map
        self.class_to_spatial = nn.Linear(class_emb_dim, class_emb_dim)

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution (accepts input + class embedding channels)
        # The class embedding is spatially tiled and concatenated
        self.conv_in = nn.Conv2d(in_channels + class_emb_dim, model_channels, 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        now_channels = model_channels

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # Residual blocks at this resolution
            down_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                down_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(now_channels, out_ch, time_emb_dim, dropout),
                        AttentionBlock(out_ch, num_heads)
                    ])
                )
                now_channels = out_ch

            self.downs.append(down_blocks)

            # Downsample (except last level)
            if i != len(channel_mult) - 1:
                self.down_samples.append(
                    nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1)
                )
            else:
                self.down_samples.append(nn.Identity())

        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout),
            AttentionBlock(now_channels, num_heads),
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)
        ])

        # Upsampling
        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult

            # Residual blocks at this resolution
            up_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = now_channels if j == 0 else out_ch
                up_blocks.append(
                    nn.ModuleList([
                        ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim, dropout),
                        AttentionBlock(out_ch, num_heads)
                    ])
                )
                now_channels = out_ch

            self.ups.append(up_blocks)

            # Upsample (except last level)
            if i != len(channel_mult) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, class_labels=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            class_labels: Class labels [B] (None for unconditional, or contains num_classes for uncond)

        Returns:
            Predicted noise [B, C, H, W]
        """
        batch_size = x.shape[0]

        # Handle class labels (use num_classes as the unconditional token)
        if class_labels is None:
            class_labels = torch.full((batch_size,), self.num_classes,
                                     device=x.device, dtype=torch.long)

        # Embed class labels
        class_emb = self.class_embedding(class_labels)  # [B, class_emb_dim]
        class_emb = self.class_to_spatial(class_emb)     # [B, class_emb_dim]

        # Tile class embedding spatially to match image dimensions
        H, W = x.shape[2], x.shape[3]
        class_spatial = class_emb[:, :, None, None].expand(-1, -1, H, W)  # [B, class_emb_dim, H, W]

        # Concatenate with input
        x = torch.cat([x, class_spatial], dim=1)  # [B, C + class_emb_dim, H, W]

        # Time embedding
        t_emb = self.time_mlp(timesteps)

        # Initial conv
        x = self.conv_in(x)

        # Downsampling with skip connections
        skips = []
        for down_blocks, down_sample in zip(self.downs, self.down_samples):
            for res, attn in down_blocks:
                x = res(x, t_emb)
                x = attn(x)
                skips.append(x)
            x = down_sample(x)

        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)

        # Upsampling with skip connections
        for up_blocks, up_sample in zip(self.ups, self.up_samples):
            for res, attn in up_blocks:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = res(x, t_emb)
                x = attn(x)
            x = up_sample(x)

        # Output
        return self.conv_out(x)
