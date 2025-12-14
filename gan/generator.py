import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    """
    Generator network that transforms random noise into images.
    Architecture: Fully connected layers with BatchNorm and LeakyReLU.
    """
    def __init__(self, latent_dim: int = 128, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.img_size = int(np.prod(img_shape))
        
        self.model = nn.Sequential(
            # Layer 1: latent_dim -> 256
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 256 -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 512 -> 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer: 1024 -> img_size
            nn.Linear(1024, self.img_size),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img_flat = self.model(z)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img