import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    """
    Discriminator network that classifies images as real or fake.
    Architecture: Fully connected layers with LeakyReLU and Dropout.
    """
    def __init__(self, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.img_size = int(np.prod(img_shape))
        
        self.model = nn.Sequential(
            # Layer 1: img_size -> 512
            nn.Linear(self.img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),
            
            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),
            
            # Output layer: 256 -> 1
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity