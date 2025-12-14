import torch
import torch.nn as nn
import numpy as np

class ConvConditionalGenerator(nn.Module):
    """
    Convolutional Generator for use with PatchGAN discriminator.
    Uses transposed convolutions for upsampling.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 10,
        img_channels: int = 1,
        feature_maps: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        input_dim = latent_dim + num_classes
        
        # Project and reshape: input_dim -> 7x7x256
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Transposed convolutions: 7x7 -> 14x14 -> 28x28
        self.conv_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final conv to get desired channels
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat([z, label_emb], dim=1)
        
        x = self.fc(gen_input)
        x = x.view(x.size(0), 256, 7, 7)
        img = self.conv_blocks(x)
        return img
