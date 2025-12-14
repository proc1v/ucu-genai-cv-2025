import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator that classifies NxN patches of the image.
    Instead of outputting a single value, outputs a grid of predictions.
    This encourages high-frequency detail in generated images.
    
    For 28x28 images, outputs a 4x4 patch grid.
    """
    def __init__(
        self,
        num_classes: int = 10,
        img_channels: int = 1,
        feature_maps: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Label embedding as a channel (expanded spatially)
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)
        
        # Input: img_channels + 1 (for label channel)
        in_channels = img_channels + 1
        
        # Convolutional layers
        # 28x28 -> 14x14 -> 7x7 -> 4x4
        self.conv_blocks = nn.Sequential(
            # Layer 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, feature_maps, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 14x14 -> 7x7
            nn.Conv2d(feature_maps, feature_maps * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 7x7 -> 4x4 (with padding adjustment)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 4x4 patch predictions
            nn.Conv2d(feature_maps * 4, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Store intermediate features for feature matching
        self.features = []
    
    def forward(
        self,
        img: torch.Tensor,
        labels: torch.Tensor,
        return_features: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        batch_size = img.size(0)
        
        # Create label channel
        label_emb = self.label_embedding(labels)
        label_channel = label_emb.view(batch_size, 1, 28, 28)
        
        # Concatenate image and label channel
        x = torch.cat([img, label_channel], dim=1)
        
        if return_features:
            features = []
            for i, layer in enumerate(self.conv_blocks):
                x = layer(x)
                if isinstance(layer, nn.LeakyReLU):
                    features.append(x.clone())
            return x, features
        else:
            return self.conv_blocks(x), None
    
    def get_features(self, img: torch.Tensor, labels: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate features for feature matching loss."""
        _, features = self.forward(img, labels, return_features=True)
        return features