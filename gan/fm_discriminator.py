import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchingDiscriminator(nn.Module):
    """
    Wrapper that adds feature extraction capability to any discriminator.
    Used for computing feature matching loss.
    """
    def __init__(
        self,
        num_classes: int = 10,
        img_channels: int = 1,
        feature_maps: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)
        
        in_channels = img_channels + 1
        
        # Define layers separately for feature extraction
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_maps * 4, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        img: torch.Tensor,
        labels: torch.Tensor,
        return_features: bool = False
    ):
        batch_size = img.size(0)
        
        label_emb = self.label_embedding(labels)
        label_channel = label_emb.view(batch_size, 1, 28, 28)
        x = torch.cat([img, label_channel], dim=1)
        
        features = []
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        output = self.output_layer(x)
        
        if return_features:
            return output, features
        return output, None

class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss: Encourages generator to match statistics
    of real images at intermediate layers of the discriminator.
    
    This helps stabilize training by providing a more informative
    gradient signal to the generator.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        real_features: list[torch.Tensor],
        fake_features: list[torch.Tensor]
    ) -> torch.Tensor:
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            # Match mean of features
            loss += F.mse_loss(fake_feat.mean(dim=0), real_feat.mean(dim=0))
        return loss
