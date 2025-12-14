import torch
import torch.nn as nn
import pytorch_lightning as pl

from gan.cond_generator import ConvConditionalGenerator
from gan.fm_discriminator import FeatureMatchingDiscriminator, FeatureMatchingLoss


class ConditionalGAN(pl.LightningModule):
    """
    PyTorch Lightning module for Conditional GAN with various discriminator options.
    
    Supports:
    - Standard conditional discriminator
    - PatchGAN discriminator
    - Feature matching loss
    """
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        img_shape: tuple = (1, 28, 28),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        discriminator_type: str = "standard",  # "standard", "patchgan", "feature_matching"
        feature_matching_weight: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.feature_matching_weight = feature_matching_weight
        
        # Select generator based on discriminator type
        if discriminator_type in ["patchgan", "feature_matching"]:
            self.generator = ConvConditionalGenerator(
                latent_dim=latent_dim,
                num_classes=num_classes,
                img_channels=img_shape[0]
            )
        else:
            raise NotImplementedError(f"Generator for discriminator type '{discriminator_type}' is not implemented.")
        
        # Select discriminator
        if discriminator_type == "feature_matching":
            self.discriminator = FeatureMatchingDiscriminator(
                num_classes=num_classes,
                img_channels=img_shape[0]
            )
        else:
            raise NotImplementedError(f"Discriminator type '{discriminator_type}' is not implemented.")
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        
        # Loss tracking
        self.g_losses = []
        self.d_losses = []
        self.fm_losses = []  # Feature matching losses
        
        # Fixed noise and labels for visualization
        self.validation_z = torch.randn(100, latent_dim)  # 10 samples per class
        self.validation_labels = torch.arange(10).repeat(10)  # 0-9 repeated 10 times
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.generator(z, labels)
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)
        
        opt_g, opt_d = self.optimizers()
        
        # Generate labels for real/fake
        if self.hparams.discriminator_type in ["patchgan", "feature_matching"]:
            # PatchGAN outputs a grid, create matching target
            d_out, _ = self.discriminator(imgs, labels)
            real_labels = torch.ones_like(d_out)
            fake_labels = torch.zeros_like(d_out)
        else:
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        opt_d.zero_grad()
        
        # Real images
        if self.hparams.discriminator_type in ["patchgan", "feature_matching"]:
            real_pred, real_features = self.discriminator(imgs, labels, return_features=True)
        else:
            real_pred = self.discriminator(imgs, labels)
            real_features = None
        
        d_real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Fake images
        fake_imgs = self.generator(z, labels).detach()
        
        if self.hparams.discriminator_type in ["patchgan", "feature_matching"]:
            fake_pred, _ = self.discriminator(fake_imgs, labels)
        else:
            fake_pred = self.discriminator(fake_imgs, labels)
        
        d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # -----------------
        # Train Generator
        # -----------------
        opt_g.zero_grad()
        
        # Generate new fake images
        gen_imgs = self.generator(z, labels)
        
        if self.hparams.discriminator_type in ["patchgan", "feature_matching"]:
            validity, fake_features = self.discriminator(gen_imgs, labels, return_features=True)
        else:
            validity = self.discriminator(gen_imgs, labels)
            fake_features = None
        
        # Adversarial loss
        g_adv_loss = self.adversarial_loss(validity, real_labels)
        
        # Feature matching loss (if applicable)
        fm_loss = torch.tensor(0.0, device=self.device)
        if self.hparams.discriminator_type == "feature_matching" and real_features is not None:
            # Recompute real features (they were computed with previous generator output)
            _, real_features = self.discriminator(imgs, labels, return_features=True)
            fm_loss = self.feature_matching_loss(real_features, fake_features)
            g_loss = g_adv_loss + self.hparams.feature_matching_weight * fm_loss
        else:
            g_loss = g_adv_loss
        
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Logging
        self.log('g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('g_adv_loss', g_adv_loss, on_epoch=True)
        
        if self.hparams.discriminator_type == "feature_matching":
            self.log('fm_loss', fm_loss, on_epoch=True)
            self.fm_losses.append(fm_loss.item())
        
        self.g_losses.append(g_loss.item())
        self.d_losses.append(d_loss.item())
        
        return {'g_loss': g_loss, 'd_loss': d_loss}
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1, b2 = self.hparams.b1, self.hparams.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        return [opt_g, opt_d], []
    
    def generate_by_class(self, class_label: int, num_images: int = 10) -> torch.Tensor:
        """Generate images for a specific class."""
        z = torch.randn(num_images, self.hparams.latent_dim, device=self.device)
        labels = torch.full((num_images,), class_label, dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.generator(z, labels)