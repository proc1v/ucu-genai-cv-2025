import torch
import torch.nn as nn
import pytorch_lightning as pl
from gan.generator import Generator
from gan.discriminator import Discriminator

class VanillaGAN(pl.LightningModule):
    """
    PyTorch Lightning module implementing Vanilla GAN training.
    Handles both generator and discriminator training with automatic optimization.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        img_shape: tuple = (1, 28, 28),
        lr: float = 0.0005,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Manual optimization for GAN training

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        
        # Networks
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
        # For tracking losses
        self.g_losses = []
        self.d_losses = []
        
        # Fixed noise for consistent visualization
        self.validation_z = torch.randn(64, latent_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.size(0)
        
        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        # Ground truth labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Sample noise
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        
        # ---------------------
        # Train Discriminator
        # ---------------------
        opt_d.zero_grad()
        
        # Real images loss
        real_pred = self.discriminator(imgs)
        d_real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Fake images loss
        fake_imgs = self.generator(z).detach()
        fake_pred = self.discriminator(fake_imgs)
        d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # -----------------
        # Train Generator
        # -----------------
        opt_g.zero_grad()
        
        # Generate fake images
        gen_imgs = self.generator(z)
        
        # Generator wants discriminator to think fake images are real
        validity = self.discriminator(gen_imgs)
        g_loss = self.adversarial_loss(validity, real_labels)
        
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Logging
        self.log('g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('d_real_acc', (real_pred > 0.5).float().mean(), on_epoch=True)
        self.log('d_fake_acc', (fake_pred < 0.5).float().mean(), on_epoch=True)
        
        # Store losses for plotting
        self.g_losses.append(g_loss.item())
        self.d_losses.append(d_loss.item())
        
        return {'g_loss': g_loss, 'd_loss': d_loss}
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        return [opt_g, opt_d], []
    
    def generate_images(self, num_images: int = 64) -> torch.Tensor:
        """Generate a batch of images for visualization."""
        z = torch.randn(num_images, self.hparams.latent_dim, device=self.device)
        with torch.no_grad():
            return self.generator(z)