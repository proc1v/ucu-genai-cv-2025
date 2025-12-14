import torch
import torchvision
import pytorch_lightning as pl
import numpy as np

class VAEGenerateCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1, num_samples=8):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save images every N epochs
        self.num_samples = num_samples  # Number of random samples to generate
        
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            
            with torch.no_grad():
                pl_module.eval()
                
                # 1. Reconstruct input images
                reconst_imgs_logits, mu, logvar = pl_module(input_imgs)
                reconst_imgs = torch.sigmoid(reconst_imgs_logits)  # ← Convert logits to [0,1]
                
                # 2. Generate random samples from prior N(0,1)
                z = torch.randn(self.num_samples, pl_module.hparams.latent_dim).to(pl_module.device)
                random_samples_logits = pl_module.decoder(z)
                random_samples = torch.sigmoid(random_samples_logits)  # ← Convert logits to [0,1]
                
                pl_module.train()
            
            # Plot reconstructions
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(
                imgs, nrow=2, normalize=True, value_range=(-1, 1)
            )
            trainer.logger.experiment.add_image(
                "Reconstructions", grid, global_step=trainer.global_step
            )
            
            # Plot random samples
            sample_grid = torchvision.utils.make_grid(
                random_samples, nrow=4, normalize=True, value_range=(-1, 1)
            )
            trainer.logger.experiment.add_image(
                "Random_Samples", sample_grid, global_step=trainer.global_step
            )
            
            # Log latent space statistics
            trainer.logger.experiment.add_histogram(
                "Latent/mu", mu, global_step=trainer.global_step
            )
            trainer.logger.experiment.add_histogram(
                "Latent/logvar", logvar, global_step=trainer.global_step
            )
            trainer.logger.experiment.add_scalar(
                "Latent/mu_mean", mu.mean(), global_step=trainer.global_step
            )
            trainer.logger.experiment.add_scalar(
                "Latent/logvar_mean", logvar.mean(), global_step=trainer.global_step
            )