import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pathlib import Path


class ImageGenerationCallback(Callback):
    """
    Callback to generate and save sample images during training.
    Useful for monitoring training progress and detecting mode collapse.
    """
    def __init__(self, output_dir: str = "gan_outputs", every_n_epochs: int = 5):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.generated_images = []  # Store for mode collapse analysis
    
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0 or trainer.current_epoch == 0:
            # Generate images
            pl_module.eval()
            with torch.no_grad():
                z = pl_module.validation_z.to(pl_module.device)
                gen_imgs = pl_module(z)
            pl_module.train()
            
            # Store for analysis
            self.generated_images.append((trainer.current_epoch, gen_imgs.cpu()))
            
            # Create grid and save
            grid = make_grid(gen_imgs, nrow=8, normalize=True, value_range=(-1, 1))
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Generated Images - Epoch {trainer.current_epoch + 1}')
            
            plt.savefig(self.output_dir / f'epoch_{trainer.current_epoch + 1:03d}.png',
                       bbox_inches='tight', dpi=150)
            plt.close()

class ConditionalImageCallback(Callback):
    """
    Callback to generate and save class-conditional sample images.
    Shows generated images organized by class.
    """
    def __init__(
        self,
        output_dir: str = "cgan_outputs",
        every_n_epochs: int = 5,
        class_names: list[str] = None
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.class_names = class_names or [str(i) for i in range(10)]
        self.generated_images = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0 or trainer.current_epoch == 0:
            pl_module.eval()
            
            with torch.no_grad():
                z = pl_module.validation_z.to(pl_module.device)
                labels = pl_module.validation_labels.to(pl_module.device)
                gen_imgs = pl_module(z, labels)
            
            pl_module.train()
            
            self.generated_images.append((trainer.current_epoch, gen_imgs.cpu(), labels.cpu()))
            
            # Create grid organized by class (10 rows, 10 columns)
            fig, axes = plt.subplots(10, 10, figsize=(12, 12))
            
            for class_idx in range(10):
                class_mask = labels.cpu() == class_idx
                class_imgs = gen_imgs[class_mask][:10]
                
                for img_idx in range(min(10, len(class_imgs))):
                    ax = axes[class_idx, img_idx]
                    img = class_imgs[img_idx].squeeze().cpu().numpy()
                    img = (img + 1) / 2  # Denormalize
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                    if img_idx == 0:
                        ax.set_ylabel(self.class_names[class_idx], fontsize=8)
            
            plt.suptitle(f'Generated Images by Class - Epoch {trainer.current_epoch + 1}', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'epoch_{trainer.current_epoch + 1:03d}.png',
                       bbox_inches='tight', dpi=150)
            plt.close()
