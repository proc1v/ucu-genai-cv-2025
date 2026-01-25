"""
Callbacks for DDPM training and evaluation.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from pathlib import Path


class DDPMSamplingCallback(Callback):
    """
    Callback to generate and visualize samples during training.
    Compares DDPM and DDIM sampling.
    """

    def __init__(
        self,
        num_samples: int = 16,
        log_every_n_epochs: int = 10,
        save_dir: str = 'plots',
        ddim_steps: int = 50
    ):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ddim_steps = ddim_steps

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate samples at the end of validation epoch"""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        pl_module.eval()
        with torch.no_grad():
            # Generate samples with DDPM
            ddpm_samples = pl_module.sample(
                num_samples=self.num_samples,
                method='ddpm',
                return_all_steps=False
            )

            # Generate samples with DDIM
            ddim_samples = pl_module.sample(
                num_samples=self.num_samples,
                method='ddim',
                num_inference_steps=self.ddim_steps,
                return_all_steps=False
            )

            # Visualize
            self._plot_samples(ddpm_samples, ddim_samples, trainer.current_epoch)

        pl_module.train()

    def _plot_samples(self, ddpm_samples, ddim_samples, epoch):
        """Plot DDPM vs DDIM samples"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # DDPM samples
        ddpm_grid = torchvision.utils.make_grid(
            ddpm_samples.cpu(), nrow=8, normalize=True, value_range=(-1, 1)
        )
        axes[0].imshow(np.transpose(ddpm_grid.numpy(), (1, 2, 0)), cmap='gray')
        axes[0].set_title(f'DDPM Samples (Epoch {epoch + 1})')
        axes[0].axis('off')

        # DDIM samples
        ddim_grid = torchvision.utils.make_grid(
            ddim_samples.cpu(), nrow=8, normalize=True, value_range=(-1, 1)
        )
        axes[1].imshow(np.transpose(ddim_grid.numpy(), (1, 2, 0)), cmap='gray')
        axes[1].set_title(f'DDIM Samples ({self.ddim_steps} steps, Epoch {epoch + 1})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'samples_epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
        plt.close()


class DiffusionProgressCallback(Callback):
    """
    Callback to visualize the diffusion process across timesteps.
    Shows how images evolve from noise to final samples.
    """

    def __init__(
        self,
        num_samples: int = 8,
        num_steps_to_show: int = 10,
        save_dir: str = 'plots',
        log_at_epochs: list = None
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_steps_to_show = num_steps_to_show
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_at_epochs = log_at_epochs or [50, 100, 200]

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate diffusion progress visualization"""
        if (trainer.current_epoch + 1) not in self.log_at_epochs:
            return

        pl_module.eval()
        with torch.no_grad():
            # Generate with all intermediate steps
            _, ddpm_steps = pl_module.sample(
                num_samples=self.num_samples,
                method='ddpm',
                return_all_steps=True
            )

            _, ddim_steps = pl_module.sample(
                num_samples=self.num_samples,
                method='ddim',
                num_inference_steps=50,
                return_all_steps=True
            )

            # Visualize progression
            self._plot_diffusion_progress(ddpm_steps, ddim_steps, trainer.current_epoch)

        pl_module.train()

    def _plot_diffusion_progress(self, ddpm_steps, ddim_steps, epoch):
        """Plot diffusion process over time"""
        # Select evenly spaced timesteps to show
        ddpm_indices = np.linspace(0, len(ddpm_steps) - 1, self.num_steps_to_show, dtype=int)
        ddim_indices = np.linspace(0, len(ddim_steps) - 1, self.num_steps_to_show, dtype=int)

        fig, axes = plt.subplots(2, self.num_steps_to_show, figsize=(20, 5))

        # DDPM progression
        for i, idx in enumerate(ddpm_indices):
            img = ddpm_steps[idx][0]  # First sample
            img = (img.cpu() + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img = img.squeeze().numpy()
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title(f'Step {idx}\n(Noise)', fontsize=8)
            elif i == self.num_steps_to_show - 1:
                axes[0, i].set_title(f'Step {idx}\n(Final)', fontsize=8)
            else:
                axes[0, i].set_title(f'Step {idx}', fontsize=8)

        # DDIM progression
        for i, idx in enumerate(ddim_indices):
            img = ddim_steps[idx][0]  # First sample
            img = (img.cpu() + 1) / 2
            img = img.squeeze().numpy()
            axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title(f'Step {idx}\n(Noise)', fontsize=8)
            elif i == self.num_steps_to_show - 1:
                axes[1, i].set_title(f'Step {idx}\n(Final)', fontsize=8)
            else:
                axes[1, i].set_title(f'Step {idx}', fontsize=8)

        axes[0, 0].set_ylabel('DDPM\n(1000 steps)', fontsize=10, rotation=0, ha='right', va='center')
        axes[1, 0].set_ylabel('DDIM\n(50 steps)', fontsize=10, rotation=0, ha='right', va='center')

        plt.suptitle(f'Diffusion Process Comparison (Epoch {epoch + 1})', fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f'diffusion_progress_epoch_{epoch + 1}.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()


class SampleQualityEvaluator(Callback):
    """
    Callback to evaluate sample quality at different diffusion steps.
    Compares DDPM and DDIM at various inference step counts.
    """

    def __init__(
        self,
        num_samples: int = 64,
        ddim_step_counts: list = None,
        save_dir: str = 'plots',
        log_at_epoch: int = None
    ):
        super().__init__()
        self.num_samples = num_samples
        self.ddim_step_counts = ddim_step_counts or [10, 20, 50, 100, 200]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_at_epoch = log_at_epoch

    def on_train_end(self, trainer, pl_module):
        """Evaluate at the end of training"""
        if self.log_at_epoch is not None and trainer.current_epoch + 1 != self.log_at_epoch:
            return

        pl_module.eval()
        with torch.no_grad():
            # Generate DDPM samples (full 1000 steps)
            ddpm_samples = pl_module.sample(
                num_samples=self.num_samples,
                method='ddpm',
                return_all_steps=False
            )

            # Generate DDIM samples with different step counts
            ddim_samples_dict = {}
            for steps in self.ddim_step_counts:
                samples = pl_module.sample(
                    num_samples=self.num_samples,
                    method='ddim',
                    num_inference_steps=steps,
                    return_all_steps=False
                )
                ddim_samples_dict[steps] = samples

            # Visualize comparison
            self._plot_comparison(ddpm_samples, ddim_samples_dict, trainer.current_epoch)

        pl_module.train()

    def _plot_comparison(self, ddpm_samples, ddim_samples_dict, epoch):
        """Plot quality comparison"""
        num_methods = 1 + len(ddim_samples_dict)
        fig, axes = plt.subplots(1, num_methods, figsize=(4 * num_methods, 4))

        if num_methods == 1:
            axes = [axes]

        # DDPM
        ddpm_grid = torchvision.utils.make_grid(
            ddpm_samples[:16].cpu(), nrow=4, normalize=True, value_range=(-1, 1)
        )
        axes[0].imshow(np.transpose(ddpm_grid.numpy(), (1, 2, 0)), cmap='gray')
        axes[0].set_title('DDPM\n(1000 steps)')
        axes[0].axis('off')

        # DDIM with different step counts
        for i, (steps, samples) in enumerate(ddim_samples_dict.items(), 1):
            grid = torchvision.utils.make_grid(
                samples[:16].cpu(), nrow=4, normalize=True, value_range=(-1, 1)
            )
            axes[i].imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap='gray')
            axes[i].set_title(f'DDIM\n({steps} steps)')
            axes[i].axis('off')

        plt.suptitle(f'Sample Quality Comparison (Epoch {epoch + 1})', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f'quality_comparison_epoch_{epoch + 1}.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()

        print(f"\nSample quality comparison saved to {self.save_dir}")
