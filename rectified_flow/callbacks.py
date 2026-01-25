"""
Callbacks for Rectified Flow training and evaluation.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from pathlib import Path


class RFSamplingCallback(Callback):
    """
    Callback to generate and visualize Rectified Flow samples during training.
    Compares Euler and Heun integration methods.
    """

    def __init__(
        self,
        num_samples: int = 16,
        log_every_n_epochs: int = 10,
        save_dir: str = 'plots/rectified_flow',
        num_steps: int = 50
    ):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_steps = num_steps

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate samples at the end of validation epoch"""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        pl_module.eval()
        with torch.no_grad():
            # Generate samples with Euler method
            euler_samples = pl_module.sample(
                num_samples=self.num_samples,
                method='euler',
                num_inference_steps=self.num_steps,
                return_all_steps=False
            )

            # Generate samples with Heun method
            heun_samples = pl_module.sample(
                num_samples=self.num_samples,
                method='heun',
                num_inference_steps=self.num_steps,
                return_all_steps=False
            )

            # Visualize
            self._plot_samples(euler_samples, heun_samples, trainer.current_epoch)

        pl_module.train()

    def _plot_samples(self, euler_samples, heun_samples, epoch):
        """Plot Euler vs Heun samples"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Euler samples
        euler_grid = torchvision.utils.make_grid(
            euler_samples.cpu(), nrow=8, normalize=True, value_range=(-1, 1)
        )
        axes[0].imshow(np.transpose(euler_grid.numpy(), (1, 2, 0)), cmap='gray')
        axes[0].set_title(f'Euler Method Samples (Epoch {epoch + 1}, {self.num_steps} steps)')
        axes[0].axis('off')

        # Heun samples
        heun_grid = torchvision.utils.make_grid(
            heun_samples.cpu(), nrow=8, normalize=True, value_range=(-1, 1)
        )
        axes[1].imshow(np.transpose(heun_grid.numpy(), (1, 2, 0)), cmap='gray')
        axes[1].set_title(f'Heun Method Samples (Epoch {epoch + 1}, {self.num_steps} steps)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'samples_epoch_{epoch + 1}.png', dpi=150, bbox_inches='tight')
        plt.close()


class FlowProgressCallback(Callback):
    """
    Callback to visualize the flow process across timesteps.
    Shows how images evolve from noise to final samples via straight-line interpolation.
    """

    def __init__(
        self,
        num_samples: int = 8,
        num_steps_to_show: int = 10,
        save_dir: str = 'plots/rectified_flow',
        log_at_epochs: list = None
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_steps_to_show = num_steps_to_show
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_at_epochs = log_at_epochs or [50, 100, 200]

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate flow progress visualization"""
        if (trainer.current_epoch + 1) not in self.log_at_epochs:
            return

        pl_module.eval()
        with torch.no_grad():
            # Generate with all intermediate steps
            _, euler_steps = pl_module.sample(
                num_samples=self.num_samples,
                method='euler',
                num_inference_steps=50,
                return_all_steps=True
            )

            _, heun_steps = pl_module.sample(
                num_samples=self.num_samples,
                method='heun',
                num_inference_steps=50,
                return_all_steps=True
            )

            # Visualize progression
            self._plot_flow_progress(euler_steps, heun_steps, trainer.current_epoch)

        pl_module.train()

    def _plot_flow_progress(self, euler_steps, heun_steps, epoch):
        """Plot flow process over time"""
        # Select evenly spaced timesteps to show
        euler_indices = np.linspace(0, len(euler_steps) - 1, self.num_steps_to_show, dtype=int)
        heun_indices = np.linspace(0, len(heun_steps) - 1, self.num_steps_to_show, dtype=int)

        fig, axes = plt.subplots(2, self.num_steps_to_show, figsize=(20, 5))

        # Euler progression
        for i, idx in enumerate(euler_indices):
            img = euler_steps[idx][0]  # First sample
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

        # Heun progression
        for i, idx in enumerate(heun_indices):
            img = heun_steps[idx][0]  # First sample
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

        axes[0, 0].set_ylabel('Euler\n(50 steps)', fontsize=10, rotation=0, ha='right', va='center')
        axes[1, 0].set_ylabel('Heun\n(50 steps)', fontsize=10, rotation=0, ha='right', va='center')

        plt.suptitle(f'Rectified Flow Process Comparison (Epoch {epoch + 1})', fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f'flow_progress_epoch_{epoch + 1}.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()


class RFQualityEvaluator(Callback):
    """
    Callback to evaluate Rectified Flow sample quality at different step counts.
    Compares Euler and Heun methods with various inference steps.
    """

    def __init__(
        self,
        num_samples: int = 64,
        step_counts: list = None,
        save_dir: str = 'plots/rectified_flow',
        log_at_epoch: int = None
    ):
        super().__init__()
        self.num_samples = num_samples
        self.step_counts = step_counts or [10, 25, 50, 100]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_at_epoch = log_at_epoch

    def on_train_end(self, trainer, pl_module):
        """Evaluate at the end of training"""
        if self.log_at_epoch is not None and trainer.current_epoch + 1 != self.log_at_epoch:
            return

        pl_module.eval()
        with torch.no_grad():
            # Generate samples with different step counts
            samples_dict = {}
            for steps in self.step_counts:
                # Euler method
                euler_samples = pl_module.sample(
                    num_samples=self.num_samples,
                    method='euler',
                    num_inference_steps=steps,
                    return_all_steps=False
                )
                samples_dict[f'euler_{steps}'] = euler_samples

                # Heun method
                heun_samples = pl_module.sample(
                    num_samples=self.num_samples,
                    method='heun',
                    num_inference_steps=steps,
                    return_all_steps=False
                )
                samples_dict[f'heun_{steps}'] = heun_samples

            # Visualize comparison
            self._plot_comparison(samples_dict, trainer.current_epoch)

        pl_module.train()

    def _plot_comparison(self, samples_dict, epoch):
        """Plot quality comparison"""
        num_methods = len(samples_dict)
        num_cols = min(num_methods, 4)
        num_rows = (num_methods + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        axes = axes.flatten() if num_methods > 1 else [axes]

        for idx, (name, samples) in enumerate(samples_dict.items()):
            if idx >= len(axes):
                break

            grid = torchvision.utils.make_grid(
                samples[:16].cpu(), nrow=4, normalize=True, value_range=(-1, 1)
            )
            axes[idx].imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap='gray')

            # Parse method and steps from name
            method, steps = name.split('_')
            axes[idx].set_title(f'{method.capitalize()}\n({steps} steps)')
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(len(samples_dict), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Rectified Flow Quality Comparison (Epoch {epoch + 1})', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / f'quality_comparison_epoch_{epoch + 1}.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()

        print(f"\nRectified Flow quality comparison saved to {self.save_dir}")
