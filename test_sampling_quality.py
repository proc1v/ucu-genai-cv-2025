"""
Quick visual test to compare DDPM vs DDIM sampling quality.
Run this after training your LDM model.
"""

import torch
import matplotlib.pyplot as plt
from ldm import LatentDiffusionModel
from vae.model import VariationalAutoencoder


def test_sampling_comparison():
    """Test and visualize DDPM vs DDIM sampling"""
    print("Testing DDPM vs DDIM sampling quality...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create a simple VAE and LDM for testing (untrained)
    vae = VariationalAutoencoder(
        base_channel_size=32,
        latent_dim=128,
        num_input_channels=1,
        width=28,
        height=28
    ).to(device)

    ldm = LatentDiffusionModel(
        vae_model=vae,
        latent_dim=128,
        model_channels=64,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule='linear',
        lr=2e-4,
        num_inference_steps=50,
        freeze_vae=True
    ).to(device)

    ldm.eval()

    print("\n" + "="*60)
    print("NOTE: These are samples from an UNTRAINED model!")
    print("They will look like noise. After training on real data,")
    print("both DDPM and DDIM should produce good samples.")
    print("="*60 + "\n")

    # Generate samples with DDPM
    print("Generating samples with DDPM (100 steps)...")
    with torch.no_grad():
        samples_ddpm = ldm.sample(num_samples=8, method='ddpm')

    # Generate samples with DDIM
    print("Generating samples with DDIM (50 steps)...")
    with torch.no_grad():
        samples_ddim = ldm.sample(num_samples=8, method='ddim', num_inference_steps=50)

    # Visualize
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    # DDPM samples (top row)
    for i in range(8):
        axes[0, i].imshow(samples_ddpm[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('DDPM\n(100 steps)', fontsize=12)

    # DDIM samples (bottom row)
    for i in range(8):
        axes[1, i].imshow(samples_ddim[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('DDIM\n(50 steps)', fontsize=12)

    plt.suptitle('DDPM vs DDIM Sampling Comparison (Untrained Model)', fontsize=14)
    plt.tight_layout()
    plt.savefig('sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison to 'sampling_comparison.png'")
    plt.show()

    # Print statistics
    print("\nSample Statistics:")
    print(f"DDPM - min: {samples_ddpm.min():.3f}, max: {samples_ddpm.max():.3f}, mean: {samples_ddpm.mean():.3f}")
    print(f"DDIM - min: {samples_ddim.min():.3f}, max: {samples_ddim.max():.3f}, mean: {samples_ddim.mean():.3f}")


if __name__ == '__main__':
    test_sampling_comparison()
