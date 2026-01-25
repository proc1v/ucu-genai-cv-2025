"""
Quick test script to verify Latent Diffusion Model implementation works correctly.
"""

import torch
from ldm import LatentDiffusionModel
from vae.model import VariationalAutoencoder


def test_vae_loading():
    """Test VAE creation and encoding/decoding"""
    print("Testing VAE initialization...")

    # Create a simple VAE for testing
    vae = VariationalAutoencoder(
        base_channel_size=32,
        latent_dim=128,
        num_input_channels=1,
        width=28,
        height=28
    )

    # Test encoding
    x = torch.randn(4, 1, 28, 28)
    mu, logvar = vae.encoder(x)
    assert mu.shape == (4, 128), f"Expected mu shape (4, 128), got {mu.shape}"
    assert logvar.shape == (4, 128), f"Expected logvar shape (4, 128), got {logvar.shape}"
    print("  ✓ VAE encoding works")

    # Test decoding
    z = torch.randn(4, 128)
    x_recon = vae.decoder(z)
    assert x_recon.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {x_recon.shape}"
    print("  ✓ VAE decoding works")

    return vae


def test_ldm_model(vae):
    """Test Latent Diffusion Model"""
    print("\nTesting Latent Diffusion Model...")

    # Create LDM with the VAE
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
    )

    # Test encoding to latent
    x = torch.randn(2, 1, 28, 28)
    z = ldm.encode_to_latent(x)
    assert z.shape == (2, 128), f"Expected latent shape (2, 128), got {z.shape}"
    print("  ✓ Encoding to latent space works")

    # Test decoding from latent
    x_recon = ldm.decode_from_latent(z)
    assert x_recon.shape == (2, 1, 28, 28), f"Expected shape (2, 1, 28, 28), got {x_recon.shape}"
    print("  ✓ Decoding from latent space works")

    # Test forward pass (noise prediction in latent space)
    z = torch.randn(2, 128)
    t = torch.randint(0, 100, (2,))
    noise_pred = ldm(z, t)
    assert noise_pred.shape == (2, 128), f"Expected noise shape (2, 128), got {noise_pred.shape}"
    print("  ✓ Noise prediction in latent space works")

    # Test training step
    batch = (x, torch.zeros(2).long())
    loss = ldm.training_step(batch, 0)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"  ✓ Training step works (loss: {loss.item():.4f})")

    return ldm


def test_ldm_sampling(ldm):
    """Test LDM sampling with DDPM and DDIM"""
    print("\nTesting LDM Sampling...")

    ldm.eval()

    # Test DDPM sampling in latent space
    print("  Testing DDPM sampling...")
    with torch.no_grad():
        samples = ldm.sample(num_samples=4, method='ddpm')
    assert samples.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {samples.shape}"
    assert samples.min() >= 0 and samples.max() <= 1, "Samples should be in range [0, 1]"
    print("  ✓ DDPM sampling works")

    # Test DDIM sampling in latent space
    print("  Testing DDIM sampling...")
    with torch.no_grad():
        samples = ldm.sample(num_samples=4, method='ddim', num_inference_steps=10)
    assert samples.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {samples.shape}"
    assert samples.min() >= 0 and samples.max() <= 1, "Samples should be in range [0, 1]"
    print("  ✓ DDIM sampling works")

    # Test sampling with return_latents
    print("  Testing sampling with latent return...")
    with torch.no_grad():
        samples, latents = ldm.sample(num_samples=4, method='ddim',
                                     num_inference_steps=10, return_latents=True)
    assert samples.shape == (4, 1, 28, 28), f"Expected samples shape (4, 1, 28, 28), got {samples.shape}"
    assert latents.shape == (4, 128), f"Expected latents shape (4, 128), got {latents.shape}"
    print("  ✓ Sampling with latent return works")


def test_device_handling():
    """Test device handling (CPU/GPU)"""
    print("\nTesting Device Handling...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    # Create VAE and LDM
    vae = VariationalAutoencoder(
        base_channel_size=32,
        latent_dim=64,
        num_input_channels=1,
        width=28,
        height=28
    ).to(device)

    ldm = LatentDiffusionModel(
        vae_model=vae,
        latent_dim=64,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        lr=2e-4
    ).to(device)

    # Test training step on device
    x = torch.randn(2, 1, 28, 28).to(device)
    labels = torch.zeros(2).long().to(device)
    batch = (x, labels)

    loss = ldm.training_step(batch, 0)
    assert loss.device.type == device, f"Loss should be on {device}"
    print(f"  ✓ Training on {device} works")

    # Test sampling on device
    ldm.eval()
    with torch.no_grad():
        samples = ldm.sample(num_samples=2, method='ddim', num_inference_steps=10)

    assert samples.device.type == device, f"Samples should be on {device}"
    print(f"  ✓ Sampling on {device} works")

    print("\n✅ Device handling tests passed!")


def test_speed_comparison():
    """Test speed comparison functionality"""
    print("\nTesting Speed Comparison...")

    vae = VariationalAutoencoder(
        base_channel_size=32,
        latent_dim=64,
        num_input_channels=1,
        width=28,
        height=28
    )

    ldm = LatentDiffusionModel(
        vae_model=vae,
        latent_dim=64,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=50,  # Fewer timesteps for faster testing
        lr=2e-4,
        num_inference_steps=10
    )

    ldm.eval()
    results = ldm.compare_sampling_speed(num_samples=2, methods=['ddpm', 'ddim'], ddim_steps=10)

    assert 'ldm_ddpm' in results, "DDPM results missing"
    assert 'ldm_ddim' in results, "DDIM results missing"
    assert results['ldm_ddpm']['time'] > 0, "DDPM time should be positive"
    assert results['ldm_ddim']['time'] > 0, "DDIM time should be positive"

    print(f"  ✓ DDPM: {results['ldm_ddpm']['time']:.3f}s ({results['ldm_ddpm']['steps']} steps)")
    print(f"  ✓ DDIM: {results['ldm_ddim']['time']:.3f}s ({results['ldm_ddim']['steps']} steps)")
    print(f"  ✓ Speedup: {results['ldm_ddpm']['time'] / results['ldm_ddim']['time']:.2f}x")

    print("\n✅ Speed comparison tests passed!")


if __name__ == '__main__':
    print("="*60)
    print("Latent Diffusion Model Implementation Tests")
    print("="*60)

    vae = test_vae_loading()
    ldm = test_ldm_model(vae)
    test_ldm_sampling(ldm)
    test_device_handling()
    test_speed_comparison()

    print("\n" + "="*60)
    print("✅ All tests passed! LDM implementation is working correctly.")
    print("="*60)
