"""
Quick test script to verify DDPM implementation works correctly.
"""

import torch
from ddpm import DDPM, DDPMScheduler, DDIMScheduler

def test_schedulers():
    """Test DDPM and DDIM schedulers"""
    print("Testing DDPM Scheduler...")

    # Create scheduler
    scheduler = DDPMScheduler(num_timesteps=1000)

    # Test forward diffusion
    x_0 = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))

    x_t, noise = scheduler.add_noise(x_0, t)
    assert x_t.shape == x_0.shape
    assert noise.shape == x_0.shape
    print("  ✓ Forward diffusion works")

    # Test reverse diffusion
    predicted_noise = torch.randn_like(noise)
    x_prev = scheduler.sample_prev_timestep(x_t, t, predicted_noise)
    assert x_prev.shape == x_t.shape
    print("  ✓ Reverse diffusion works")

    # Test DDIM scheduler
    print("\nTesting DDIM Scheduler...")
    ddim_scheduler = DDIMScheduler(scheduler, num_inference_steps=50)

    x_t = torch.randn(4, 1, 28, 28)
    predicted_noise = torch.randn_like(x_t)
    x_prev = ddim_scheduler.sample_prev_timestep(x_t, 25, predicted_noise)
    assert x_prev.shape == x_t.shape
    print("  ✓ DDIM sampling works")

    print("\n✅ All scheduler tests passed!")


def test_model():
    """Test DDPM model"""
    print("\nTesting DDPM Model...")

    model = DDPM(
        in_channels=1,              # Grayscale images
        model_channels=64,          # Base UNet channels
        channel_mult=(1, 2, 4),     # Channel multipliers per level
        num_res_blocks=2,           # Residual blocks per level
        num_timesteps=100,         # Diffusion timesteps
        beta_start=1e-4,            # Starting noise level
        beta_end=0.02,              # Ending noise level
        beta_schedule='linear',     # Noise schedule type
        lr=2e-4,                    # Learning rate
        num_inference_steps=50,     # DDIM steps for inference
        dropout=0.1,                # Dropout probability
        num_heads=4                 # Attention heads
    )

    # Test forward pass
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 100, (2,))

    noise_pred = model(x, t)
    assert noise_pred.shape == x.shape
    print("  ✓ Model forward pass works")

    # Test DDPM sampling
    print("  Testing DDPM sampling...")
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples=4, method='ddpm')
    assert samples.shape == (4, 1, 28, 28)
    print("  ✓ DDPM sampling works")

    # Test DDIM sampling
    print("  Testing DDIM sampling...")
    with torch.no_grad():
        samples = model.sample(num_samples=4, method='ddim', num_inference_steps=10)
    assert samples.shape == (4, 1, 28, 28)
    print("  ✓ DDIM sampling works")

    print("\n✅ All model tests passed!")


def test_device_handling():
    """Test device handling (CPU/GPU)"""
    print("\nTesting Device Handling...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model = DDPM(
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        lr=2e-4
    )
    model = model.to(device)

    # Test training step
    x = torch.randn(2, 1, 28, 28).to(device)
    labels = torch.zeros(2).long().to(device)

    batch = (x, labels)
    loss = model.training_step(batch, 0)

    assert loss.device.type == device
    print(f"  ✓ Training on {device} works")

    # Test sampling on device
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples=2, method='ddim', num_inference_steps=10)

    assert samples.device.type == device
    print(f"  ✓ Sampling on {device} works")

    print("\n✅ Device handling tests passed!")


if __name__ == '__main__':
    print("="*60)
    print("DDPM Implementation Tests")
    print("="*60)

    test_schedulers()
    test_model()
    test_device_handling()

    print("\n" + "="*60)
    print("✅ All tests passed! DDPM implementation is working correctly.")
    print("="*60)
