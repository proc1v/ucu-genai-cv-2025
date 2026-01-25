"""
Quick test script to verify Rectified Flow implementation works correctly.
Tests both pixel-space and latent-space Rectified Flow models.
"""

import torch
from rectified_flow import RectifiedFlow, LatentRectifiedFlow
from vae.model import VariationalAutoencoder


def test_flow_scheduler():
    """Test Rectified Flow scheduler"""
    print("Testing Rectified Flow Scheduler...")

    from rectified_flow.flow_scheduler import RectifiedFlowScheduler

    scheduler = RectifiedFlowScheduler(num_timesteps=1000)

    # Test forward flow
    x_1 = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))

    x_t, x_0, velocity = scheduler.add_flow(x_1, t)

    assert x_t.shape == (4, 1, 28, 28), f"Expected x_t shape (4, 1, 28, 28), got {x_t.shape}"
    assert x_0.shape == (4, 1, 28, 28), f"Expected x_0 shape (4, 1, 28, 28), got {x_0.shape}"
    assert velocity.shape == (4, 1, 28, 28), f"Expected velocity shape (4, 1, 28, 28), got {velocity.shape}"

    # Verify interpolation
    t_normalized = t.float() / 999.0
    t_normalized = t_normalized.view(-1, 1, 1, 1)
    expected_x_t = t_normalized * x_1 + (1 - t_normalized) * x_0
    assert torch.allclose(x_t, expected_x_t, atol=1e-6), "Interpolation formula incorrect"

    # Verify velocity
    expected_velocity = x_1 - x_0
    assert torch.allclose(velocity, expected_velocity), "Velocity should be x_1 - x_0"

    print("  ✓ Forward flow (interpolation) works")
    print("  ✓ Velocity computation works")
    print()


def test_rectified_flow_model():
    """Test pixel-space Rectified Flow model"""
    print("Testing Pixel-Space Rectified Flow Model...")

    # Create model
    model = RectifiedFlow(
        in_channels=1,
        model_channels=64,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        lr=2e-4,
        num_inference_steps=20
    )

    # Test forward pass (velocity prediction)
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 100, (2,))
    velocity_pred = model(x, t)
    assert velocity_pred.shape == (2, 1, 28, 28), f"Expected shape (2, 1, 28, 28), got {velocity_pred.shape}"
    print("  ✓ Velocity prediction works")

    # Test training step
    batch = (x, torch.zeros(2).long())
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"  ✓ Training step works (loss: {loss.item():.4f})")

    # Test sampling with Euler method
    print("  Testing Euler sampling...")
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples=4, method='euler', num_inference_steps=10)
    assert samples.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {samples.shape}"
    print("  ✓ Euler sampling works")

    # Test sampling with Heun method
    print("  Testing Heun sampling...")
    with torch.no_grad():
        samples = model.sample(num_samples=4, method='heun', num_inference_steps=10)
    assert samples.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {samples.shape}"
    print("  ✓ Heun sampling works")

    # Test sampling with all steps
    print("  Testing sampling with all steps...")
    with torch.no_grad():
        final_samples, all_steps = model.sample(
            num_samples=4, method='euler', num_inference_steps=10, return_all_steps=True
        )
    assert final_samples.shape == (4, 1, 28, 28), f"Expected final shape (4, 1, 28, 28), got {final_samples.shape}"
    assert len(all_steps) == 11, f"Expected 11 steps (initial + 10 steps), got {len(all_steps)}"
    print("  ✓ Sampling with all steps works")

    print()
    return model


def test_latent_rectified_flow_model():
    """Test latent-space Rectified Flow model"""
    print("Testing Latent-Space Rectified Flow Model...")

    # Create VAE for testing
    vae = VariationalAutoencoder(
        base_channel_size=32,
        latent_dim=64,
        num_input_channels=1,
        width=28,
        height=28
    )

    # Test VAE encoding
    x = torch.randn(4, 1, 28, 28)
    mu, logvar = vae.encoder(x)
    assert mu.shape == (4, 64), f"Expected mu shape (4, 64), got {mu.shape}"
    print("  ✓ VAE encoding works")

    # Create Latent Rectified Flow
    lrf = LatentRectifiedFlow(
        vae_model=vae,
        latent_dim=64,
        model_channels=64,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        lr=2e-4,
        num_inference_steps=20,
        freeze_vae=True
    )

    # Test encoding to latent
    z = lrf.encode_to_latent(x)
    assert z.shape == (4, 64), f"Expected latent shape (4, 64), got {z.shape}"
    print("  ✓ Encoding to latent space works")

    # Test decoding from latent
    x_recon = lrf.decode_from_latent(z)
    assert x_recon.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {x_recon.shape}"
    print("  ✓ Decoding from latent space works")

    # Test forward pass (velocity prediction in latent space)
    z_test = torch.randn(2, 64)
    t = torch.randint(0, 100, (2,))
    velocity_pred = lrf(z_test, t)
    assert velocity_pred.shape == (2, 64), f"Expected velocity shape (2, 64), got {velocity_pred.shape}"
    print("  ✓ Velocity prediction in latent space works")

    # Test training step
    batch = (x, torch.zeros(4).long())
    loss = lrf.training_step(batch, 0)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"  ✓ Training step works (loss: {loss.item():.4f})")

    # Test sampling with Euler method
    print("  Testing Euler sampling in latent space...")
    lrf.eval()
    with torch.no_grad():
        samples = lrf.sample(num_samples=4, method='euler', num_inference_steps=10)
    assert samples.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {samples.shape}"
    assert samples.min() >= 0 and samples.max() <= 1, "Samples should be in range [0, 1]"
    print("  ✓ Euler sampling in latent space works")

    # Test sampling with Heun method
    print("  Testing Heun sampling in latent space...")
    with torch.no_grad():
        samples = lrf.sample(num_samples=4, method='heun', num_inference_steps=10)
    assert samples.shape == (4, 1, 28, 28), f"Expected shape (4, 1, 28, 28), got {samples.shape}"
    print("  ✓ Heun sampling in latent space works")

    # Test sampling with latent return
    print("  Testing sampling with latent return...")
    with torch.no_grad():
        samples, latents = lrf.sample(
            num_samples=4, method='euler', num_inference_steps=10, return_latents=True
        )
    assert samples.shape == (4, 1, 28, 28), f"Expected samples shape (4, 1, 28, 28), got {samples.shape}"
    assert latents.shape == (4, 64), f"Expected latents shape (4, 64), got {latents.shape}"
    print("  ✓ Sampling with latent return works")

    print()
    return lrf


def test_device_handling():
    """Test device handling (CPU/GPU)"""
    print("Testing Device Handling...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    # Create VAE and Latent RF
    vae = VariationalAutoencoder(
        base_channel_size=32,
        latent_dim=64,
        num_input_channels=1,
        width=28,
        height=28
    ).to(device)

    lrf = LatentRectifiedFlow(
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

    loss = lrf.training_step(batch, 0)
    assert loss.device.type == device, f"Loss should be on {device}"
    print(f"  ✓ Training on {device} works")

    # Test sampling on device
    lrf.eval()
    with torch.no_grad():
        samples = lrf.sample(num_samples=2, method='euler', num_inference_steps=10)

    assert samples.device.type == device, f"Samples should be on {device}"
    print(f"  ✓ Sampling on {device} works")

    print()


def test_speed_comparison():
    """Test speed comparison functionality"""
    print("Testing Speed Comparison...")

    # Create model
    model = RectifiedFlow(
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=50,
        lr=2e-4,
        num_inference_steps=10
    )

    model.eval()
    results = model.compare_sampling_speed(
        num_samples=2,
        methods=['euler', 'heun'],
        step_counts=[5, 10]
    )

    assert 'rf_euler_5' in results, "Euler 5 steps results missing"
    assert 'rf_euler_10' in results, "Euler 10 steps results missing"
    assert 'rf_heun_5' in results, "Heun 5 steps results missing"
    assert 'rf_heun_10' in results, "Heun 10 steps results missing"

    for key, value in results.items():
        assert value['time'] > 0, f"{key} time should be positive"
        print(f"  ✓ {key}: {value['time']:.3f}s ({value['steps']} steps)")

    print()


if __name__ == '__main__':
    print("=" * 60)
    print("Rectified Flow Implementation Tests")
    print("=" * 60)
    print()

    test_flow_scheduler()
    test_rectified_flow_model()
    test_latent_rectified_flow_model()
    test_device_handling()
    test_speed_comparison()

    print("=" * 60)
    print("✅ All tests passed! Rectified Flow implementation is working correctly.")
    print("=" * 60)
    print()
