"""
Quick test script to verify CFG implementation works correctly.
Tests both input concatenation and cross-attention conditioning methods.
"""

import torch
from ddpm import DDPMCFG


def test_concat_model():
    """Test DDPM-CFG with input concatenation conditioning"""
    print("Testing DDPM-CFG with Input Concatenation...")

    model = DDPMCFG(
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        num_classes=10,
        class_emb_dim=64,
        conditioning_type='concat',
        cfg_dropout=0.1,
        cfg_scale=3.0,
        lr=2e-4
    )

    # Test forward pass with conditioning
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 100, (4,))
    labels = torch.randint(0, 10, (4,))

    noise_pred = model(x, t, labels)
    assert noise_pred.shape == x.shape
    print("  ✓ Conditional forward pass works")

    # Test forward pass without conditioning (unconditional)
    noise_pred_uncond = model(x, t, None)
    assert noise_pred_uncond.shape == x.shape
    print("  ✓ Unconditional forward pass works")

    # Test training step
    batch = (x, labels)
    loss = model.training_step(batch, 0)
    assert loss.item() >= 0
    print("  ✓ Training step works")

    # Test sampling with CFG
    print("  Testing CFG sampling...")
    model.eval()
    with torch.no_grad():
        # Sample with specific class labels
        class_labels = torch.tensor([0, 1, 2, 3])
        samples = model.sample(
            num_samples=4,
            class_labels=class_labels,
            method='ddim',
            num_inference_steps=10,
            cfg_scale=2.0
        )
    assert samples.shape == (4, 1, 28, 28)
    print("  ✓ CFG sampling with labels works")

    # Test sampling without labels (random classes)
    with torch.no_grad():
        samples = model.sample(
            num_samples=4,
            method='ddim',
            num_inference_steps=10,
            cfg_scale=2.0
        )
    assert samples.shape == (4, 1, 28, 28)
    print("  ✓ CFG sampling without labels works")

    # Test different CFG scales
    with torch.no_grad():
        samples_no_cfg = model.sample(num_samples=2, cfg_scale=0.0, method='ddim', num_inference_steps=5)
        samples_low_cfg = model.sample(num_samples=2, cfg_scale=1.0, method='ddim', num_inference_steps=5)
        samples_high_cfg = model.sample(num_samples=2, cfg_scale=5.0, method='ddim', num_inference_steps=5)

    assert samples_no_cfg.shape == (2, 1, 28, 28)
    assert samples_low_cfg.shape == (2, 1, 28, 28)
    assert samples_high_cfg.shape == (2, 1, 28, 28)
    print("  ✓ Different CFG scales work")

    print("\n✅ All concat conditioning tests passed!")


def test_crossattn_model():
    """Test DDPM-CFG with cross-attention conditioning"""
    print("\nTesting DDPM-CFG with Cross-Attention...")

    model = DDPMCFG(
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        num_classes=10,
        class_emb_dim=128,
        conditioning_type='cross_attn',
        cfg_dropout=0.1,
        cfg_scale=3.0,
        lr=2e-4
    )

    # Test forward pass with conditioning
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 100, (4,))
    labels = torch.randint(0, 10, (4,))

    noise_pred = model(x, t, labels)
    assert noise_pred.shape == x.shape
    print("  ✓ Conditional forward pass works")

    # Test forward pass without conditioning (unconditional)
    noise_pred_uncond = model(x, t, None)
    assert noise_pred_uncond.shape == x.shape
    print("  ✓ Unconditional forward pass works")

    # Test training step
    batch = (x, labels)
    loss = model.training_step(batch, 0)
    assert loss.item() >= 0
    print("  ✓ Training step works")

    # Test sampling with CFG
    print("  Testing CFG sampling...")
    model.eval()
    with torch.no_grad():
        # Sample with specific class labels
        class_labels = torch.tensor([0, 1, 2, 3])
        samples = model.sample(
            num_samples=4,
            class_labels=class_labels,
            method='ddim',
            num_inference_steps=10,
            cfg_scale=2.0
        )
    assert samples.shape == (4, 1, 28, 28)
    print("  ✓ CFG sampling with labels works")

    # Test sampling without labels (random classes)
    with torch.no_grad():
        samples = model.sample(
            num_samples=4,
            method='ddim',
            num_inference_steps=10,
            cfg_scale=2.0
        )
    assert samples.shape == (4, 1, 28, 28)
    print("  ✓ CFG sampling without labels works")

    # Test different CFG scales
    with torch.no_grad():
        samples_no_cfg = model.sample(num_samples=2, cfg_scale=0.0, method='ddim', num_inference_steps=5)
        samples_low_cfg = model.sample(num_samples=2, cfg_scale=1.0, method='ddim', num_inference_steps=5)
        samples_high_cfg = model.sample(num_samples=2, cfg_scale=5.0, method='ddim', num_inference_steps=5)

    assert samples_no_cfg.shape == (2, 1, 28, 28)
    assert samples_low_cfg.shape == (2, 1, 28, 28)
    assert samples_high_cfg.shape == (2, 1, 28, 28)
    print("  ✓ Different CFG scales work")

    print("\n✅ All cross-attention conditioning tests passed!")


def test_device_handling():
    """Test device handling (CPU/GPU)"""
    print("\nTesting Device Handling...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    # Test concat model on device
    model_concat = DDPMCFG(
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        num_classes=10,
        conditioning_type='concat',
        lr=2e-4
    )
    model_concat = model_concat.to(device)

    x = torch.randn(2, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (2,)).to(device)

    batch = (x, labels)
    loss = model_concat.training_step(batch, 0)
    assert loss.device.type == device
    print(f"  ✓ Concat model training on {device} works")

    # Test sampling on device
    model_concat.eval()
    with torch.no_grad():
        samples = model_concat.sample(
            num_samples=2,
            method='ddim',
            num_inference_steps=5,
            cfg_scale=2.0
        )
    assert samples.device.type == device
    print(f"  ✓ Concat model sampling on {device} works")

    # Test cross-attention model on device
    model_attn = DDPMCFG(
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=100,
        num_classes=10,
        conditioning_type='cross_attn',
        lr=2e-4
    )
    model_attn = model_attn.to(device)

    loss = model_attn.training_step(batch, 0)
    assert loss.device.type == device
    print(f"  ✓ Cross-attention model training on {device} works")

    model_attn.eval()
    with torch.no_grad():
        samples = model_attn.sample(
            num_samples=2,
            method='ddim',
            num_inference_steps=5,
            cfg_scale=2.0
        )
    assert samples.device.type == device
    print(f"  ✓ Cross-attention model sampling on {device} works")

    print("\n✅ Device handling tests passed!")


def test_cfg_behavior():
    """Test that CFG actually affects outputs"""
    print("\nTesting CFG Behavior...")

    model = DDPMCFG(
        in_channels=1,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        num_timesteps=50,
        num_classes=10,
        conditioning_type='cross_attn',
        lr=2e-4
    )

    model.eval()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    class_labels = torch.tensor([5, 5])

    # Sample with different CFG scales from same initial noise
    with torch.no_grad():
        torch.manual_seed(123)
        samples_cfg_0 = model.sample(
            num_samples=2,
            class_labels=class_labels,
            method='ddim',
            num_inference_steps=5,
            cfg_scale=0.0
        )

        torch.manual_seed(123)
        samples_cfg_3 = model.sample(
            num_samples=2,
            class_labels=class_labels,
            method='ddim',
            num_inference_steps=5,
            cfg_scale=3.0
        )

        torch.manual_seed(123)
        samples_cfg_7 = model.sample(
            num_samples=2,
            class_labels=class_labels,
            method='ddim',
            num_inference_steps=5,
            cfg_scale=7.0
        )

    # Outputs should be different with different CFG scales
    diff_0_3 = (samples_cfg_0 - samples_cfg_3).abs().mean().item()
    diff_3_7 = (samples_cfg_3 - samples_cfg_7).abs().mean().item()

    print(f"  Mean difference (CFG 0.0 vs 3.0): {diff_0_3:.6f}")
    print(f"  Mean difference (CFG 3.0 vs 7.0): {diff_3_7:.6f}")

    # Check that outputs are actually different
    assert diff_0_3 > 0.001, "CFG scale should affect output"
    assert diff_3_7 > 0.001, "CFG scale should affect output"

    print("  ✓ CFG scale affects generation as expected")

    print("\n✅ CFG behavior tests passed!")


if __name__ == '__main__':
    print("=" * 60)
    print("DDPM-CFG Implementation Tests")
    print("=" * 60)

    test_concat_model()
    test_crossattn_model()
    test_device_handling()
    test_cfg_behavior()

    print("\n" + "=" * 60)
    print("✅ All tests passed! CFG implementation is working correctly.")
    print("=" * 60)
    print("\nYou can now train the models using:")
    print("  - Input concatenation: notebooks/cfg_mnist_concat.ipynb")
    print("  - Cross-attention: notebooks/cfg_mnist_crossattn.ipynb")
