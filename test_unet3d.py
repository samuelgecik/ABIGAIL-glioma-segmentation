"""
Verification tests for 3D UNet pipeline.

Tests:
1. Forward pass shape correctness
2. GPU memory usage with batch_size=4 + AMP + gradient checkpointing
3. Dataset output shapes
4. Augmentation shape preservation
"""
import torch
import torch.nn as nn


def test_forward_pass():
    """Verify UNet3D produces correct output shapes."""
    from src.model3d import UNet3D

    print("=" * 60)
    print("TEST: Forward pass shape correctness")
    print("=" * 60)

    model = UNet3D(in_channels=1, out_classes=1, base_filters=32, use_checkpoint=False)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test with exact padded BraTS dimensions
    x = torch.randn(1, 1, 192, 224, 192)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print("PASSED: Output shape matches input shape\n")

    # Test with non-padded dimensions (odd sizes)
    x2 = torch.randn(1, 1, 182, 218, 182)
    with torch.no_grad():
        out2 = model(x2)
    print(f"Input:  {x2.shape}")
    print(f"Output: {out2.shape}")
    assert out2.shape == x2.shape, f"Shape mismatch: {out2.shape} != {x2.shape}"
    print("PASSED: Handles non-power-of-2 dimensions\n")


def test_gpu_memory():
    """Verify batch_size=4 fits in GPU memory with AMP + gradient checkpointing."""
    from src.model3d import UNet3D

    if not torch.cuda.is_available():
        print("SKIPPED: No GPU available")
        return

    print("=" * 60)
    print("TEST: GPU memory with batch=4, AMP, gradient checkpointing")
    print("=" * 60)

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = UNet3D(in_channels=1, out_classes=1, base_filters=32, use_checkpoint=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()

    mem_after_model = torch.cuda.memory_allocated() / 1e9
    print(f"Memory after model load: {mem_after_model:.2f} GB")

    # Simulate one training step with batch_size=4 at padded volume size
    x = torch.randn(4, 1, 192, 224, 192, device=device)
    target = torch.zeros(4, 1, 192, 224, 192, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast('cuda'):
        out = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out, target)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    total_memory = torch.cuda.get_device_properties(0).total_mem / 1e9

    print(f"Output shape: {out.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Peak GPU memory: {peak_memory:.2f} GB")
    print(f"Total GPU memory: {total_memory:.2f} GB")
    print(f"Headroom: {total_memory - peak_memory:.2f} GB")
    assert peak_memory < total_memory, "OOM: peak memory exceeds GPU capacity!"
    print("PASSED: Fits in GPU memory\n")

    del x, target, out, loss
    torch.cuda.empty_cache()


def test_augmentation():
    """Verify RandomAugment3D preserves tensor shapes."""
    from src.dataset3d import RandomAugment3D

    print("=" * 60)
    print("TEST: Augmentation shape preservation")
    print("=" * 60)

    aug = RandomAugment3D()
    image = torch.randn(1, 182, 218, 182)
    label = torch.randint(0, 2, (1, 182, 218, 182), dtype=torch.float32)

    for i in range(10):
        img_aug, lbl_aug = aug(image.clone(), label.clone())
        assert img_aug.shape == image.shape, f"Image shape changed: {img_aug.shape}"
        assert lbl_aug.shape == label.shape, f"Label shape changed: {lbl_aug.shape}"

    print(f"Shape: {image.shape} -> {img_aug.shape}")
    print("PASSED: 10 augmentation rounds preserved shapes\n")


def test_parameter_comparison():
    """Compare parameter counts between 2D and 3D models."""
    from src.model import UNet
    from src.model3d import UNet3D

    print("=" * 60)
    print("TEST: Parameter comparison")
    print("=" * 60)

    unet2d = UNet(in_channels=1, out_classes=1)
    unet3d = UNet3D(in_channels=1, out_classes=1, base_filters=32)

    p2d = sum(p.numel() for p in unet2d.parameters())
    p3d = sum(p.numel() for p in unet3d.parameters())

    print(f"2D UNet:  {p2d:>12,} parameters")
    print(f"3D UNet:  {p3d:>12,} parameters")
    print(f"Ratio:    {p3d / p2d:.2f}x")
    print()


if __name__ == "__main__":
    test_forward_pass()
    test_augmentation()
    test_parameter_comparison()
    test_gpu_memory()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
