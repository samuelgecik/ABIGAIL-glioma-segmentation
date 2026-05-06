"""
Test script to verify DeepLab v3 model and CLI integration.
"""
import torch
from pathlib import Path
from collections import defaultdict
from src.model import DeepLabV3, UNet


VALID_ARCHITECTURES = ('unet', 'deeplabv3', 'nestedunet')


def test_deeplabv3():
    """Test DeepLab v3 model instantiation and forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Test DeepLab v3
    print("Testing DeepLab v3 model...")
    model = DeepLabV3(in_channels=1, out_classes=1).to(device)

    # Create a dummy input (batch_size=2, channels=1, height=256, width=256)
    dummy_input = torch.randn(2, 1, 256, 256).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  ✓ DeepLab v3 test passed!\n")

    # Test UNet for comparison
    print("Testing UNet model (for comparison)...")
    unet = UNet(in_channels=1, out_classes=1, up_sample_mode='conv_transpose').to(device)

    with torch.no_grad():
        unet_output = unet(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {unet_output.shape}")
    print(f"  Output range: [{unet_output.min():.3f}, {unet_output.max():.3f}]")

    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"  Total parameters: {unet_params:,}")
    print("  ✓ UNet test passed!\n")


def test_filename_parsing():
    """Test that inference filename parsing works for all architectures."""
    print("Testing filename parsing logic...")

    model_dir = Path("saved_models")
    if not model_dir.exists():
        print("  SKIP: saved_models/ not found")
        return

    for model_arch in VALID_ARCHITECTURES:
        # Replicate the filtering/parsing logic from inference.py
        if model_arch == 'unet':
            glob_pattern = "best_*.pt"
            arch_prefixes = tuple(f"best_{a}_" for a in VALID_ARCHITECTURES if a != 'unet')
        else:
            glob_pattern = f"best_{model_arch}_*.pt"
            arch_prefixes = None

        model_files = list(model_dir.glob(glob_pattern))
        if arch_prefixes:
            model_files = [f for f in model_files if not f.name.startswith(arch_prefixes)]

        if not model_files:
            print(f"  {model_arch}: no model files found (OK if not trained yet)")
            continue

        prefix_parts = 1 if model_arch == 'unet' else 2
        timestamp_groups = defaultdict(list)
        for model_file in model_files:
            parts = model_file.stem.split("_")
            if len(parts) >= prefix_parts + 2:
                orientation = parts[prefix_parts]
                timestamp = "_".join(parts[prefix_parts + 1:])
                timestamp_groups[timestamp].append((orientation, model_file))

        found = False
        for ts, models in sorted(timestamp_groups.items(), reverse=True):
            orientations = {ori for ori, _ in models}
            if {"axial", "coronal", "sagittal"}.issubset(orientations):
                print(f"  {model_arch}: complete set at {ts}")
                # Verify orientations parsed correctly
                for ori, _ in models:
                    assert ori in ("axial", "coronal", "sagittal"), f"Bad orientation: {ori}"
                found = True
                break

        if not found:
            print(f"  {model_arch}: no complete set found (OK if not all orientations trained)")

    print("  ✓ Filename parsing test passed!\n")


if __name__ == "__main__":
    test_deeplabv3()
    test_filename_parsing()
    print("All tests completed successfully!")
