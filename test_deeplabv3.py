"""
Test script to verify DeepLab v3 model can be instantiated and used.
"""
import torch
from src.model import DeepLabV3, UNet

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
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_deeplabv3()
