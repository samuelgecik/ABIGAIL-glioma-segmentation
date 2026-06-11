import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import segmentation
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from src.unet_nested import NestedUNet
from src.model3d import UNet3D

class DeepLabV3(nn.Module):
    def __init__(self, in_channels=1, out_classes=1):
        """
        DeepLab v3 with ResNet-101 backbone from torch hub.
        Adapted for single-channel input and custom number of output classes.
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            out_classes: Number of output classes (default: 1 for binary segmentation)
        """
        super(DeepLabV3, self).__init__()
        
        # Load DeepLab v3 with ResNet-101 from torchvision
        self.model = segmentation.deeplabv3_resnet101(weights=None)
        
        # Modify the first conv layer to accept single-channel input if needed
        if in_channels != 3:
            # Get the original first conv layer
            original_conv1 = self.model.backbone.conv1
            # Create new conv1 with desired input channels
            # Keep same out_channels, kernel_size, stride, padding, bias as original
            self.model.backbone.conv1 = nn.Conv2d(
                in_channels, 
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
        
        # Modify the classifier to output the desired number of classes
        # DeepLab v3 has a classifier with a final conv layer
        self.model.classifier[4] = nn.Conv2d(256, out_classes, kernel_size=1)
        
        # Also modify the auxiliary classifier if it exists and is not None
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, out_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through DeepLab v3.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
            
        Returns:
            Output tensor of shape (batch_size, out_classes, H, W)
        """
        # DeepLab v3 returns a dictionary with 'out' key during training
        # and 'aux' key if auxiliary classifier is present
        output = self.model(x)
        
        # Extract the main output
        if isinstance(output, dict):
            return output['out']
        return output

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=2, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(in_channels, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)

        dy = skip_input.size(2) - x.size(2)
        dx = skip_input.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2,  # left, right
                          dy // 2, dy - dy // 2])  # top,  bottom

        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


def build_segformer(
    num_classes: int = 4,
    in_channels: int = 1,
    variant: str = "b2",
    pretrained: bool = True,
) -> SegformerForSemanticSegmentation:
    """
    Build a SegFormer segmentation model.

    Parameters
    ----------
    num_classes : number of segmentation classes (4 for BraTS).
    in_channels : input channels (1 for single MRI modality).
    variant     : model size – "b0" (3.7M) ... "b5" (84M).
                  "b2" (~27M) is a good speed/accuracy trade-off.
    pretrained  : if True, load ImageNet pre-trained weights and adapt
                  the first conv + final head to our task.
    """
    hub_name = f"nvidia/segformer-{variant}-finetuned-ade-512-512"

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            hub_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        cfg = SegformerConfig.from_pretrained(hub_name, num_labels=num_classes)
        model = SegformerForSemanticSegmentation(cfg)

    # Adapt first conv layer if input channels != 3
    if in_channels != 3:
        _adapt_first_conv_to_channels(model, in_channels, pretrained)

    return model


def _adapt_first_conv_to_channels(model: nn.Module, in_channels: int, pretrained: bool = True) -> None:
    """Find the first Conv2d that expects 3 RGB channels and adapt it to `in_channels`.

    Locating the layer by its signature (in_channels == 3) rather than by a hard-coded
    attribute path keeps this robust across `transformers` versions, which have changed
    the SegFormer module layout (e.g. `segformer.encoder.patch_embeddings[0].proj` in
    4.x vs `segformer.stages.0.patch_embeddings.proj` in 5.x).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels,
                module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
            )
            if pretrained:
                with torch.no_grad():
                    # Average the pre-trained RGB weights into the new single channel
                    new_conv.weight.copy_(module.weight.mean(dim=1, keepdim=True))
                    if module.bias is not None:
                        new_conv.bias.copy_(module.bias)
            # Replace in the module tree
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            return
    raise RuntimeError("No Conv2d with in_channels=3 found in model")


class SegFormerWrapper(nn.Module):
    """
    SegFormer with the same interface as UNet / DeepLabV3 / NestedUNet:

        model = SegFormerWrapper(in_channels=1, out_classes=1).to(device)

    Parameters
    ----------
    in_channels : input channels (1 for single MRI modality).
    out_classes : output channels / classes (1 for binary segmentation).
    variant     : model size – "b0" (3.7M) … "b5" (84M).  Default "b2".
    pretrained  : load ImageNet pre-trained encoder weights.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 1,
        variant: str = "b2",
        pretrained: bool = True,
    ):
        super().__init__()
        self.out_classes = out_classes
        self.model = build_segformer(
            num_classes=out_classes,
            in_channels=in_channels,
            variant=variant,
            pretrained=pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, in_channels, H, W)
        returns: (B, out_classes, H, W) logits at full input resolution
        """
        out = self.model(pixel_values=x)
        logits = out.logits  # (B, out_classes, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


# Registry of supported 2D segmentation architectures (binary, single-channel).
VALID_ARCHITECTURES = ("unet", "deeplabv3", "nestedunet", "segformer")


def build_model(model_arch: str, in_channels: int = 1, out_classes: int = 1) -> nn.Module:
    """Instantiate a 2D segmentation model by architecture name.

    Single source of truth shared by training, validation inference, and
    single-case prediction, so registering a new architecture only requires
    adding it here and to ``VALID_ARCHITECTURES``.
    """
    if model_arch == "deeplabv3":
        return DeepLabV3(in_channels=in_channels, out_classes=out_classes)
    if model_arch == "segformer":
        return SegFormerWrapper(in_channels=in_channels, out_classes=out_classes)
    if model_arch == "nestedunet":
        return NestedUNet(in_ch=in_channels, out_ch=out_classes)
    if model_arch == "unet":
        return UNet(in_channels=in_channels, out_classes=out_classes, up_sample_mode="conv_transpose")
    raise ValueError(
        f"Unknown model_arch '{model_arch}'. Choose from {VALID_ARCHITECTURES}."
    )

