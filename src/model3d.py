import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.double_conv = DoubleConv3D(in_channels, out_channels)
        self.down_sample = nn.MaxPool3d(2)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            skip_out = checkpoint(self.double_conv, x, use_reentrant=False)
        else:
            skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode='conv_transpose',
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose3d(
                in_channels - out_channels, in_channels - out_channels,
                kernel_size=2, stride=2,
            )
        elif up_sample_mode == 'trilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (use 'conv_transpose' or 'trilinear')")
        self.double_conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        # Handle odd-dimensional mismatches across all 3 spatial dims
        dd = skip_input.size(2) - x.size(2)
        dh = skip_input.size(3) - x.size(3)
        dw = skip_input.size(4) - x.size(4)
        if dd or dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2,
                          dh // 2, dh - dh // 2,
                          dd // 2, dd - dd // 2])
        x = torch.cat([x, skip_input], dim=1)
        if self.use_checkpoint and self.training:
            return checkpoint(self.double_conv, x, use_reentrant=False)
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_classes=1, base_filters=32,
                 up_sample_mode='conv_transpose', use_checkpoint=True):
        super().__init__()
        f = base_filters  # 32

        # Encoder
        self.down_conv1 = DownBlock3D(in_channels, f, use_checkpoint)
        self.down_conv2 = DownBlock3D(f, f * 2, use_checkpoint)
        self.down_conv3 = DownBlock3D(f * 2, f * 4, use_checkpoint)
        self.down_conv4 = DownBlock3D(f * 4, f * 8, use_checkpoint)

        # Bottleneck
        self.bottleneck = DoubleConv3D(f * 8, f * 8)
        self._use_checkpoint = use_checkpoint

        # Decoder
        self.up_conv4 = UpBlock3D(f * 8 + f * 8, f * 8, up_sample_mode, use_checkpoint)
        self.up_conv3 = UpBlock3D(f * 8 + f * 4, f * 4, up_sample_mode, use_checkpoint)
        self.up_conv2 = UpBlock3D(f * 4 + f * 2, f * 2, up_sample_mode, use_checkpoint)
        self.up_conv1 = UpBlock3D(f * 2 + f, f, up_sample_mode, use_checkpoint)

        # Final
        self.conv_last = nn.Conv3d(f, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)

        # Bottleneck
        if self._use_checkpoint and self.training:
            x = checkpoint(self.bottleneck, x, use_reentrant=False)
        else:
            x = self.bottleneck(x)

        # Decoder
        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)

        return self.conv_last(x)
