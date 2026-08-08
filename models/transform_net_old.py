import torch
import torch.nn as nn


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


def conv_block(in_channels, out_channels, kernel_size, stride, use_instance_norm=True):
    layers = [
        ReflectionConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
    ]
    if use_instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ReflectionConv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            ReflectionConv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class TransformNetwork(nn.Module):
    """Feed-forward image transformation network for style transfer.

    The network follows a downsample-residual-upsample layout:
    - initial convolution
    - two downsampling convolutions with stride 2
    - five residual blocks
    - two transposed convolutions for upsampling
    - final convolution to RGB output
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            ReflectionConv2d(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            ReflectionConv2d(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            nn.Upsample(scale_factor=2.0, mode='nearest'),
            ReflectionConv2d(128, 64, kernel_size=3, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2.0, mode='nearest'),
            ReflectionConv2d(64, 32, kernel_size=3, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)
