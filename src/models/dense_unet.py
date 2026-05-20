from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=0,
                dilation=dilation,
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _valid_groups(channels: int, groups: int) -> int:
    groups = min(groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class GNAct(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_valid_groups(channels, groups), channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        dilation: int = 1,
    ):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.conv1 = ReflectConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = GNAct(out_channels, groups=groups)
        self.conv2 = ReflectConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_valid_groups(out_channels, groups), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.act(y + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block = ResConvBlock(in_channels, out_channels, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()
        self.block = ResConvBlock(in_channels + skip_channels, out_channels, groups=groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DenseUNetSmallEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        latent_channels: int = 32,
        groups: int = 8,
        final_norm: bool = True,
    ):
        super().__init__()
        self.stem = ResConvBlock(in_channels, width, groups=groups)
        self.down1 = DownBlock(width, width * 2, groups=groups)
        self.down2 = DownBlock(width * 2, width * 4, groups=groups)
        self.bottleneck = ResConvBlock(width * 4, width * 4, groups=groups, dilation=1)
        self.up1 = UpBlock(in_channels=width * 4, skip_channels=width * 2, out_channels=width * 2, groups=groups)
        self.up2 = UpBlock(in_channels=width * 2, skip_channels=width, out_channels=width, groups=groups)
        self.head = nn.Conv2d(width, latent_channels, kernel_size=1)
        self.final_norm = nn.GroupNorm(_valid_groups(latent_channels, groups), latent_channels) if final_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        z = self.bottleneck(s2)
        z = self.up1(z, s1)
        z = self.up2(z, s0)
        z = self.head(z)
        z = self.final_norm(z)
        return z
