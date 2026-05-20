import torch
import torch.nn as nn
from typing import Optional


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FullResEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(latent_channels, dilation=1),
            ResidualBlock(latent_channels, dilation=1),
            ResidualBlock(latent_channels, dilation=2),
            ResidualBlock(latent_channels, dilation=2),
            ResidualBlock(latent_channels, dilation=4),
            ResidualBlock(latent_channels, dilation=4),
            ResidualBlock(latent_channels, dilation=8),
            ResidualBlock(latent_channels, dilation=8),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x


class ConvNeXtDenseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
        use_reflect_padding: bool = True,
    ):
        super().__init__()
        pad = int(kernel_size // 2)
        if use_reflect_padding:
            self.dwconv = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=0, groups=channels),
            )
        else:
            self.dwconv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, expansion * channels)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(expansion * channels, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtDenseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        use_reflect_padding: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1) if use_reflect_padding else nn.Identity(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0 if use_reflect_padding else 1),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ConvNeXtDenseBlock(
                    channels=hidden_channels,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    use_reflect_padding=use_reflect_padding,
                )
                for _ in range(int(depth))
            ]
        )
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.final_norm = LayerNorm2d(latent_channels) if final_norm else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x


class ResCNNBlock(nn.Module):
    def __init__(self, channels: int, hidden: Optional[int] = None):
        super().__init__()
        if hidden is None:
            hidden = channels
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=hidden),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResCNNDenseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 6,
        final_norm: bool = True,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResCNNBlock(hidden_channels) for _ in range(int(depth))])
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.final_norm = nn.GroupNorm(num_groups=1, num_channels=latent_channels) if final_norm else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x
