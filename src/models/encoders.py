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


def _valid_groups(channels: int, groups: int) -> int:
    g = max(1, int(groups))
    while channels % g != 0 and g > 1:
        g -= 1
    return g


def make_norm2d(channels: int, norm_type: str = "layernorm", norm_groups: int = 8, norm_eps: float = 1e-6) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "layernorm":
        return LayerNorm2d(channels, eps=float(norm_eps))
    if kind == "groupnorm":
        return nn.GroupNorm(_valid_groups(channels, norm_groups), channels, eps=float(norm_eps))
    raise ValueError(f"Unsupported norm_type={norm_type}. Use 'layernorm' or 'groupnorm'.")


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
    def __init__(
        self,
        channels: int,
        hidden: Optional[int] = None,
        norm_type: str = "groupnorm",
        norm_groups: int = 1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if hidden is None:
            hidden = channels
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=0),
            make_norm2d(hidden, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=0),
            make_norm2d(channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
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
        norm_type: str = "groupnorm",
        norm_groups: int = 1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0),
            make_norm2d(hidden_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResCNNBlock(
                    hidden_channels,
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                )
                for _ in range(int(depth))
            ]
        )
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.final_norm = (
            make_norm2d(latent_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps)
            if final_norm
            else nn.Identity()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x


class PyramidResDilatedBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        norm_type: str = "layernorm",
        norm_groups: int = 8,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        pad = int(dilation)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            make_norm2d(channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            make_norm2d(channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class PyramidResDilatedEncoder(nn.Module):
    """
    Legacy pyramid encoder: residual 3x3 dilated conv stack.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 6,
        final_norm: bool = True,
        norm_type: str = "layernorm",
        norm_groups: int = 8,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            make_norm2d(hidden_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8]
        blocks = []
        for i in range(int(depth)):
            blocks.append(
                PyramidResDilatedBlock(
                    hidden_channels,
                    dilation=dilations[i % len(dilations)],
                    norm_type=norm_type,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.final_norm = (
            make_norm2d(latent_channels, norm_type=norm_type, norm_groups=norm_groups, norm_eps=norm_eps)
            if final_norm
            else nn.Identity()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x


class DilatedConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        pad = int(dilation) * (int(kernel_size) - 1) // 2
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
            padding_mode="reflect",
        )
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, int(expansion) * channels, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(int(expansion) * channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(float(layer_scale_init) * torch.ones(1, channels, 1, 1))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.gamma * x
        return residual + x


class PyramidConvNeXtDilatedEncoder(nn.Module):
    """
    Encoder for multiscale pyramid cubes using dilated ConvNeXt blocks.
    Expects BCHW with channels containing per-scale maps + mask-token maps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        latent_channels: int = 32,
        depth: int = 10,
        final_norm: bool = True,
        norm_type: str = "layernorm",
        norm_groups: int = 8,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        # keep signature compatibility with builder; this architecture uses LayerNorm2d internally
        _ = (norm_type, norm_groups, norm_eps)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            LayerNorm2d(hidden_channels),
        )
        dilations = [1, 2, 4, 8, 16]
        blocks = []
        for i in range(int(depth)):
            d = dilations[i % len(dilations)]
            blocks.append(
                DilatedConvNeXtBlock(
                    channels=hidden_channels,
                    dilation=d,
                    kernel_size=7,
                    expansion=4,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.final_norm = LayerNorm2d(latent_channels) if final_norm else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_norm(x)
        return x
