from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm3d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3)


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, mlp_ratio: float = 4.0):
        super().__init__()
        pad = kernel_size // 2
        hidden = int(channels * mlp_ratio)
        self.dw = nn.Conv3d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.norm = LayerNorm3d(channels)
        self.pw1 = nn.Conv3d(channels, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, channels, 1)

        nn.init.zeros_(self.pw2.weight)
        if self.pw2.bias is not None:
            nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + y


class ScaleGateMixer3D(nn.Module):
    def __init__(self, channels: int, num_scales: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * num_scales, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, num_scales, 1),
        )

    def forward(self, x):
        b, s, c, d, h, w = x.shape
        flat = x.reshape(b, s * c, d, h, w)
        weights = torch.softmax(self.gate(flat), dim=1)
        y = (x * weights[:, :, None]).sum(dim=1)
        return y


class ScaleAwareConvNeXt3DEncoder(nn.Module):
    def __init__(
        self,
        num_scales: int,
        out_channels: int = 16,
        scale_channels: int = 8,
        depth: int = 3,
        kernel_size: int = 5,
        fusion: str = "gate",
        stride: int = 1,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.fusion = str(fusion)

        self.scale_stem = nn.Sequential(
            nn.Conv3d(2, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(scale_channels, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        if self.fusion == "gate":
            self.fuse = ScaleGateMixer3D(scale_channels, self.num_scales)
            fused_channels = scale_channels
        elif self.fusion == "concat":
            self.fuse = None
            fused_channels = scale_channels * self.num_scales
        else:
            raise ValueError(f"Unknown fusion={fusion}")

        self.proj = nn.Conv3d(fused_channels, out_channels, kernel_size=1)

        blocks = []
        if int(stride) > 1:
            blocks.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=int(stride), padding=1))
        for _ in range(int(depth)):
            blocks.append(ConvNeXtBlock3D(out_channels, kernel_size=int(kernel_size)))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, fields, mask_tokens=None):
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        b, s, d, h, w = fields.shape
        if s != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {s}")

        x = torch.stack([fields, mask_tokens], dim=2)
        x = x.reshape(b * s, 2, d, h, w)
        x = self.scale_stem(x)
        cs = x.shape[1]
        x = x.reshape(b, s, cs, d, h, w)

        if self.fusion == "gate":
            x = self.fuse(x)
        else:
            x = x.reshape(b, s * cs, d, h, w)

        x = self.proj(x)
        x = self.blocks(x)
        return x
