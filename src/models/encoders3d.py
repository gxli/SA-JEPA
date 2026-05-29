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


# ---------------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation) – 3D scale-conditioned shared ConvNeXt
# ---------------------------------------------------------------------------


class ScaleFiLM3d(nn.Module):
    """Per-scale affine modulation for B,S,C,D,H,W tensors.

    Learns (gamma, beta) per scale via a tiny embedding, initialized to zero.
    """

    def __init__(self, num_scales: int, channels: int):
        super().__init__()
        self.emb = nn.Embedding(int(num_scales), int(channels) * 2)
        nn.init.zeros_(self.emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B,S,C,D,H,W  ->  B,S,C,D,H,W"""
        B, S, C, D, H, W = x.shape
        idx = torch.arange(S, device=x.device)
        gb = self.emb(idx)  # S, 2C
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.view(1, S, C, 1, 1, 1)
        beta = beta.view(1, S, C, 1, 1, 1)
        return x * (1.0 + gamma) + beta


class SharedScaleConvNeXtStage3d(nn.Module):
    """Apply a single shared ConvNeXt3D block to every scale, with optional FiLM."""

    def __init__(self, block: nn.Module, num_scales: int, channels: int, use_film: bool = True):
        super().__init__()
        self.block = block
        self.film = ScaleFiLM3d(num_scales, channels) if use_film else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B,S,C,D,H,W  ->  B,S,C,D,H,W"""
        B, S, C, D, H, W = x.shape
        x = self.film(x)
        y = x.reshape(B * S, C, D, H, W)
        y = self.block(y)
        y = y.reshape(B, S, C, D, H, W)
        return y


class PerScaleAdapter3d(nn.Module):
    """Tiny per-scale 1x1x1 conv adapter (residual, zero-init)."""

    def __init__(self, num_scales: int, channels: int):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 1),
                nn.GELU(),
                nn.Conv3d(channels, channels, 1),
            )
            for _ in range(int(num_scales))
        ])
        for a in self.adapters:
            nn.init.zeros_(a[-1].weight)
            if a[-1].bias is not None:
                nn.init.zeros_(a[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B,S,C,D,H,W  ->  B,S,C,D,H,W"""
        B, S, C, D, H, W = x.shape
        outs = []
        for s in range(S):
            xs = x[:, s]
            outs.append(xs + self.adapters[s](xs))
        return torch.stack(outs, dim=1)


class ScaleFiLMConvNeXt3DEncoder(nn.Module):
    """3D scale-aware encoder: shared ConvNeXt3D blocks + scale FiLM.

    Input:
      fields:      B x S x D x H x W
      mask_tokens: B x S x D x H x W

    Pipeline:
      1. Per-scale stem  ->  B, S, scale_channels, D, H, W
      2. Stack of SharedScaleConvNeXtStage3d blocks (shared weights + FiLM)
      3. Optional PerScaleAdapter3d after each shared block
      4. Fusion (gate or concat)  ->  B, fused, D, H, W
      5. Proj + optional stride
    """

    def __init__(
        self,
        num_scales: int,
        out_channels: int = 16,
        scale_channels: int = 8,
        depth: int = 3,
        kernel_size: int = 5,
        fusion: str = "gate",
        use_film: bool = True,
        use_per_scale_adapters: bool = False,
        stride: int = 1,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.fusion = str(fusion)

        # Per-scale stem (applied independently to each scale's [field, mask])
        self.scale_stem = nn.Sequential(
            nn.Conv3d(2, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(scale_channels, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Shared ConvNeXt3D blocks with FiLM
        self.use_film = bool(use_film)
        self.use_per_scale_adapters = bool(use_per_scale_adapters)
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            blk = ConvNeXtBlock3D(scale_channels, kernel_size=int(kernel_size))
            stage = SharedScaleConvNeXtStage3d(
                block=blk,
                num_scales=self.num_scales,
                channels=scale_channels,
                use_film=self.use_film,
            )
            self.blocks.append(stage)
            if self.use_per_scale_adapters:
                self.blocks.append(
                    PerScaleAdapter3d(self.num_scales, scale_channels)
                )

        # Fusion
        if self.fusion == "gate":
            self.fuse = ScaleGateMixer3D(scale_channels, self.num_scales)
            fused_channels = scale_channels
        elif self.fusion == "concat":
            self.fuse = None
            fused_channels = scale_channels * self.num_scales
        else:
            raise ValueError(f"Unknown fusion={fusion}")

        # Head
        head_layers = []
        if int(stride) > 1:
            head_layers.append(
                nn.Conv3d(fused_channels, out_channels, kernel_size=3, stride=int(stride), padding=1)
            )
        else:
            head_layers.append(nn.Conv3d(fused_channels, out_channels, kernel_size=1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, fields, mask_tokens=None):
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        b, s, d, h, w = fields.shape
        if s != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {s}")

        # Per-scale stem
        x = torch.stack([fields, mask_tokens], dim=2)  # B, S, 2, D, H, W
        x = x.reshape(b * s, 2, d, h, w)
        x = self.scale_stem(x)
        cs = x.shape[1]
        x = x.reshape(b, s, cs, d, h, w)  # B, S, C, D, H, W

        # Shared ConvNeXt3D blocks with FiLM
        for blk in self.blocks:
            x = blk(x)

        # Fusion
        if self.fusion == "gate":
            x = self.fuse(x)  # B, C, D, H, W
        else:
            x = x.reshape(b, s * cs, d, h, w)  # B, S*C, D, H, W

        # Head
        x = self.head(x)
        return x
