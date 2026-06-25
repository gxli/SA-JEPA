from __future__ import annotations

import torch
import torch.nn as nn


def _depth_window_ranges(full_depth: int, out_start: int, out_depth: int, radii: list[int]) -> list[tuple[int, int]]:
    out_lo = int(out_start)
    out_hi = int(out_start) + int(out_depth) - 1
    ranges: list[tuple[int, int]] = []
    remaining = sum(int(r) for r in radii)
    for radius in radii:
        remaining -= int(radius)
        lo = max(0, out_lo - remaining)
        hi = min(int(full_depth) - 1, out_hi + remaining)
        ranges.append((lo, hi))
    return ranges


def _crop_depth_global(x: torch.Tensor, current_offset: int, lo: int, hi: int) -> tuple[torch.Tensor, int]:
    local_lo = max(0, int(lo) - int(current_offset))
    local_hi = min(int(x.shape[-3]) - 1, int(hi) - int(current_offset))
    if local_hi < local_lo:
        raise RuntimeError(
            f"Invalid depth crop lo={lo} hi={hi} offset={current_offset} tensor_depth={x.shape[-3]}"
        )
    return x[..., local_lo : local_hi + 1, :, :], int(lo)


# ---------------------------------------------------------------------------
# Norm layers
# ---------------------------------------------------------------------------

class LayerNorm3d(nn.Module):
    """LayerNorm over channels for B,C,D,H,W tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # B,D,H,W,C
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3)  # B,C,D,H,W


class GRN3D(nn.Module):
    """Global Response Normalization for 3D (ConvNeXt V2 style).

    Operates in channels-last layout (B,D,H,W,C).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, int(dim)))
        self.eps = float(eps)

    def forward(self, x):
        # x: B,D,H,W,C
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ---------------------------------------------------------------------------
# ConvNeXt3D Block
# ---------------------------------------------------------------------------

class ConvNeXtBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        mlp_ratio: float = 4.0,
        use_grn: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2
        hidden = int(channels * mlp_ratio)
        self.use_grn = bool(use_grn)

        self.dw = nn.Conv3d(channels, channels, kernel_size, padding=pad, groups=channels, padding_mode="replicate")
        self.norm = LayerNorm3d(channels)
        self.pw1 = nn.Conv3d(channels, hidden, 1)
        self.act = nn.GELU()
        # GRN operates in channels-last: B,D,H,W,C
        self.grn = GRN3D(hidden) if self.use_grn else nn.Identity()
        self.pw2 = nn.Conv3d(hidden, channels, 1)

        nn.init.zeros_(self.pw2.weight)
        if self.pw2.bias is not None:
            nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        y = self.dw(x)                     # B,C,D,H,W
        y = self.norm(y)                   # LayerNorm3d (permute → norm → permute)
        y = self.pw1(y)
        y = self.act(y)
        y = y.permute(0, 2, 3, 4, 1)      # B,D,H,W,C for GRN
        y = self.grn(y)
        y = y.permute(0, 4, 1, 2, 3)      # B,C,D,H,W
        y = self.pw2(y)
        return x + y

    def forward_depth_window(self, x: torch.Tensor, current_offset: int, out_lo: int, out_hi: int):
        y = self.dw(x)
        y, next_offset = _crop_depth_global(y, current_offset, out_lo, out_hi)
        residual, _ = _crop_depth_global(x, current_offset, out_lo, out_hi)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = y.permute(0, 2, 3, 4, 1)
        y = self.grn(y)
        y = y.permute(0, 4, 1, 2, 3)
        y = self.pw2(y)
        return residual + y, next_offset


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ScaleAware ConvNeXt3D Encoder  (WITH norm flags — mirrors 2D)
# ---------------------------------------------------------------------------

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
        *,
        use_grn: bool = True,
        stem_norm: bool = True,
        norm_per_scale: bool = True,
        adapter_norm: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.fusion = str(fusion)
        self.norm_per_scale = bool(norm_per_scale)
        self.adapter_norm = bool(adapter_norm)

        # Stem: [field, mask_token] → scale_channels
        stem_layers = [
            nn.Conv3d(2, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(scale_channels, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if stem_norm:
            stem_layers.append(LayerNorm3d(scale_channels))
        self.scale_stem = nn.Sequential(*stem_layers)
        self.per_scale_norm = LayerNorm3d(scale_channels) if self.norm_per_scale else nn.Identity()

        # Fusion
        if self.fusion == "gate":
            self.fuse = ScaleGateMixer3D(scale_channels, self.num_scales)
            fused_channels = scale_channels
        elif self.fusion == "concat":
            self.fuse = None
            fused_channels = scale_channels * self.num_scales
        else:
            raise ValueError(f"Unknown fusion={fusion}")

        # Proj
        self.proj = nn.Conv3d(fused_channels, out_channels, kernel_size=1)

        # Blocks
        blocks = []
        for _ in range(int(depth)):
            blocks.append(ConvNeXtBlock3D(out_channels, kernel_size=int(kernel_size), use_grn=use_grn))
        self.blocks = nn.Sequential(*blocks)
        self._depth_radii = [1, 1] + [int(kernel_size) // 2 for _ in range(int(depth))]

        # Final norm
        self.final_norm = LayerNorm3d(out_channels) if final_norm else nn.Identity()

    def forward(self, fields, mask_tokens=None):
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        b, s, d, h, w = fields.shape
        if s != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {s}")

        x = torch.stack([fields, mask_tokens], dim=2)  # B, S, 2, D, H, W
        x = x.reshape(b * s, 2, d, h, w)
        x = self.scale_stem(x)
        cs = x.shape[1]
        x = x.reshape(b, s, cs, d, h, w)  # B, S, C, D, H, W

        # Per-scale norm before fusion
        if self.norm_per_scale:
            x = self.per_scale_norm(x.reshape(b * s, cs, d, h, w)).reshape(b, s, cs, d, h, w)

        # Fusion
        if self.fusion == "gate":
            x = self.fuse(x)
        else:
            x = x.reshape(b, s * cs, d, h, w)

        x = self.proj(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return x

    def forward_depth_window(self, fields, mask_tokens=None, out_start: int = 0, out_depth: int | None = None):
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        b, s, d, h, w = fields.shape
        if s != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {s}")
        out_depth = int(out_depth if out_depth is not None else d)
        ranges = _depth_window_ranges(d, int(out_start), out_depth, self._depth_radii)
        range_i = 0
        offset = 0

        x = torch.stack([fields, mask_tokens], dim=2).reshape(b * s, 2, d, h, w)
        for layer in self.scale_stem:
            x = layer(x)
            if isinstance(layer, nn.Conv3d) and int(layer.kernel_size[0]) > 1:
                lo, hi = ranges[range_i]
                x, offset = _crop_depth_global(x, offset, lo, hi)
                range_i += 1
        cs = x.shape[1]
        cur_d = x.shape[-3]
        x = x.reshape(b, s, cs, cur_d, h, w)

        if self.norm_per_scale:
            x = self.per_scale_norm(x.reshape(b * s, cs, cur_d, h, w)).reshape(b, s, cs, cur_d, h, w)

        if self.fusion == "gate":
            x = self.fuse(x)
        else:
            x = x.reshape(b, s * cs, cur_d, h, w)

        x = self.proj(x)
        for block in self.blocks:
            lo, hi = ranges[range_i]
            x, offset = block.forward_depth_window(x, offset, lo, hi)
            range_i += 1
        x = self.final_norm(x)
        return x


# ---------------------------------------------------------------------------
# FiLM + Shared ConvNeXt3D Encoder  (WITH norm flags)
# ---------------------------------------------------------------------------

class ScaleFiLM3d(nn.Module):
    """Per-scale affine modulation for B,S,C,D,H,W tensors."""

    def __init__(self, num_scales: int, channels: int):
        super().__init__()
        self.emb = nn.Embedding(int(num_scales), int(channels) * 2)
        nn.init.zeros_(self.emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, D, H, W = x.shape
        idx = torch.arange(S, device=x.device)
        gb = self.emb(idx)
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
        B, S, C, D, H, W = x.shape
        x = self.film(x)
        y = x.reshape(B * S, C, D, H, W)
        y = self.block(y)
        y = y.reshape(B, S, C, D, H, W)
        return y

    def forward_depth_window(self, x: torch.Tensor, current_offset: int, out_lo: int, out_hi: int):
        B, S, C, D, H, W = x.shape
        x = self.film(x)
        y = x.reshape(B * S, C, D, H, W)
        y, next_offset = self.block.forward_depth_window(y, current_offset, out_lo, out_hi)
        y = y.reshape(B, S, C, y.shape[-3], H, W)
        return y, next_offset


class PerScaleAdapter3d(nn.Module):
    """Tiny per-scale 1x1x1 conv adapter (residual, zero-init)."""

    def __init__(self, num_scales: int, channels: int, use_norm: bool = True):
        super().__init__()
        self.use_norm = bool(use_norm)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 1),
                nn.GELU(),
                nn.Conv3d(channels, channels, 1),
            )
            for _ in range(int(num_scales))
        ])
        self.norms = nn.ModuleList([LayerNorm3d(channels) for _ in range(int(num_scales))])
        for a in self.adapters:
            nn.init.zeros_(a[-1].weight)
            if a[-1].bias is not None:
                nn.init.zeros_(a[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, D, H, W = x.shape
        outs = []
        for s in range(S):
            xs = x[:, s]
            adapt = self.adapters[s](xs)
            if self.use_norm:
                adapt = self.norms[s](adapt)
            outs.append(xs + adapt)
        return torch.stack(outs, dim=1)


class ScaleFiLMConvNeXt3DEncoder(nn.Module):
    """3D scale-aware encoder: shared ConvNeXt3D blocks + scale FiLM + norms.

    Input:
      fields:      B x S x D x H x W
      mask_tokens: B x S x D x H x W

    Norm flags mirror the 2D CDDScaleAwareConvNeXtEncoder:
      stem_norm, norm_per_scale, adapter_norm, final_norm, use_grn
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
        *,
        use_grn: bool = True,
        stem_norm: bool = True,
        norm_per_scale: bool = True,
        adapter_norm: bool = True,
        final_norm: bool = True,
    ):
        super().__init__()
        self.num_scales = int(num_scales)
        self.fusion = str(fusion)
        self.norm_per_scale = bool(norm_per_scale)

        # Per-scale stem  [field, mask] → scale_channels
        stem_layers = [
            nn.Conv3d(2, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(scale_channels, scale_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if stem_norm:
            stem_layers.append(LayerNorm3d(scale_channels))
        self.scale_stem = nn.Sequential(*stem_layers)
        self.per_scale_norm = LayerNorm3d(scale_channels) if self.norm_per_scale else nn.Identity()

        # Shared ConvNeXt3D blocks with FiLM
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            blk = ConvNeXtBlock3D(scale_channels, kernel_size=int(kernel_size), use_grn=use_grn)
            stage = SharedScaleConvNeXtStage3d(
                block=blk,
                num_scales=self.num_scales,
                channels=scale_channels,
                use_film=use_film,
            )
            self.blocks.append(stage)
            if use_per_scale_adapters:
                self.blocks.append(
                    PerScaleAdapter3d(self.num_scales, scale_channels, use_norm=adapter_norm)
                )

        self._depth_radii = [1, 1] + [int(kernel_size) // 2 for _ in range(int(depth))]

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
        self.head = nn.Conv3d(fused_channels, out_channels, kernel_size=1)

        # Final norm
        self.final_norm = LayerNorm3d(out_channels) if final_norm else nn.Identity()

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

        # Per-scale norm
        if self.norm_per_scale:
            x = self.per_scale_norm(x.reshape(b * s, cs, d, h, w)).reshape(b, s, cs, d, h, w)

        # Shared ConvNeXt3D blocks + adapters
        for blk in self.blocks:
            x = blk(x)

        # Fusion
        if self.fusion == "gate":
            x = self.fuse(x)  # B, C, D, H, W
        else:
            x = x.reshape(b, s * cs, d, h, w)

        # Head + final norm
        x = self.head(x)
        x = self.final_norm(x)
        return x

    def forward_depth_window(self, fields, mask_tokens=None, out_start: int = 0, out_depth: int | None = None):
        if mask_tokens is None:
            mask_tokens = torch.zeros_like(fields)

        b, s, d, h, w = fields.shape
        if s != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {s}")
        out_depth = int(out_depth if out_depth is not None else d)
        ranges = _depth_window_ranges(d, int(out_start), out_depth, self._depth_radii)
        range_i = 0
        offset = 0

        x = torch.stack([fields, mask_tokens], dim=2).reshape(b * s, 2, d, h, w)
        for layer in self.scale_stem:
            x = layer(x)
            if isinstance(layer, nn.Conv3d) and int(layer.kernel_size[0]) > 1:
                lo, hi = ranges[range_i]
                x, offset = _crop_depth_global(x, offset, lo, hi)
                range_i += 1
        cs = x.shape[1]
        cur_d = x.shape[-3]
        x = x.reshape(b, s, cs, cur_d, h, w)

        if self.norm_per_scale:
            x = self.per_scale_norm(x.reshape(b * s, cs, cur_d, h, w)).reshape(b, s, cs, cur_d, h, w)

        for blk in self.blocks:
            if isinstance(blk, SharedScaleConvNeXtStage3d):
                lo, hi = ranges[range_i]
                x, offset = blk.forward_depth_window(x, offset, lo, hi)
                range_i += 1
            else:
                x = blk(x)

        if self.fusion == "gate":
            x = self.fuse(x)
        else:
            x = x.reshape(b, s * cs, x.shape[-3], h, w)

        x = self.head(x)
        x = self.final_norm(x)
        return x
