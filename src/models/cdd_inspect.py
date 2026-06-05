from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDDOperatorFeatures2D(nn.Module):
    """
    Pure tensor transform for CDD/operator attributes.

    Supports:
    - 2D pyramid input: B,S,H,W
    - 3D pyramid input: B,S,D,H,W (default expected)

    Returns:
        {
            "x": cdd,
            "grad_x": gx,
            "grad_y": gy,
            "gradmag": gradmag,
            "lap": lap,
            "abslap": abslap,
            "local_mean": local_mean,
            "local_std": local_std,
            "stack": concatenated features along channel/scale axis
        }

    No plotting, no saving, no file I/O.
    """

    def __init__(
        self,
        features: Iterable[str] = ("x", "gradmag", "abslap", "local_std"),
        local_std_kernel: int = 7,
        eps: float = 1e-6,
        padding_mode: str = "reflect",
        normalize_stack: bool = False,
        expect_3d_pyramid: bool = True,
        apply_lognorm: bool = False,
        log_eps: float = 1e-30,
        log_std_floor_mult: float = 0.05,
        lognorm_mode: str = "auto",
        unified_lognorm: bool = False,
        lognorm_on_stack: bool = False,
    ):
        super().__init__()

        self.features = tuple(features)
        self.local_std_kernel = int(local_std_kernel)
        self.eps = float(eps)
        self.padding_mode = str(padding_mode)
        self.normalize_stack = bool(normalize_stack)
        self.expect_3d_pyramid = bool(expect_3d_pyramid)
        self.apply_lognorm = bool(apply_lognorm)
        self.log_eps = float(log_eps)
        self.log_std_floor_mult = float(log_std_floor_mult)
        self.lognorm_mode = str(lognorm_mode).lower()
        self.unified_lognorm = bool(unified_lognorm)
        self.lognorm_on_stack = bool(lognorm_on_stack)
        if self.lognorm_mode not in {"auto", "positive", "signed"}:
            raise ValueError("lognorm_mode must be one of: 'auto', 'positive', 'signed'.")

        if self.local_std_kernel % 2 == 0:
            raise ValueError("local_std_kernel must be odd.")

        allowed = {
            "x",
            "grad_x",
            "grad_y",
            "gradmag",
            "lap",
            "abslap",
            "local_mean",
            "local_std",
        }
        unknown = set(self.features) - allowed
        if unknown:
            raise ValueError(f"Unknown operator features: {sorted(unknown)}")

        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ) / 8.0
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ) / 8.0
        lap = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32,
        )

        self.register_buffer("kx", kx[None, None])
        self.register_buffer("ky", ky[None, None])
        self.register_buffer("lap_kernel", lap[None, None])

    @property
    def multiplier(self) -> int:
        return len(self.features)

    def out_channels(self, cdd_channels: int) -> int:
        return int(cdd_channels) * self.multiplier

    def _pad(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return x
        return F.pad(x, (pad, pad, pad, pad), mode=self.padding_mode)

    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        w = kernel.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        x_pad = self._pad(x, 1)
        return F.conv2d(x_pad, w, groups=c)

    def _local_mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.local_std_kernel
        pad = k // 2

        mean = F.avg_pool2d(self._pad(x, pad), kernel_size=k, stride=1)
        mean2 = F.avg_pool2d(self._pad(x * x, pad), kernel_size=k, stride=1)

        var = torch.clamp(mean2 - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        return mean, std

    def _normalize_per_channel(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sd = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        structural_eps = max(self.eps, 0.01)
        return (x - mu) / sd.clamp_min(structural_eps)

    def _lognorm_positive(self, x: torch.Tensor) -> torch.Tensor:
        eps = max(1e-30, float(self.log_eps))
        base = torch.clamp(x, min=0.0)
        base_std = torch.std(base, dim=(-2, -1), keepdim=True)
        log_floor = torch.clamp(base_std * float(self.log_std_floor_mult), min=eps)
        return torch.log(base + log_floor)

    def _lognorm_signed(self, x: torch.Tensor) -> torch.Tensor:
        eps = max(1e-30, float(self.log_eps))
        base = torch.abs(x)
        base_std = torch.std(base, dim=(-2, -1), keepdim=True)
        structural_std_floor = torch.clamp(base_std, min=1e-3)
        log_floor = torch.clamp(structural_std_floor * float(self.log_std_floor_mult), min=eps)
        # log1p(base / floor) is always >= 0, preserving monotonicity and sign.
        # log(base + floor) would flip sign when base + floor < 1.
        return torch.sign(x) * torch.log1p(base / log_floor)

    def _lognorm_positive_with_floor(self, x: torch.Tensor, log_floor: torch.Tensor) -> torch.Tensor:
        base = torch.clamp(x, min=0.0)
        return torch.log(base + log_floor)

    def _lognorm_signed_with_floor(self, x: torch.Tensor, log_floor: torch.Tensor) -> torch.Tensor:
        base = torch.abs(x)
        return torch.sign(x) * torch.log1p(base / log_floor)

    def _compute_unified_floor(self, maps: dict[str, torch.Tensor]) -> torch.Tensor:
        eps = max(1e-30, float(self.log_eps))
        # Use non-negative base from x, same spirit as the main CDD path.
        base = torch.clamp(maps["x"], min=0.0)
        base_std = torch.std(base, dim=(-2, -1), keepdim=True)
        structural_std_floor = torch.clamp(base_std, min=1e-3)
        return torch.clamp(structural_std_floor * float(self.log_std_floor_mult), min=eps)

    def _compute_stack_floor(self, stack: torch.Tensor) -> torch.Tensor:
        eps = max(1e-30, float(self.log_eps))
        base = torch.abs(stack)
        base_std = torch.std(base, dim=(-2, -1), keepdim=True)
        structural_std_floor = torch.clamp(base_std, min=1e-3)
        return torch.clamp(structural_std_floor * float(self.log_std_floor_mult), min=eps)

    def _apply_lognorm_maps(self, maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        signed_keys = {"grad_x", "grad_y", "lap"}
        shared_floor = self._compute_unified_floor(maps) if self.unified_lognorm else None
        out = {}
        for key, value in maps.items():
            if self.lognorm_mode == "positive":
                if shared_floor is None:
                    out[key] = self._lognorm_positive(value)
                else:
                    out[key] = self._lognorm_positive_with_floor(value, shared_floor)
            elif self.lognorm_mode == "signed":
                if shared_floor is None:
                    out[key] = self._lognorm_signed(value)
                else:
                    out[key] = self._lognorm_signed_with_floor(value, shared_floor)
            else:
                if key in signed_keys:
                    if shared_floor is None:
                        out[key] = self._lognorm_signed(value)
                    else:
                        out[key] = self._lognorm_signed_with_floor(value, shared_floor)
                else:
                    if shared_floor is None:
                        out[key] = self._lognorm_positive(value)
                    else:
                        out[key] = self._lognorm_positive_with_floor(value, shared_floor)
        return out

    def _flatten_spatial2d(self, cdd: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if cdd.ndim == 4:
            return cdd, tuple(cdd.shape)
        if cdd.ndim == 5:
            b, s, d, h, w = cdd.shape
            return cdd.permute(0, 2, 1, 3, 4).reshape(b * d, s, h, w), tuple(cdd.shape)
        raise ValueError(
            f"CDDOperatorFeatures2D expects B,S,H,W or B,S,D,H,W; got {tuple(cdd.shape)}"
        )

    def _restore_shape(self, x: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
        if len(original_shape) == 4:
            return x
        b, _, d, h, w = original_shape
        c = x.shape[1]
        return x.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, cdd: torch.Tensor) -> dict[str, torch.Tensor]:
        cdd2d, original_shape = self._flatten_spatial2d(cdd)
        if self.expect_3d_pyramid and len(original_shape) != 5:
            raise ValueError(
                f"expect_3d_pyramid=True expects B,S,D,H,W input; got {tuple(original_shape)}"
            )

        cdd2d = torch.nan_to_num(cdd2d, nan=0.0, posinf=0.0, neginf=0.0)

        gx = self._depthwise_filter(cdd2d, self.kx)
        gy = self._depthwise_filter(cdd2d, self.ky)
        gradmag = torch.sqrt(gx * gx + gy * gy + self.eps)

        lap = self._depthwise_filter(cdd2d, self.lap_kernel)
        abslap = lap.abs()

        local_mean, local_std = self._local_mean_std(cdd2d)

        all_maps_2d = {
            "x": cdd2d,
            "grad_x": gx,
            "grad_y": gy,
            "gradmag": gradmag,
            "lap": lap,
            "abslap": abslap,
            "local_mean": local_mean,
            "local_std": local_std,
        }
        if self.apply_lognorm:
            all_maps_2d = self._apply_lognorm_maps(all_maps_2d)

        stack_parts = [all_maps_2d[name] for name in self.features]
        stack = torch.cat(stack_parts, dim=1)
        if self.apply_lognorm and self.lognorm_on_stack:
            # Apply lognorm directly to network input stack with a data-based floor.
            stack_floor = self._compute_stack_floor(stack)
            stack = torch.sign(stack) * torch.log1p(torch.abs(stack) / stack_floor)
        if self.normalize_stack:
            stack = self._normalize_per_channel(stack)

        out = {k: self._restore_shape(v, original_shape) for k, v in all_maps_2d.items()}
        out["stack"] = self._restore_shape(stack, original_shape)
        return out
