from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cdd_inspect import CDDOperatorFeatures2D
from .encoders import ConvNeXtDenseEncoder


def resolve_opnet_dilations(
    cdd_scales: tuple[float, ...],
    depth: int,
    mode: str = "half_cdd_scale",
    explicit_dilations=None,
    max_dilation: int = 16,
) -> tuple[int, ...]:
    depth = int(depth)
    max_dilation = int(max_dilation)
    if explicit_dilations is not None:
        dilations = [int(d) for d in explicit_dilations]
    else:
        mode = str(mode).lower()
        if mode == "none":
            dilations = [1] * depth
        elif mode == "half_cdd_scale":
            dilations = [
                max(1, min(max_dilation, int(round(float(s) / 2.0))))
                for s in cdd_scales
            ]
        elif mode == "cdd_scale":
            dilations = [
                max(1, min(max_dilation, int(round(float(s)))))
                for s in cdd_scales
            ]
        elif mode == "powers_of_two":
            dilations = [2**i for i in range(depth)]
        else:
            raise ValueError(
                f"Unknown opnet_dilation_mode={mode}. "
                "Use 'none', 'half_cdd_scale', 'cdd_scale', or 'powers_of_two'."
            )

    if len(dilations) < depth:
        reps = (depth + len(dilations) - 1) // len(dilations)
        dilations = (dilations * reps)[:depth]
    elif len(dilations) > depth:
        dilations = dilations[:depth]
    return tuple(int(d) for d in dilations)


class CDDOpNetEncoder(nn.Module):
    """
    CDD-OpNet encoder.

    Pipeline (per forward pass):
      1) log_cdd = log(clamp(cdd, min=0) + floor), floor from data std.
      2) gradmag(log_cdd) * scale
      3) lap(log_cdd) * scale^2 (signed, no abs)
      4) concat features -> ConvNeXt encoder

    Optional mask-token channels can be concatenated after feature construction.
    """

    def __init__(
        self,
        field_channels: int,
        scales: tuple[float, ...],
        latent_channels: int,
        hidden_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        use_reflect_padding: bool = True,
        final_norm: bool = True,
        include_mask_tokens: bool = True,
        log_eps: float = 1e-30,
        log_std_floor_mult: float = 0.05,
        op_smoothing_mode: str = "sqrt_scale",
        op_smoothing_mult: float = 1.0,
        op_smoothing_padding_mode: str = "reflect",
        opnet_dilation_mode: str = "half_cdd_scale",
        opnet_dilations=None,
        opnet_max_dilation: int = 16,
        cache_primitives: bool = True,
        cache_detach: bool = True,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.include_mask_tokens = bool(include_mask_tokens)
        self.log_eps = float(log_eps)
        self.log_std_floor_mult = float(log_std_floor_mult)
        self.op_smoothing_mode = str(op_smoothing_mode).lower()
        self.op_smoothing_mult = float(op_smoothing_mult)
        self.op_smoothing_padding_mode = str(op_smoothing_padding_mode).lower()
        if self.op_smoothing_mode not in {"none", "sqrt_scale", "scale"}:
            raise ValueError("op_smoothing_mode must be one of: 'none', 'sqrt_scale', 'scale'")
        if self.op_smoothing_padding_mode not in {"reflect", "replicate", "constant"}:
            raise ValueError("op_smoothing_padding_mode must be one of: 'reflect', 'replicate', 'constant'")
        self.cache_primitives = bool(cache_primitives)
        self.cache_detach = bool(cache_detach)

        scales = tuple(float(s) for s in scales)
        if len(scales) != self.field_channels:
            raise ValueError(
                f"CDDOpNetEncoder scales length must equal field_channels; "
                f"got len(scales)={len(scales)} field_channels={self.field_channels}"
            )
        self.register_buffer("scale_tensor", torch.tensor(scales, dtype=torch.float32).view(1, -1, 1, 1))
        self.opnet_dilations = resolve_opnet_dilations(
            cdd_scales=scales,
            depth=depth,
            mode=opnet_dilation_mode,
            explicit_dilations=opnet_dilations,
            max_dilation=opnet_max_dilation,
        )

        # We only need gradmag and signed lap from log_cdd.
        self.ops = CDDOperatorFeatures2D(
            features=("gradmag", "lap"),
            expect_3d_pyramid=False,
            apply_lognorm=False,
        )

        in_ch = 2 * self.field_channels
        if self.include_mask_tokens:
            in_ch += self.field_channels

        self.encoder = ConvNeXtDenseEncoder(
            in_channels=in_ch,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            depth=depth,
            kernel_size=kernel_size,
            expansion=expansion,
            use_reflect_padding=use_reflect_padding,
            final_norm=final_norm,
            dilations=self.opnet_dilations,
        )
        # Static-like inspection cache (updated each forward when enabled).
        self.last_log_cdd = None
        self.last_log_cdd_smooth = None
        self.last_grad_scaled = None
        self.last_lap_scaled = None
        self.last_primitives = None

    def _cache(self, name: str, tensor: torch.Tensor) -> None:
        if not self.cache_primitives:
            return
        value = tensor.detach() if self.cache_detach else tensor
        setattr(self, name, value)

    def get_cached_primitives(self) -> dict[str, torch.Tensor] | None:
        if self.last_primitives is None:
            return None
        return {
            "log_cdd": self.last_log_cdd,
            "log_cdd_smooth": self.last_log_cdd_smooth,
            "grad_scaled": self.last_grad_scaled,
            "lap_scaled": self.last_lap_scaled,
            "primitives": self.last_primitives,
        }

    def _log_cdd(self, field: torch.Tensor) -> torch.Tensor:
        eps = max(1e-30, self.log_eps)
        base = torch.clamp(field, min=0.0)
        base_std = torch.std(base, dim=(-2, -1), keepdim=True)
        floor = torch.clamp(base_std * self.log_std_floor_mult, min=eps)
        return torch.log(base + floor)

    def _gaussian_blur_single(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return x
        radius = max(1, int(round(3.0 * float(sigma))))
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=x.dtype, device=x.device) - radius
        g = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
        g = g / g.sum().clamp_min(1e-30)
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum().clamp_min(1e-30)
        weight = kernel.view(1, 1, size, size)
        x_pad = F.pad(x, (radius, radius, radius, radius), mode=self.op_smoothing_padding_mode)
        return F.conv2d(x_pad, weight, padding=0)

    def _smooth_for_operators(self, log_cdd: torch.Tensor) -> torch.Tensor:
        if self.op_smoothing_mode == "none":
            return log_cdd
        b, s, h, w = log_cdd.shape
        out = torch.empty_like(log_cdd)
        scales = self.scale_tensor.to(device=log_cdd.device, dtype=log_cdd.dtype).view(-1)
        for i in range(s):
            scale_i = float(scales[i].item())
            if self.op_smoothing_mode == "scale":
                sigma = self.op_smoothing_mult * scale_i
            else:
                sigma = self.op_smoothing_mult * (scale_i ** 0.5)
            out[:, i : i + 1] = self._gaussian_blur_single(log_cdd[:, i : i + 1], sigma=sigma)
        return out

    def forward(self, field: torch.Tensor, mask_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError(f"Expected field B,C,H,W, got {tuple(field.shape)}")
        if field.shape[1] != self.field_channels:
            raise ValueError(f"Expected {self.field_channels} field channels, got {field.shape[1]}")

        log_cdd = self._log_cdd(field)
        log_cdd_for_ops = self._smooth_for_operators(log_cdd)
        attrs = self.ops(log_cdd_for_ops)

        scale_tensor = self.scale_tensor.to(device=field.device, dtype=field.dtype)
        grad_scaled = attrs["gradmag"] * scale_tensor
        lap_scaled = attrs["lap"] * (scale_tensor * scale_tensor)
        primitives = torch.cat([grad_scaled, lap_scaled], dim=1)
        x = primitives
        self._cache("last_log_cdd", log_cdd)
        self._cache("last_log_cdd_smooth", log_cdd_for_ops)
        self._cache("last_grad_scaled", grad_scaled)
        self._cache("last_lap_scaled", lap_scaled)
        self._cache("last_primitives", primitives)

        if self.include_mask_tokens:
            if mask_tokens is None:
                mask_tokens = torch.zeros_like(field)
            if mask_tokens.shape != field.shape:
                raise ValueError(
                    f"mask_tokens shape must match field shape. "
                    f"field={tuple(field.shape)} mask={tuple(mask_tokens.shape)}"
                )
            x = torch.cat([x, mask_tokens], dim=1)

        return self.encoder(x)
