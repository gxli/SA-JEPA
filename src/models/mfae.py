from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFAE2D(nn.Module):
    """
    Multiscale Field Attribute Encoding for 2D fields.
    """

    def __init__(
        self,
        scales=(1, 2, 4),
        features=("x", "gradmag", "abslap", "local_std"),
        eps: float = 1e-6,
        padding_mode: str = "reflect",
        normalize_attributes: bool = False,
    ):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)
        self.features = tuple(str(f) for f in features)
        self.eps = float(eps)
        self.padding_mode = str(padding_mode)
        self.normalize_attributes = bool(normalize_attributes)

        allowed = {"x", "gradmag", "abslap", "local_mean", "local_std"}
        unknown = set(self.features) - allowed
        if unknown:
            raise ValueError(f"Unknown MFAE2D features: {sorted(unknown)}")

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
        self.register_buffer("lap", lap[None, None])

    @property
    def multiplier(self) -> int:
        return len(self.scales) * len(self.features)

    def out_channels(self, in_channels: int) -> int:
        return int(in_channels) * self.multiplier

    def _pad2d(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return x
        return F.pad(x, (pad, pad, pad, pad), mode=self.padding_mode)

    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        w = kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        x_pad = self._pad2d(x, 1)
        return F.conv2d(x_pad, w, groups=c)

    def _smooth_for_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale <= 1:
            return x
        k = 2 * int(scale) + 1
        pad = k // 2
        return F.avg_pool2d(self._pad2d(x, pad), kernel_size=k, stride=1)

    def _local_mean_std(self, x: torch.Tensor, k: int):
        if k <= 1:
            return x, torch.zeros_like(x)
        pad = k // 2
        mean = F.avg_pool2d(self._pad2d(x, pad), kernel_size=k, stride=1)
        mean2 = F.avg_pool2d(self._pad2d(x * x, pad), kernel_size=k, stride=1)
        var = torch.clamp(mean2 - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        return mean, std

    def _normalize_per_channel(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sd = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - mu) / sd.clamp_min(self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"MFAE2D expects B,C,H,W input, got {tuple(x.shape)}")

        outs = []
        for s in self.scales:
            xs = self._smooth_for_scale(x, s)

            gx = self._depthwise_filter(xs, self.kx)
            gy = self._depthwise_filter(xs, self.ky)
            gradmag = torch.sqrt(gx * gx + gy * gy + self.eps)
            abslap = self._depthwise_filter(xs, self.lap).abs()

            local_k = 2 * int(s) + 1
            local_mean, local_std = self._local_mean_std(xs, local_k)
            fmap = {
                "x": xs,
                "gradmag": gradmag,
                "abslap": abslap,
                "local_mean": local_mean,
                "local_std": local_std,
            }

            for name in self.features:
                y = fmap[name]
                if self.normalize_attributes:
                    y = self._normalize_per_channel(y)
                outs.append(y)

        return torch.cat(outs, dim=1)
