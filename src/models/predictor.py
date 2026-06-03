import torch.nn as nn

from .encoders import LayerNorm2d


class FullResPredictor(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        hidden: int = 64,
        use_layernorm: bool = False,
        kernel_size: int = 3,
        spatial_conv: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = bool(residual)
        if not spatial_conv:
            # Channel-only: 1x1 -> LayerNorm -> GELU -> 1x1, zero-init last conv, residual.
            norm = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
            self.net = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                norm,
                nn.GELU(),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )
            nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.net[-1].bias)
            return

        k = int(kernel_size)
        if k <= 0 or (k % 2) == 0:
            raise ValueError(f"FullResPredictor kernel_size must be a positive odd integer, got {kernel_size}")
        pad = k // 2
        norm1 = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
        norm2 = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
        mid_conv_kwargs = {"kernel_size": k, "padding": pad}
        if k > 1:
            mid_conv_kwargs["padding_mode"] = "reflect"
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            norm1,
            nn.GELU(),
            nn.Conv2d(hidden, hidden, **mid_conv_kwargs),
            norm2,
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y
