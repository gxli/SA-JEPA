import torch.nn as nn

from .encoders import LayerNorm2d


class FullResPredictor(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        hidden: int = 64,
        use_layernorm: bool = False,
        kernel_size: int = 3,
    ):
        super().__init__()
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
        return self.net(x)
