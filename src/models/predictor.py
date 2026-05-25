import torch.nn as nn

from .encoders import LayerNorm2d


class FullResPredictor(nn.Module):
    def __init__(self, channels: int = 32, hidden: int = 64, use_layernorm: bool = False):
        super().__init__()
        norm1 = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
        norm2 = LayerNorm2d(hidden) if use_layernorm else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            norm1,
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            norm2,
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)
