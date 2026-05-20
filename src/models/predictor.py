import torch.nn as nn


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: B,C,H,W -> normalize channel dimension per spatial location
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class FullResPredictor(nn.Module):
    def __init__(self, channels: int = 32, hidden: int = 64, use_layernorm: bool = False):
        super().__init__()
        norm1 = ChannelLayerNorm2d(hidden) if use_layernorm else nn.Identity()
        norm2 = ChannelLayerNorm2d(hidden) if use_layernorm else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            norm1,
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            norm2,
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)
