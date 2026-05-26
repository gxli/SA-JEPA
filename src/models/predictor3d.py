from __future__ import annotations

import torch.nn as nn


class FullResPredictor3D(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
        )

    def forward(self, x):
        return self.net(x)
