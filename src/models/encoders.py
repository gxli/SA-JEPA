import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FullResEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(latent_channels, dilation=1),
            ResidualBlock(latent_channels, dilation=1),
            ResidualBlock(latent_channels, dilation=2),
            ResidualBlock(latent_channels, dilation=2),
            ResidualBlock(latent_channels, dilation=4),
            ResidualBlock(latent_channels, dilation=4),
            ResidualBlock(latent_channels, dilation=8),
            ResidualBlock(latent_channels, dilation=8),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x
