import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    def __init__(self, in_channels=None, proj_dim: int = 256, out_size=(224, 224)):
        super().__init__()
        if in_channels is None:
            in_channels = [96, 192, 384, 768]
        self.out_size = out_size

        self.projections = nn.ModuleList(
            [nn.Conv2d(c, proj_dim, kernel_size=1) for c in in_channels]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(proj_dim * 4, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(proj_dim, 1, kernel_size=1),
        )

    def forward(self, features):
        target_size = features[0].shape[2:]

        projected = []
        for i, feat in enumerate(features):
            p = self.projections[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)

        fused = torch.cat(projected, dim=1)
        mask_lowres = self.fusion(fused)
        return F.interpolate(mask_lowres, size=self.out_size, mode="bilinear", align_corners=False)
