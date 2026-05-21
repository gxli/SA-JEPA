from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import ConvNeXtDenseEncoder
from .mfae import MFAE2D


class MFAEConvNeXtDenseEncoder(nn.Module):
    """
    MFAE-ConvNeXt: Multiscale Field Attribute Encoding ConvNeXt.
    """

    def __init__(
        self,
        field_channels: int,
        latent_channels: int,
        hidden_channels: int = 32,
        depth: int = 3,
        kernel_size: int = 5,
        expansion: int = 4,
        use_reflect_padding: bool = True,
        final_norm: bool = True,
        mfae_scales=(1, 2, 4),
        mfae_features=("x", "gradmag", "abslap", "local_std"),
        mfae_normalize_attributes: bool = False,
        include_mask_tokens: bool = False,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.include_mask_tokens = bool(include_mask_tokens)

        self.mfae = MFAE2D(
            scales=mfae_scales,
            features=mfae_features,
            normalize_attributes=mfae_normalize_attributes,
        )

        in_ch = self.mfae.out_channels(self.field_channels)
        if self.include_mask_tokens:
            in_ch += self.field_channels

        self.convnext = ConvNeXtDenseEncoder(
            in_channels=in_ch,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            depth=depth,
            kernel_size=kernel_size,
            expansion=expansion,
            use_reflect_padding=use_reflect_padding,
            final_norm=final_norm,
        )

    def forward(self, field: torch.Tensor, mask_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError(f"Expected field B,C,H,W, got {tuple(field.shape)}")
        if field.shape[1] != self.field_channels:
            raise ValueError(f"Expected {self.field_channels} field channels, got {field.shape[1]}")

        x = self.mfae(field)
        if self.include_mask_tokens:
            if mask_tokens is None:
                mask_tokens = torch.zeros_like(field)
            if mask_tokens.shape != field.shape:
                raise ValueError(
                    f"mask_tokens shape must match field shape. "
                    f"field={tuple(field.shape)} mask={tuple(mask_tokens.shape)}"
                )
            x = torch.cat([x, mask_tokens], dim=1)

        return self.convnext(x)
