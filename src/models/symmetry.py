from __future__ import annotations

import itertools

import torch
import torch.nn as nn


def _pad_to_square(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims (H,W) to square max(H,W). Returns (padded, (orig_h, orig_w))."""
    B, C, H, W = x.shape
    if H == W:
        return x, (H, W)
    S = max(H, W)
    pad_h = S - H
    pad_w = S - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return torch.nn.functional.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0,
    ), (H, W)


def _crop_from_square(x: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    """Crop back from square to original (H, W)."""
    H, W = orig_shape
    cur_h, cur_w = x.shape[-2:]
    top = max(0, (cur_h - H) // 2)
    left = max(0, (cur_w - W) // 2)
    return x[..., top:top + H, left:left + W]


def symmetric_forward_2d(encoder: nn.Module, x: torch.Tensor, return_var: bool = False, **kwargs):
    """
    Four-way rotational group average for dense 2D spatial features.

    Single batched forward pass: pads to square, stacks all 4 rotations into
    batch dim, runs the encoder once (CuDNN-optimized), then unrotates, crops
    back, and averages.  Handles non-square (H ≠ W) inputs transparently.

    When return_var=True, also returns the per-pixel variance across the 4 rotation
    views as a regularisation signal (shape matches the averaged output).
    """
    if x.ndim < 4:
        raise ValueError(f"symmetric_forward_2d expects at least 4D input, got {tuple(x.shape)}")

    B, C, H, W = x.shape

    # Pad to square so rot90 doesn't change spatial dims
    x_sq, orig_shape = _pad_to_square(x)
    _, _, HS, WS = x_sq.shape

    # Stack all 4 rotations in the batch dimension → (B*4, C, S, S)
    x_rot = torch.cat([torch.rot90(x_sq, k=k, dims=(-2, -1)) for k in range(4)], dim=0)

    # Apply same pad+rotation to spatial tensor kwargs
    kw_stacked = {}
    for name, val in kwargs.items():
        if torch.is_tensor(val) and val.ndim >= 2 and val.shape[-2:] == (H, W):
            val_sq, _ = _pad_to_square(val)
            kw_stacked[name] = torch.cat([torch.rot90(val_sq, k=k, dims=(-2, -1)) for k in range(4)], dim=0)
        else:
            kw_stacked[name] = val

    # Single forward pass
    feat_stacked = encoder(x_rot, **kw_stacked)

    # Unstack, inverse-rotate, crop back to original H,W, and stack
    feats = torch.chunk(feat_stacked, chunks=4, dim=0)
    feats_inv = []
    for k in range(4):
        f = _crop_from_square(torch.rot90(feats[k], k=-k, dims=(-2, -1)), orig_shape)
        feats_inv.append(f)
    feats_stacked_inv = torch.stack(feats_inv, dim=0)  # (4, B, C_out, H, W)

    avg = feats_stacked_inv.mean(dim=0)
    if return_var:
        var = feats_stacked_inv.var(dim=0, unbiased=False).clamp(min=0.0)
        return avg, var
    return avg


def symmetric_forward_3d(encoder: nn.Module, x: torch.Tensor, return_var: bool = False, **kwargs):
    """
    Eight-way flip group average for dense 3D spatial features.

    Single batched forward pass: stacks all 8 flip configurations into batch dim,
    runs the encoder once, then unflips and averages.
    Flips are involutions, so the same flip axes align encoder outputs back to
    the original D/H/W layout.
    """
    if x.ndim < 5:
        raise ValueError(f"symmetric_forward_3d expects at least 5D input, got {tuple(x.shape)}")

    B, C, D, H, W = x.shape
    spatial_dims = (-3, -2, -1)

    # Build all 8 flip configurations (2^3 = 8)
    all_dims_list = []
    for r in range(len(spatial_dims) + 1):
        for dims in itertools.combinations(spatial_dims, r):
            all_dims_list.append(dims)

    # Stack all 8 flips in the batch dimension → (B*8, C, D, H, W)
    x_flips = []
    for dims in all_dims_list:
        if dims:
            x_flips.append(torch.flip(x, dims=dims))
        else:
            x_flips.append(x)
    x_stacked = torch.cat(x_flips, dim=0)

    # Apply same flips to spatial tensor kwargs
    kw_stacked = {}
    for name, val in kwargs.items():
        if torch.is_tensor(val) and val.ndim >= 3 and val.shape[-3:] == (D, H, W):
            parts = []
            for dims in all_dims_list:
                if dims:
                    parts.append(torch.flip(val, dims=dims))
                else:
                    parts.append(val)
            kw_stacked[name] = torch.cat(parts, dim=0)
        else:
            kw_stacked[name] = val

    # Single forward pass
    feat_stacked = encoder(x_stacked, **kw_stacked)

    # Unstack, unflip, and average
    feats = torch.chunk(feat_stacked, chunks=len(all_dims_list), dim=0)
    feats_inv = []
    for k, dims in enumerate(all_dims_list):
        if dims:
            feats_inv.append(torch.flip(feats[k], dims=dims))
        else:
            feats_inv.append(feats[k])
    feats_stacked_inv = torch.stack(feats_inv, dim=0)  # (8, B, C_out, D_out, H_out, W_out)

    avg = feats_stacked_inv.mean(dim=0)
    if return_var:
        var = feats_stacked_inv.var(dim=0, unbiased=False).clamp(min=0.0)
        return avg, var
    return avg
