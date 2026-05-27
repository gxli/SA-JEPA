from __future__ import annotations

import itertools

import torch
import torch.nn as nn


def _transform_tensor_kwargs(
    kwargs: dict,
    spatial_shape: tuple[int, ...],
    transform,
) -> dict:
    out = {}
    for name, val in kwargs.items():
        if torch.is_tensor(val) and val.ndim >= len(spatial_shape) and tuple(val.shape[-len(spatial_shape) :]) == spatial_shape:
            out[name] = transform(val)
        else:
            out[name] = val
    return out


def symmetric_forward_2d(encoder: nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Four-way rotational group average for dense 2D spatial features.

    Inputs and tensor kwargs with matching HxW spatial shape are rotated together.
    Encoder outputs are inverse-rotated before averaging, preserving the original
    feature-map layout.
    """
    if x.ndim < 4:
        raise ValueError(f"symmetric_forward_2d expects at least 4D input, got {tuple(x.shape)}")

    spatial_shape = tuple(x.shape[-2:])
    accum = None
    for k in range(4):
        rot = lambda t, kk=k: torch.rot90(t, k=kk, dims=(-2, -1))
        x_rot = rot(x)
        kw = _transform_tensor_kwargs(kwargs, spatial_shape, rot)
        feat = encoder(x_rot, **kw)
        feat = torch.rot90(feat, k=-k, dims=(-2, -1))
        accum = feat if accum is None else accum + feat
    return accum / 4.0


def symmetric_forward_3d(encoder: nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Eight-way flip group average for dense 3D spatial features.

    Flips are involutions, so the same flip axes align encoder outputs back to
    the original D/H/W layout.
    """
    if x.ndim < 5:
        raise ValueError(f"symmetric_forward_3d expects at least 5D input, got {tuple(x.shape)}")

    spatial_shape = tuple(x.shape[-3:])
    spatial_dims = (-3, -2, -1)
    accum = None
    count = 0
    for r in range(len(spatial_dims) + 1):
        for dims in itertools.combinations(spatial_dims, r):
            if dims:
                flip = lambda t, dd=dims: torch.flip(t, dims=dd)
                x_flip = flip(x)
                kw = _transform_tensor_kwargs(kwargs, spatial_shape, flip)
                feat = encoder(x_flip, **kw)
                feat = flip(feat)
            else:
                feat = encoder(x, **kwargs)
            accum = feat if accum is None else accum + feat
            count += 1
    return accum / float(count)
