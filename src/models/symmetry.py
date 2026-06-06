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


def _stack_spatial_kwargs(kwargs: dict, spatial_shape: tuple[int, ...], view_fns: list, pad_2d: bool = False) -> list[dict]:
    per_view = [dict() for _ in view_fns]
    for name, val in kwargs.items():
        if torch.is_tensor(val) and val.ndim >= len(spatial_shape) and val.shape[-len(spatial_shape):] == spatial_shape:
            if pad_2d:
                val, _ = _pad_to_square(val)
            for i, fn in enumerate(view_fns):
                per_view[i][name] = fn(val)
        else:
            for i in range(len(view_fns)):
                per_view[i][name] = val
    return per_view


def _encode_view_chunks(
    encoder: nn.Module,
    view_inputs: list[torch.Tensor],
    view_kwargs: list[dict],
    inverse_fns: list,
    max_views_per_forward: int,
) -> torch.Tensor:
    max_views = max(1, int(max_views_per_forward))
    aligned = []
    for start in range(0, len(view_inputs), max_views):
        xs = view_inputs[start:start + max_views]
        kws = view_kwargs[start:start + max_views]
        x_batch = torch.cat(xs, dim=0)
        kw_batch = {}
        for name in kws[0].keys():
            vals = [kw[name] for kw in kws]
            if torch.is_tensor(vals[0]) and all(torch.is_tensor(v) and v.shape == vals[0].shape for v in vals):
                kw_batch[name] = torch.cat(vals, dim=0)
            else:
                kw_batch[name] = vals[0]
        feat_batch = encoder(x_batch, **kw_batch)
        for local_i, feat in enumerate(torch.chunk(feat_batch, chunks=len(xs), dim=0)):
            aligned.append(inverse_fns[start + local_i](feat))
    return torch.stack(aligned, dim=0)


def symmetric_forward_2d(
    encoder: nn.Module,
    x: torch.Tensor,
    return_var: bool = False,
    max_views_per_forward: int = 1,
    **kwargs,
):
    """
    Four-way rotational group average for dense 2D spatial features.

    Pads to square, evaluates the four rotations in view chunks, unrotates,
    crops back, and averages. Handles non-square (H != W) inputs transparently.

    When return_var=True, also returns the per-pixel variance across the 4 rotation
    views as a regularisation signal (shape matches the averaged output).
    """
    if x.ndim < 4:
        raise ValueError(f"symmetric_forward_2d expects at least 4D input, got {tuple(x.shape)}")

    B, C, H, W = x.shape

    x_sq, orig_shape = _pad_to_square(x)
    view_fns = [lambda t, k=k: torch.rot90(t, k=k, dims=(-2, -1)) for k in range(4)]
    inverse_fns = [
        lambda t, k=k: _crop_from_square(torch.rot90(t, k=-k, dims=(-2, -1)), orig_shape)
        for k in range(4)
    ]
    view_inputs = [fn(x_sq) for fn in view_fns]
    view_kwargs = _stack_spatial_kwargs(kwargs, (H, W), view_fns, pad_2d=True)
    feats_stacked_inv = _encode_view_chunks(
        encoder,
        view_inputs,
        view_kwargs,
        inverse_fns,
        max_views_per_forward=max_views_per_forward,
    )

    avg = feats_stacked_inv.mean(dim=0)
    if return_var:
        var = feats_stacked_inv.var(dim=0, unbiased=False).clamp(min=0.0)
        return avg, var
    return avg


def symmetric_forward_3d(
    encoder: nn.Module,
    x: torch.Tensor,
    return_var: bool = False,
    max_views_per_forward: int = 1,
    **kwargs,
):
    """
    Eight-way flip group average for dense 3D spatial features.

    Evaluates the 8 flip configurations in view chunks, then unflips and averages.
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

    view_fns = [
        (lambda t, dims=dims: torch.flip(t, dims=dims) if dims else t)
        for dims in all_dims_list
    ]
    inverse_fns = view_fns
    view_inputs = [fn(x) for fn in view_fns]
    view_kwargs = _stack_spatial_kwargs(kwargs, (D, H, W), view_fns, pad_2d=False)
    feats_stacked_inv = _encode_view_chunks(
        encoder,
        view_inputs,
        view_kwargs,
        inverse_fns,
        max_views_per_forward=max_views_per_forward,
    )

    avg = feats_stacked_inv.mean(dim=0)
    if return_var:
        var = feats_stacked_inv.var(dim=0, unbiased=False).clamp(min=0.0)
        return avg, var
    return avg
