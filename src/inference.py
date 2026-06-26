from __future__ import annotations

import json
import os
import csv
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from src.losses import representation_dense_energy
from src.models.masking import extract_location_patches, norm_per_sample_channel, prepare_context_batch

# Bump this on every inference-affecting change so session logs show which code ran.
INFERENCE_VERSION = "v2-border-nan-3d-fix-2025"
from src.utils.viz import _target_location_yx


def _save_npz(path: str, arr: np.ndarray) -> None:
    """Save a single array as compressed .npz (zip archive, key='arr').

    Replaces np.save(…, .npy) to drastically reduce disk footprint for spatial
    maps that are sparse, zero-heavy, or contain structured data.
    """
    np.savez_compressed(path, arr=arr)


def _iter_tta_views_2d(x: torch.Tensor, mode: str):
    m = str(mode).lower().strip()
    if m in ("none", "", "off"):
        yield "id", x
        return
    if m in ("flip4", "4fold", "4-fold"):
        yield "id", x
        yield "fx", torch.flip(x, dims=(-1,))
        yield "fy", torch.flip(x, dims=(-2,))
        yield "fxy", torch.flip(x, dims=(-2, -1))
        return
    if m in ("d4", "dihedral8", "8fold", "8-fold", "rotflip8"):
        for k in range(4):
            xr = torch.rot90(x, k=k, dims=(-2, -1))
            yield f"r{k}", xr
            yield f"r{k}fx", torch.flip(xr, dims=(-1,))
        return
    if m in ("rot4", "rot", "rot90"):
        for k in range(4):
            yield f"r{k}", torch.rot90(x, k=k, dims=(-2, -1))
        return
    raise ValueError(f"Unsupported inference TTA mode: {mode}")


def _apply_tta_2d(name: str, z: torch.Tensor) -> torch.Tensor:
    """Invert a TTA view transform produced by _iter_tta_views_2d."""
    if name == "id":
        return z
    if name == "fx":
        return torch.flip(z, dims=(-1,))
    if name == "fy":
        return torch.flip(z, dims=(-2,))
    if name == "fxy":
        return torch.flip(z, dims=(-2, -1))
    if name.startswith("r") and name[1:2].isdigit():
        rest = name[1:]
        if rest.endswith("fx"):
            k = int(rest[:-2])
            return torch.rot90(torch.flip(z, dims=(-1,)), k=-k, dims=(-2, -1))
        k = int(rest)
        return torch.rot90(z, k=-k, dims=(-2, -1))
    raise ValueError(name)


def _forward_tta_streaming_2d(
    *,
    x: torch.Tensor,
    mode: str,
    forward_one: Callable[[torch.Tensor, torch.Tensor | None], dict],
    cdd: torch.Tensor | None = None,
    keys=("pred_map", "gt_map", "context_map"),
) -> tuple[dict, int]:
    """Run TTA one view at a time and average aligned maps without stacking views."""
    first_out = None
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    n_views = 0
    align_keys = set(keys) | {
        "x_clean_raw",
        "x_context_raw",
        "x_clean",
        "x_context",
        "network_context_in",
        "network_target_in",
        "target_mask_map",
        "cdd_channels_orig",
        "cdd_channels_masked",
        "dip_field_per_channel",
        "pyramid_mask_token",
    }
    cdd_iter = _iter_tta_views_2d(cdd, mode) if cdd is not None else None
    for name, xv in _iter_tta_views_2d(x, mode):
        cdv = next(cdd_iter)[1] if cdd_iter is not None else None
        out_v = forward_one(xv, cdv)
        n_views += 1
        for key in align_keys:
            if out_v.get(key) is not None:
                out_v[key] = _apply_tta_2d(name, out_v[key])
        if first_out is None:
            first_out = out_v
        for key in keys:
            value = out_v.get(key)
            if value is None:
                continue
            if key not in sums:
                sums[key] = value
                counts[key] = 1
            else:
                sums[key].add_(value)
                counts[key] += 1
        if out_v is not first_out:
            del out_v
    if first_out is None:
        raise ValueError("TTA produced no views")
    for key, value in sums.items():
        first_out[key] = value.div(float(max(1, counts[key])))
    return first_out, n_views


def _mask_invalid_targets_from_input(
    *,
    outputs: dict,
    x_input: torch.Tensor,
    patch_size: int,
    invalid_values=(0.0, "nan"),
) -> int:
    loc = outputs["target_locations"]
    valid = outputs["target_valid"]
    if x_input.dim() != 4 or x_input.shape[1] < 1:
        return 0
    bsz, ksz, _ = loc.shape
    h, w = int(x_input.shape[-2]), int(x_input.shape[-1])
    half_lo = int(patch_size) // 2
    half_hi = int(patch_size) - half_lo
    invalid_specs = tuple(invalid_values) if invalid_values is not None else tuple()
    updated = valid.clone()
    numeric_specs = []
    check_nan = False
    for spec in invalid_specs:
        if isinstance(spec, str) and spec.lower() == "nan":
            check_nan = True
        else:
            try:
                numeric_specs.append(float(spec))
            except (TypeError, ValueError):
                continue
    loc_y, loc_x = _target_location_yx(loc)
    loc_y = loc_y.long()
    loc_x = loc_x.long()
    y0 = loc_y - half_lo
    x0 = loc_x - half_lo
    y1 = loc_y + half_hi
    x1 = loc_x + half_hi
    in_bounds = (y0 >= 0) & (x0 >= 0) & (y1 <= h) & (x1 <= w)
    invalid_target = valid & ~in_bounds
    if int(patch_size) > 0 and h >= int(patch_size) and w >= int(patch_size):
        patches = F.unfold(x_input[:, :1], kernel_size=int(patch_size)).transpose(1, 2)
        linear = (y0.clamp(0, h - int(patch_size)) * (w - int(patch_size) + 1) + x0.clamp(0, w - int(patch_size))).clamp_min(0)
        gather_idx = linear.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        target_patches = patches.gather(1, gather_idx)
        invalid_mask = torch.zeros_like(target_patches, dtype=torch.bool)
        if check_nan:
            invalid_mask |= ~torch.isfinite(target_patches)
        for spec in numeric_specs:
            invalid_mask |= torch.isclose(
                target_patches,
                torch.tensor(spec, device=target_patches.device, dtype=target_patches.dtype),
            )
        invalid_target |= valid & in_bounds & invalid_mask.all(dim=-1)
    else:
        invalid_target |= valid
    updated[invalid_target] = False
    outputs["target_valid"] = updated
    return int(invalid_target.sum().item())


def _accumulate_point_energy(
    *,
    outputs: dict,
    energy_sum: torch.Tensor,
    count_map: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[float, int]:
    pred = outputs["pred_patches"]
    gt = outputs["gt_patches"].detach()
    loc = outputs["target_locations"]
    valid = outputs["target_valid"]
    h, w = int(image_size[0]), int(image_size[1])
    reduce_dims = tuple(range(2, pred.dim()))
    err = (pred - gt).pow(2).mean(dim=reduce_dims)  # B,K
    cy = loc[..., 0].long()
    cx = loc[..., 1].long()
    in_bounds = valid & (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
    if not bool(in_bounds.any().item()):
        return 0.0, 0
    b_idx = torch.arange(err.shape[0], device=err.device).unsqueeze(1).expand_as(err)
    flat_idx = (b_idx * h * w + cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1))[in_bounds]
    values = err[in_bounds].to(dtype=energy_sum.dtype)
    energy_sum.view(-1).scatter_add_(0, flat_idx.reshape(-1), values.reshape(-1))
    count_map.view(-1).scatter_add_(0, flat_idx.reshape(-1), torch.ones_like(values, dtype=count_map.dtype).reshape(-1))
    return float(values.sum().item()), int(values.numel())


def _target_center_map(
    target_locations: torch.Tensor,
    target_valid: torch.Tensor,
    image_size: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tloc = torch.as_tensor(target_locations)
    tvalid = torch.as_tensor(target_valid).bool()
    bsz, _, _ = tloc.shape
    h, w = int(image_size[0]), int(image_size[1])
    tmap = torch.zeros((bsz, 1, h, w), dtype=dtype)
    loc_y, loc_x = _target_location_yx(tloc)
    for bi in range(bsz):
        for ki in range(tloc.shape[1]):
            if not bool(tvalid[bi, ki].item()):
                continue
            cy = int(loc_y[bi, ki].item())
            cx = int(loc_x[bi, ki].item())
            if 0 <= cy < h and 0 <= cx < w:
                # 3x3 cross so targets are visible when downscaled.
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            tmap[bi, 0, yy, xx] = 1.0
    return tmap


def _apply_nan_boundary_frame(x: torch.Tensor, border_px: int) -> torch.Tensor:
    if border_px <= 0:
        return x
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    out = x.clone()
    _, _, h, w = out.shape
    b = int(max(0, min(border_px, h // 2, w // 2)))
    if b <= 0:
        return out
    out[:, :, :b, :] = float("nan")
    out[:, :, h - b :, :] = float("nan")
    out[:, :, :, :b] = float("nan")
    out[:, :, :, w - b :] = float("nan")
    return out


def _tile_starts_2d(length: int, window: int, overlap: int) -> list[int]:
    window = max(1, min(int(window), int(length)))
    overlap = min(max(0, int(overlap)), max(0, window - 1))
    step = max(1, window - overlap)
    starts = list(range(0, max(1, int(length) - window + 1), step))
    tail = max(0, int(length) - window)
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return starts


def _encoder_receptive_field_2d(model) -> int:
    depth = int(getattr(model, "encoder_depth", 4))
    kernel = int(getattr(model, "encoder_kernel_size", 7))
    dilations = getattr(model, "convnext_layer_dilations", None)
    if dilations is None:
        dil_list = [1] * depth
    else:
        dil_list = [int(d) for d in dilations]
        if len(dil_list) < depth:
            reps = (depth + len(dil_list) - 1) // max(1, len(dil_list))
            dil_list = (dil_list * reps)[:depth]
        else:
            dil_list = dil_list[:depth]
    fov = 1 + 2 + 2
    for dilation in dil_list:
        fov += max(0, kernel - 1) * max(1, int(dilation))
    return max(1, int(fov))


def _dense_output_receptive_field_2d(model) -> int:
    rf = int(_encoder_receptive_field_2d(model))
    predictor = getattr(model, "predictor", None)
    if bool(getattr(model, "predictor_spatial_conv", False)) and predictor is not None:
        pred_kernel = 3
        try:
            for module in predictor.modules():
                if isinstance(module, torch.nn.Conv2d) and module.kernel_size[0] > 1:
                    pred_kernel = max(pred_kernel, int(module.kernel_size[0]))
        except Exception:
            pred_kernel = 3
        rf += max(0, int(pred_kernel) - 1)
    return max(1, int(rf))


def _visual_border_half_fov(model) -> int:
    rf = int(getattr(model, "encoder_receptive_field_depth", 0) or 0)
    if rf <= 0:
        rf = int(_encoder_receptive_field_2d(model))
    return max(0, rf // 2)


def _dense_forward_2d_tile(
    model,
    x_tile: torch.Tensor,
    cdd_tile: torch.Tensor | None,
    log_floor: torch.Tensor,
    *,
    cdd_preencoded: bool = False,
):
    post_log = bool(getattr(model, "post_log_transform", False))
    x_enc = torch.log(torch.clamp(x_tile, min=0.0) + log_floor) if post_log else x_tile

    encoder_type = str(getattr(model, "encoder_type", ""))
    if encoder_type == "cdd_scaleaware_convnext":
        if cdd_tile is None:
            raise RuntimeError("cdd_scaleaware_convnext tiled inference requires CDD channels")
        cdd_enc = cdd_tile if cdd_preencoded else (torch.log(torch.clamp(cdd_tile, min=0.0) + log_floor) if post_log else cdd_tile)
        if (not cdd_preencoded) and bool(getattr(model, "scaleaware_norm_per_scale", False)):
            cdd_enc = norm_per_sample_channel(cdd_enc)
        zero = torch.zeros_like(cdd_enc)
        context_map = model.context_encoder(cdd_enc, mask_tokens=zero)
        with torch.no_grad():
            gt_base = model.target_encoder(cdd_enc, mask_tokens=zero)
    elif encoder_type in ("convnext_dense_pyramid", "escnn_c4_pyramid"):
        if cdd_tile is None:
            raise RuntimeError(f"{encoder_type} tiled inference requires CDD channels")
        cdd_enc = torch.log(torch.clamp(cdd_tile, min=0.0) + log_floor) if post_log else cdd_tile
        zero = torch.zeros_like(cdd_enc)
        enc = torch.cat([cdd_enc, zero], dim=1)
        context_map = model.context_encoder(enc)
        with torch.no_grad():
            gt_base = model.target_encoder(enc)
    elif encoder_type == "convnext_dense_masktoken":
        zero = torch.zeros_like(x_enc[:, :1])
        enc = torch.cat([x_enc, zero], dim=1)
        context_map = model.context_encoder(enc)
        with torch.no_grad():
            gt_base = model.target_encoder(enc)
    else:
        context_map = model.context_encoder(x_enc)
        with torch.no_grad():
            gt_base = model.target_encoder(x_enc)

    context_proj = model.projector(context_map)
    pred_map = model.predictor(context_proj)
    with torch.no_grad():
        gt_map = model.target_projector(gt_base)
    energy = representation_dense_energy(pred_map, gt_map)
    return {
        "x_clean": x_enc,
        "x_context": x_enc,
        "context_map": context_map,
        "pred_map": pred_map,
        "gt_map": gt_map,
        "energy_rel_sym": energy["energy_rel_sym"],
        "energy_raw": energy["energy_raw"],
        "energy_rel_gt": energy["energy_rel_gt"],
        "energy_cosine": energy["energy_cosine"],
    }


def _prepare_full_mask_debug_2d(
    *,
    model,
    x_raw_cpu: torch.Tensor,
    cdd_raw_cpu: torch.Tensor | None,
    lattice_shift_override: tuple[int, int] = (0, 0),
    target_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build full-frame mask/debug tensors without using them as encoder input."""
    mask_scale, mask_box_size = model.sample_mask_params(device=x_raw_cpu.device)
    # Resize target_mask to match the full-frame image if shapes differ.
    if target_mask is not None:
        tgt_h, tgt_w = int(x_raw_cpu.shape[-2]), int(x_raw_cpu.shape[-1])
        if target_mask.dim() == 2:
            target_mask = target_mask.unsqueeze(0).unsqueeze(0)
        elif target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        if target_mask.shape[-2:] != (tgt_h, tgt_w):
            target_mask = F.interpolate(
                target_mask.float(),
                size=(tgt_h, tgt_w),
                mode="nearest",
            ).bool()
        else:
            target_mask = target_mask.to(dtype=torch.bool)
    invalid_pixel_mask = ~torch.isfinite(x_raw_cpu)
    border_margin = int(model.encoder_receptive_field()) // 2 if hasattr(model, "encoder_receptive_field") else 0
    border = int(max(0, min(border_margin, int(x_raw_cpu.shape[-2]) // 2, int(x_raw_cpu.shape[-1]) // 2)))
    if border > 0:
        invalid_pixel_mask = invalid_pixel_mask.clone()
        invalid_pixel_mask[:, :, :border, :] = True
        invalid_pixel_mask[:, :, int(x_raw_cpu.shape[-2]) - border :, :] = True
        invalid_pixel_mask[:, :, :, :border] = True
        invalid_pixel_mask[:, :, :, int(x_raw_cpu.shape[-1]) - border :] = True
    context_data = prepare_context_batch(
        x_clean=x_raw_cpu,
        sigmas=model.sigmas,
        mask_fraction=model.mask_fraction,
        mask_scale=float(mask_scale),
        spacing_scale=model.spacing_scale,
        global_shift=model.global_shift,
        align_scales=model.align_scales,
        mask_box_size=int(mask_box_size),
        mask_box_size_range=model.mask_box_size_range,
        random_mask_box_per_target=getattr(model, "random_mask_box_per_target", False),
        manual_mask_box_sizes=model.manual_mask_box_sizes,
        cdd_mode=model.cdd_mode,
        cdd_constrained=model.cdd_constrained,
        cdd_sm_mode=model.cdd_sm_mode,
        cdd_append_last_residual=model.cdd_append_last_residual,
        cdd_pre_log_transform=model.cdd_pre_log_transform,
        patch_size=model.patch_size,
        return_debug=True,
        enable_grid_jitter=False,
        enable_target_dithering=False,
        lattice_shift_override=lattice_shift_override,
        target_invalid_region_skip=model.target_invalid_region_skip,
        target_invalid_region_values=model.target_invalid_region_values,
        target_sampling_mode=model.target_sampling_mode,
        priority_top_percent=model.priority_top_percent,
        priority_n_target=model.priority_n_target,
        priority_min_targets_per_map=model.priority_min_targets_per_map,
        priority_dithering_pixels=model.priority_dithering_pixels,
        priority_candidate_oversample=model.priority_candidate_oversample,
        target_nonoverlap=getattr(model, "target_nonoverlap", False),
        target_allow_partial_overlap=getattr(model, "target_allow_partial_overlap", 0.0),
        mask_box_hardcap=getattr(model, "mask_box_hardcap", None),
        cdd_use_gpu=False,
        cdd_orig_in=cdd_raw_cpu,
        use_cdd=cdd_raw_cpu is not None,
        invalid_pixel_mask_in=invalid_pixel_mask,
        target_mask=target_mask,
    )
    debug = context_data[4] if len(context_data) > 4 else {}
    out = {
        "target_locations": context_data[1],
        "target_scales": context_data[2],
        "target_valid": context_data[3],
    }
    if target_mask is not None:
        out["target_allowed_mask_map"] = target_mask.to(dtype=torch.float32)
    if "mask_map" in debug:
        out["target_mask_map"] = debug["mask_map"].unsqueeze(1)
    for src, dst in (
        ("cdd_channels_orig", "cdd_channels_orig"),
        ("cdd_channels_masked", "cdd_channels_masked"),
        ("dip_field_per_channel", "dip_field_per_channel"),
    ):
        if src in debug:
            out[dst] = debug[src]
    if "dip_field_per_channel" in out:
        out["pyramid_mask_token"] = out["dip_field_per_channel"]
    for key in (
        "priority_good_candidates",
        "priority_nonzero_mean",
        "priority_auto_base_targets",
        "priority_effective_targets",
    ):
        if key in debug:
            out[key] = debug[key]
    return out


def _run_tiled_dense_inference_2d(
    *,
    model,
    x_raw: torch.Tensor,
    cdd_raw: torch.Tensor | None,
    tile_size: int,
    tile_overlap: int | None,
    config_name: str,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    b, _, h, w = x_raw.shape
    if b != 1:
        raise RuntimeError("Tiled 2D dashboard inference currently expects batch size 1")
    rf = _dense_output_receptive_field_2d(model)
    margin = min(rf // 2, max(0, (int(tile_size) - 1) // 2))
    overlap_eff = max(2 * margin, int(tile_overlap) if tile_overlap is not None else 0)
    overlap_eff = min(max(0, overlap_eff), max(0, int(tile_size) - 1))
    y_starts = _tile_starts_2d(h, tile_size, overlap_eff)
    x_starts = _tile_starts_2d(w, tile_size, overlap_eff)
    eps = max(1e-6, float(getattr(model, "log_eps", 1.0)))
    if bool(getattr(model, "post_log_transform", False)):
        base = torch.clamp(x_raw, min=0.0)
        base_std = torch.std(base, dim=(-2, -1), keepdim=True)
        log_floor = torch.clamp(base_std * float(getattr(model, "cdd_log_std_floor_mult", 0.05)), min=eps)
    else:
        log_floor = torch.ones((b, 1, 1, 1), dtype=x_raw.dtype, device=x_raw.device)
    cdd_source = cdd_raw
    cdd_preencoded = False
    if cdd_raw is not None and str(getattr(model, "encoder_type", "")) == "cdd_scaleaware_convnext":
        cdd_source = cdd_raw.to(device, non_blocking=True)
        if bool(getattr(model, "post_log_transform", False)):
            cdd_source = torch.log(torch.clamp(cdd_source, min=0.0) + log_floor.to(device, non_blocking=True))
        if bool(getattr(model, "scaleaware_norm_per_scale", False)):
            cdd_source = norm_per_sample_channel(cdd_source)
        cdd_source = cdd_source.detach().cpu()
        cdd_preencoded = True

    print(
        f"[{config_name}] tiled 2D dense inference: tile={tile_size} "
        f"y_tiles={len(y_starts)} x_tiles={len(x_starts)} dense_output_rf={rf} "
        f"discard_margin={margin} overlap={overlap_eff} "
        f"cdd_preencoded_full_field={cdd_preencoded}"
    )
    sums: dict[str, torch.Tensor] = {}
    weight = torch.zeros((b, 1, h, w), dtype=torch.float32)
    for y0 in y_starts:
        ye = min(y0 + int(tile_size), h)
        valid_h = int(ye - y0)
        wy0 = 0 if y0 == 0 else min(margin, valid_h)
        wy1 = valid_h if ye == h else max(wy0, valid_h - min(margin, valid_h))
        if wy1 <= wy0:
            continue
        for x0 in x_starts:
            xe = min(x0 + int(tile_size), w)
            valid_w = int(xe - x0)
            wx0 = 0 if x0 == 0 else min(margin, valid_w)
            wx1 = valid_w if xe == w else max(wx0, valid_w - min(margin, valid_w))
            if wx1 <= wx0:
                continue
            x_tile = x_raw[:, :, y0:ye, x0:xe]
            cdd_tile = None if cdd_source is None else cdd_source[:, :, y0:ye, x0:xe]
            pad_h = max(0, int(tile_size) - int(x_tile.shape[-2]))
            pad_w = max(0, int(tile_size) - int(x_tile.shape[-1]))
            if pad_h or pad_w:
                min_dim = min(x_tile.shape[-2:])
                # reflect mode requires pad < input_dim; fall back to replicate
                # when the tile is smaller than the padding (e.g. 256 px image
                # with a 512 px tile size needs 256 px of padding).
                if min_dim > max(pad_h, pad_w):
                    pad_mode = "reflect" if min_dim > 1 else "replicate"
                else:
                    pad_mode = "replicate"
                x_tile = F.pad(x_tile, (0, pad_w, 0, pad_h), mode=pad_mode)
                if cdd_tile is not None:
                    cdd_tile = F.pad(cdd_tile, (0, pad_w, 0, pad_h), mode=pad_mode)
            tile_out = _dense_forward_2d_tile(
                model,
                x_tile.to(device, non_blocking=True),
                None if cdd_tile is None else cdd_tile.to(device, non_blocking=True),
                log_floor.to(device, non_blocking=True),
                cdd_preencoded=cdd_preencoded,
            )
            out_y0, out_y1 = y0 + wy0, y0 + wy1
            out_x0, out_x1 = x0 + wx0, x0 + wx1
            for key, value in tile_out.items():
                value_cpu = value[:, :, :valid_h, :valid_w].detach().cpu()
                if key not in sums:
                    sums[key] = torch.zeros((b, int(value_cpu.shape[1]), h, w), dtype=torch.float32)
                sums[key][:, :, out_y0:out_y1, out_x0:out_x1] += value_cpu[:, :, wy0:wy1, wx0:wx1]
            weight[:, :, out_y0:out_y1, out_x0:out_x1] += 1.0
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not sums:
        raise RuntimeError(f"[{config_name}] tiled 2D dense inference produced no tiles")
    weight = weight.clamp_min(1.0)
    return {key: value / weight for key, value in sums.items()}


def _load_cdd_cache_for_image(session_dir: str, full_h: int, full_w: int, h: int, w: int,
                              use_log: bool = False) -> torch.Tensor | None:
    """Load CDD from disk cache, returning a (1, S, h, w) tile from center."""
    import glob as _glob
    cdd_dir = os.path.join(session_dir, "cdd_cache")
    if not os.path.isdir(cdd_dir):
        return None
    # Prefer new .npz format; fall back to legacy .npy.
    candidates_npz = _glob.glob(os.path.join(cdd_dir, "*.npz"))
    candidates_npy = _glob.glob(os.path.join(cdd_dir, "*.npy"))
    if candidates_npz:
        loaded = np.load(candidates_npz[0])
        variant = "transformed" if use_log else "untransformed"
        if variant not in loaded:
            return None
        cdd_full = loaded[variant].astype(np.float32)  # (S, H, W)
    elif candidates_npy:
        cdd_full = np.load(candidates_npy[0]).astype(np.float32)  # legacy (S, H, W)
    else:
        return None
    if cdd_full.ndim != 3:
        return None
    y0 = max(0, (cdd_full.shape[-2] - h) // 2)
    x0 = max(0, (cdd_full.shape[-1] - w) // 2)
    tile = cdd_full[:, y0:y0 + h, x0:x0 + w]
    return torch.from_numpy(tile).unsqueeze(0)  # (1, S, h, w)


def _first_full_resolution_batch(dataloader):
    """Fetch a single batch for diagnostic visualization.

    Always returns the native sample. Post-training diagnostics must preserve
    field of view; memory control belongs to tiled inference, not cropping.
    """
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__getitem__"):
        return None
    old = {}

    for name, value in (
        ("crop_mode", "none"),
        ("crop_size", None),
        ("d4_augment", False),
        ("crop_min_valid_fraction", 0.0),
    ):
        if hasattr(dataset, name):
            old[name] = getattr(dataset, name)
            setattr(dataset, name, value)
    try:
        sample = dataset[0]
    finally:
        for name, value in old.items():
            setattr(dataset, name, value)
    if isinstance(sample, (tuple, list)):
        return tuple(x.unsqueeze(0) if torch.is_tensor(x) and x.dim() in (3, 4) else x for x in sample)
    if torch.is_tensor(sample):
        return sample.unsqueeze(0) if sample.dim() in (3, 4) else sample
    return None


def run_post_training_inference(
    *,
    model,
    dataloader,
    session_dir: str,
    config_name: str,
    visit_counts,
    force_recompute_inference: bool,
    inference_mask_passes: int,
    mask_inference: bool,
    target_mask: torch.Tensor | None = None,
    inference_mask_border: bool | int | float = True,
    compute_jepa_energy_fn: Callable,
    compute_target_energy_map_fn: Callable,
    inference_tta_enabled: bool = False,
    inference_tta_mode: str = "flip4",
    max_diagnostic_size: int | None = None,
    tile_size: int | None = 512,
    tile_overlap: int | None = None,
    inference_discard_margin: int | None = None,
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
    print(f"[{config_name}] inference_version={INFERENCE_VERSION}", flush=True)
    if (not force_recompute_inference) and os.path.exists(inference_outputs_path):
        print(
            f"[{config_name}] inference_outputs.pt already exists; "
            "skipping post-training dashboard-sample inference "
            "(set train.force_recompute_inference=true to recompute)"
        )
        return session_dir

    inference_required = [
        inference_outputs_path,
        os.path.join(session_dir, "network_input_clean.npz"),
        os.path.join(session_dir, "network_input_context.npz"),
        os.path.join(session_dir, "pred_map.npz"),
        os.path.join(session_dir, "gt_map.npz"),
        os.path.join(session_dir, "target_energy_map.npz"),
        os.path.join(session_dir, "jepa_energy_summary.json"),
    ]
    if (not force_recompute_inference) and all(os.path.exists(p) for p in inference_required):
        print(
            f"[{config_name}] inference artifacts already exist; "
            "skipping post-training dashboard-sample inference "
            "(set train.force_recompute_inference=true to recompute)"
        )
        return session_dir

    try:
        dataloader_len = len(dataloader)
    except TypeError:
        dataloader_len = None
    if dataloader_len is not None and dataloader_len > 1:
        print(
            f"[{config_name}] post_training_inference scope=dashboard_sample_batch "
            f"using first batch only out of {dataloader_len}; use src.inference_from_session "
            "for explicit dataset inference"
        )
    else:
        print(f"[{config_name}] post_training_inference scope=dashboard_sample_batch")
    model.eval()
    with torch.no_grad():
        print(f"[{config_name}] post_training_inference loading sample batch")
        if max_diagnostic_size is not None:
            try:
                if int(max_diagnostic_size) > 0:
                    print(
                        f"[{config_name}] post_training_inference ignoring "
                        f"inference_max_diagnostic_size={int(max_diagnostic_size)} "
                        "because post-training inference always preserves full image shape"
                    )
            except (TypeError, ValueError):
                pass
        raw_batch = _first_full_resolution_batch(dataloader)
        if raw_batch is None:
            raw_batch = next(iter(dataloader))
        else:
            print(f"[{config_name}] post_training_inference using uncropped first image for dashboard")
        if isinstance(raw_batch, (tuple, list)) and len(raw_batch) == 2 and raw_batch[1] is not None:
            cdd_raw, x_raw = raw_batch
            cdd_raw = cdd_raw.to(next(model.parameters()).device)
        else:
            cdd_raw = None
            x_raw = raw_batch if not isinstance(raw_batch, (tuple, list)) else raw_batch[0]
        x_raw = x_raw.to(next(model.parameters()).device)
        use_tiled_2d = (
            x_raw.dim() == 4
            and tile_size is not None
            and int(tile_size) > 0
        )
        if bool(mask_inference) and use_tiled_2d:
            print(
                f"[{config_name}] tiled dense inference with mask diagnostics: "
                "encoder runs clean full-frame tiles; mask/debug tensors are "
                "computed full-frame and applied downstream"
            )
        shifts = [(0, 0)]
        print(f"[{config_name}] post_training_inference model forward deterministic_shifts=1")
        outputs = None
        energy_sum = None
        count_map = None
        total_energy = 0.0
        total_valid = 0
        invalid_region_skip = bool(getattr(model, "target_invalid_region_skip", False))
        invalid_region_values = tuple(getattr(model, "target_invalid_region_values", (0.0, "nan")))
        patch_size = int(getattr(model, "patch_size", 2))
        tta_view_count = 1
        shift_sums: dict[str, torch.Tensor] = {}
        shift_counts: dict[str, int] = {}
        if use_tiled_2d:
            h, w = int(x_raw.shape[-2]), int(x_raw.shape[-1])
            probe_h = min(int(tile_size), h)
            probe_w = min(int(tile_size), w)
            x_probe = x_raw[:, :, :probe_h, :probe_w]
            cdd_probe = None if cdd_raw is None else cdd_raw[:, :, :probe_h, :probe_w]
            outputs = model(
                x_probe,
                return_debug=True,
                enable_grid_jitter=False,
                enable_target_dithering=False,
                lattice_shift_override=(0, 0),
                mask_inference=False,
                cdd_orig=cdd_probe,
            )
            mask_debug_full = {}
            if bool(mask_inference):
                # Prefer the CDD cache from disk to avoid on-the-fly
                # decomposition on the full image (slow for large fields).
                cdd_for_mask = None
                if cdd_raw is None:
                    cdd_cache_dir = os.path.join(session_dir, "cdd_cache")
                    if os.path.isdir(cdd_cache_dir):
                        import glob as _glob
                        _candidates = _glob.glob(os.path.join(cdd_cache_dir, "*.npy"))
                        if _candidates:
                            cdd_for_mask = torch.from_numpy(
                                np.load(_candidates[0])
                            ).float().unsqueeze(0)
                else:
                    cdd_for_mask = cdd_raw
                mask_debug_full = _prepare_full_mask_debug_2d(
                    model=model,
                    x_raw_cpu=x_raw.detach().cpu(),
                    cdd_raw_cpu=None if cdd_for_mask is None else cdd_for_mask.detach().cpu(),
                    lattice_shift_override=(0, 0),
                    target_mask=target_mask,
                )
            tiled = _run_tiled_dense_inference_2d(
                model=model,
                x_raw=x_raw.detach().cpu(),
                cdd_raw=None if cdd_raw is None else cdd_raw.detach().cpu(),
                tile_size=int(tile_size),
                tile_overlap=tile_overlap,
                config_name=config_name,
            )
            for key in ("x_clean", "x_context", "context_map", "pred_map", "gt_map"):
                outputs[key] = tiled[key].to(x_raw.device)
            outputs["x_clean_raw"] = x_raw
            outputs["x_context_raw"] = x_raw
            for key in (
                "network_context_in",
                "network_target_in",
                "target_mask_map",
                "cdd_channels_orig",
                "cdd_channels_masked",
                "dip_field_per_channel",
                "pyramid_mask_token",
            ):
                outputs.pop(key, None)
            for key, value in mask_debug_full.items():
                outputs[key] = value.to(x_raw.device) if torch.is_tensor(value) else value
            outputs["pred_patches"] = extract_location_patches(
                outputs["pred_map"],
                outputs["target_locations"],
                patch_size=int(getattr(model, "patch_size", 1)),
            )
            outputs["gt_patches"] = extract_location_patches(
                outputs["gt_map"],
                outputs["target_locations"],
                patch_size=int(getattr(model, "patch_size", 1)),
            )
            if cdd_raw is not None:
                outputs.setdefault("cdd_channels_orig", cdd_raw)
                outputs.setdefault("cdd_channels_masked", cdd_raw)
                zero_cdd = torch.zeros_like(cdd_raw)
                outputs.setdefault("dip_field_per_channel", zero_cdd)
                outputs.setdefault("pyramid_mask_token", zero_cdd)
            bsz = int(x_raw.shape[0])
            energy_sum = tiled["energy_rel_sym"].to(x_raw.device, dtype=x_raw.dtype)
            count_map = torch.ones_like(energy_sum)
            total_energy = float(tiled["energy_rel_sym"].sum().item())
            total_valid = int(max(1, tiled["energy_rel_sym"].numel()))
            outputs["_tiled_dense_energy"] = {k: v.to(x_raw.device) for k, v in tiled.items() if k.startswith("energy_")}
        else:
            for pi, shift in enumerate(shifts):
                return_debug_next = pi == 0

                def _forward_one_tta(xv: torch.Tensor, cdv: torch.Tensor | None) -> dict:
                    nonlocal return_debug_next
                    return_debug = return_debug_next
                    return_debug_next = False
                    return model(
                        xv,
                        return_debug=return_debug,
                        enable_grid_jitter=False,
                        enable_target_dithering=False,
                        lattice_shift_override=shift,
                        mask_inference=bool(mask_inference),
                        cdd_orig=cdv,
                    )

                if bool(inference_tta_enabled):
                    out_i, tta_view_count = _forward_tta_streaming_2d(
                        x=x_raw,
                        mode=inference_tta_mode,
                        forward_one=_forward_one_tta,
                        cdd=cdd_raw,
                    )
                else:
                    out_i = _forward_one_tta(x_raw, cdd_raw)
                for key in ("pred_map", "gt_map", "context_map"):
                    value = out_i.get(key)
                    if value is None:
                        continue
                    if key not in shift_sums:
                        shift_sums[key] = value
                        shift_counts[key] = 1
                    else:
                        shift_sums[key].add_(value)
                        shift_counts[key] += 1
                if invalid_region_skip:
                    _mask_invalid_targets_from_input(
                        outputs=out_i,
                        x_input=x_raw,
                        patch_size=patch_size,
                        invalid_values=invalid_region_values,
                    )
                if outputs is None:
                    outputs = out_i
                    h, w = outputs["x_clean"].shape[-2:]
                    bsz = outputs["x_clean"].shape[0]
                    energy_sum = torch.zeros((bsz, 1, h, w), device=outputs["x_clean"].device, dtype=outputs["x_clean"].dtype)
                    count_map = torch.zeros_like(energy_sum)
                e_tot_i, n_val_i = _accumulate_point_energy(
                    outputs=out_i,
                    energy_sum=energy_sum,
                    count_map=count_map,
                    image_size=(h, w),
                )
                total_energy += float(e_tot_i)
                total_valid += int(n_val_i)
        assert outputs is not None and energy_sum is not None and count_map is not None
        # Average aligned maps across deterministic shifts to stabilize dashboard embeddings.
        if len(shifts) > 1:
            for key, value in shift_sums.items():
                outputs[key] = value.div(float(max(1, shift_counts[key])))

    inference_outputs = {
        "inference_scope": "dashboard_sample_batch",
        "mask_inference": bool(mask_inference),
        "patch_size": int(getattr(model, "patch_size", 1)),
        "inference_input_shape": tuple(int(v) for v in outputs["x_clean"].shape[-2:]),
        "inference_pred_shape": tuple(int(v) for v in outputs["pred_map"].shape[-2:]),
        "inference_num_dataloader_batches_seen": 1,
        "inference_num_dataloader_batches_available": dataloader_len if dataloader_len is not None else -1,
        "x_clean_raw": outputs.get("x_clean_raw", outputs["x_clean"])[:8].detach().cpu(),
        "x_context_raw": outputs.get("x_context_raw", outputs["x_context"])[:8].detach().cpu(),
        "x_clean": outputs["x_clean"][:8].detach().cpu(),
        "x_context": outputs["x_context"][:8].detach().cpu(),
        "target_locations": outputs["target_locations"][:8].detach().cpu(),
        "target_scales": outputs["target_scales"][:8].detach().cpu(),
        "target_valid": outputs["target_valid"][:8].detach().cpu(),
        "pred_map": outputs["pred_map"][:2].detach().cpu(),
        "gt_map": outputs["gt_map"][:2].detach().cpu(),
        "context_map": outputs.get("context_map", outputs["pred_map"])[:2].detach().cpu(),
        "pred_patches": outputs["pred_patches"][:2].detach().cpu(),
        "gt_patches": outputs["gt_patches"][:2].detach().cpu(),
    }
    if "network_context_in" in outputs:
        inference_outputs["network_context_in"] = outputs["network_context_in"][:8].detach().cpu()
    if "network_target_in" in outputs:
        inference_outputs["network_target_in"] = outputs["network_target_in"][:8].detach().cpu()
    if "target_mask_map" in outputs:
        inference_outputs["target_mask_map"] = outputs["target_mask_map"][:8].detach().cpu()
    if "target_allowed_mask_map" in outputs:
        inference_outputs["target_allowed_mask_map"] = outputs["target_allowed_mask_map"][:8].detach().cpu()
    if "cdd_channels_orig" in outputs:
        inference_outputs["cdd_channels_orig"] = outputs["cdd_channels_orig"][:8].detach().cpu()
    if "cdd_channels_masked" in outputs:
        inference_outputs["cdd_channels_masked"] = outputs["cdd_channels_masked"][:8].detach().cpu()
    if "dip_field_per_channel" in outputs:
        inference_outputs["dip_field_per_channel"] = outputs["dip_field_per_channel"][:8].detach().cpu()
    if "pyramid_mask_token" in outputs:
        inference_outputs["pyramid_mask_token"] = outputs["pyramid_mask_token"][:8].detach().cpu()
    for k in (
        "priority_good_candidates",
        "priority_nonzero_mean",
        "priority_auto_base_targets",
        "priority_effective_targets",
    ):
        if k in outputs:
            inference_outputs[k] = outputs[k][:8].detach().cpu()
    energy_scalar = float(total_energy / max(1, total_valid))
    energy_scalar_norm = compute_jepa_energy_fn(outputs, normalize=True)
    # Dense full-image energy from prediction/target maps. For tiled 2D
    # inference, use the stitched energy maps computed tile-by-tile so the
    # dashboard never falls back to a one-shot full-frame pass.
    if "_tiled_dense_energy" in outputs:
        e_map_dense = {
            "energy_rel_sym": outputs["_tiled_dense_energy"]["energy_rel_sym"],
            "energy_raw": outputs["_tiled_dense_energy"]["energy_raw"],
            "energy_rel_gt": outputs["_tiled_dense_energy"]["energy_rel_gt"],
            "energy_cosine": outputs["_tiled_dense_energy"]["energy_cosine"],
        }
    else:
        e_map_dense = compute_target_energy_map_fn(
            outputs,
            image_size=(int(outputs["x_clean"].shape[-2]), int(outputs["x_clean"].shape[-1])),
        )
    # Keep point-sampled target energy as a secondary diagnostic.
    e_map_points = energy_sum / count_map.clamp_min(1.0)
    inference_outputs["jepa_energy"] = torch.tensor(energy_scalar, dtype=torch.float32)
    inference_outputs["jepa_energy_normalized"] = torch.tensor(energy_scalar_norm, dtype=torch.float32)
    inference_outputs["target_energy_map"] = e_map_dense["energy_rel_sym"][:8].detach().cpu()
    inference_outputs["target_energy_raw_map"] = e_map_dense["energy_raw"][:8].detach().cpu()
    inference_outputs["target_energy_rel_gt_map"] = e_map_dense["energy_rel_gt"][:8].detach().cpu()
    inference_outputs["target_energy_cosine_map"] = e_map_dense["energy_cosine"][:8].detach().cpu()
    inference_outputs["target_energy_point_map"] = e_map_points[:8].detach().cpu()
    inference_outputs["target_energy_count_map"] = count_map[:8].detach().cpu()

    # Encoder FOV border: true/auto → half-FOV; number → that many px; false/0 → skip.
    if isinstance(inference_mask_border, bool):
        fov_border = int(_visual_border_half_fov(model)) if inference_mask_border else 0
    else:
        fov_border = int(max(0, float(inference_mask_border)))
    if fov_border > 0:
        for key in ("pred_map", "gt_map", "context_map",
                    "target_energy_map", "target_energy_raw_map",
                    "target_energy_rel_gt_map", "target_energy_cosine_map",
                    "target_energy_point_map"):
            if key in inference_outputs:
                inference_outputs[key] = _apply_nan_boundary_frame(
                    inference_outputs[key], fov_border
                )

    # target_map is a center-point diagnostic. Keep sampled mask footprints in
    # target_mask_map/mask_cube only; do not reuse them as target locations.
    h, w = inference_outputs["x_clean"].shape[-2:]
    tmap = _target_center_map(
        inference_outputs["target_locations"],
        inference_outputs["target_valid"],
        (h, w),
        dtype=inference_outputs["x_clean"].dtype,
    )
    inference_outputs["target_map"] = tmap
    torch.save(inference_outputs, inference_outputs_path)
    print(f"[{config_name}] saved inference_outputs.pt")

    _save_npz(os.path.join(session_dir, "network_input_clean.npz"), inference_outputs["x_clean"].numpy())
    _save_npz(os.path.join(session_dir, "network_input_context.npz"), inference_outputs["x_context"].numpy())
    _save_npz(os.path.join(session_dir, "network_input_clean_raw.npz"), inference_outputs["x_clean_raw"].numpy())
    _save_npz(os.path.join(session_dir, "network_input_context_raw.npz"), inference_outputs["x_context_raw"].numpy())
    if "network_context_in" in inference_outputs:
        _save_npz(
            os.path.join(session_dir, "network_context_in.npz"),
            inference_outputs["network_context_in"].numpy(),
        )
    if "network_target_in" in inference_outputs:
        _save_npz(
            os.path.join(session_dir, "network_target_in.npz"),
            inference_outputs["network_target_in"].numpy(),
        )
    _save_npz(os.path.join(session_dir, "target_valid.npz"), inference_outputs["target_valid"].numpy())
    if "target_mask_map" in inference_outputs:
        _save_npz(os.path.join(session_dir, "target_mask_map.npz"), inference_outputs["target_mask_map"].numpy())
    if "target_allowed_mask_map" in inference_outputs:
        _save_npz(os.path.join(session_dir, "target_allowed_mask_map.npz"), inference_outputs["target_allowed_mask_map"].numpy())
    if "cdd_channels_orig" in inference_outputs:
        _save_npz(os.path.join(session_dir, "cdd_channels_orig.npz"), inference_outputs["cdd_channels_orig"].numpy())
    if "cdd_channels_masked" in inference_outputs:
        _save_npz(os.path.join(session_dir, "cdd_channels_masked.npz"), inference_outputs["cdd_channels_masked"].numpy())
    if "dip_field_per_channel" in inference_outputs:
        _save_npz(
            os.path.join(session_dir, "dip_field_per_channel.npz"),
            inference_outputs["dip_field_per_channel"].numpy(),
        )
        # Backward-compatible dashboard artifact name.
        _save_npz(
            os.path.join(session_dir, "pyramid_mask_token.npz"),
            inference_outputs["dip_field_per_channel"].numpy(),
        )
    if "pyramid_mask_token" in inference_outputs:
        _save_npz(os.path.join(session_dir, "pyramid_mask_token.npz"), inference_outputs["pyramid_mask_token"].numpy())
    if visit_counts is not None:
        _save_npz(os.path.join(session_dir, "visited_target_frequency.npz"), visit_counts.astype(np.float32))
    _save_npz(os.path.join(session_dir, "target_energy_map.npz"), inference_outputs["target_energy_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_raw_map.npz"), inference_outputs["target_energy_raw_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_rel_gt_map.npz"), inference_outputs["target_energy_rel_gt_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_cosine_map.npz"), inference_outputs["target_energy_cosine_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_point_map.npz"), inference_outputs["target_energy_point_map"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_count_map.npz"), inference_outputs["target_energy_count_map"].numpy())
    encoder_rf = int(_encoder_receptive_field_2d(model))
    dense_output_rf = int(_dense_output_receptive_field_2d(model))
    if inference_discard_margin is None or str(inference_discard_margin).strip().lower() == "auto":
        # Tiled inference already handles tile-boundary overlap via tile_overlap.
        # No discard margin is needed; a positive value leaks into downstream
        # dashboard border logic where it is misinterpreted as a crop signal.
        discard_margin = 0
    else:
        discard_margin = int(max(0, int(inference_discard_margin)))
    with open(os.path.join(session_dir, "jepa_energy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "jepa_energy": float(energy_scalar),
                "jepa_energy_normalized": float(energy_scalar_norm),
                "inference_mask_passes": int(len(shifts)),
                "mask_inference": bool(mask_inference),
                "inference_tta_enabled": bool(inference_tta_enabled),
                "inference_tta_mode": str(inference_tta_mode),
                "inference_tta_views": int(tta_view_count),
                "inference_tiled_dense_2d": bool(use_tiled_2d),
                "inference_tile_size": None if tile_size is None else int(tile_size),
                "inference_tile_overlap": None if tile_overlap is None else int(tile_overlap),
                "inference_input_shape": list(inference_outputs["inference_input_shape"]),
                "inference_pred_shape": list(inference_outputs["inference_pred_shape"]),
                "inference_encoder_receptive_field": int(encoder_rf),
                "inference_dense_output_receptive_field": int(dense_output_rf),
                "inference_discard_margin": int(discard_margin),
                "target_allowed_mask_present": bool("target_allowed_mask_map" in inference_outputs),
                "target_allowed_mask_fraction": (
                    float(inference_outputs["target_allowed_mask_map"].float().mean().item())
                    if "target_allowed_mask_map" in inference_outputs
                    else None
                ),
                "target_valid_fraction": float(inference_outputs["target_valid"].float().mean().item()),
            },
            f,
            indent=2,
        )
    if "priority_effective_targets" in inference_outputs:
        active_target_fraction = float(getattr(model, "mask_fraction", 1.0))
        priority_n_target_cfg = getattr(model, "priority_n_target", 20)
        summary = {
            "active_target_fraction": float(active_target_fraction),
            "priority_n_target_config": priority_n_target_cfg,
            "priority_min_targets_per_map_config": int(getattr(model, "priority_min_targets_per_map", 0)),
            "priority_good_candidates_mean": float(inference_outputs.get("priority_good_candidates", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_good_candidates", None) is not None and inference_outputs["priority_good_candidates"].numel() > 0
            else 0.0,
            "priority_nonzero_mean_mean": float(inference_outputs.get("priority_nonzero_mean", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_nonzero_mean", None) is not None and inference_outputs["priority_nonzero_mean"].numel() > 0
            else 1.0,
            "priority_auto_base_targets_mean": float(inference_outputs.get("priority_auto_base_targets", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_auto_base_targets", None) is not None and inference_outputs["priority_auto_base_targets"].numel() > 0
            else 0.0,
            "priority_effective_targets_mean": float(inference_outputs.get("priority_effective_targets", torch.tensor([])).float().mean().item())
            if inference_outputs.get("priority_effective_targets", None) is not None and inference_outputs["priority_effective_targets"].numel() > 0
            else 0.0,
        }
        with open(os.path.join(session_dir, "target_selection_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    _save_npz(os.path.join(session_dir, "pred_map.npz"), inference_outputs["pred_map"].numpy())
    _save_npz(os.path.join(session_dir, "gt_map.npz"), inference_outputs["gt_map"].numpy())
    pred_norm = inference_outputs["pred_map"].norm(dim=1).numpy()
    gt_norm = inference_outputs["gt_map"].norm(dim=1).numpy()
    err_norm = (inference_outputs["pred_map"] - inference_outputs["gt_map"]).norm(dim=1).numpy()
    _save_npz(os.path.join(session_dir, "pred_latent_norm.npz"), pred_norm)
    _save_npz(os.path.join(session_dir, "gt_latent_norm.npz"), gt_norm)
    _save_npz(os.path.join(session_dir, "pred_gt_latent_error_norm.npz"), err_norm)
    print(
        f"[{config_name}] post_training_artifacts_saved session_dir={session_dir} "
        f"(run scripts/session_to_dash.py to generate plots/dashboards)"
    )
    return session_dir


def run_post_training_inference_3d(
    *,
    model,
    dataloader,
    session_dir: str,
    config_name: str,
    force_recompute_inference: bool,
    cdd_cache: dict | None = None,
    inference_depth: int | None = None,
    slice_index: int | None = None,
    spatial_tile_size: int | None = 64,
    spatial_overlap: int | None = None,
    inference_tta_enabled: bool = False,
    inference_tta_mode: str = "flip4",
    inference_mask_border: bool | int | float = True,
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
    print(f"[{config_name}] inference_version={INFERENCE_VERSION}", flush=True)
    if (not force_recompute_inference) and os.path.exists(inference_outputs_path):
        summary_path = os.path.join(session_dir, "jepa_energy_summary.json")
        if cdd_cache:
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                if bool(summary.get("inference_3d_full_xy_slice", False)):
                    return session_dir
            except Exception:
                pass
        else:
            return session_dir

    model.eval()
    with torch.no_grad():
        if cdd_cache:
            (path, _), cdd_vol_raw = next(iter(cdd_cache.items()))
            if isinstance(cdd_vol_raw, dict):
                cdd_vol = cdd_vol_raw["untransformed"]
            else:
                cdd_vol = cdd_vol_raw
            cdd_np = np.asarray(cdd_vol, dtype=np.float32)
            if cdd_np.ndim != 4:
                raise ValueError(f"Expected cached CDD cube SxDxHxW, got {tuple(cdd_np.shape)} for {path}")
            _, d, h, w = cdd_np.shape
            min_depth = int(getattr(model, "required_input_depth", getattr(model, "encoder_receptive_field_depth", 1)))
            slab_depth = int(min_depth)
            z = int(d // 2 if slice_index is None else slice_index)
            z = max(0, min(d - 1, z))
            before = slab_depth // 2
            after = slab_depth - before - 1
            req0 = z - before
            req1 = z + after + 1
            src0 = max(0, req0)
            src1 = min(d, req1)
            pad_front = src0 - req0
            pad_back = req1 - src1
            vol_cpu = torch.from_numpy(cdd_np[:, src0:src1]).unsqueeze(0).float()
            if pad_front or pad_back:
                vol_cpu = F.pad(vol_cpu, (0, 0, 0, 0, pad_front, pad_back), mode="replicate")
            device = next(model.parameters()).device
            tile_size = max(1, int(spatial_tile_size or max(h, w)))
            encoder_rf = int(getattr(model, "encoder_receptive_field_depth", 1) or 1)
            margin = min(max(0, encoder_rf // 2), max(0, (tile_size - 1) // 2))
            overlap_eff = max(2 * margin, int(spatial_overlap) if spatial_overlap is not None else 0)
            overlap_eff = min(max(0, overlap_eff), max(0, tile_size - 1))
            y_starts = _tile_starts_2d(h, tile_size, overlap_eff)
            x_starts = _tile_starts_2d(w, tile_size, overlap_eff)
            print(
                f"[{config_name}] 3D slice inference: source={path} "
                f"slice={z}/{d} input_slab_depth={slab_depth} full_xy=({h}, {w}) "
                f"spatial_tile={tile_size} y_tiles={len(y_starts)} x_tiles={len(x_starts)} "
                f"encoder_rf={encoder_rf} discard_margin={margin} overlap={overlap_eff}"
            )

            log_floor_cpu = None
            if bool(getattr(model, "post_log_transform", False)):
                eps = max(1e-6, float(getattr(model, "log_eps", 1.0)))
                base = torch.clamp(vol_cpu, min=0.0)
                base_std = torch.std(base, dim=(-3, -2, -1), keepdim=True)
                log_floor_cpu = torch.clamp(
                    base_std * float(getattr(model, "cdd_log_std_floor_mult", 0.05)),
                    min=eps,
                )
            center_idx = int(slab_depth // 2)
            pred_sum = None
            gt_sum = None
            ctx_sum = None
            tta_view_count = 1
            if bool(inference_tta_enabled):
                tta_view_count = sum(1 for _name, _x in _iter_tta_views_2d(torch.empty(1, 1, 1, 2, 2), inference_tta_mode))
                print(
                    f"[{config_name}] 3D slice inference TTA: enabled mode={inference_tta_mode} "
                    f"views={tta_view_count}"
                )
            weight = torch.zeros((1, 1, 1, h, w), dtype=torch.float32)
            for y0 in y_starts:
                ye = min(y0 + tile_size, h)
                valid_h = int(ye - y0)
                wy0 = 0 if y0 == 0 else min(margin, valid_h)
                wy1 = valid_h if ye == h else max(wy0, valid_h - min(margin, valid_h))
                if wy1 <= wy0:
                    continue
                for x0 in x_starts:
                    xe = min(x0 + tile_size, w)
                    valid_w = int(xe - x0)
                    wx0 = 0 if x0 == 0 else min(margin, valid_w)
                    wx1 = valid_w if xe == w else max(wx0, valid_w - min(margin, valid_w))
                    if wx1 <= wx0:
                        continue
                    tile_cpu = vol_cpu[:, :, :, y0:ye, x0:xe]
                    pad_h = max(0, tile_size - int(tile_cpu.shape[-2]))
                    pad_w = max(0, tile_size - int(tile_cpu.shape[-1]))
                    if pad_h or pad_w:
                        tile_cpu = F.pad(tile_cpu, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
                    fields_cpu = model.make_fields(tile_cpu) if hasattr(model, "make_fields") else tile_cpu
                    if log_floor_cpu is not None:
                        fields_cpu = torch.log(torch.clamp(fields_cpu, min=0.0) + log_floor_cpu)
                    fields = fields_cpu.to(device, non_blocking=True)
                    ctx_acc = None
                    gt_acc = None
                    pred_acc = None
                    view_iter = _iter_tta_views_2d(fields, inference_tta_mode) if bool(inference_tta_enabled) else (("id", fields),)
                    actual_views = 0
                    for view_name, fields_view in view_iter:
                        zero_mask_tokens = torch.zeros_like(fields_view)
                        context_view = model.context_encoder(fields_view, mask_tokens=zero_mask_tokens)
                        gt_view = model.target_encoder(fields_view, mask_tokens=zero_mask_tokens)
                        pred_view = model.predictor3d(context_view)
                        context_view = _apply_tta_2d(view_name, context_view)
                        gt_view = _apply_tta_2d(view_name, gt_view)
                        pred_view = _apply_tta_2d(view_name, pred_view)
                        if ctx_acc is None:
                            ctx_acc = context_view
                            gt_acc = gt_view
                            pred_acc = pred_view
                        else:
                            ctx_acc = ctx_acc + context_view
                            gt_acc = gt_acc + gt_view
                            pred_acc = pred_acc + pred_view
                        actual_views += 1
                        if view_name != "id":
                            del fields_view
                        del zero_mask_tokens, context_view, gt_view, pred_view
                    if actual_views <= 0 or ctx_acc is None or gt_acc is None or pred_acc is None:
                        raise RuntimeError(f"[{config_name}] 3D TTA produced no views")
                    inv_views = 1.0 / float(actual_views)
                    context_map_full = ctx_acc * inv_views
                    gt_map_full = gt_acc * inv_views
                    pred_map_full = pred_acc * inv_views
                    pred_tile = pred_map_full[:, :, center_idx : center_idx + 1, :valid_h, :valid_w].detach().cpu()
                    gt_tile = gt_map_full[:, :, center_idx : center_idx + 1, :valid_h, :valid_w].detach().cpu()
                    ctx_tile = context_map_full[:, :, center_idx : center_idx + 1, :valid_h, :valid_w].detach().cpu()
                    del fields, ctx_acc, gt_acc, pred_acc, context_map_full, gt_map_full, pred_map_full
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if pred_sum is None:
                        pred_sum = torch.zeros((1, int(pred_tile.shape[1]), 1, h, w), dtype=torch.float32)
                        gt_sum = torch.zeros((1, int(gt_tile.shape[1]), 1, h, w), dtype=torch.float32)
                        ctx_sum = torch.zeros((1, int(ctx_tile.shape[1]), 1, h, w), dtype=torch.float32)
                    out_y0, out_y1 = y0 + wy0, y0 + wy1
                    out_x0, out_x1 = x0 + wx0, x0 + wx1
                    pred_sum[:, :, :, out_y0:out_y1, out_x0:out_x1] += pred_tile[:, :, :, wy0:wy1, wx0:wx1]
                    gt_sum[:, :, :, out_y0:out_y1, out_x0:out_x1] += gt_tile[:, :, :, wy0:wy1, wx0:wx1]
                    ctx_sum[:, :, :, out_y0:out_y1, out_x0:out_x1] += ctx_tile[:, :, :, wy0:wy1, wx0:wx1]
                    weight[:, :, :, out_y0:out_y1, out_x0:out_x1] += 1.0
            if pred_sum is None or gt_sum is None or ctx_sum is None:
                raise RuntimeError(f"[{config_name}] 3D slice tiled inference produced no tiles")
            pred_map = pred_sum / weight.clamp_min(1.0)
            gt_map = gt_sum / weight.clamp_min(1.0)
            context_map = ctx_sum / weight.clamp_min(1.0)
            x_clean = vol_cpu.sum(dim=1, keepdim=True)[:, :, center_idx : center_idx + 1]
            x_context = x_clean
            mask_cube = torch.zeros((1, 1, 1, h, w), dtype=x_clean.dtype)
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.long),
                torch.arange(w, dtype=torch.long),
                indexing="ij",
            )
            zz = torch.zeros_like(yy)
            target_locations = torch.stack([zz, yy, xx], dim=-1).reshape(1, h * w, 3)
            target_valid = torch.ones((1, h * w), dtype=torch.bool)
            target_scales = torch.ones((1, h * w), dtype=x_clean.dtype)
            pred_patches = pred_map[:, :, :, h // 2 : h // 2 + 1, w // 2 : w // 2 + 1].unsqueeze(1)
            gt_patches = gt_map[:, :, :, h // 2 : h // 2 + 1, w // 2 : w // 2 + 1].unsqueeze(1)
            outputs = {
                "pred_map": pred_map,
                "gt_map": gt_map,
                "context_map": context_map,
                "x_clean": x_clean,
                "x_context": x_context,
                "mask_cube": mask_cube,
                "target_locations": target_locations,
                "target_valid": target_valid,
                "target_scales": target_scales,
                "pred_patches": pred_patches,
                "gt_patches": gt_patches,
                "selected_slab_start_index": torch.tensor([src0], dtype=torch.long),
                "selected_slab_depth": torch.tensor([slab_depth], dtype=torch.long),
                "_inference_3d_full_xy_slice": True,
                "_inference_slice_index": z,
                "_inference_input_slab_depth": slab_depth,
                "_inference_spatial_tile_size": tile_size,
                "_inference_spatial_overlap": overlap_eff,
                "_inference_discard_margin": margin,
                "_inference_tta_enabled": bool(inference_tta_enabled),
                "_inference_tta_mode": str(inference_tta_mode),
                "_inference_tta_views": int(tta_view_count),
                "_inference_source_path": path,
            }
        else:
            x = next(iter(dataloader))
            x = x.to(next(model.parameters()).device)
            outputs = model(x)

    pred_map = outputs["pred_map"][:1].detach().cpu()
    gt_map = outputs["gt_map"][:1].detach().cpu()
    context_map = outputs["context_map"][:1].detach().cpu()
    x_clean = outputs["x_clean"][:1].detach().cpu()
    x_context = outputs["x_context"][:1].detach().cpu()
    mask_cube = outputs["mask_cube"][:1].detach().cpu()
    if pred_map.dim() != 5:
        raise ValueError(f"Expected center-slab pred_map BxCxDxHxW, got {tuple(pred_map.shape)}")
    center_slab_middle_index = int(pred_map.shape[2] // 2)
    slab_energy = representation_dense_energy(pred_map, gt_map)

    inference_outputs = {
        "x_clean": x_clean,
        "x_context": x_context,
        "mask_cube": mask_cube,
        "pred_map": pred_map,
        "gt_map": gt_map,
        "context_map": context_map,
        "target_locations": outputs["target_locations"][:1].detach().cpu(),
        "target_valid": outputs["target_valid"][:1].detach().cpu(),
        "target_scales": outputs.get("target_scales", torch.ones_like(outputs["target_valid"], dtype=x_clean.dtype))[:1].detach().cpu(),
        "pred_patches": outputs["pred_patches"][:1].detach().cpu(),
        "gt_patches": outputs["gt_patches"][:1].detach().cpu(),
        "target_energy_map": slab_energy["energy_rel_sym"],
        "target_energy_raw_map": slab_energy["energy_raw"],
        "target_energy_rel_gt_map": slab_energy["energy_rel_gt"],
        "target_energy_cosine_map": slab_energy["energy_cosine"],
        "center_slab_middle_index": torch.tensor(center_slab_middle_index, dtype=torch.int64),
    }
    if "target_mask_map" in outputs:
        inference_outputs["target_mask_map"] = outputs["target_mask_map"][:1].detach().cpu()
    inference_outputs["selected_slab_start_index"] = outputs["selected_slab_start_index"][:1].detach().cpu()
    inference_outputs["selected_slab_depth"] = outputs["selected_slab_depth"][:1].detach().cpu()
    inference_outputs["inference_3d_full_xy_slice"] = bool(outputs.get("_inference_3d_full_xy_slice", False))

    # Encoder FOV border: true/auto → half-FOV; number → that many px; false/0 → skip.
    if isinstance(inference_mask_border, bool):
        fov_border_3d = (int(getattr(model, "encoder_receptive_field_depth", 1) or 1) // 2) if inference_mask_border else 0
    else:
        fov_border_3d = int(max(0, float(inference_mask_border)))
    if fov_border_3d > 0:
        for key in ("pred_map", "gt_map", "context_map"):
            if key in inference_outputs and inference_outputs[key].dim() == 5 and inference_outputs[key].shape[2] == 1:
                inference_outputs[key] = _apply_nan_boundary_frame(
                    inference_outputs[key].squeeze(2), fov_border_3d
                ).unsqueeze(2)

    if fov_border_3d > 0:
        for key in ("target_energy_map", "target_energy_raw_map",
                    "target_energy_rel_gt_map", "target_energy_cosine_map"):
            if key in inference_outputs:
                t = inference_outputs[key]
                if t.dim() == 5 and t.shape[2] == 1:
                    t = _apply_nan_boundary_frame(t.squeeze(2), fov_border_3d).unsqueeze(2)
                elif t.dim() == 4:
                    t = _apply_nan_boundary_frame(t, fov_border_3d)
                inference_outputs[key] = t

    torch.save(inference_outputs, inference_outputs_path)

    _save_npz(os.path.join(session_dir, "network_input_clean_3d.npz"), x_clean.numpy())
    _save_npz(os.path.join(session_dir, "network_input_context_3d.npz"), x_context.numpy())
    _save_npz(os.path.join(session_dir, "mask_cube_3d.npz"), mask_cube.numpy())
    _save_npz(os.path.join(session_dir, "pred_map_3d.npz"), pred_map.numpy())
    _save_npz(os.path.join(session_dir, "gt_map_3d.npz"), gt_map.numpy())
    _save_npz(os.path.join(session_dir, "context_map_3d.npz"), context_map.numpy())
    _save_npz(os.path.join(session_dir, "target_energy_map_slab.npz"), slab_energy["energy_rel_sym"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_raw_map_slab.npz"), slab_energy["energy_raw"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_rel_gt_map_slab.npz"), slab_energy["energy_rel_gt"].numpy())
    _save_npz(os.path.join(session_dir, "target_energy_cosine_map_slab.npz"), slab_energy["energy_cosine"].numpy())
    with open(os.path.join(session_dir, "jepa_energy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "inference_3d_full_xy_slice": bool(outputs.get("_inference_3d_full_xy_slice", False)),
                "inference_slice_index": outputs.get("_inference_slice_index"),
                "inference_input_slab_depth": outputs.get("_inference_input_slab_depth"),
                "inference_spatial_tile_size": outputs.get("_inference_spatial_tile_size"),
                "inference_spatial_overlap": outputs.get("_inference_spatial_overlap"),
                "inference_encoder_receptive_field": int(getattr(model, "encoder_receptive_field_depth", 1) or 1),
                "inference_discard_margin": outputs.get("_inference_discard_margin"),
                "inference_tta_enabled": bool(outputs.get("_inference_tta_enabled", False)),
                "inference_tta_mode": outputs.get("_inference_tta_mode", "off"),
                "inference_tta_views": int(outputs.get("_inference_tta_views", 1)),
                "inference_source_path": outputs.get("_inference_source_path"),
                "inference_output_shape": list(pred_map.shape),
            },
            f,
            indent=2,
        )
    print(f"[{config_name}] saved 3D inference artifacts")
    return session_dir
