from __future__ import annotations

import json
import os
import csv
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from src.losses import representation_dense_energy


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
    loc_y = loc[..., 0].long()
    loc_x = loc[..., 1].long()
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


def _first_full_resolution_batch(dataloader, max_diagnostic_size: int = 768):
    """Fetch a single batch for diagnostic visualization.

    Enforces a max size to prevent GPU OOM and 100MB+ HTML crashes
    on very large images (e.g. 2000×2000+).
    """
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__getitem__"):
        return None
    old = {}

    # Check original image size to see if we need to enforce a crop
    needs_safety_crop = False
    try:
        sample_path = dataset.sample_index[0][0]
        shape = dataset._probe_file_shape(sample_path)
        if shape[-2] > max_diagnostic_size or shape[-1] > max_diagnostic_size:
            needs_safety_crop = True
    except Exception:
        pass

    target_crop_mode = "center" if needs_safety_crop else "none"
    target_crop_size = max_diagnostic_size if needs_safety_crop else None

    for name, value in (
        ("crop_mode", target_crop_mode),
        ("crop_size", target_crop_size),
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
    viz_crop_border: bool,
    viz_crop_border_px: int | None,
    compute_jepa_energy_fn: Callable,
    compute_target_energy_map_fn: Callable,
    inference_visit_batches: int = 32,
    training_d4_augment: bool = False,
    inference_tta_enabled: bool = False,
    inference_tta_mode: str = "flip4",
    max_diagnostic_size: int = 768,
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
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
        raw_batch = _first_full_resolution_batch(dataloader, max_diagnostic_size=max_diagnostic_size)
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
        if bool(mask_inference):
            raise ValueError(
                "Post-training lattice-sweep mask inference has been removed. "
                "Run post-training inference with mask_inference=false and use "
                "the dense prediction/target maps for energy diagnostics."
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
    # Dense full-image energy from lattice prediction/target maps.
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

    if bool(viz_crop_border):
        if viz_crop_border_px is None:
            auto_border = int(max(getattr(model, "sigmas", (16.0,))))
        else:
            auto_border = int(max(0, viz_crop_border_px))
        inference_outputs["target_energy_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_map"], auto_border
        )
        inference_outputs["target_energy_raw_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_raw_map"], auto_border
        )
        inference_outputs["target_energy_rel_gt_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_rel_gt_map"], auto_border
        )
        inference_outputs["target_energy_cosine_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_cosine_map"], auto_border
        )
        inference_outputs["target_energy_point_map"] = _apply_nan_boundary_frame(
            inference_outputs["target_energy_point_map"], auto_border
        )

    if "target_mask_map" in inference_outputs:
        tmap = inference_outputs["target_mask_map"]
    else:
        # Fallback map should represent target centers as points, not squares.
        tloc = inference_outputs["target_locations"]
        tvalid = inference_outputs["target_valid"]
        bsz, _, _ = tloc.shape
        h, w = inference_outputs["x_clean"].shape[-2:]
        tmap = torch.zeros((bsz, 1, h, w), dtype=inference_outputs["x_clean"].dtype)
        for bi in range(bsz):
            for ki in range(tloc.shape[1]):
                if not bool(tvalid[bi, ki].item()):
                    continue
                cy = int(tloc[bi, ki, 0].item())
                cx = int(tloc[bi, ki, 1].item())
                if 0 <= cy < h and 0 <= cx < w:
                    # 3×3 cross so targets are visible when downscaled
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = cy + dy, cx + dx
                            if 0 <= yy < h and 0 <= xx < w:
                                tmap[bi, 0, yy, xx] = 1.0
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
    if "cdd_channels_orig" in inference_outputs:
        _save_npz(os.path.join(session_dir, "cdd_channels_orig.npz"), inference_outputs["cdd_channels_orig"].numpy())
    if "cdd_channels_masked" in inference_outputs:
        _save_npz(os.path.join(session_dir, "cdd_channels_masked.npz"), inference_outputs["cdd_channels_masked"].numpy())
        # Requested artifact: one example masked channel cube for quick inspection.
        _save_npz(
            os.path.join(session_dir, "example_masked_channel_cube.npz"),
            inference_outputs["cdd_channels_masked"][0].numpy().astype(np.float32),
        )
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
    # Canonical target-visit heatmap from inference loader (d4_augment is forced off there).
    # This avoids mirrored-symmetry artefacts from training-time augmentation logs.
    visit_h = int(inference_outputs["x_clean"].shape[-2])
    visit_w = int(inference_outputs["x_clean"].shape[-1])
    canonical_visit_counts = np.zeros((visit_h, visit_w), dtype=np.float32)
    canonical_rows = []
    max_visit_batches = int(inference_visit_batches)
    if max_visit_batches < 0:
        max_visit_batches = 0
    dev = next(model.parameters()).device
    with torch.no_grad():
        for ib, batch in enumerate(dataloader):
            if max_visit_batches > 0 and ib >= max_visit_batches:
                break
            if isinstance(batch, (tuple, list)) and len(batch) == 2 and batch[1] is not None:
                cdd_b, xb = batch
                cdd_b = cdd_b.to(dev)
            else:
                cdd_b = None
                xb = batch if not isinstance(batch, (tuple, list)) else batch[0]
            xb = xb.to(dev)
            outb = model(
                xb,
                return_debug=False,
                enable_grid_jitter=False,
                mask_inference=bool(mask_inference),
                cdd_orig=cdd_b,
            )
            tloc_b = outb["target_locations"].detach().cpu().numpy()
            tvalid_b = outb["target_valid"].detach().cpu().numpy().astype(bool)
            tscale_b = outb["target_scales"].detach().cpu().numpy()
            for bi in range(tloc_b.shape[0]):
                for ki in range(tloc_b.shape[1]):
                    if not bool(tvalid_b[bi, ki]):
                        continue
                    yy = int(tloc_b[bi, ki, 0])
                    xx = int(tloc_b[bi, ki, 1])
                    if 0 <= yy < visit_h and 0 <= xx < visit_w:
                        canonical_visit_counts[yy, xx] += 1.0
                        canonical_rows.append(
                            [int(ib), int(bi), int(ki), int(yy), int(xx), float(tscale_b[bi, ki])]
                        )
    _save_npz(
        os.path.join(session_dir, "visited_target_frequency_canonical.npz"),
        canonical_visit_counts.astype(np.float32),
    )
    with open(
        os.path.join(session_dir, "visited_target_locations_canonical.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["inference_batch", "sample_idx", "target_idx", "y", "x", "scale"])
        if canonical_rows:
            w.writerows(canonical_rows)
    if bool(training_d4_augment):
        print(
            f"[{config_name}] canonical_visit_map_saved "
            "(built from d4_augment=false inference loader)"
        )
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
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
    if (not force_recompute_inference) and os.path.exists(inference_outputs_path):
        return session_dir

    model.eval()
    with torch.no_grad():
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
    inference_outputs["selected_slab_start_index"] = outputs["selected_slab_start_index"][:1].detach().cpu()
    inference_outputs["selected_slab_depth"] = outputs["selected_slab_depth"][:1].detach().cpu()
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
    print(f"[{config_name}] saved 3D inference artifacts")
    return session_dir


def run_full_volume_inference_3d(
    *,
    model,
    cdd_cache: dict,
    session_dir: str,
    config_name: str,
    device: torch.device,
    slab_depth: int = 8,
    overlap: int = 4,
    post_log_transform: bool = True,
    log_eps: float = 1.0,
    cdd_log_std_floor_mult: float = 0.05,
) -> str:
    """Run encoder on the full precomputed CDD volume and save context map.

    Slides the encoder over the depth axis in overlapping slabs, stitches
    results by averaging in the overlap region. Saves the full (C, D, H, W)
    context_map_3d as .npz.
    """
    if not cdd_cache:
        print(f"[{config_name}] full-volume inference: no CDD cache, skipping")
        return session_dir

    model.eval()
    with torch.no_grad():
        for (path, _), cdd_vol in cdd_cache.items():
            # cdd_vol: (S, D, H, W) numpy float32
            s, d, h, w = cdd_vol.shape
            vol_t = torch.from_numpy(cdd_vol).unsqueeze(0).to(device)  # (1, S, D, H, W)

            # Apply post_log_transform to match training input distribution
            if post_log_transform:
                eps_f = max(1e-6, float(log_eps))
                vol_clamp = vol_t.clamp(min=0.0)
                base_std = torch.std(vol_clamp, dim=(-3, -2, -1), keepdim=True)
                log_floor = torch.clamp(base_std * float(cdd_log_std_floor_mult), min=eps_f)
                vol_t = torch.log(vol_clamp + log_floor)

            full_volume_training = bool(getattr(model, "full_volume_training", False))
            target_depth = d if full_volume_training else max(1, min(int(getattr(model, "slab_depth", slab_depth)), slab_depth))
            context_margin = 0 if full_volume_training else max(0, (slab_depth - target_depth) // 2)
            target_overlap = min(max(0, int(overlap)), max(0, target_depth - 1))
            step = max(1, target_depth - target_overlap)
            starts = list(range(0, max(1, d - slab_depth + 1), step))
            tail_start = max(0, d - slab_depth)
            if not starts or starts[-1] != tail_start:
                starts.append(tail_start)

            ctx_sum = None
            ctx_weight = torch.zeros((1, 1, d, 1, 1), device=device)

            for z0 in starts:
                ze = min(z0 + slab_depth, d)
                slab = vol_t[:, :, z0:ze]  # (1, S, slab, H, W)
                valid_depth = int(ze - z0)
                if slab.shape[2] < slab_depth:
                    pad_needed = slab_depth - slab.shape[2]
                    pad_mode = "reflect" if slab.shape[2] > 1 else "replicate"
                    slab = F.pad(slab, (0, 0, 0, 0, 0, pad_needed), mode=pad_mode)

                mask_tokens = torch.zeros_like(slab)
                ctx = model.context_encoder(slab, mask_tokens=mask_tokens)[:, :, :valid_depth]
                if ctx_sum is None:
                    ctx_sum = torch.zeros((1, int(ctx.shape[1]), d, int(ctx.shape[3]), int(ctx.shape[4])), device=device)
                write_rel0 = 0 if z0 == 0 else min(context_margin, valid_depth)
                write_rel1 = valid_depth if ze == d else max(write_rel0, min(valid_depth, slab_depth - context_margin))
                if write_rel1 <= write_rel0:
                    continue
                out_z0 = z0 + write_rel0
                out_z1 = z0 + write_rel1
                ctx_sum[:, :, out_z0:out_z1] += ctx[:, :, write_rel0:write_rel1]
                ctx_weight[:, :, out_z0:out_z1] += 1.0

            if ctx_sum is None:
                raise RuntimeError(f"[{config_name}] full-volume inference produced no slabs for {path}")
            ctx_avg = ctx_sum / ctx_weight.clamp_min(1.0)

            name = os.path.basename(path).replace('.npy', '')
            out_path = os.path.join(session_dir, f"{name}_context_map_3d.npz")
            _save_npz(out_path, ctx_avg.squeeze(0).cpu().numpy())
            print(f"[{config_name}] full-volume context map saved: {out_path}")

    return session_dir
