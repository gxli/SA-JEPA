from __future__ import annotations

import json
import os
import csv
from typing import Callable

import numpy as np
import torch


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
    # Avoid per-target GPU sync from repeated .item() calls.
    loc_cpu = loc.detach().to("cpu").numpy()
    valid_cpu = valid.detach().to("cpu").numpy()
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
    n_masked = 0
    for bi in range(bsz):
        for ki in range(ksz):
            if not bool(valid_cpu[bi, ki]):
                continue
            cy = int(loc_cpu[bi, ki, 0])
            cx = int(loc_cpu[bi, ki, 1])
            y0 = cy - half_lo
            y1 = cy + half_hi
            x0 = cx - half_lo
            x1 = cx + half_hi
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                updated[bi, ki] = False
                n_masked += 1
                continue
            patch = x_input[bi, 0, y0:y1, x0:x1]
            invalid_mask = torch.zeros_like(patch, dtype=torch.bool)
            if check_nan:
                invalid_mask |= ~torch.isfinite(patch)
            for spec in numeric_specs:
                invalid_mask |= torch.isclose(patch, torch.tensor(spec, device=patch.device, dtype=patch.dtype))
            if bool(torch.all(invalid_mask).item()):
                updated[bi, ki] = False
                n_masked += 1
    outputs["target_valid"] = updated
    return int(n_masked)


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
    err = (pred - gt).pow(2).mean(dim=(2, 3, 4))  # B,K
    # Avoid per-target GPU sync from repeated .item() calls.
    loc_cpu = loc.detach().to("cpu").numpy()
    valid_cpu = valid.detach().to("cpu").numpy()
    err_cpu = err.detach().to("cpu").numpy()
    total = 0.0
    n_valid = 0
    bsz, ksz = err.shape
    for bi in range(bsz):
        for ki in range(ksz):
            if not bool(valid_cpu[bi, ki]):
                continue
            cy = int(loc_cpu[bi, ki, 0])
            cx = int(loc_cpu[bi, ki, 1])
            if 0 <= cy < h and 0 <= cx < w:
                v = float(err_cpu[bi, ki])
                energy_sum[bi, 0, cy, cx] += v
                count_map[bi, 0, cy, cx] += 1.0
                total += v
                n_valid += 1
    return total, n_valid


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
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
    dashboard_html_path = os.path.join(session_dir, "dashboard.html")
    if (not force_recompute_inference) and os.path.exists(dashboard_html_path):
        print(
            f"[{config_name}] dashboard already exists; "
            "skipping post-training inference (set train.force_recompute_inference=true to recompute)"
        )
        return session_dir

    inference_required = [
        inference_outputs_path,
        os.path.join(session_dir, "network_input_clean.npy"),
        os.path.join(session_dir, "network_input_context.npy"),
        os.path.join(session_dir, "pred_map.npy"),
        os.path.join(session_dir, "gt_map.npy"),
        os.path.join(session_dir, "target_energy_map.npy"),
        os.path.join(session_dir, "jepa_energy_summary.json"),
    ]
    if (not force_recompute_inference) and all(os.path.exists(p) for p in inference_required):
        print(
            f"[{config_name}] inference artifacts already exist; "
            "skipping post-training inference (set train.force_recompute_inference=true to recompute)"
        )
        return session_dir

    print(f"[{config_name}] post_training_inference begin")
    model.eval()
    with torch.no_grad():
        print(f"[{config_name}] post_training_inference loading sample batch")
        x_raw = next(iter(dataloader))
        x_raw = x_raw.to(next(model.parameters()).device)
        # Deterministic lattice sweep is only meaningful when mask inference is enabled.
        largest_sigma = float(max(getattr(model, "sigmas", (16.0,))))
        mask_scale = float(getattr(model, "mask_scale", 1.0))
        mask_box_size = int(getattr(model, "mask_box_size", 16))
        max_box = round(largest_sigma * mask_scale + mask_box_size)
        spacing = int(
            max(
                1,
                round(float(max_box) * float(getattr(model, "spacing_scale", 1.5))),
            )
        )
        if bool(mask_inference):
            # TODO(cleanup): lattice sweep is a workaround for Gaussian masking's
            # inability to produce a dense error map in a single pass. Replace
            # with discrete block masking (MAE-style) so one forward pass is
            # enough. Remove this sweep once block masking is validated.
            import warnings
            warnings.warn(
                "Lattice sweep inference is deprecated and will be removed. "
                "Switch to block masking for single-pass dense energy maps.",
                FutureWarning,
                stacklevel=2,
            )
            all_shifts = [(dy, dx) for dy in range(spacing) for dx in range(spacing)]
            n_passes = max(1, int(inference_mask_passes))
            shifts = all_shifts if n_passes <= 0 else all_shifts[: min(len(all_shifts), n_passes)]
        else:
            shifts = [(0, 0)]
        print(f"[{config_name}] post_training_inference model forward deterministic_shifts={len(shifts)} spacing={spacing}")
        outputs = None
        energy_sum = None
        count_map = None
        total_energy = 0.0
        total_valid = 0
        invalid_region_skip = bool(getattr(model, "target_invalid_region_skip", False))
        invalid_region_values = tuple(getattr(model, "target_invalid_region_values", (0.0, "nan")))
        patch_size = int(getattr(model, "patch_size", 2))
        for pi, shift in enumerate(shifts):
            out_i = model(
                x_raw,
                return_debug=(pi == 0),
                forced_grid_shift=shift,
                enable_grid_jitter=False,
                mask_inference=bool(mask_inference),
            )
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

    inference_outputs = {
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
    if "target_mask_map" in outputs:
        inference_outputs["target_mask_map"] = outputs["target_mask_map"][:8].detach().cpu()
    if "cdd_channels_orig" in outputs:
        inference_outputs["cdd_channels_orig"] = outputs["cdd_channels_orig"][:8].detach().cpu()
    if "cdd_channels_masked" in outputs:
        inference_outputs["cdd_channels_masked"] = outputs["cdd_channels_masked"][:8].detach().cpu()
    if "pyramid_mask_token" in outputs:
        inference_outputs["pyramid_mask_token"] = outputs["pyramid_mask_token"][:8].detach().cpu()
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
    inference_outputs["target_energy_map"] = e_map_dense[:8].detach().cpu()
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
                    tmap[bi, 0, cy, cx] = 1.0
    inference_outputs["target_map"] = tmap
    torch.save(inference_outputs, inference_outputs_path)
    print(f"[{config_name}] saved inference_outputs.pt")

    np.save(os.path.join(session_dir, "network_input_clean.npy"), inference_outputs["x_clean"].numpy())
    np.save(os.path.join(session_dir, "network_input_context.npy"), inference_outputs["x_context"].numpy())
    np.save(os.path.join(session_dir, "network_input_clean_raw.npy"), inference_outputs["x_clean_raw"].numpy())
    np.save(os.path.join(session_dir, "network_input_context_raw.npy"), inference_outputs["x_context_raw"].numpy())
    np.save(os.path.join(session_dir, "target_valid.npy"), inference_outputs["target_valid"].numpy())
    if "target_mask_map" in inference_outputs:
        np.save(os.path.join(session_dir, "target_mask_map.npy"), inference_outputs["target_mask_map"].numpy())
    if "cdd_channels_orig" in inference_outputs:
        np.save(os.path.join(session_dir, "cdd_channels_orig.npy"), inference_outputs["cdd_channels_orig"].numpy())
    if "cdd_channels_masked" in inference_outputs:
        np.save(os.path.join(session_dir, "cdd_channels_masked.npy"), inference_outputs["cdd_channels_masked"].numpy())
        # Requested artifact: one example masked channel cube for quick inspection.
        np.save(
            os.path.join(session_dir, "example_masked_channel_cube.npy"),
            inference_outputs["cdd_channels_masked"][0].numpy().astype(np.float32),
        )
    if "pyramid_mask_token" in inference_outputs:
        np.save(os.path.join(session_dir, "pyramid_mask_token.npy"), inference_outputs["pyramid_mask_token"].numpy())
    if visit_counts is not None:
        np.save(os.path.join(session_dir, "visited_target_frequency.npy"), visit_counts.astype(np.float32))
    np.save(os.path.join(session_dir, "target_energy_map.npy"), inference_outputs["target_energy_map"].numpy())
    np.save(os.path.join(session_dir, "target_energy_point_map.npy"), inference_outputs["target_energy_point_map"].numpy())
    np.save(os.path.join(session_dir, "target_energy_count_map.npy"), inference_outputs["target_energy_count_map"].numpy())
    with open(os.path.join(session_dir, "jepa_energy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "jepa_energy": float(energy_scalar),
                "jepa_energy_normalized": float(energy_scalar_norm),
                "inference_mask_passes": int(len(shifts)),
                "inference_grid_spacing": int(spacing),
                "mask_inference": bool(mask_inference),
            },
            f,
            indent=2,
        )
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
        for ib, xb in enumerate(dataloader):
            if max_visit_batches > 0 and ib >= max_visit_batches:
                break
            xb = xb.to(dev)
            outb = model(
                xb,
                return_debug=False,
                enable_grid_jitter=False,
                mask_inference=bool(mask_inference),
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
    np.save(
        os.path.join(session_dir, "visited_target_frequency_canonical.npy"),
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
    np.save(os.path.join(session_dir, "pred_map.npy"), inference_outputs["pred_map"].numpy())
    np.save(os.path.join(session_dir, "gt_map.npy"), inference_outputs["gt_map"].numpy())
    pred_norm = inference_outputs["pred_map"].norm(dim=1).numpy()
    gt_norm = inference_outputs["gt_map"].norm(dim=1).numpy()
    err_norm = (inference_outputs["pred_map"] - inference_outputs["gt_map"]).norm(dim=1).numpy()
    np.save(os.path.join(session_dir, "pred_latent_norm.npy"), pred_norm)
    np.save(os.path.join(session_dir, "gt_latent_norm.npy"), gt_norm)
    np.save(os.path.join(session_dir, "pred_gt_latent_error_norm.npy"), err_norm)
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

    mid = int(pred_map.shape[2] // 2)
    energy_map_mid = (pred_map[:, :, mid] - gt_map[:, :, mid]).pow(2).mean(dim=1, keepdim=True)

    inference_outputs = {
        "x_clean": x_clean,
        "x_context": x_clean,
        "pred_map": pred_map,
        "gt_map": gt_map,
        "context_map": context_map,
        "target_locations": outputs["target_locations"][:1].detach().cpu(),
        "target_valid": outputs["target_valid"][:1].detach().cpu(),
        "target_scales": outputs.get("target_scales", torch.ones_like(outputs["target_valid"], dtype=x_clean.dtype))[:1].detach().cpu(),
        "pred_patches": outputs["pred_patches"][:1].detach().cpu(),
        "gt_patches": outputs["gt_patches"][:1].detach().cpu(),
        "target_energy_map": energy_map_mid,
        "middle_slice_index": torch.tensor(mid, dtype=torch.int64),
    }
    torch.save(inference_outputs, inference_outputs_path)

    np.save(os.path.join(session_dir, "network_input_clean_3d.npy"), x_clean.numpy())
    np.save(os.path.join(session_dir, "pred_map_3d.npy"), pred_map.numpy())
    np.save(os.path.join(session_dir, "gt_map_3d.npy"), gt_map.numpy())
    np.save(os.path.join(session_dir, "context_map_3d.npy"), context_map.numpy())
    np.save(os.path.join(session_dir, "target_energy_map_mid_slice.npy"), energy_map_mid.numpy())
    print(f"[{config_name}] saved 3D inference artifacts")
    return session_dir
