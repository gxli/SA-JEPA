from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np
import torch


def run_post_training_inference(
    *,
    model,
    dataloader,
    session_dir: str,
    config_name: str,
    visit_counts,
    force_recompute_inference: bool,
    compute_jepa_energy_fn: Callable,
    compute_target_energy_map_fn: Callable,
) -> str:
    inference_outputs_path = os.path.join(session_dir, "inference_outputs.pt")
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
        print(f"[{config_name}] post_training_inference model forward")
        outputs = model(x_raw)

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
    energy_scalar = compute_jepa_energy_fn(outputs, normalize=False)
    energy_scalar_norm = compute_jepa_energy_fn(outputs, normalize=True)
    e_map = compute_target_energy_map_fn(outputs, image_size=outputs["x_clean"].shape[-2:])
    inference_outputs["jepa_energy"] = torch.tensor(energy_scalar, dtype=torch.float32)
    inference_outputs["jepa_energy_normalized"] = torch.tensor(energy_scalar_norm, dtype=torch.float32)
    inference_outputs["target_energy_map"] = e_map[:8].detach().cpu()

    tloc = inference_outputs["target_locations"]
    tvalid = inference_outputs["target_valid"]
    patch_size = int(inference_outputs["pred_patches"].shape[-1])
    half = patch_size // 2
    bsz, _, _ = tloc.shape
    h, w = inference_outputs["x_clean"].shape[-2:]
    tmap = torch.zeros((bsz, 1, h, w), dtype=inference_outputs["x_clean"].dtype)
    for bi in range(bsz):
        for ki in range(tloc.shape[1]):
            if not bool(tvalid[bi, ki].item()):
                continue
            cy = int(tloc[bi, ki, 0].item())
            cx = int(tloc[bi, ki, 1].item())
            y0 = cy - half
            y1 = cy + half
            x0 = cx - half
            x1 = cx + half
            if y0 < 0:
                y0 = 0
                y1 = patch_size
            if x0 < 0:
                x0 = 0
                x1 = patch_size
            if y1 > h:
                y1 = h
                y0 = h - patch_size
            if x1 > w:
                x1 = w
                x0 = w - patch_size
            tmap[bi, 0, y0:y1, x0:x1] = 1.0
    inference_outputs["target_map"] = tmap
    torch.save(inference_outputs, inference_outputs_path)
    print(f"[{config_name}] saved inference_outputs.pt")

    np.save(os.path.join(session_dir, "network_input_clean.npy"), inference_outputs["x_clean"].numpy())
    np.save(os.path.join(session_dir, "network_input_context.npy"), inference_outputs["x_context"].numpy())
    np.save(os.path.join(session_dir, "network_input_clean_raw.npy"), inference_outputs["x_clean_raw"].numpy())
    np.save(os.path.join(session_dir, "network_input_context_raw.npy"), inference_outputs["x_context_raw"].numpy())
    np.save(os.path.join(session_dir, "target_valid.npy"), inference_outputs["target_valid"].numpy())
    if visit_counts is not None:
        np.save(os.path.join(session_dir, "visited_target_frequency.npy"), visit_counts.astype(np.float32))
    np.save(os.path.join(session_dir, "target_energy_map.npy"), inference_outputs["target_energy_map"].numpy())
    with open(os.path.join(session_dir, "jepa_energy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "jepa_energy": float(energy_scalar),
                "jepa_energy_normalized": float(energy_scalar_norm),
            },
            f,
            indent=2,
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

