from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.cdd_import import import_constrained_diffusion, safe_constrained_diffusion_decomposition


def _get_context_encoder(model):
    if hasattr(model, "context_encoder"):
        return model.context_encoder
    if hasattr(model, "encoder"):
        return model.encoder
    raise AttributeError("Model has neither context_encoder nor encoder")


def _encode_context(model, x_pyr, mask_tokens=None):
    encoder = _get_context_encoder(model)
    if mask_tokens is None:
        try:
            return encoder(x_pyr)
        except TypeError:
            return encoder(x_pyr, mask_tokens=torch.zeros_like(x_pyr))
    try:
        return encoder(x_pyr, mask_tokens=mask_tokens)
    except TypeError:
        try:
            return encoder(x_pyr, mask_tokens)
        except TypeError:
            return encoder(torch.cat([x_pyr, mask_tokens], dim=1))


def _encode_direct_context(model, x: torch.Tensor) -> torch.Tensor:
    encoder = _get_context_encoder(model)
    return encoder(x)


@torch.no_grad()
def build_cdd_reconstructed_image_variants(
    reference_input: torch.Tensor,
    *,
    source_image: torch.Tensor | None = None,
    sigmas: Sequence[float],
    cdd_mode: str = "log",
    cdd_constrained: bool = True,
    cdd_sm_mode: str = "reflect",
    cdd_gaussian_backend: str = "cuda",
    cdd_append_last_residual: bool = True,
    post_log_transform: bool = True,
    log_eps: float = 1.0,
    cdd_log_std_floor_mult: float = 0.05,
    perturb_channel: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Build image-mode probe variants by CDD editing then reconstructing.

    For image encoders, the model sees a regular image tensor, not CDD channels.
    This helper decomposes the raw image, builds both drop-scale and scale-only
    reconstructed images, applies the same post-log transform used by the model,
    and returns two ``B,V,C,H,W`` variant stacks with all other network-input
    channels preserved.
    """

    if reference_input.ndim != 4:
        raise ValueError(f"Expected reference_input B,C,H,W, got {tuple(reference_input.shape)}")
    if not sigmas:
        raise ValueError("sigmas must be non-empty for CDD reconstructed image variants")
    B, C, H, W = reference_input.shape
    ch = int(perturb_channel)
    if ch < 0 or ch >= C:
        raise ValueError(f"perturb_channel={ch} outside C={C}")
    if source_image is None:
        raise ValueError(
            "source_image is required for image CDD scale probe. "
            "Refusing to decompose reference_input because it may already be post-log/network input."
        )
    if source_image.ndim == 3:
        source_image = source_image.unsqueeze(1)
    if source_image.ndim != 4:
        raise ValueError(f"Expected source_image B,C,H,W or B,H,W, got {tuple(source_image.shape)}")
    if source_image.shape[0] != B or source_image.shape[-2:] != (H, W):
        raise ValueError(
            f"source_image must match reference batch/spatial shape; "
            f"reference={tuple(reference_input.shape)} source={tuple(source_image.shape)}"
        )
    source_ch = min(ch, int(source_image.shape[1]) - 1)
    source_check = source_image.detach().float()
    if not torch.isfinite(source_check).all():
        raise ValueError("source_image contains non-finite values; expected raw normalized image")
    src_min = float(source_check.amin().item())
    src_max = float(source_check.amax().item())
    if src_min < -1e-5 or src_max > 1.0 + 1e-4:
        raise ValueError(
            f"source_image does not look like raw normalized 0-1 data "
            f"(min={src_min:.6g}, max={src_max:.6g})"
        )

    cdd = import_constrained_diffusion(allow_monai=False)
    cdd_num_channels = int(len(sigmas))
    variant_names = [f"drop_cdd_sigma_{float(s):g}" for s in sigmas]
    drop_variants = []
    only_variants = []
    device = reference_input.device
    dtype = reference_input.dtype
    ref_cpu = reference_input.detach().float().cpu()
    source_cpu = source_image.detach().float().cpu()

    def _encode_like_model(recon: np.ndarray, raw_source: np.ndarray) -> np.ndarray:
        recon = np.asarray(recon, dtype=np.float32)
        if not bool(post_log_transform):
            return recon
        eps = max(1e-6, float(log_eps))
        raw_clamped = np.clip(np.asarray(raw_source, dtype=np.float32), a_min=0.0, a_max=None)
        log_floor = max(eps, float(np.std(raw_clamped)) * float(cdd_log_std_floor_mult))
        return np.log(np.clip(recon, a_min=0.0, a_max=None) + log_floor).astype(np.float32)

    for bi in range(B):
        arr = np.asarray(source_cpu[bi, source_ch].numpy(), dtype=np.float32)
        channels_arr, residual, _scales_used = safe_constrained_diffusion_decomposition(
            cdd,
            arr,
            num_channels=cdd_num_channels,
            max_scale=max(float(s) for s in sigmas),
            min_scale=min(float(s) for s in sigmas),
            mode=str(cdd_mode),
            constrained=bool(cdd_constrained),
            sm_mode=str(cdd_sm_mode),
            return_scales=True,
            verbose=False,
            use_gpu=False,
            gaussian_backend=str(cdd_gaussian_backend),
        )
        channels = np.asarray(channels_arr[:cdd_num_channels], dtype=np.float32)
        if channels.shape[0] < cdd_num_channels:
            raise RuntimeError(
                f"CDD returned {channels.shape[0]} bands, expected {cdd_num_channels}"
            )
        residual_arr = np.asarray(residual, dtype=np.float32) if residual is not None else np.zeros_like(arr, dtype=np.float32)
        full_recon = np.sum(channels, axis=0, dtype=np.float32) + residual_arr
        if bool(cdd_append_last_residual):
            # Match training's CDD channel convention while keeping reconstruction exact.
            channels = channels.copy()
            channels[-1] = channels[-1] + residual_arr
            full_recon = np.sum(channels, axis=0, dtype=np.float32)

        sample_drop_variants = []
        sample_only_variants = []
        for si in range(cdd_num_channels):
            drop_var = ref_cpu[bi].clone()
            recon_drop = full_recon - channels[si]
            drop_var[ch] = torch.from_numpy(_encode_like_model(recon_drop, arr))
            sample_drop_variants.append(drop_var)

            only_var = ref_cpu[bi].clone()
            recon_only = channels[si]
            only_var[ch] = torch.from_numpy(_encode_like_model(recon_only, arr))
            sample_only_variants.append(only_var)
        drop_variants.append(torch.stack(sample_drop_variants, dim=0))
        only_variants.append(torch.stack(sample_only_variants, dim=0))

    return (
        torch.stack(drop_variants, dim=0).to(device=device, dtype=dtype),
        torch.stack(only_variants, dim=0).to(device=device, dtype=dtype),
        variant_names,
    )


@torch.no_grad()
def _normalize_channel_map(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / (z.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def _effective_rank(z: torch.Tensor, max_points: int = 50000, eps: float = 1e-12) -> float:
    if z.ndim < 4:
        raise ValueError(f"Expected dense feature map B,C,... got {tuple(z.shape)}")
    c = z.shape[1]
    x = z.detach().float().permute(0, *range(2, z.ndim), 1).reshape(-1, c)
    if x.shape[0] > max_points:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_points]
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    try:
        evals = torch.linalg.eigvalsh(cov).clamp_min(0)
    except (NotImplementedError, RuntimeError):
        evals = torch.linalg.eigvalsh(cov.cpu()).clamp_min(0).to(cov.device)
    p = evals / evals.sum().clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy).item())


@torch.no_grad()
def probe_image_response(
    model,
    reference_input: torch.Tensor,
    variant_inputs: Optional[torch.Tensor] = None,
    scale_only_inputs: Optional[torch.Tensor] = None,
    variant_names: Optional[Sequence[str]] = None,
    out_dir: str | Path = "scale_response_report",
    run_name: str = "probe",
    include_predictor: bool = True,
    max_rank_points: int = 50000,
    perturb_channel: int = 0,
) -> Dict:
    """Probe image-mode encoder response to named input variants.

    ``reference_input`` is the actual ConvNeXt image-mode network input
    (typically ``[clean_image, zero_mask]``). ``variant_inputs`` may be a
    ``B,V,C,H,W`` stack. If omitted, the probe compares the reference against a
    copy with one channel zeroed.
    """

    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if reference_input.ndim != 4:
        raise ValueError(f"Expected reference_input B,C,H,W, got {tuple(reference_input.shape)}")
    B, C, H, W = reference_input.shape
    if not (0 <= int(perturb_channel) < C):
        raise ValueError(f"perturb_channel={perturb_channel} outside C={C}")

    if variant_inputs is None:
        dropped = reference_input.clone()
        dropped[:, int(perturb_channel)] = 0.0
        variant_inputs = torch.stack([reference_input, dropped], dim=1)
        if variant_names is None:
            variant_names = ["original_image", "cdd_removed_image"]
    elif variant_inputs.ndim == 4:
        variant_inputs = variant_inputs.unsqueeze(1)
    if variant_inputs.ndim != 5:
        raise ValueError(f"Expected variant_inputs B,V,C,H,W, got {tuple(variant_inputs.shape)}")
    if variant_inputs.shape[0] != B or variant_inputs.shape[2:] != (C, H, W):
        raise ValueError(
            "variant_inputs must match reference batch/channels/spatial shape; "
            f"reference={tuple(reference_input.shape)} variants={tuple(variant_inputs.shape)}"
        )
    if scale_only_inputs is not None:
        if scale_only_inputs.ndim == 4:
            scale_only_inputs = scale_only_inputs.unsqueeze(1)
        if scale_only_inputs.ndim != 5:
            raise ValueError(f"Expected scale_only_inputs B,V,C,H,W, got {tuple(scale_only_inputs.shape)}")
        if scale_only_inputs.shape != variant_inputs.shape:
            raise ValueError(
                "scale_only_inputs must match variant_inputs shape; "
                f"scale_only={tuple(scale_only_inputs.shape)} variants={tuple(variant_inputs.shape)}"
            )
    V = int(variant_inputs.shape[1])
    if variant_names is None:
        variant_names = [f"variant_{i}" for i in range(V)]
    if len(variant_names) != V:
        raise ValueError(f"variant_names length {len(variant_names)} != V={V}")

    z_ref = _encode_direct_context(model, reference_input)
    z_ref_n = _normalize_channel_map(z_ref)

    pred_ref = None
    pred_ref_n = None
    if include_predictor and hasattr(model, "predictor"):
        try:
            z_in_ref = model.projector(z_ref) if hasattr(model, "projector") else z_ref
            pred_ref = model.predictor(z_in_ref)
            pred_ref_n = _normalize_channel_map(pred_ref)
        except Exception:
            pred_ref = None
            pred_ref_n = None

    sensitivity_maps = []
    scale_only_sim_maps = []
    pred_sensitivity_maps = []

    def _restore_input_hw(map_bhw: torch.Tensor) -> torch.Tensor:
        if map_bhw.shape[-2:] == (H, W):
            return map_bhw
        return F.interpolate(
            map_bhw.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    for v in range(V):
        x_var = variant_inputs[:, v]
        z_var = _encode_direct_context(model, x_var)
        z_var_n = _normalize_channel_map(z_var)

        diff = (z_ref_n - z_var_n).pow(2).sum(dim=1).sqrt()
        sensitivity_maps.append(_restore_input_hw(diff))

        if scale_only_inputs is not None:
            x_one = scale_only_inputs[:, v]
            z_one = _encode_direct_context(model, x_one)
            z_one_n = _normalize_channel_map(z_one)
            sim = (z_ref_n * z_one_n).sum(dim=1)
        else:
            sim = (z_ref_n * z_var_n).sum(dim=1)
        scale_only_sim_maps.append(_restore_input_hw(sim))

        if pred_ref_n is not None:
            z_in_var = model.projector(z_var) if hasattr(model, "projector") else z_var
            pred_var = model.predictor(z_in_var)
            pred_var_n = _normalize_channel_map(pred_var)
            pred_diff = (pred_ref_n - pred_var_n).pow(2).sum(dim=1).sqrt()
            pred_sensitivity_maps.append(_restore_input_hw(pred_diff))

    sensitivity_maps = torch.stack(sensitivity_maps, dim=1)
    scale_only_sim_maps = torch.stack(scale_only_sim_maps, dim=1)
    pred_sensitivity_maps = torch.stack(pred_sensitivity_maps, dim=1) if pred_sensitivity_maps else None

    sens_global = sensitivity_maps.mean(dim=(0, 2, 3))
    sim_global = scale_only_sim_maps.mean(dim=(0, 2, 3))
    pred_sens_global = pred_sensitivity_maps.mean(dim=(0, 2, 3)) if pred_sensitivity_maps is not None else None
    sens_frac = sens_global / sens_global.sum().clamp_min(1e-12)
    pred_sens_frac = (
        pred_sens_global / pred_sens_global.sum().clamp_min(1e-12)
        if pred_sens_global is not None
        else None
    )
    winner_map = sensitivity_maps[0].argmax(dim=0)

    report = {
        "run_name": run_name,
        "probe_mode": "image",
        "probe_variant_mode": (
            "cdd_reconstruct_drop_and_scale_only_post_log"
            if scale_only_inputs is not None
            else "image_variant_drop_only"
        ),
        "input_shape": list(reference_input.shape),
        "variant_input_shape": list(variant_inputs.shape),
        "scale_only_input_shape": None if scale_only_inputs is None else list(scale_only_inputs.shape),
        "feature_shape": list(z_ref.shape),
        "scale_names": list(variant_names),
        "context_effective_rank": _effective_rank(z_ref, max_points=max_rank_points),
        "scale_drop_sensitivity": {
            name: float(sens_global[i].item()) for i, name in enumerate(variant_names)
        },
        "scale_drop_sensitivity_fraction": {
            name: float(sens_frac[i].item()) for i, name in enumerate(variant_names)
        },
        "scale_only_similarity_to_full": {
            name: float(sim_global[i].item()) for i, name in enumerate(variant_names)
        },
        "dominant_context_scale": variant_names[int(torch.argmax(sens_global).item())],
    }

    if pred_ref is not None:
        report["predictor_effective_rank"] = _effective_rank(pred_ref, max_points=max_rank_points)
    if pred_sens_global is not None:
        report["pred_scale_drop_sensitivity"] = {
            name: float(pred_sens_global[i].item()) for i, name in enumerate(variant_names)
        }
        report["pred_scale_drop_sensitivity_fraction"] = {
            name: float(pred_sens_frac[i].item()) for i, name in enumerate(variant_names)
        }
        report["dominant_pred_scale"] = variant_names[int(torch.argmax(pred_sens_global).item())]

    save_obj = {
        "sensitivity_maps": sensitivity_maps.detach().cpu(),
        "scale_only_sim_maps": scale_only_sim_maps.detach().cpu(),
        "winner_map": winner_map.detach().cpu(),
        "input_map": reference_input[0, 0].detach().cpu(),
        "variant_inputs": variant_inputs.detach().cpu(),
        "z_full": z_ref.detach().cpu(),
    }
    if scale_only_inputs is not None:
        save_obj["scale_only_inputs"] = scale_only_inputs.detach().cpu()
    if pred_sensitivity_maps is not None:
        save_obj["pred_sensitivity_maps"] = pred_sensitivity_maps.detach().cpu()
    if pred_ref is not None:
        save_obj["pred_full"] = pred_ref.detach().cpu()

    torch.save(save_obj, out_dir / f"{run_name}_scale_response.pt")

    with open(out_dir / f"{run_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append(f"Image-response report: {run_name}")
    lines.append(f"input shape:   {tuple(reference_input.shape)}")
    lines.append(f"feature shape: {tuple(z_ref.shape)}")
    lines.append(f"context effective rank: {report['context_effective_rank']:.4f}")
    if "predictor_effective_rank" in report:
        lines.append(f"predictor effective rank: {report['predictor_effective_rank']:.4f}")
    lines.append("")
    lines.append("Context image-variant sensitivity:")
    for name in variant_names:
        val = report["scale_drop_sensitivity"][name]
        frac = report["scale_drop_sensitivity_fraction"][name]
        simv = report["scale_only_similarity_to_full"][name]
        lines.append(f"  {name:>18s}: diff={val:.6f}  frac={frac:.3f}  sim={simv:.3f}")
    lines.append(f"dominant context variant: {report['dominant_context_scale']}")
    if "pred_scale_drop_sensitivity" in report:
        lines.append("")
        lines.append("Predictor image-variant sensitivity:")
        for name in variant_names:
            val = report["pred_scale_drop_sensitivity"][name]
            frac = report["pred_scale_drop_sensitivity_fraction"][name]
            lines.append(f"  {name:>18s}: diff={val:.6f}  frac={frac:.3f}")
        lines.append(f"dominant pred variant: {report['dominant_pred_scale']}")

    with open(out_dir / f"{run_name}_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    return report


@torch.no_grad()
def probe_scale_response(
    model,
    x_pyr: torch.Tensor,
    mask_tokens: Optional[torch.Tensor] = None,
    scale_names: Optional[Sequence[str]] = None,
    out_dir: str | Path = "scale_response_report",
    run_name: str = "probe",
    include_predictor: bool = True,
    max_rank_points: int = 50000,
) -> Dict:
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if x_pyr.ndim != 4:
        raise ValueError(f"Expected x_pyr B,S,H,W, got {tuple(x_pyr.shape)}")

    B, S, H, W = x_pyr.shape
    if scale_names is None:
        scale_names = [f"scale_{i}" for i in range(S)]
    if len(scale_names) != S:
        raise ValueError(f"scale_names length {len(scale_names)} != S={S}")

    if mask_tokens is None:
        mask_tokens = torch.zeros_like(x_pyr)

    z_full = _encode_context(model, x_pyr, mask_tokens=mask_tokens)
    z_full_n = _normalize_channel_map(z_full)

    pred_full = None
    pred_full_n = None
    if include_predictor and hasattr(model, "predictor"):
        try:
            z_in_full = model.projector(z_full) if hasattr(model, "projector") else z_full
            pred_full = model.predictor(z_in_full)
            pred_full_n = _normalize_channel_map(pred_full)
        except Exception:
            pred_full = None
            pred_full_n = None

    sensitivity_maps = []
    scale_only_sim_maps = []
    pred_sensitivity_maps = []

    def _restore_input_hw(map_bhw: torch.Tensor) -> torch.Tensor:
        if map_bhw.shape[-2:] == (H, W):
            return map_bhw
        return F.interpolate(
            map_bhw.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    for s in range(S):
        x_drop = x_pyr.clone()
        x_drop[:, s] = 0.0
        z_drop = _encode_context(model, x_drop, mask_tokens=mask_tokens)
        z_drop_n = _normalize_channel_map(z_drop)

        diff = (z_full_n - z_drop_n).pow(2).sum(dim=1).sqrt()
        sensitivity_maps.append(_restore_input_hw(diff))

        if pred_full_n is not None:
            z_in_drop = model.projector(z_drop) if hasattr(model, "projector") else z_drop
            pred_drop = model.predictor(z_in_drop)
            pred_drop_n = _normalize_channel_map(pred_drop)
            pred_diff = (pred_full_n - pred_drop_n).pow(2).sum(dim=1).sqrt()
            pred_sensitivity_maps.append(_restore_input_hw(pred_diff))

        x_one = torch.zeros_like(x_pyr)
        x_one[:, s] = x_pyr[:, s]
        z_one = _encode_context(model, x_one, mask_tokens=mask_tokens)
        z_one_n = _normalize_channel_map(z_one)
        sim = (z_full_n * z_one_n).sum(dim=1)
        scale_only_sim_maps.append(_restore_input_hw(sim))

    sensitivity_maps = torch.stack(sensitivity_maps, dim=1)  # B,S,H,W
    scale_only_sim_maps = torch.stack(scale_only_sim_maps, dim=1)

    if pred_sensitivity_maps:
        pred_sensitivity_maps = torch.stack(pred_sensitivity_maps, dim=1)
    else:
        pred_sensitivity_maps = None

    sens_global = sensitivity_maps.mean(dim=(0, 2, 3))
    sim_global = scale_only_sim_maps.mean(dim=(0, 2, 3))
    pred_sens_global = None
    if pred_sensitivity_maps is not None:
        pred_sens_global = pred_sensitivity_maps.mean(dim=(0, 2, 3))

    sens_frac = sens_global / sens_global.sum().clamp_min(1e-12)
    if pred_sens_global is not None:
        pred_sens_frac = pred_sens_global / pred_sens_global.sum().clamp_min(1e-12)
    else:
        pred_sens_frac = None

    winner_map = sensitivity_maps[0].argmax(dim=0)  # H,W — per-location dominant scale

    report = {
        "run_name": run_name,
        "input_shape": list(x_pyr.shape),
        "feature_shape": list(z_full.shape),
        "scale_names": list(scale_names),
        "context_effective_rank": _effective_rank(z_full, max_points=max_rank_points),
        "scale_drop_sensitivity": {
            name: float(sens_global[i].item()) for i, name in enumerate(scale_names)
        },
        "scale_drop_sensitivity_fraction": {
            name: float(sens_frac[i].item()) for i, name in enumerate(scale_names)
        },
        "scale_only_similarity_to_full": {
            name: float(sim_global[i].item()) for i, name in enumerate(scale_names)
        },
        "dominant_context_scale": scale_names[int(torch.argmax(sens_global).item())],
    }

    if pred_full is not None:
        report["predictor_effective_rank"] = _effective_rank(pred_full, max_points=max_rank_points)
    if pred_sens_global is not None:
        report["pred_scale_drop_sensitivity"] = {
            name: float(pred_sens_global[i].item()) for i, name in enumerate(scale_names)
        }
        report["pred_scale_drop_sensitivity_fraction"] = {
            name: float(pred_sens_frac[i].item()) for i, name in enumerate(scale_names)
        }
        report["dominant_pred_scale"] = scale_names[int(torch.argmax(pred_sens_global).item())]

    save_obj = {
        "sensitivity_maps": sensitivity_maps.detach().cpu(),
        "scale_only_sim_maps": scale_only_sim_maps.detach().cpu(),
        "winner_map": winner_map.detach().cpu(),
        "input_map": x_pyr[0].sum(dim=0).detach().cpu(),
        "z_full": z_full.detach().cpu(),
    }
    if pred_sensitivity_maps is not None:
        save_obj["pred_sensitivity_maps"] = pred_sensitivity_maps.detach().cpu()
    if pred_full is not None:
        save_obj["pred_full"] = pred_full.detach().cpu()

    torch.save(save_obj, out_dir / f"{run_name}_scale_response.pt")

    with open(out_dir / f"{run_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append(f"Scale-response report: {run_name}")
    lines.append(f"input shape:   {tuple(x_pyr.shape)}")
    lines.append(f"feature shape: {tuple(z_full.shape)}")
    lines.append(f"context effective rank: {report['context_effective_rank']:.4f}")
    if "predictor_effective_rank" in report:
        lines.append(f"predictor effective rank: {report['predictor_effective_rank']:.4f}")
    lines.append("")
    lines.append("Context scale-drop sensitivity:")
    for name in scale_names:
        val = report["scale_drop_sensitivity"][name]
        frac = report["scale_drop_sensitivity_fraction"][name]
        simv = report["scale_only_similarity_to_full"][name]
        lines.append(f"  {name:>12s}: drop={val:.6f}  frac={frac:.3f}  only_sim={simv:.3f}")
    lines.append(f"dominant context scale: {report['dominant_context_scale']}")
    if "pred_scale_drop_sensitivity" in report:
        lines.append("")
        lines.append("Predictor scale-drop sensitivity:")
        for name in scale_names:
            val = report["pred_scale_drop_sensitivity"][name]
            frac = report["pred_scale_drop_sensitivity_fraction"][name]
            lines.append(f"  {name:>12s}: drop={val:.6f}  frac={frac:.3f}")
        lines.append(f"dominant pred scale: {report['dominant_pred_scale']}")

    with open(out_dir / f"{run_name}_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    return report
