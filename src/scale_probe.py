from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


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
    evals = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = evals / evals.sum().clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy).item())


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
            pred_full = model.predictor(z_full)
            pred_full_n = _normalize_channel_map(pred_full)
        except Exception:
            pred_full = None
            pred_full_n = None

    sensitivity_maps = []
    scale_only_sim_maps = []
    pred_sensitivity_maps = []

    for s in range(S):
        x_drop = x_pyr.clone()
        x_drop[:, s] = 0.0
        z_drop = _encode_context(model, x_drop, mask_tokens=mask_tokens)
        z_drop_n = _normalize_channel_map(z_drop)

        diff = (z_full_n - z_drop_n).pow(2).sum(dim=1).sqrt()
        sensitivity_maps.append(diff)

        if pred_full_n is not None:
            pred_drop = model.predictor(z_drop)
            pred_drop_n = _normalize_channel_map(pred_drop)
            pred_diff = (pred_full_n - pred_drop_n).pow(2).sum(dim=1).sqrt()
            pred_sensitivity_maps.append(pred_diff)

        x_one = torch.zeros_like(x_pyr)
        x_one[:, s] = x_pyr[:, s]
        z_one = _encode_context(model, x_one, mask_tokens=mask_tokens)
        z_one_n = _normalize_channel_map(z_one)
        sim = (z_full_n * z_one_n).sum(dim=1)
        scale_only_sim_maps.append(sim)

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


def save_scale_response_plots(
    session_dir: str,
    scale_names: Optional[Sequence[str]] = None,
    run_name: str = "probe",
) -> list[str]:
    """Load scale_response.pt and save per-scale sensitivity PNGs + winner map.

    Returns list of saved image paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    pt_path = os.path.join(session_dir, f"{run_name}_scale_response.pt")
    if not os.path.exists(pt_path):
        return []

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    sensitivity_maps = data["sensitivity_maps"]  # B,S,H,W
    winner_map = data.get("winner_map")  # H,W or None

    B, S, H, W = sensitivity_maps.shape
    if scale_names is None:
        scale_names = [f"scale_{i}" for i in range(S)]

    saved = []

    # Per-scale sensitivity maps (sample 0).
    fig, axes = plt.subplots(1, S, figsize=(4 * S, 4))
    if S == 1:
        axes = [axes]
    vmax = float(sensitivity_maps[0].max().item())
    for i in range(S):
        im = axes[i].imshow(
            sensitivity_maps[0, i].numpy(),
            cmap="inferno",
            origin="upper",
            vmin=0,
            vmax=vmax,
        )
        axes[i].set_title(f"{scale_names[i]} sensitivity")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = os.path.join(session_dir, f"{run_name}_scale_sensitivity_maps.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Scale-only similarity maps.
    if "scale_only_sim_maps" in data:
        sim_maps = data["scale_only_sim_maps"]
        fig, axes = plt.subplots(1, S, figsize=(4 * S, 4))
        if S == 1:
            axes = [axes]
        for i in range(S):
            im = axes[i].imshow(sim_maps[0, i].numpy(), cmap="RdYlBu_r", origin="upper")
            axes[i].set_title(f"{scale_names[i]} only-sim")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = os.path.join(session_dir, f"{run_name}_scale_only_similarity.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # Winner map.
    if winner_map is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.get_cmap("tab10", S)
        bounds = list(range(S + 1))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(winner_map.numpy(), cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, ticks=list(range(S)), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(scale_names)
        ax.set_title("dominant scale per location")
        ax.axis("off")
        fig.tight_layout()
        p = os.path.join(session_dir, f"{run_name}_scale_winner_map.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # Predictor sensitivity maps if present.
    if "pred_sensitivity_maps" in data:
        pred_maps = data["pred_sensitivity_maps"]
        fig, axes = plt.subplots(1, S, figsize=(4 * S, 4))
        if S == 1:
            axes = [axes]
        pvmax = float(pred_maps[0].max().item())
        for i in range(S):
            im = axes[i].imshow(
                pred_maps[0, i].numpy(),
                cmap="inferno",
                origin="upper",
                vmin=0,
                vmax=pvmax,
            )
            axes[i].set_title(f"pred {scale_names[i]} sensitivity")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = os.path.join(session_dir, f"{run_name}_pred_scale_sensitivity_maps.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved
