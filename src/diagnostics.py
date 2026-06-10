from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


MAX_RANK_TOKENS = 65536


def _subsample_rows_np(z: np.ndarray, max_rows: int = MAX_RANK_TOKENS) -> np.ndarray:
    if z.shape[0] <= int(max_rows):
        return z
    rng = np.random.default_rng(42)
    idx = rng.choice(z.shape[0], size=int(max_rows), replace=False)
    return z[idx]


def _subsample_rows_torch(z: torch.Tensor, max_rows: int = MAX_RANK_TOKENS) -> torch.Tensor:
    if z.shape[0] <= int(max_rows):
        return z
    idx = torch.randperm(z.shape[0], device=z.device)[:int(max_rows)]
    return z.index_select(0, idx)


def compute_effective_rank_from_features(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        return 0.0
    z = _subsample_rows_np(z)
    z = z - z.mean(axis=0, keepdims=True)
    denom = max(1, z.shape[0] - 1)
    try:
        svals = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        jitter_scale = max(float(np.std(z)), 1e-12) * 1e-5
        z_jitter = z + np.random.normal(0, jitter_scale, z.shape)
        svals = np.linalg.svd(z_jitter, full_matrices=False, compute_uv=False)
    evals = np.square(svals) / denom
    s = float(evals.sum())
    if s <= 0.0:
        return 0.0
    p = evals / s
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    return float(np.exp(h))


def spectral_rank_stats(fmap: torch.Tensor, eps: float = 1e-12, dead_thresh: float = 1e-5) -> dict:
    """
    fmap: C,H,W or B,C,H,W
    Computes covariance spectral stats over spatial tokens.
    """
    if fmap.ndim == 3:
        z = fmap.permute(1, 2, 0).reshape(-1, fmap.shape[0])
    elif fmap.ndim == 4:
        z = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])
    else:
        raise ValueError(f"Expected C,H,W or B,C,H,W, got {tuple(fmap.shape)}")

    z = z.float()
    z = _subsample_rows_torch(z)
    z = z - z.mean(dim=0, keepdim=True)

    ch_std = z.std(dim=0, unbiased=False)
    dead = ch_std < float(dead_thresh)
    dead_frac = dead.float().mean().item()

    denom = max(1, z.shape[0] - 1)
    try:
        eig_raw = torch.linalg.svdvals(z)
    except RuntimeError:
        jitter_scale = z.std(unbiased=False).clamp_min(1e-12) * 1e-5
        z_jitter = z + torch.randn_like(z) * jitter_scale
        eig_raw = torch.linalg.svdvals(z_jitter)
    eig = eig_raw.pow(2).div(float(denom)).clamp_min(0)

    total = eig.sum().clamp_min(eps)
    p = eig / total
    entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    erank = torch.exp(entropy)

    top1 = p[:1].sum()
    top4 = p[:4].sum()
    top8 = p[:8].sum()
    participation = total.pow(2) / eig.pow(2).sum().clamp_min(eps)

    return {
        "erank": float(erank.item()),
        "manifold_size": float(participation.item()),
        "top1_energy": float(top1.item()),
        "top4_energy": float(top4.item()),
        "top8_energy": float(top8.item()),
        "dead_channel_fraction": float(dead_frac),
        "dead_channel_count": int(dead.sum().item()),
        "mean_channel_std": float(ch_std.mean().item()),
        "min_channel_std": float(ch_std.min().item()),
        "max_channel_std": float(ch_std.max().item()),
        "dead_channel_threshold": float(dead_thresh),
    }


def rank_dashboard(outputs: dict) -> dict:
    context = outputs.get("context_map")
    pred = outputs["pred_map"]
    gt = outputs["gt_map"]

    out = {}
    if context is not None:
        out["context"] = spectral_rank_stats(torch.as_tensor(context))
    out["pred"] = spectral_rank_stats(torch.as_tensor(pred))
    out["gt"] = spectral_rank_stats(torch.as_tensor(gt))
    out["rank_match_ratio"] = out["pred"]["erank"] / max(out["gt"]["erank"], 1e-12)
    out["volume_match_ratio"] = out["pred"]["manifold_size"] / max(
        out["gt"]["manifold_size"], 1e-12
    )
    return out


def compute_error_by_scale(outputs: dict) -> dict[float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    scales = outputs["target_scales"].detach()  # B,K
    valid = outputs["target_valid"].detach()  # B,K

    # Per-target MSE averaged over C,P,P
    reduce_dims = tuple(range(2, pred.dim()))
    mse_bk = torch.mean((pred - gt) ** 2, dim=reduce_dims)  # B,K
    out = defaultdict(list)
    b, k = mse_bk.shape
    for bi in range(b):
        for ki in range(k):
            if not bool(valid[bi, ki].item()):
                continue
            s = round(float(scales[bi, ki].item()), 6)
            out[s].append(float(mse_bk[bi, ki].item()))
    return {float(s): float(np.mean(v)) for s, v in out.items() if len(v) > 0}
