from __future__ import annotations

import torch
import torch.nn.functional as F


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("offdiag expects square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _flatten_vicreg_samples(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    spatial_mode: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.dim() not in (5, 6):
        raise ValueError(f"Expected pred/gt rank 5 (2D patches) or 6 (3D cubes), got pred.rank={pred.dim()}")
    b, k, c = pred.shape[:3]
    spatial_shape = pred.shape[3:]
    spatial_n = 1
    for s in spatial_shape:
        spatial_n *= int(s)
    mode = str(spatial_mode).lower()
    if mode == "pooled":
        reduce_dims = tuple(range(3, pred.dim()))
        pred_v = pred.mean(dim=reduce_dims)  # B,K,C
        gt_v = gt.mean(dim=reduce_dims)  # B,K,C
        vm = valid.reshape(-1)
        z1 = pred_v.reshape(-1, c)[vm]
        z2 = gt_v.reshape(-1, c)[vm]
        return z1, z2
    if mode != "dense":
        raise ValueError(f"Unsupported spatial_mode={spatial_mode}. Use 'dense' or 'pooled'.")
    if pred.dim() == 5:
        pred_v = pred.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
        gt_v = gt.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
    else:
        pred_v = pred.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
        gt_v = gt.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, spatial_n, 1).reshape(-1)
    z1 = pred_v.reshape(-1, c)[vm]
    z2 = gt_v.reshape(-1, c)[vm]
    return z1, z2


def extract_valid_pooled_embeddings(outputs: dict, key: str = "pred_patches") -> torch.Tensor:
    patches = outputs[key]  # B,K,C,P,P
    valid = outputs["target_valid"]  # B,K
    _, _, c = patches.shape[:3]
    pooled = patches.mean(dim=tuple(range(3, patches.dim())))  # B,K,C
    vm = valid.reshape(-1)
    z = pooled.reshape(-1, c)[vm]
    return z


def sketched_sigreg_loss(z: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """
    Lightweight SIGReg-style isotropic Gaussian regularization.
    Encourages projected embeddings to have mean 0 and variance 1.
    """
    if z.numel() == 0:
        return z.sum() * 0.0
    if z.shape[0] < 2:
        return z.sum() * 0.0

    z = z - z.mean(dim=0, keepdim=True)
    c = z.shape[1]
    sketch_dim = int(max(1, sketch_dim))
    a = torch.randn((c, sketch_dim), device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-6)
    y = z @ a  # N,sketch_dim

    mean_loss = y.mean(dim=0).pow(2).mean()
    var_loss = (y.var(dim=0, unbiased=False) - 1.0).pow(2).mean()
    return mean_loss + var_loss


def compute_sim_var_cov(outputs: dict, spatial_mode: str = "dense") -> tuple[float, float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    valid = outputs["target_valid"].detach()  # B,K

    z1, z2 = _flatten_vicreg_samples(pred, gt, valid, spatial_mode=spatial_mode)
    if z1.numel() == 0 or z2.numel() == 0:
        return 0.0, 0.0, 0.0

    # sim: cosine similarity (higher is better)
    sim = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean()

    if z1.shape[0] < 2:
        return float(sim.item()), 0.0, 0.0

    # var: VICReg variance regularizer term (lower is better; 0 ideal)
    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-4)
    var_term = 0.5 * (torch.relu(1.0 - std_z1).mean() + torch.relu(1.0 - std_z2).mean())

    # cov: VICReg covariance regularizer term (lower is better; 0 ideal)
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)
    cov_z1 = (z1c.T @ z1c) / max(1, z1c.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / max(1, z2c.shape[0] - 1)
    cov_term = 0.5 * ((_offdiag(cov_z1).pow(2).mean()) + (_offdiag(cov_z2).pow(2).mean()))

    return float(sim.item()), float(var_term.item()), float(cov_term.item())


def compute_sim_var_cov_torch(outputs: dict, spatial_mode: str = "dense") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = outputs["pred_patches"]  # keep graph
    gt = outputs["gt_patches"]  # keep graph (target branch already no-grad in forward)
    valid = outputs["target_valid"]

    z1, z2 = _flatten_vicreg_samples(pred, gt, valid, spatial_mode=spatial_mode)
    if z1.numel() == 0 or z2.numel() == 0:
        z = pred.sum() * 0.0
        return z, z, z

    sim = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean()
    if z1.shape[0] < 2:
        z = sim * 0.0
        return sim, z, z

    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-4)
    var_term = 0.5 * (torch.relu(1.0 - std_z1).mean() + torch.relu(1.0 - std_z2).mean())

    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)
    cov_z1 = (z1c.T @ z1c) / max(1, z1c.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / max(1, z2c.shape[0] - 1)
    cov_term = 0.5 * ((_offdiag(cov_z1).pow(2).mean()) + (_offdiag(cov_z2).pow(2).mean()))
    return sim, var_term, cov_term


def compute_raw_mse_and_norm_err(outputs: dict) -> tuple[float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    valid = outputs["target_valid"].detach()  # B,K

    if pred.dim() not in (5, 6):
        raise ValueError(f"Expected pred/gt rank 5 or 6, got {pred.dim()}")
    b, k, c = pred.shape[:3]
    spatial_shape = pred.shape[3:]
    spatial_n = 1
    for s in spatial_shape:
        spatial_n *= int(s)
    if pred.dim() == 5:
        pred_v = pred.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
        gt_v = gt.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
    else:
        pred_v = pred.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
        gt_v = gt.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, spatial_n, 1).reshape(-1)
    z1 = pred_v.reshape(-1, c)[vm]
    z2 = gt_v.reshape(-1, c)[vm]
    if z1.numel() == 0 or z2.numel() == 0:
        return 0.0, 0.0

    raw_mse = torch.mean((z1 - z2) ** 2)
    norm_err = torch.mean(torch.abs(torch.norm(z1, dim=1) - torch.norm(z2, dim=1)))
    return float(raw_mse.item()), float(norm_err.item())


def compute_jepa_energy(outputs: dict, normalize: bool = False) -> float:
    pred = outputs["pred_patches"]
    gt = outputs["gt_patches"].detach()
    valid = outputs["target_valid"]
    if normalize:
        pred = torch.nn.functional.normalize(pred, dim=2)
        gt = torch.nn.functional.normalize(gt, dim=2)
    reduce_dims = tuple(range(2, pred.dim()))
    energy_per_target = (pred - gt).pow(2).mean(dim=reduce_dims)
    if bool(valid.any()):
        return float(energy_per_target[valid].mean().item())
    return 0.0


def representation_dense_energy(pred_map: torch.Tensor, gt_map: torch.Tensor, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    diff = pred_map - gt_map
    diff2 = diff.pow(2)

    raw = diff2.mean(dim=1, keepdim=True)

    gt_norm2 = gt_map.pow(2).sum(dim=1, keepdim=True)
    pred_norm2 = pred_map.pow(2).sum(dim=1, keepdim=True)

    rel_gt = diff2.sum(dim=1, keepdim=True) / gt_norm2.clamp_min(eps)
    rel_sym = diff2.sum(dim=1, keepdim=True) / (0.5 * (gt_norm2 + pred_norm2)).clamp_min(eps)

    cos = 1.0 - F.cosine_similarity(pred_map, gt_map, dim=1, eps=eps).unsqueeze(1)

    return {
        "energy_raw": raw,
        "energy_rel_gt": rel_gt,
        "energy_rel_sym": rel_sym,
        "energy_cosine": cos,
    }


def representation_patch_energy(pred_patches: torch.Tensor, gt_patches: torch.Tensor, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    diff = pred_patches - gt_patches

    raw = diff.pow(2).mean(dim=(2, 3, 4))

    diff2 = diff.pow(2).sum(dim=(2, 3, 4))
    gt2 = gt_patches.pow(2).sum(dim=(2, 3, 4))
    pred2 = pred_patches.pow(2).sum(dim=(2, 3, 4))

    rel_gt = diff2 / gt2.clamp_min(eps)
    rel_sym = diff2 / (0.5 * (gt2 + pred2)).clamp_min(eps)

    cos = 1.0 - F.cosine_similarity(
        pred_patches.flatten(2),
        gt_patches.flatten(2),
        dim=2,
        eps=eps,
    )

    return {
        "energy_raw": raw,
        "energy_rel_gt": rel_gt,
        "energy_rel_sym": rel_sym,
        "energy_cosine": cos,
    }


def compute_target_energy_map(outputs: dict, image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    pred_map = outputs["pred_map"]
    gt_map = outputs["gt_map"].detach()
    h, w = int(image_size[0]), int(image_size[1])

    result = representation_dense_energy(pred_map, gt_map)

    for key in result:
        if result[key].shape[-2:] != (h, w):
            result[key] = F.interpolate(result[key], size=(h, w), mode="bilinear", align_corners=False)
    return result
