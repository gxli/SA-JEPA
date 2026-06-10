from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize_patches(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize per-target patch/cube tensors across all non-B/K dimensions."""
    if tensor.dim() < 3:
        raise ValueError(f"Expected tensor with B,K,... shape, got {tuple(tensor.shape)}")
    b, k = tensor.shape[:2]
    patch_shape = tensor.shape[2:]
    return F.normalize(tensor.reshape(b, k, -1), dim=2).reshape(b, k, *patch_shape)


def parse_spread_regularizer_config(train_cfg: dict) -> dict[str, float | str]:
    removed_flat_keys = {
        "spread_regularizer_weight",
        "spread_sketch_dim",
        "embed_spread_target",
        "spread_regularizer_eps",
        "spread_regularizer_noise_std",
    }
    stale_keys = removed_flat_keys & set(train_cfg)
    assert not stale_keys, f"Use train.spread_regularizer instead of flat keys: {sorted(stale_keys)}"
    cfg = dict(train_cfg.get("spread_regularizer", {}))
    expected_keys = {"type", "target", "weight", "target_std", "eps", "sketch_dim", "sketch_seed"}
    extra_keys = set(cfg) - expected_keys
    assert not extra_keys, f"Unsupported train.spread_regularizer keys: {sorted(extra_keys)}"
    sigreg_type = str(cfg.get("type", "std_hinge"))
    sigreg_target = str(cfg.get("target", "context"))
    sigreg_weight = float(cfg.get("weight", 0.0))
    assert sigreg_type in {"std_hinge", "weak_sigreg", "sketched_sigreg"}
    assert sigreg_target == "context"
    assert sigreg_weight >= 0
    target_std = float(cfg.get("target_std", 1.0))
    eps = float(cfg.get("eps", 1e-4))
    sketch_dim = int(cfg.get("sketch_dim", 64))
    sketch_seed = int(cfg.get("sketch_seed", 0))
    assert target_std >= 0
    assert eps > 0
    assert sketch_dim > 0
    assert sketch_seed >= 0
    return {
        "type": sigreg_type,
        "target": sigreg_target,
        "weight": sigreg_weight,
        "target_std": target_std,
        "eps": eps,
        "sketch_dim": sketch_dim,
        "sketch_seed": sketch_seed,
    }


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("offdiag expects square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _reshape_patch_pairs(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    spatial_mode: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten B×K×C×P×P (or ×P×P×P) to (N, C) tokens, filter by valid."""
    if pred.dim() not in (5, 6):
        raise ValueError(f"Expected pred/gt rank 5 or 6, got {pred.dim()}")
    b, k, c = pred.shape[:3]
    spatial_shape = pred.shape[3:]
    spatial_n = 1
    for s in spatial_shape:
        spatial_n *= int(s)
    mode = str(spatial_mode).lower()
    if mode not in ("dense", "pooled"):
        raise ValueError(f"Unsupported spatial_mode={spatial_mode}. Use 'dense' or 'pooled'.")
    if mode == "pooled":
        reduce_dims = tuple(range(3, pred.dim()))
        pred_v = pred.mean(dim=reduce_dims)
        gt_v = gt.mean(dim=reduce_dims)
        vm = valid.reshape(-1)
        z1 = pred_v.reshape(-1, c)[vm]
        z2 = gt_v.reshape(-1, c)[vm]
        return z1, z2
    # dense mode: keep spatial tokens
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


def extract_valid_pooled_embeddings(outputs: dict, key: str = "context_patches") -> torch.Tensor:
    patches = outputs[key]  # Pre-predictor context embeddings by default: B,K,C,...
    valid = outputs["target_valid"]  # B,K
    _, _, c = patches.shape[:3]
    pooled = patches.mean(dim=tuple(range(3, patches.dim())))  # B,K,C
    vm = valid.reshape(-1)
    z = pooled.reshape(-1, c)[vm]
    return z


def extract_valid_dense_embeddings(outputs: dict, key: str = "context_patches") -> torch.Tensor:
    patches = outputs[key]  # B,K,C,... spatial patch tokens
    valid = outputs["target_valid"]  # B,K
    if patches.dim() not in (5, 6):
        raise ValueError(f"Expected {key} rank 5 or 6, got {patches.dim()}")
    b, k, c = patches.shape[:3]
    spatial_shape = patches.shape[3:]
    spatial_n = 1
    for size in spatial_shape:
        spatial_n *= int(size)
    if patches.dim() == 5:
        z = patches.permute(0, 1, 3, 4, 2).reshape(b, k, spatial_n, c)
    else:
        z = patches.permute(0, 1, 3, 4, 5, 2).reshape(b, k, spatial_n, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, spatial_n, 1).reshape(-1)
    return z.reshape(-1, c)[vm]


def spread_regularizer_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Standard-deviation hinge on context embeddings.
    Gradients remain useful close to collapse without hidden projection modes.
    """
    z = z.float()  # cast to fp32 to avoid underflow in fp16
    if z.numel() == 0 or z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    n = z.shape[0]
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + float(eps))
    loss = torch.relu(float(target_std) - std).mean()

    effective_n = 256.0
    if n > effective_n:
        grad_scale = float(n) / effective_n
        loss = loss * grad_scale - loss.detach() * (grad_scale - 1.0)

    return loss


def weak_sigreg_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    sketch_dim: int = 64,
    eps: float = 1e-4,
    sketch_seed: int = 0,
) -> torch.Tensor:
    """Sketched isotropic Gaussian regularization on context embeddings.

    Applies the variance hinge on the full embedding dimension for consistent
    anti-collapse gradients, then uses a rotating sketch only for the cheaper
    off-diagonal covariance penalty.
    """
    del sketch_seed  # Kept as a config key for reproducibility metadata/backward compatibility.
    z = z.float()
    if z.numel() == 0 or z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    n, c = z.shape
    z = z - z.mean(dim=0, keepdim=True)

    std_full = torch.sqrt(z.var(dim=0, unbiased=False) + float(eps))
    var_loss = torch.relu(float(target_std) - std_full).mean()

    k = min(int(sketch_dim), int(c))
    if c > k:
        sketch = torch.randn(c, k, device=z.device, dtype=z.dtype)
        sketch, _ = torch.linalg.qr(sketch, mode="reduced")
        z_proj = z @ sketch
    else:
        z_proj = z

    cov = (z_proj.t() @ z_proj) / (float(n - 1) + float(eps))
    cov = cov - torch.diag(cov.diag())
    cov_loss = cov.pow(2).sum() / float(k)
    loss = var_loss + cov_loss

    effective_n = 256.0
    if n > effective_n:
        grad_scale = float(n) / effective_n
        loss = loss * grad_scale - loss.detach() * (grad_scale - 1.0)

    return loss


def sketched_sigreg_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    sketch_dim: int = 64,
    eps: float = 1e-6,
    sketch_seed: int = 0,
) -> torch.Tensor:
    del sketch_seed  # Preserve config shape; projection is intentionally stochastic.
    z = z.float()
    if z.numel() == 0 or z.shape[0] < 2:
        return z.sum() * 0.0

    z = z - z.mean(dim=0, keepdim=True)
    c = z.shape[1]
    sketch_dim = int(max(1, sketch_dim))
    a = torch.randn((c, sketch_dim), device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-6)
    y = z @ a
    projected_var_loss = (y.var(dim=0, unbiased=False) - float(target_std) ** 2).pow(2).mean()

    # The historical projected variance loss has zero gradient at exact
    # collapse because d(var(z @ A))/dz is proportional to z. Keep the old
    # isotropic pressure, but add a direct std hinge so near-zero embeddings
    # have an escape gradient instead of sitting at loss ~= 1 forever.
    # If all target embeddings are exactly identical, std(z) is high-loss but
    # zero-gradient. Add tiny loss-local jitter so collapsed dimensions get a
    # deterministic escape direction through the std hinge; the jitter is not
    # applied to the model outputs or diagnostics.
    escape_jitter = max(float(eps) ** 0.5, 1e-3)
    z_escape = z + escape_jitter * torch.randn_like(z)
    std_full = torch.sqrt(z_escape.var(dim=0, unbiased=False) + float(eps))
    escape_loss = torch.relu(float(target_std) - std_full).mean()

    std_corr = torch.sqrt(z.var(dim=0, unbiased=False) + float(eps))
    z_corr = z / std_corr.clamp_min(float(eps)).unsqueeze(0)
    corr = (z_corr.t() @ z_corr) / float(max(1, z.shape[0] - 1))
    corr = corr - torch.diag(corr.diag())
    if c > 1:
        decorrelation_loss = corr.pow(2).sum() / float(c * (c - 1))
    else:
        decorrelation_loss = corr.sum() * 0.0
    return projected_var_loss + escape_loss + decorrelation_loss


def compute_spread_regularizer_loss(z: torch.Tensor, cfg: dict[str, float | str]) -> torch.Tensor:
    sigreg_type = str(cfg.get("type", "std_hinge"))
    if sigreg_type == "std_hinge":
        return spread_regularizer_loss(
            z,
            target_std=float(cfg.get("target_std", 1.0)),
            eps=float(cfg.get("eps", 1e-4)),
        )
    if sigreg_type == "weak_sigreg":
        return weak_sigreg_loss(
            z,
            target_std=float(cfg.get("target_std", 1.0)),
            sketch_dim=int(cfg.get("sketch_dim", 64)),
            eps=float(cfg.get("eps", 1e-4)),
            sketch_seed=int(cfg.get("sketch_seed", 0)),
        )
    if sigreg_type == "sketched_sigreg":
        return sketched_sigreg_loss(
            z,
            target_std=float(cfg.get("target_std", 1.0)),
            sketch_dim=int(cfg.get("sketch_dim", 64)),
            eps=float(cfg.get("eps", 1e-6)),
            sketch_seed=int(cfg.get("sketch_seed", 0)),
        )
    raise ValueError(f"Unsupported spread regularizer type: {sigreg_type}")


def compute_output_spread_regularizer_loss(
    outputs: dict,
    cfg: dict[str, float | str],
    *,
    include_predictor: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "context_patches" not in outputs:
        raise KeyError("SIGReg requires outputs['context_patches']; refusing to regularize predictor-only outputs")
    z_ctx = extract_valid_pooled_embeddings(outputs, key="context_patches")
    loss = compute_spread_regularizer_loss(z_ctx, cfg)
    if include_predictor:
        z_pred = extract_valid_pooled_embeddings(outputs, key="pred_patches")
        loss = loss + compute_spread_regularizer_loss(z_pred, cfg)
    return loss, z_ctx


@torch.no_grad()
def embedding_spread_stats(
    z: torch.Tensor,
    target_std: float = 1.0,
    dead_channel_threshold: float = 1e-5,
) -> dict[str, float]:
    """Compact collapse diagnostics for pooled context embeddings."""
    z = z.detach().float()
    if z.numel() == 0:
        return {
            "embed_spread_mean": 0.0,
            "embed_spread_min": 0.0,
            "embed_under_spread_frac": 1.0,
            "dead_channel_count": 0,
            "context_manifold_size": 0.0,
        }
    z = z - z.mean(dim=0, keepdim=True)
    var = z.var(dim=0, unbiased=False)
    cov = (z.T @ z) / max(1, int(z.shape[0]))
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    eig_sum = eig.sum()
    rank = 0.0
    if float(eig_sum.item()) > 1e-20:
        p = eig / eig_sum
        p = p[p > 1e-20]
        if p.numel() > 0:
            rank = float(torch.exp(-(p * p.log()).sum()).item())
    std = torch.sqrt(var + 1e-12)
    return {
        "embed_spread_mean": float(std.mean().item()),
        "embed_spread_min": float(std.min().item()),
        "embed_under_spread_frac": float((std < float(target_std)).float().mean().item()),
        "dead_channel_count": int((std < float(dead_channel_threshold)).sum().item()),
        "context_manifold_size": rank,
    }


def _compute_sim_var_cov_tensors(
    outputs: dict,
    spatial_mode: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = outputs["pred_patches"].float()  # keep graph, cast to fp32 for safe norm
    ctx = outputs.get("context_patches", outputs["pred_patches"]).float()  # regularize before predictor
    gt = outputs["gt_patches"].float()  # keep graph (target branch already no-grad in forward)
    valid = outputs["target_valid"]

    z_pred, z_gt = _reshape_patch_pairs(pred, gt, valid, spatial_mode=spatial_mode)
    z_ctx, _ = _reshape_patch_pairs(ctx, gt, valid, spatial_mode=spatial_mode)
    if z_pred.numel() == 0 or z_gt.numel() == 0:
        z = pred.sum() * 0.0
        return z, z, z

    sim = torch.nn.functional.cosine_similarity(z_pred, z_gt, dim=1).mean()
    if z_pred.shape[0] < 2:
        z = sim * 0.0
        return sim, z, z

    std_ctx = torch.sqrt(z_ctx.var(dim=0, unbiased=False) + 1e-4)
    std_gt = torch.sqrt(z_gt.var(dim=0, unbiased=False) + 1e-4)
    var_term = 0.5 * (torch.relu(1.0 - std_ctx).mean() + torch.relu(1.0 - std_gt).mean())

    z1c = z_ctx - z_ctx.mean(dim=0, keepdim=True)
    z2c = z_gt - z_gt.mean(dim=0, keepdim=True)
    cov_z1 = (z1c.T @ z1c) / max(1, z1c.shape[0] - 1)
    cov_z2 = (z2c.T @ z2c) / max(1, z2c.shape[0] - 1)
    dim = max(1, int(cov_z1.shape[0]))
    cov_term = 0.5 * ((_offdiag(cov_z1).pow(2).sum() / dim) + (_offdiag(cov_z2).pow(2).sum() / dim))
    return sim, var_term, cov_term


def compute_sim_var_cov(outputs: dict, spatial_mode: str = "dense") -> tuple[float, float, float]:
    values = _compute_sim_var_cov_tensors(outputs, spatial_mode=spatial_mode)
    return tuple(float(value.detach().item()) for value in values)


def compute_sim_var_cov_torch(outputs: dict, spatial_mode: str = "dense") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _compute_sim_var_cov_tensors(outputs, spatial_mode=spatial_mode)


def compute_raw_mse_and_norm_err(outputs: dict) -> tuple[float, float]:
    pred = outputs["pred_patches"].detach().float()
    gt = outputs["gt_patches"].detach().float()
    valid = outputs["target_valid"].detach()
    z1, z2 = _reshape_patch_pairs(pred, gt, valid, spatial_mode="dense")
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
        b, k = pred.shape[:2]
        patch_shape = pred.shape[2:]
        pred = torch.nn.functional.normalize(pred.reshape(b, k, -1), dim=2).reshape(b, k, *patch_shape)
        gt = torch.nn.functional.normalize(gt.reshape(b, k, -1), dim=2).reshape(b, k, *patch_shape)
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

    cos = (1.0 - F.cosine_similarity(pred_map, gt_map, dim=1, eps=eps).unsqueeze(1)).clamp_min(0.0)

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
