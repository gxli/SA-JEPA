import csv
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.dataset import JEPADataset
from src.models.build_jepa import PyramidGridJEPA


def _compute_pca_2d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(x)
    except Exception:
        x_t = torch.from_numpy(x.astype(np.float32))
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        u, s, _ = torch.pca_lowrank(x_t, q=2)
        return (u[:, :2] * s[:2]).cpu().numpy()


def _compute_pca_3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=3).fit_transform(x)
    except Exception:
        u, s, _ = np.linalg.svd(x.astype(np.float64), full_matrices=False)
        z = (u[:, :3] * s[:3]).astype(np.float32)
        return z


def _compute_umap_nd(x: np.ndarray, n_components: int = 3) -> np.ndarray:
    try:
        from cuml.manifold import UMAP as CuMLUMAP

        return CuMLUMAP(n_components=n_components, random_state=42).fit_transform(x)
    except Exception:
        pass

    try:
        import torchdr

        if hasattr(torchdr, "UMAP"):
            model = torchdr.UMAP(n_components=n_components)
            z = model.fit_transform(torch.from_numpy(x.astype(np.float32)))
            if isinstance(z, torch.Tensor):
                return z.cpu().numpy()
            return np.asarray(z)
    except Exception:
        pass

    try:
        import umap

        return umap.UMAP(n_components=n_components, random_state=42).fit_transform(x)
    except Exception:
        pass

    if n_components == 2:
        return _compute_pca_2d(x)
    p2 = _compute_pca_2d(x)
    z = np.zeros((p2.shape[0], n_components), dtype=np.float32)
    z[:, :2] = p2.astype(np.float32)
    return z


def _save_latent_overview_html(session_dir: str, pca_points: np.ndarray, umap_points: np.ndarray, h: int, w: int) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def rgb_from_points(points: np.ndarray):
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        rng = np.maximum(maxs - mins, 1e-12)
        norm = np.clip((points - mins) / rng, 0.0, 1.0)
        rgb_u8 = (norm * 255.0).astype(np.uint8)
        colors = [f"rgb({r},{g},{b})" for r, g, b in rgb_u8]
        return rgb_u8.reshape(h, w, 3), colors

    pca_img, pca_colors = rgb_from_points(pca_points)
    umap_img, umap_colors = rgb_from_points(umap_points)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}], [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=["PCA Color Map", "PCA XYZ", "UMAP Color Map", "UMAP XYZ"],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Image(z=pca_img), row=1, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=pca_points[:, 0], y=pca_points[:, 1], z=pca_points[:, 2], mode="markers", marker={"size": 2, "color": pca_colors}
        ),
        row=1,
        col=2,
    )
    fig.add_trace(go.Image(z=umap_img), row=2, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=umap_points[:, 0], y=umap_points[:, 1], z=umap_points[:, 2], mode="markers", marker={"size": 2, "color": umap_colors}
        ),
        row=2,
        col=2,
    )
    fig.update_layout(
        title="Latent Overview: PCA/UMAP Color Maps vs XYZ",
        template="plotly_white",
        width=1400,
        height=900,
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3", "aspectmode": "cube"},
        scene2={"xaxis_title": "U1", "yaxis_title": "U2", "zaxis_title": "U3", "aspectmode": "cube"},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=2, col=1)
    out_path = os.path.join(session_dir, "latent_overview_4panel.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def save_inference_dashboard(session_dir: str, outputs: dict) -> str:
    x_clean_raw = outputs.get("x_clean_raw", outputs["x_clean"])
    x_context_raw = outputs.get("x_context_raw", outputs["x_context"])
    x_clean = outputs["x_clean"]
    x_context = outputs["x_context"]
    target_locations = outputs["target_locations"]
    target_scales = outputs["target_scales"]
    target_valid = outputs["target_valid"]
    pred_map = outputs["pred_map"]
    gt_map = outputs["gt_map"]
    context_map = outputs.get("context_map")

    orig = x_clean_raw[0, 0].detach().cpu().numpy()
    ctx = x_context_raw[0, 0].detach().cpu().numpy()

    # Render sampled target locations for first sample.
    target_vis = np.zeros_like(orig, dtype=np.float32)
    for i in range(target_locations.shape[1]):
        cy = int(target_locations[0, i, 0].item())
        cx = int(target_locations[0, i, 1].item())
        if 0 <= cy < target_vis.shape[0] and 0 <= cx < target_vis.shape[1]:
            target_vis[cy, cx] = 1.0

    pred_vec = pred_map.detach().cpu().permute(0, 2, 3, 1).reshape(-1, pred_map.shape[1]).numpy()
    gt_vec = gt_map.detach().cpu().permute(0, 2, 3, 1).reshape(-1, gt_map.shape[1]).numpy()
    x = np.concatenate([pred_vec, gt_vec], axis=0)
    y = np.concatenate(
        [np.zeros(pred_vec.shape[0], dtype=np.int32), np.ones(gt_vec.shape[0], dtype=np.int32)], axis=0
    )

    pca_cache = os.path.join(session_dir, "pca_embeddings.npy")
    umap_cache = os.path.join(session_dir, "umap_embeddings.npy")
    if os.path.exists(pca_cache):
        try:
            pca_2d = np.load(pca_cache)
        except Exception:
            pca_2d = _compute_pca_2d(x)
            np.save(pca_cache, pca_2d)
    else:
        pca_2d = _compute_pca_2d(x)
        np.save(pca_cache, pca_2d)
    if os.path.exists(umap_cache):
        try:
            umap_3d = np.load(umap_cache)
        except Exception:
            umap_3d = _compute_umap_nd(x, n_components=3)
            np.save(umap_cache, umap_3d)
    else:
        umap_3d = _compute_umap_nd(x, n_components=3)
        np.save(umap_cache, umap_3d)
    # Session plot compatibility artifacts.
    results_dir = os.path.join(session_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "latent_vectors_full.npy"), x.astype(np.float32))
    np.save(os.path.join(results_dir, "umap_x.npy"), umap_3d[:, 0].astype(np.float32))
    np.save(os.path.join(results_dir, "umap_y.npy"), umap_3d[:, 1].astype(np.float32))
    np.save(os.path.join(results_dir, "umap_z.npy"), umap_3d[:, 2].astype(np.float32))

    def _save_branch_embeddings(branch_name: str, fmap: torch.Tensor):
        # Use sample-0 dense latent map (H*W tokens) for branch-specific plotly 2D color + 3D scatter.
        h_map = int(fmap.shape[-2])
        w_map = int(fmap.shape[-1])
        z = fmap[0].detach().cpu().permute(1, 2, 0).reshape(-1, fmap.shape[1]).numpy().astype(np.float32)
        pca3 = _compute_pca_3d(z).astype(np.float32)
        umap3 = _compute_umap_nd(z, n_components=3).astype(np.float32)
        np.save(os.path.join(results_dir, f"{branch_name}_spatial_shape.npy"), np.asarray([h_map, w_map], dtype=np.int64))
        np.save(os.path.join(results_dir, f"{branch_name}_latent_vectors_full.npy"), z)
        np.save(os.path.join(results_dir, f"{branch_name}_pca_xyz.npy"), pca3)
        np.save(os.path.join(results_dir, f"{branch_name}_pca_x.npy"), pca3[:, 0])
        np.save(os.path.join(results_dir, f"{branch_name}_pca_y.npy"), pca3[:, 1])
        np.save(os.path.join(results_dir, f"{branch_name}_pca_z.npy"), pca3[:, 2])
        np.save(os.path.join(results_dir, f"{branch_name}_umap_x.npy"), umap3[:, 0])
        np.save(os.path.join(results_dir, f"{branch_name}_umap_y.npy"), umap3[:, 1])
        np.save(os.path.join(results_dir, f"{branch_name}_umap_z.npy"), umap3[:, 2])

    _save_branch_embeddings("predict", pred_map)
    _save_branch_embeddings("target", gt_map)
    if context_map is not None:
        _save_branch_embeddings("context", context_map)

    # Spatial latent map overview (sample-0 pred map only): PCA/UMAP colormap + XYZ scatter.
    pred0 = pred_map[0].detach().cpu().permute(1, 2, 0).reshape(-1, pred_map.shape[1]).numpy().astype(np.float32)
    pca0_3d = _compute_pca_3d(pred0)
    umap0_3d = _compute_umap_nd(pred0, n_components=3)
    latent_html_path = _save_latent_overview_html(session_dir, pca0_3d, umap0_3d, pred_map.shape[-2], pred_map.shape[-1])

    # Historical target-location heatmap loaded from session CSV log.
    hist_vis = np.zeros_like(orig, dtype=np.float32)
    hist_path = os.path.join(session_dir, "visited_target_locations.csv")
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cy = int(float(row["y"]))
                    cx = int(float(row["x"]))
                    if 0 <= cy < hist_vis.shape[0] and 0 <= cx < hist_vis.shape[1]:
                        hist_vis[cy, cx] += 1.0
        except Exception:
            pass
    if float(hist_vis.max()) > 0.0:
        hist_vis = hist_vis / float(hist_vis.max())

    # PNG dashboard export is disabled; use HTML dashboards only.
    return latent_html_path


def save_loss_curve(session_dir: str):
    metrics_path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return None
    x_ep = []
    total = []
    jepa = []
    pixel = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_ep.append(float(row["epoch"]) + 0.001 * float(row["batch"]))
            total.append(float(row["total_loss"]))
            jepa.append(float(row["loss_jepa"]))
            pixel.append(float(row["loss_pixel"]))
    if len(x_ep) == 0:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(x_ep, total, label="total_loss")
    ax.plot(x_ep, jepa, label="loss_jepa")
    ax.plot(x_ep, pixel, label="loss_pixel")
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("epoch + 0.001*batch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    out_path = os.path.join(session_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("offdiag expects square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def compute_sim_var_cov(outputs: dict) -> tuple[float, float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    valid = outputs["target_valid"].detach()  # B,K

    b, k, c, p, _ = pred.shape
    # reshape to (B,K,P*P,C), then gather valid rows
    pred_v = pred.permute(0, 1, 3, 4, 2).reshape(b, k, p * p, c)
    gt_v = gt.permute(0, 1, 3, 4, 2).reshape(b, k, p * p, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, p * p, 1).reshape(-1)
    z1 = pred_v.reshape(-1, c)[vm]
    z2 = gt_v.reshape(-1, c)[vm]
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


def compute_raw_mse_and_norm_err(outputs: dict) -> tuple[float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    valid = outputs["target_valid"].detach()  # B,K

    b, k, c, p, _ = pred.shape
    pred_v = pred.permute(0, 1, 3, 4, 2).reshape(b, k, p * p, c)
    gt_v = gt.permute(0, 1, 3, 4, 2).reshape(b, k, p * p, c)
    vm = valid.unsqueeze(-1).unsqueeze(-1).expand(b, k, p * p, 1).reshape(-1)
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
    energy_per_target = (pred - gt).pow(2).mean(dim=(2, 3, 4))
    if bool(valid.any()):
        return float(energy_per_target[valid].mean().item())
    return 0.0


def compute_target_energy_map(outputs: dict, image_size: tuple[int, int]) -> torch.Tensor:
    pred = outputs["pred_patches"]
    gt = outputs["gt_patches"].detach()
    loc = outputs["target_locations"]
    valid = outputs["target_valid"]
    h, w = int(image_size[0]), int(image_size[1])
    b, k, _, _, _ = pred.shape
    energy_map = torch.zeros((b, 1, h, w), device=pred.device, dtype=pred.dtype)
    count_map = torch.zeros((b, 1, h, w), device=pred.device, dtype=pred.dtype)
    err = (pred - gt).pow(2).mean(dim=(2, 3, 4))
    for bi in range(b):
        for ki in range(k):
            if not bool(valid[bi, ki]):
                continue
            y = int(loc[bi, ki, 0].item())
            x = int(loc[bi, ki, 1].item())
            if 0 <= y < h and 0 <= x < w:
                energy_map[bi, 0, y, x] += err[bi, ki]
                count_map[bi, 0, y, x] += 1.0
    energy_map = energy_map / count_map.clamp_min(1.0)
    return energy_map


def compute_error_by_scale(outputs: dict) -> dict[float, float]:
    pred = outputs["pred_patches"].detach()  # B,K,C,P,P
    gt = outputs["gt_patches"].detach()  # B,K,C,P,P
    scales = outputs["target_scales"].detach()  # B,K
    valid = outputs["target_valid"].detach()  # B,K

    # Per-target MSE averaged over C,P,P
    mse_bk = torch.mean((pred - gt) ** 2, dim=(2, 3, 4))  # B,K
    out = defaultdict(list)
    b, k = mse_bk.shape
    for bi in range(b):
        for ki in range(k):
            if not bool(valid[bi, ki].item()):
                continue
            s = round(float(scales[bi, ki].item()), 6)
            out[s].append(float(mse_bk[bi, ki].item()))
    return {float(s): float(np.mean(v)) for s, v in out.items() if len(v) > 0}


@torch.no_grad()
def evaluate_validation(model: PyramidGridJEPA, val_loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict:
    model.eval()
    n = 0
    loss_sum = 0.0
    sim_sum = 0.0
    scale_mse = defaultdict(list)
    for batch_idx, x_raw in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_raw = x_raw.to(device, non_blocking=True)
        outputs = model(x_raw)
        loss = model.compute_loss(outputs)
        sim_val, _, _ = compute_sim_var_cov(outputs)
        ebs = compute_error_by_scale(outputs)
        for s, v in ebs.items():
            scale_mse[s].append(float(v))
        loss_sum += float(loss.item())
        sim_sum += float(sim_val)
        n += 1

    if n == 0:
        return {"val_loss": 0.0, "val_sim": 0.0, "val_error_by_scale": {}}
    return {
        "val_loss": loss_sum / n,
        "val_sim": sim_sum / n,
        "val_error_by_scale": {float(s): float(np.mean(v)) for s, v in scale_mse.items()},
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_session_dir(root: str, config_name: str) -> str:
    path = os.path.join(root, config_name)
    os.makedirs(path, exist_ok=True)
    return path


def run_training(config: dict, config_name: str, sessions_root: str = "sessions") -> str:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print(
        f"[{config_name}] backend_discovered device={device.type} "
        f"cuda_available={torch.cuda.is_available()} mps_available={mps_available}"
    )

    train_cfg = config["train"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    session_dir = make_session_dir(sessions_root, config_name)
    os.makedirs(session_dir, exist_ok=True)
    model_ckpt_path = os.path.join(session_dir, "model_last.pt")
    resume_ckpt_path = os.path.join(session_dir, "checkpoint_last.pt")
    resume_from_existing = os.path.exists(model_ckpt_path)

    with open(os.path.join(session_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    blur_mode = model_cfg.get("blur_mode", "gaussian")
    dataset_log_transform = bool(data_cfg.get("log_transform", True))
    dataset_apply_cdd = True
    model_post_log = True
    # Pipeline policy:
    # - blur_mode == cdd:
    #   dataset: normalize only
    #   model: CDD mask + shared log
    # - blur_mode == gaussian:
    #   dataset: normalize + mandatory dataset CDD, no dataset log
    #   model: gaussian masking + shared optional log
    if blur_mode == "cdd":
        # REQUIRED policy: dataset normalize-only; model does CDD masking + shared log.
        dataset_log_transform = False
        dataset_apply_cdd = False
        model_post_log = bool(data_cfg.get("log_transform", True))
    elif blur_mode == "gaussian":
        # Gaussian path: dataset CDD yes, dataset pre-log no, model shared-log yes.
        dataset_apply_cdd = True
        dataset_log_transform = False
        model_post_log = bool(data_cfg.get("log_transform", True))

    print(
        f"[{config_name}] mode: blur_mode={blur_mode} "
        f"dataset_apply_cdd={dataset_apply_cdd} dataset_log_transform={dataset_log_transform} "
        f"post_log_transform={model_post_log} cdd_mode={model_cfg.get('cdd_mode', data_cfg.get('cdd_mode', 'log'))}"
    )

    model = PyramidGridJEPA(
        latent_channels=model_cfg.get("latent_channels", 32),
        predictor_hidden=model_cfg.get("predictor_hidden"),
        patch_size=model_cfg.get("patch_size", 2),
        sigmas=tuple(model_cfg.get("sigmas", [2, 4, 8, 16])),
        cell_sizes=tuple(model_cfg.get("cell_sizes", [16, 32, 64, 128])),
        max_targets_per_image=model_cfg.get("max_targets_per_image", 16),
        mask_fraction=model_cfg.get("mask_fraction", 0.20),
        spacing_mult=model_cfg.get("spacing_mult", 1.5),
        box_sigma_mult=model_cfg.get("box_sigma_mult", 4.0),
        mask_scale=model_cfg.get("mask_scale", 1.0),
        min_mask_scale=model_cfg.get("min_mask_scale", 0.0),
        spacing_scale=model_cfg.get("spacing_scale", 2.0),
        full_grid=model_cfg.get("full_grid", True),
        global_shift=model_cfg.get("global_shift", True),
        align_scales=model_cfg.get("align_scales", True),
        constant_mask_box=model_cfg.get("constant_mask_box", True),
        mask_box_size=model_cfg.get("mask_box_size", 16),
        blur_mode=blur_mode,
        cdd_mode=model_cfg.get("cdd_mode", "log"),
        cdd_constrained=model_cfg.get("cdd_constrained", True),
        cdd_sm_mode=model_cfg.get("cdd_sm_mode", "reflect"),
        mask_fill_mode=model_cfg.get("mask_fill_mode", "zero"),
        dip_sigma_mult=model_cfg.get("dip_sigma_mult", 1.0),
        post_log_transform=model_cfg.get("post_log_transform", model_post_log),
        log_eps=model_cfg.get("log_eps", float(data_cfg.get("log_eps", 1.0))),
        cdd_log_std_floor_mult=model_cfg.get("cdd_log_std_floor_mult", 0.05),
        ema_momentum=model_cfg.get("ema_momentum", train_cfg.get("momentum", 0.996)),
        normalize_loss=model_cfg.get("normalize_loss", True),
    ).to(device)
    start_epoch = 0
    resume_state = None
    if os.path.exists(resume_ckpt_path):
        resume_state = torch.load(resume_ckpt_path, map_location=device)
        if "model_state_dict" in resume_state:
            model.load_state_dict(resume_state["model_state_dict"], strict=False)
        start_epoch = int(resume_state.get("epoch", 0))
        print(f"resume_checkpoint={resume_ckpt_path} start_epoch={start_epoch}")
    elif resume_from_existing:
        model.load_state_dict(torch.load(model_ckpt_path, map_location=device), strict=False)
        print(f"resume_model={model_ckpt_path}")

    scale_max = float(max(model_cfg.get("sigmas", [2, 4, 8, 16])))
    auto_roll_max = max(1, int(round(scale_max * float(model_cfg.get("mask_scale", 1.0)) * float(model_cfg.get("spacing_scale", 2.0)))))

    dataset = JEPADataset(
        num_samples=data_cfg.get("num_samples", 2000),
        image_size=data_cfg.get("image_size", 256),
        data_root=data_cfg.get("data_root", "data"),
        npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
        log_transform=dataset_log_transform,
        log_eps=data_cfg.get("log_eps", 1.0),
        cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
        cdd_strength=data_cfg.get("cdd_strength", 1.0),
        cdd_clip=data_cfg.get("cdd_clip", True),
        norm_before_cdd=data_cfg.get("norm_before_cdd", True),
        cdd_mode=data_cfg.get("cdd_mode", "log"),
        cdd_constrained=data_cfg.get("cdd_constrained", True),
        cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
        apply_cdd=dataset_apply_cdd,
        cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
        cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
        cube_slice_index=data_cfg.get("cube_slice_index", 0),
        random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
        cache_cdd=bool(data_cfg.get("cache_cdd", True)),
        cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
        cdd_mem_cache_max=int(data_cfg.get("cdd_mem_cache_max", 64)),
        cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
    )
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    val_fraction = min(max(val_fraction, 0.0), 0.95)
    total_idx = list(dataset.sample_index)
    n_total = len(total_idx)
    n_val_idx = int(round(n_total * val_fraction)) if n_total > 1 else 0
    if val_fraction > 0.0 and n_val_idx == 0 and n_total > 1:
        n_val_idx = 1
    n_train_idx = max(1, n_total - n_val_idx)
    train_idx = total_idx[:n_train_idx]
    val_idx = total_idx[n_train_idx:] if n_val_idx > 0 else []

    train_dataset = dataset
    train_dataset.sample_index = train_idx
    train_dataset.num_samples = int(train_cfg.get("num_samples", data_cfg.get("num_samples", 2000)))

    val_dataset = None
    if len(val_idx) > 0:
        val_dataset = JEPADataset(
            num_samples=max(1, int(train_cfg.get("val_num_samples", max(16, int(0.25 * train_dataset.num_samples))))),
            image_size=data_cfg.get("image_size", 256),
            data_root=data_cfg.get("data_root", "data"),
            npy_pattern=data_cfg.get("npy_pattern", "*.npy"),
            log_transform=dataset_log_transform,
            log_eps=data_cfg.get("log_eps", 1.0),
            cdd_scales=data_cfg.get("cdd_scales", [2, 4, 8, 16]),
            cdd_strength=data_cfg.get("cdd_strength", 1.0),
            cdd_clip=data_cfg.get("cdd_clip", True),
            norm_before_cdd=data_cfg.get("norm_before_cdd", True),
            cdd_mode=data_cfg.get("cdd_mode", "log"),
            cdd_constrained=data_cfg.get("cdd_constrained", True),
            cdd_sm_mode=data_cfg.get("cdd_sm_mode", "reflect"),
            apply_cdd=dataset_apply_cdd,
            cube_slice_strategy=data_cfg.get("cube_slice_strategy", "random"),
            cube_slice_axis=data_cfg.get("cube_slice_axis", 0),
            cube_slice_index=data_cfg.get("cube_slice_index", 0),
            random_roll_max=int(max(0, data_cfg.get("random_roll_max", auto_roll_max))),
            cache_cdd=bool(data_cfg.get("cache_cdd", True)),
            cdd_cache_dir=data_cfg.get("cdd_cache_dir"),
            cdd_mem_cache_max=int(data_cfg.get("cdd_mem_cache_max", 64)),
            cache_random_slices=bool(data_cfg.get("cache_random_slices", False)),
        )
        val_dataset.sample_index = val_idx
    print(
        f"[{config_name}] dataset_split total_index={n_total} train_index={len(train_idx)} "
        f"val_index={len(val_idx)} val_fraction={val_fraction:.3f}"
    )
    print(
        f"[{config_name}] data_jitter random_roll_max={dataset.random_roll_max} "
        f"(symmetric inclusive roll in [-max,+max])"
    )
    requested_workers = int(train_cfg.get("num_workers", 4))
    # macOS/MPS-safe default: avoid multiprocessing worker hangs unless explicitly set.
    if "num_workers" in train_cfg:
        num_workers = requested_workers
    else:
        num_workers = 4 if device.type == "cuda" else 0
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(num_workers > 0)
    print(
        f"[{config_name}] dataloader_setup num_workers={num_workers} "
        f"pin_memory={pin_memory} persistent_workers={persistent_workers}"
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    use_cuda_amp = torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_cuda_amp)
    if resume_state is not None:
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if "scaler_state_dict" in resume_state and use_cuda_amp:
            scaler.load_state_dict(resume_state["scaler_state_dict"])

    epochs = train_cfg.get("epochs", 20)
    log_interval = train_cfg.get("log_interval", 10)

    metrics_path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "batch",
                    "total_loss",
                    "loss_jepa",
                    "loss_pixel",
                    "sim",
                    "var",
                    "cov",
                    "raw_mse",
                    "norm_err",
                    "time_sec",
                ]
            )
    masked_scales_log_path = os.path.join(session_dir, "masked_scales_log.csv")
    if not os.path.exists(masked_scales_log_path):
        with open(masked_scales_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "scale", "count"])
    epoch_summary_path = os.path.join(session_dir, "epoch_summary.csv")
    if not os.path.exists(epoch_summary_path):
        with open(epoch_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_sim", "val_error_by_scale_json"])
    visited_targets_log_path = os.path.join(session_dir, "visited_target_locations.csv")
    if not os.path.exists(visited_targets_log_path):
        with open(visited_targets_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "batch", "sample_idx", "target_idx", "y", "x", "scale"])

    model.train()
    start = time.time()
    visit_counts = None
    if start_epoch >= int(epochs):
        print(f"[{config_name}] checkpoint epoch {start_epoch} already >= configured epochs {epochs}, skipping training loop")
    for epoch in range(start_epoch, epochs):
        epoch_total = 0.0
        epoch_jepa = 0.0
        epoch_pixel = 0.0
        epoch_sim = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_batches = 0
        for batch_idx, x_raw in enumerate(dataloader):
            x_raw = x_raw.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_cuda_amp):
                outputs = model(x_raw)
                loss_jepa = model.compute_loss(outputs)
                total_loss = loss_jepa
                loss_pixel = torch.zeros_like(total_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model.update_target_encoder()
            sim_val, var_val, cov_val = compute_sim_var_cov(outputs)
            raw_mse_val, norm_err_val = compute_raw_mse_and_norm_err(outputs)

            elapsed = time.time() - start
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    batch_idx,
                    float(total_loss.item()),
                    float(loss_jepa.item()),
                    float(loss_pixel.item()),
                    float(sim_val),
                    float(var_val),
                    float(cov_val),
                    float(raw_mse_val),
                    float(norm_err_val),
                    round(elapsed, 4),
                ])
            # Save masked-scale usage as training log in session dir.
            scales = outputs["target_scales"].detach().cpu().numpy()
            valid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
            valid_scales = scales[valid]
            if valid_scales.size > 0:
                uniq, cnt = np.unique(np.round(valid_scales.astype(np.float32), 6), return_counts=True)
                with open(masked_scales_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for s, c in zip(uniq.tolist(), cnt.tolist()):
                        writer.writerow([epoch + 1, batch_idx, float(s), int(c)])
            # Save visited target locations for full-session diagnostics.
            tloc = outputs["target_locations"].detach().cpu().numpy()
            tvalid = outputs["target_valid"].detach().cpu().numpy().astype(bool)
            tscale = outputs["target_scales"].detach().cpu().numpy()
            if visit_counts is None:
                hh, ww = int(outputs["x_clean"].shape[-2]), int(outputs["x_clean"].shape[-1])
                visit_counts = np.zeros((hh, ww), dtype=np.float32)
            for bi in range(tloc.shape[0]):
                for ki in range(tloc.shape[1]):
                    if not bool(tvalid[bi, ki]):
                        continue
                    yy = int(tloc[bi, ki, 0])
                    xx = int(tloc[bi, ki, 1])
                    if 0 <= yy < visit_counts.shape[0] and 0 <= xx < visit_counts.shape[1]:
                        visit_counts[yy, xx] += 1.0
            with open(visited_targets_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                bsz = tloc.shape[0]
                ksz = tloc.shape[1]
                for bi in range(bsz):
                    for ki in range(ksz):
                        if not bool(tvalid[bi, ki]):
                            continue
                        writer.writerow(
                            [
                                epoch + 1,
                                batch_idx,
                                bi,
                                ki,
                                int(tloc[bi, ki, 0]),
                                int(tloc[bi, ki, 1]),
                                float(tscale[bi, ki]),
                            ]
                        )
        if visit_counts is not None:
            np.save(os.path.join(session_dir, "visited_target_frequency.npy"), visit_counts.astype(np.float32))

            if batch_idx % log_interval == 0:
                print(
                    f"[{config_name}] Epoch {epoch + 1}/{epochs} Batch {batch_idx}/{len(dataloader)} "
                    f"total={total_loss.item():.4f} jepa={loss_jepa.item():.4f} pixel={loss_pixel.item():.4f} "
                    f"sim={sim_val:.4f} var={var_val:.4f} cov={cov_val:.4f} "
                    f"raw_mse={raw_mse_val:.4f} norm_err={norm_err_val:.4f}"
                )
            epoch_total += float(total_loss.item())
            epoch_jepa += float(loss_jepa.item())
            epoch_pixel += float(loss_pixel.item())
            epoch_sim += float(sim_val)
            epoch_var += float(var_val)
            epoch_cov += float(cov_val)
            epoch_batches += 1

        if epoch_batches > 0:
            print(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} summary "
                f"avg_total={epoch_total/epoch_batches:.4f} "
                f"avg_jepa={epoch_jepa/epoch_batches:.4f} "
                f"avg_pixel={epoch_pixel/epoch_batches:.4f} "
                f"avg_sim={epoch_sim/epoch_batches:.4f} "
                f"avg_var={epoch_var/epoch_batches:.4f} "
                f"avg_cov={epoch_cov/epoch_batches:.4f}"
            )
        val_loss = 0.0
        val_sim = 0.0
        val_error_by_scale = {}
        if val_loader is not None:
            v = evaluate_validation(
                model=model,
                val_loader=val_loader,
                device=device,
                max_batches=train_cfg.get("val_max_batches"),
            )
            val_loss = float(v["val_loss"])
            val_sim = float(v["val_sim"])
            val_error_by_scale = dict(v["val_error_by_scale"])
            print(
                f"[{config_name}] Epoch {epoch + 1}/{epochs} validation "
                f"val_loss={val_loss:.4f} val_sim={val_sim:.4f} "
                f"val_error_by_scale={json.dumps(val_error_by_scale, sort_keys=True)}"
            )
        with open(epoch_summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    round(epoch_total / max(1, epoch_batches), 8),
                    round(val_loss, 8),
                    round(val_sim, 8),
                    json.dumps(val_error_by_scale, sort_keys=True),
                ]
            )
        model.train()
        # Save resumable checkpoint at the end of every epoch.
        torch.save(
            {
                "epoch": int(epoch + 1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config_name": config_name,
            },
            resume_ckpt_path,
        )
        # Keep model_last in sync for inference-only resume paths.
        torch.save(model.state_dict(), model_ckpt_path)
        print(f"[{config_name}] checkpoint_saved={resume_ckpt_path} epoch={epoch + 1}")

    torch.save(model.state_dict(), os.path.join(session_dir, "model_last.pt"))

    print(f"[{config_name}] post_training_inference begin")
    model.eval()
    with torch.no_grad():
        print(f"[{config_name}] post_training_inference loading sample batch")
        x_raw = next(iter(dataloader))
        x_raw = x_raw.to(device)
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
    energy_scalar = compute_jepa_energy(outputs, normalize=False)
    energy_scalar_norm = compute_jepa_energy(outputs, normalize=True)
    e_map = compute_target_energy_map(outputs, image_size=outputs["x_clean"].shape[-2:])
    inference_outputs["jepa_energy"] = torch.tensor(energy_scalar, dtype=torch.float32)
    inference_outputs["jepa_energy_normalized"] = torch.tensor(energy_scalar_norm, dtype=torch.float32)
    inference_outputs["target_energy_map"] = e_map[:8].detach().cpu()
    # Canonical visualization target map for downstream plotting tools.
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
    torch.save(inference_outputs, os.path.join(session_dir, "inference_outputs.pt"))
    print(f"[{config_name}] saved inference_outputs.pt")

    # Explicitly save the exact network inputs for quick external inspection.
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

    print(f"[{config_name}] building dashboard artifacts (pca/umap may take time on first run)")
    dashboard_path = save_inference_dashboard(session_dir, inference_outputs)
    print(f"dashboard_saved={dashboard_path}")
    loss_curve_path = save_loss_curve(session_dir)
    if loss_curve_path is not None:
        print(f"loss_curve_saved={loss_curve_path}")

    return session_dir
