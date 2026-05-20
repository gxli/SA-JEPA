import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _compute_pca_2d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(x)
    except Exception:
        x_t = torch.from_numpy(x.astype(np.float32))
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        u, s, _ = torch.pca_lowrank(x_t, q=2)
        return (u[:, :2] * s[:2]).cpu().numpy()


def _compute_umap_2d(x: np.ndarray) -> np.ndarray:
    try:
        from cuml.manifold import UMAP as CuMLUMAP

        return CuMLUMAP(n_components=2, random_state=42).fit_transform(x)
    except Exception:
        pass

    try:
        import torchdr

        if hasattr(torchdr, "UMAP"):
            model = torchdr.UMAP(n_components=2)
            z = model.fit_transform(torch.from_numpy(x.astype(np.float32)))
            if isinstance(z, torch.Tensor):
                return z.cpu().numpy()
            return np.asarray(z)
    except Exception:
        pass

    try:
        import umap

        return umap.UMAP(n_components=2, random_state=42).fit_transform(x)
    except Exception:
        pass

    return _compute_pca_2d(x)


def _canonical_from_outputs(outputs: dict) -> dict:
    """
    Canonical fields:
      orig: HxW
      context: HxW
      target: HxW
      pred_latent: BxCxHxW
      gt_latent: BxCxHxW
    """
    if "x_clean" in outputs and "x_context" in outputs and "pred_map" in outputs and "gt_map" in outputs:
        x_clean = outputs["x_clean"]
        x_context = outputs["x_context"]
        pred_latent = outputs["pred_map"]
        gt_latent = outputs["gt_map"]
        orig = x_clean[0, 0].detach().cpu().numpy()
        context = x_context[0, 0].detach().cpu().numpy()
        if "target_map" in outputs:
            target = outputs["target_map"][0, 0].detach().cpu().numpy()
        else:
            target_locations = outputs["target_locations"]
            h, w = orig.shape
            target = np.zeros((h, w), dtype=np.float32)
            for i in range(target_locations.shape[1]):
                cy = int(target_locations[0, i, 0].item())
                cx = int(target_locations[0, i, 1].item())
                if 0 <= cy < h and 0 <= cx < w:
                    target[cy, cx] = 1.0
        return {
            "orig": orig,
            "context": context,
            "target": target,
            "pred_latent": pred_latent,
            "gt_latent": gt_latent,
        }

    # Legacy segmentation-like schema fallback.
    x_raw = outputs.get("x_raw")
    true_mask = outputs["true_mask"]
    pred_mask_logits = outputs["pred_mask_logits"]
    pred_latent = outputs["pred_latent"]
    gt_latent = outputs["gt_latent"]
    if x_raw is not None:
        orig = x_raw[0, 0].detach().cpu().numpy()
        context = x_raw[0, 1].detach().cpu().numpy() if x_raw.shape[1] > 1 else orig
    else:
        h, w = true_mask.shape[-2], true_mask.shape[-1]
        orig = np.zeros((h, w), dtype=np.float32)
        context = np.zeros((h, w), dtype=np.float32)
    target = true_mask[0, 0].detach().cpu().numpy()
    _ = pred_mask_logits
    return {
        "orig": orig,
        "context": context,
        "target": target,
        "pred_latent": pred_latent,
        "gt_latent": gt_latent,
    }


def compute_dash_data(session_dir: str, overwrite: bool = False) -> str:
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    out_npz = os.path.join(session_dir, "dash_data.npz")
    if os.path.exists(out_npz) and not overwrite:
        return out_npz
    if not os.path.exists(inf_path):
        raise FileNotFoundError(f"Missing inference outputs: {inf_path}")

    outputs = torch.load(inf_path, map_location="cpu")
    canon = _canonical_from_outputs(outputs)
    orig = canon["orig"]
    blurred = canon["context"]
    target = canon["target"]
    pred_latent = canon["pred_latent"]
    gt_latent = canon["gt_latent"]
    pred_mask = pred_latent[0].detach().cpu().norm(dim=0).numpy()

    # Full 2D pixel-by-pixel embedding collection for inference visualization.
    pred_vec = pred_latent.detach().cpu().permute(0, 2, 3, 1).reshape(-1, pred_latent.shape[1]).numpy()
    gt_vec = gt_latent.detach().cpu().permute(0, 2, 3, 1).reshape(-1, gt_latent.shape[1]).numpy()
    x = np.concatenate([pred_vec, gt_vec], axis=0)
    y = np.concatenate(
        [np.zeros(pred_vec.shape[0], dtype=np.int32), np.ones(gt_vec.shape[0], dtype=np.int32)], axis=0
    )

    pca_cache = os.path.join(session_dir, "pca_embeddings.npy")
    umap_cache = os.path.join(session_dir, "umap_embeddings.npy")
    pca_2d = None
    umap_2d = None
    if os.path.exists(pca_cache):
        try:
            pca_2d = np.load(pca_cache)
        except Exception:
            pca_2d = None
    if os.path.exists(umap_cache):
        try:
            umap_2d = np.load(umap_cache)
        except Exception:
            umap_2d = None

    expected_n = x.shape[0]
    if pca_2d is None or pca_2d.shape[0] != expected_n or pca_2d.shape[1] < 2:
        pca_2d = _compute_pca_2d(x)
        np.save(pca_cache, pca_2d)
    if umap_2d is None or umap_2d.shape[0] != expected_n or umap_2d.shape[1] < 2:
        umap_2d = _compute_umap_2d(x)
        np.save(umap_cache, umap_2d)

    np.savez_compressed(
        out_npz,
        orig=orig,
        blurred=blurred,
        target=target,
        pred_mask=pred_mask,
        y=y,
        pca_2d=pca_2d,
        umap_2d=umap_2d,
    )
    return out_npz


def plot_dash(session_dir: str, overwrite: bool = False) -> str:
    npz_path = os.path.join(session_dir, "dash_data.npz")
    out_png = os.path.join(session_dir, "dashboard.png")
    if os.path.exists(out_png) and not overwrite:
        return out_png
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing computed dash data: {npz_path}")

    data = np.load(npz_path)
    orig = data["orig"]
    blurred = data["blurred"]
    target = data["target"]
    pred_mask = data["pred_mask"]
    y = data["y"]
    pca_2d = data["pca_2d"]
    umap_2d = data["umap_2d"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    axes[0].imshow(orig, cmap="viridis")
    axes[0].set_title("Input (Log-Norm)")
    axes[0].axis("off")
    axes[1].imshow(blurred, cmap="viridis")
    axes[1].set_title("CDD Blurred/Add-Back")
    axes[1].axis("off")
    axes[2].imshow(target, cmap="gray")
    axes[2].set_title("Target Locations")
    axes[2].axis("off")
    axes[3].imshow(pred_mask, cmap="magma")
    axes[3].set_title("Predicted Mask (Sigmoid)")
    axes[3].axis("off")
    axes[4].scatter(pca_2d[y == 0, 0], pca_2d[y == 0, 1], s=8, alpha=0.7, label="Pred")
    axes[4].scatter(pca_2d[y == 1, 0], pca_2d[y == 1, 1], s=8, alpha=0.7, label="GT")
    axes[4].set_title("PCA (Latent Tokens)")
    axes[4].legend(loc="best", fontsize=8)
    axes[5].scatter(umap_2d[y == 0, 0], umap_2d[y == 0, 1], s=8, alpha=0.7, label="Pred")
    axes[5].scatter(umap_2d[y == 1, 0], umap_2d[y == 1, 1], s=8, alpha=0.7, label="GT")
    axes[5].set_title("UMAP (Latent Tokens)")
    axes[5].legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Build dashboards from existing sessions")
    parser.add_argument("--sessions-dir", type=str, default="sessions")
    parser.add_argument("--stage", type=str, choices=["compute", "plot", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output exists")
    args = parser.parse_args()

    if not os.path.isdir(args.sessions_dir):
        raise FileNotFoundError(f"Sessions dir not found: {args.sessions_dir}")

    for name in sorted(os.listdir(args.sessions_dir)):
        session_dir = os.path.join(args.sessions_dir, name)
        if not os.path.isdir(session_dir):
            continue
        inf_path = os.path.join(session_dir, "inference_outputs.pt")
        if not os.path.exists(inf_path):
            continue

        if args.stage in ("compute", "all"):
            npz_path = compute_dash_data(session_dir, overwrite=args.overwrite)
            print(f"dash_data_saved={npz_path}")
        if args.stage in ("plot", "all"):
            if not os.path.exists(os.path.join(session_dir, "dash_data.npz")):
                compute_dash_data(session_dir, overwrite=args.overwrite)
            png_path = plot_dash(session_dir, overwrite=args.overwrite)
            print(f"dashboard_saved={png_path}")


if __name__ == "__main__":
    main()
