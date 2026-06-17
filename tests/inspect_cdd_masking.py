"""Investigate CDD + masking consistency on the C12 MHD test data.

Uses the gen_139 ms=1.2 config defaults to inspect scale channels,
mask footprints, target sampling, and embedding spread.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import numpy as np
import torch

from src.train import load_config
from src.utils.npy import _safe_load_npy
from src.models.masking import prepare_context_batch, _effective_mask_box_size


def main():
    # Load config and data
    cfg = load_config("configs/examples/mhd_2d_ms1p2.json")
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    npy_pattern = data_cfg.get("npy_pattern", "")
    sigmas = model_cfg.get("sigmas", [2, 4, 8, 16])
    mask_scale = float(model_cfg.get("mask_size_scaling", 1.2))
    mask_box_size = int(model_cfg.get("mask_size", 0))
    cdd_mode = str(model_cfg.get("cdd_mode", "log"))
    cdd_constrained = bool(model_cfg.get("cdd_constrained", True))
    inner_target = int(model_cfg.get("patch_size", 3))

    print("=" * 60)
    print("CDD + Masking Consistency Check")
    print("=" * 60)
    print(f"  data:           {npy_pattern}")
    print(f"  sigmas:         {sigmas}")
    print(f"  mask_scale:     {mask_scale}")
    print(f"  mask_box_size:  {mask_box_size}")
    print(f"  cdd_mode:       {cdd_mode}")
    print(f"  cdd_constrained:{cdd_constrained}")
    print(f"  target patch:   {inner_target}×{inner_target}")

    # Load data
    data_path = os.path.join("data", npy_pattern)
    arr = _safe_load_npy(data_path, mmap_mode="r").astype(np.float32)
    print(f"\nField shape: {arr.shape}  dtype: {arr.dtype}")
    print(f"  nan count:  {int(np.isnan(arr).sum())}")
    print(f"  min: {float(np.nanmin(arr)):.4f}  max: {float(np.nanmax(arr)):.4f}  mean: {float(np.nanmean(arr)):.4f}  std: {float(np.nanstd(arr)):.4f}")
    arr = np.nan_to_num(arr, nan=0.0)

    # Normalize like dataset pipeline
    amin, amax = float(arr.min()), float(arr.max())
    if amax - amin > 1e-20:
        arr = (arr - amin) / (amax - amin)
    print(f"  after normalize01: min={arr.min():.4f} max={arr.max():.4f}")

    # CDD decomposition
    import constrained_diffusion as cdd
    use_gpu = torch.cuda.is_available()
    print(f"\n--- CDD Decomposition (use_gpu={use_gpu}) ---")
    channels, residual = cdd.constrained_diffusion_decomposition(
        arr,
        num_channels=len(sigmas),
        max_scale=max(sigmas),
        mode=cdd_mode,
        constrained=cdd_constrained,
        sm_mode="reflect",
        verbose=False,
        use_gpu=use_gpu,
    )
    S, H, W = channels.shape
    print(f"  CDD channels: {S}  shape: ({S}, {H}, {W})")
    print(f"  residual shape: {residual.shape}")
    for i in range(S):
        ch = channels[i]
        print(f"    ch[{i}] sigma≈{sigmas[i]:>2}:  min={ch.min():.4f}  max={ch.max():.4f}  mean={ch.mean():.4f}  std={ch.std():.4f}  dead={(ch.std() < 1e-5)}")
    print(f"    residual:  min={residual.min():.4f}  max={residual.max():.4f}  mean={residual.mean():.4f}  std={residual.std():.4f}")

    # Mask box sizes
    print(f"\n--- Mask Footprints ---")
    for sigma in sigmas:
        box = _effective_mask_box_size(
            sigma=float(sigma),
            mask_scale=mask_scale,
            mask_box_size=mask_box_size,
            inner_target_size=inner_target,
        )
        fraction = (box * box) / (H * W) * 100
        print(f"  sigma={sigma:>2}  mask_scale={mask_scale}  box={box:>3}px  ({box}×{box})  coverage={fraction:.1f}%")

    # Quick target sampling test
    print(f"\n--- Target Sampling (single sample, mask_fraction=1.0) ---")
    x_clean = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # B=1, C=1, H, W
    result = prepare_context_batch(
        x_clean=x_clean,
        sigmas=sigmas,
        mask_fraction=1.0,
        mask_scale=mask_scale,
        mask_box_size=mask_box_size,
        cdd_mode=cdd_mode,
        cdd_constrained=cdd_constrained,
    )
    x_context, tloc, tscale, tvalid, debug = result
    n_targets = int(tvalid.sum().item()) if tvalid is not None else 0
    print(f"  total targets: {n_targets}")
    print(f"  target locations shape: {tloc.shape if tloc is not None else 'N/A'}")
    print(f"  target valid shape:    {tvalid.shape if tvalid is not None else 'N/A'}")
    if tvalid is not None:
        per_scale = []
        for si in range(len(sigmas)):
            n = int(tvalid[:, si].sum().item())
            per_scale.append(n)
        print(f"  targets per scale: {per_scale}")

    # --- Latent embedding extraction ---
    print(f"\n--- Latent Embedding Extraction ---")
    session_dir = cfg.get("_session_dir", None)
    out_dir = "examples/output"
    os.makedirs(out_dir, exist_ok=True)

    # Try loading the inference outputs from the default session
    inf_path = os.path.join(session_dir, "inference_outputs.pt") if session_dir else None
    if inf_path and os.path.exists(inf_path):
        outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
        ctx_map = outputs.get("context_map")
        pred_map = outputs.get("pred_map")
        latent = (ctx_map if ctx_map is not None else pred_map).squeeze(0).cpu().numpy()
        print(f"  loaded latent from session: shape={latent.shape}")
    else:
        # Quick forward pass with a minimal model
        print("  no session found, running quick forward pass...")
        from src.models.build_jepa import PyramidGridJEPA
        model = PyramidGridJEPA(
            latent_channels=32,
            predictor_hidden=96,
            sigmas=tuple(sigmas),
            mask_scale=mask_scale,
            encoder_depth=int(model_cfg.get("encoder_depth", 4)),
            predictor_layernorm=bool(model_cfg.get("predictor_layernorm", True)),
            normalize_loss_l2=bool(model_cfg.get("normalize_loss_l2", False)),
        )
        model.eval()
        with torch.no_grad():
            x_in = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            outputs = model(x_in)
            ctx_map = outputs.get("context_map")
            latent = (ctx_map if ctx_map is not None else outputs.get("pred_map")).squeeze(0).cpu().numpy()
        print(f"  forward pass latent: shape={latent.shape}")

    # Save latent as .npy
    latent_path = os.path.join(out_dir, "latent_embedding.npy")
    np.save(latent_path, latent.astype(np.float32))
    print(f"  saved: {latent_path}")

    # UMAP on latent
    print(f"  computing UMAP 2D...")
    C, H_lat, W_lat = latent.shape
    pixels = latent.reshape(C, -1).T  # (H*W, C)
    # Subsample for speed if needed
    max_samples = 10000
    if pixels.shape[0] > max_samples:
        idx = np.random.default_rng(42).choice(pixels.shape[0], max_samples, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels
        idx = np.arange(pixels.shape[0])

    try:
        import torchdr
        device_umap = "mps" if torch.backends.mps.is_available() else "cpu"
        reducer = torchdr.UMAP(n_neighbors=50, min_dist=0.2, device=device_umap)
        emb_sample = reducer.fit_transform(torch.from_numpy(pixels_sample).float().to(device_umap))
        emb_sample = emb_sample.cpu().numpy()
    except Exception:
        import umap
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.2, metric="euclidean", random_state=42)
        emb_sample = reducer.fit_transform(pixels_sample)

    # Save UMAP
    umap_path = os.path.join(out_dir, "latent_umap_2d.npy")
    np.save(umap_path, emb_sample.astype(np.float32))
    print(f"  saved: {umap_path}  shape={emb_sample.shape}")

    # Quick UMAP stats
    print(f"  UMAP range: x=[{emb_sample[:,0].min():.2f}, {emb_sample[:,0].max():.2f}]  y=[{emb_sample[:,1].min():.2f}, {emb_sample[:,1].max():.2f}]")

    print(f"\n{'='*60}")
    print("Consistency check complete. Outputs in examples/output/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
