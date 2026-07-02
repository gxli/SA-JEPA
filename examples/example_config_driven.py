#!/usr/bin/env python3
"""Config-driven MHD example — loads config from YAML file."""
import os, sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, ROOT)
import numpy as np
import torch

from sajepa import ScaleAwareJEPA
from scripts.session_umap_interactive import build_html


def _write_interactive_umap_full_latent(session_dir: str, branch: str = "predict") -> str:
    """Build UMAP-display/full-latent-similarity review HTML from dashboard artifacts."""
    session = Path(session_dir)
    dash_path = session / "dash_data.npz"
    inf_path = session / "inference_outputs.pt"
    if not dash_path.exists():
        raise FileNotFoundError(f"missing {dash_path}; generate dashboard first")
    if not inf_path.exists():
        raise FileNotFoundError(f"missing {inf_path}; run inference first")

    stems = {
        "predict": ("pred_umap3d", "pred_map"),
        "masked_predict": ("masked_pred_umap3d", "masked_pred_map"),
        "target": ("gt_umap3d", "gt_map"),
        "context": ("context_umap3d", "context_map"),
    }
    if branch not in stems:
        raise ValueError(f"unsupported branch={branch!r}; expected one of {sorted(stems)}")
    umap_key, latent_key = stems[branch]

    results = session / "results"
    results.mkdir(exist_ok=True)
    with np.load(dash_path, allow_pickle=True) as dash:
        umap = np.asarray(dash[umap_key], dtype=np.float32)

    outputs = torch.load(inf_path, map_location="cpu", weights_only=False)
    latent = outputs.get(latent_key)
    if latent is None and latent_key == "context_map":
        latent = outputs.get("pred_map")
    if latent is None:
        raise KeyError(f"{latent_key} missing from inference outputs")
    latent = torch.as_tensor(latent)[0].detach().cpu().numpy().astype(np.float32)
    if latent.ndim == 4:
        latent = latent[:, latent.shape[1] // 2]
    if latent.ndim != 3:
        raise RuntimeError(f"{latent_key} must reduce to CxHxW, got shape={latent.shape}")
    h, w = latent.shape[-2:]
    if umap.shape != (h * w, 3):
        raise RuntimeError(f"{umap_key} shape {umap.shape} does not match latent map {(h, w)}")

    umap_chw = np.transpose(umap.reshape(h, w, 3), (2, 0, 1)).astype(np.float32)
    umap_path = results / f"{branch}_strict_umap_xyz.npy"
    latent_path = results / f"{branch}_latent_vectors_full.npy"
    html_path = results / f"interactive_umap_display_full_latent_similarity_{branch}.html"
    np.save(umap_path, umap_chw)
    np.save(latent_path, latent)
    build_html(
        umap_path,
        html_path,
        1.0,
        99.0,
        similarity_input_path=latent_path,
        display_label="umap",
        similarity_label="full_latent",
    )
    return str(html_path)


def main() -> None:
    model = ScaleAwareJEPA(config=os.path.join(ROOT, "configs", "examples", "mhd_example.yaml"))
    model.train(config_name="example_config_driven", sessions_dir=os.path.join(ROOT, "sessions"), dashboard=True)
    dashboard_model = os.environ.get("SAJEPA_EXAMPLE_DASHBOARD_MODEL", "umap").strip().lower()
    model.generate_dashboard(model=dashboard_model)
    dashboard = os.path.join(model.session_dir, "dashboard.html")
    interactive_html = _write_interactive_umap_full_latent(model.session_dir, branch="predict")
    print(
        "\nDone."
        f"\n  session:             {model.session_dir}"
        f"\n  dashboard[{dashboard_model}]:  {dashboard}"
        f"\n  interactive_review:  {interactive_html}"
    )


if __name__ == "__main__":
    main()
