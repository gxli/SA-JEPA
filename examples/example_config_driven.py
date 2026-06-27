#!/usr/bin/env python3
"""Config-driven MHD example — loads config from YAML file."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, ROOT)
from sajepa import ScaleAwareJEPA


def main() -> None:
    model = ScaleAwareJEPA(config=os.path.join(ROOT, "configs", "examples", "mhd_example.yaml"))
    model.train(config_name="example_config_driven", sessions_dir=os.path.join(ROOT, "sessions"), dashboard=True)
    dashboard = os.path.join(model.session_dir, "dashboard.html")
    umap_npy = os.path.join(model.session_dir, "results", "predict_umap_xyz.npy")
    umap_html = os.path.join(model.session_dir, "results", "interactive_umap_predict.html")
    if os.path.exists(umap_npy):
        model.save_interactive_umap(umap_npy, umap_html)
    print(f"\nDone.\n  session:          {model.session_dir}\n  dashboard:        {dashboard}\n  interactive_umap: {umap_html}")


if __name__ == "__main__":
    main()
