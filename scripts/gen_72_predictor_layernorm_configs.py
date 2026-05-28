#!/usr/bin/env python3
"""Generate gen_72 predictor/projector ablation configs with target_nonoverlap + mask_box_hardcap."""
from __future__ import annotations

import json
import os
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")

BASE_CONFIG = "../base_pyramid_scaleaware_convnext.json"
MASK_BOX_RANGE = (3, 11)

DATASETS = {
    "mhd": {"label": "mhd"},
    "ngc": {
        "label": "ngc",
        "npy_pattern": "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy",
    },
}

# (mask_size_scaling, mask_box_size, label_suffix)
MASK_VARIANTS = [
    (0.8, 0, "pyramid_ms0p8"),
    (1.2, 0, "pyramid_ms1p2"),
    (0.0, list(MASK_BOX_RANGE), "random_mbox"),
]

SPATIAL_CONV_VALUES = [True, False]
LAYERNORM_VALUES = [True, False]
PROJECTOR_CONV_VALUES = [True, False]
MASK_BOX_HARDCAP = 24


def _base_config() -> dict:
    return {
        "base_config": BASE_CONFIG,
        "model": {
            "mode": "pyramid",
            "model_key": "cdd_scaleaware_convnext",
            "mask_spacing_scaling": 2.0,
            "normalize_loss_l2": True,
            "use_symmetric_feature_loss": True,
            "priority_top_percent": 15.0,
            "target_sampling_mode": "priority_sampling",
            "priority_n_target": "auto",
            "priority_min_targets_per_map": 10,
            "priority_dithering_pixels": 6,
            "target_nonoverlap": True,
            "mask_box_hardcap": MASK_BOX_HARDCAP,
        },
        "train": {
            "epochs": 10,
            "log_interval": 1,
            "jepa_loss_weight": 100.0,
            "vicreg_var_weight": 0.0,
            "vicreg_cov_weight": 0.0,
            "sigreg_weight": 1.0,
            "sigreg_sketch_dim": 64,
            "inference_tta_enabled": True,
            "inference_tta_mode": "flip4",
            "umap": {"l2_normalize": True},
        },
    }


def write_config(name: str, cfg: dict) -> str:
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deepcopy(cfg), f, indent=2)
        f.write("\n")
    return path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    run = 0
    for dkey, ds in DATASETS.items():
        for mscale, mbox, mask_label in MASK_VARIANTS:
            for sc in SPATIAL_CONV_VALUES:
                for pln in LAYERNORM_VALUES:
                    for pc in PROJECTOR_CONV_VALUES:
                        run += 1
                        cfg = _base_config()
                        cfg["model"]["mask_size_scaling"] = mscale
                        cfg["model"]["mask_box_size"] = mbox

                        if "npy_pattern" in ds:
                            cfg["data"] = cfg.get("data", {})
                            cfg["data"]["npy_pattern"] = ds["npy_pattern"]

                        tags = []
                        if sc:
                            tags.append("spatial")
                        else:
                            cfg["model"]["predictor_spatial_conv"] = False
                            tags.append("channel_only")

                        if pln:
                            tags.append("pln_true")
                        else:
                            cfg["model"]["predictor_layernorm"] = False
                            tags.append("pln_false")

                        if pc:
                            tags.append("proj_on")
                        else:
                            cfg["model"]["projector_conv"] = False
                            tags.append("proj_off")

                        name = f"gen_72_run_{run}_{dkey}_symmetric_sigreg_l2_{mask_label}_{'_'.join(tags)}"
                        paths.append(write_config(name, cfg))

    for path in paths:
        print(os.path.relpath(path, ROOT))
    print(f"\nTotal: {len(paths)} configs written to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
