#!/usr/bin/env python3
"""Rename gen_96 configs so filenames expose every recovery-ablation knob."""
from __future__ import annotations

import json
import os
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "configs", "experiments")


def _float_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _dataset_tag(cfg: dict) -> str:
    pattern = str(cfg["data"]["npy_pattern"])
    if pattern == "ngc3627_12m+7m+tp_co21_strict_mom0.npy_sm.npy":
        return "ngc"
    if pattern == "chengdu.npy":
        return "chengdu"
    if pattern == "C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy":
        return "c12"
    raise ValueError(f"Add a gen_96 dataset filename tag for npy_pattern={pattern!r}")


def canonical_config_name(run: int, cfg: dict) -> str:
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    pred_type = "pred3x3" if bool(model["predictor_spatial_conv"]) else "pred1x1"
    sym_path = bool(model["use_symmetric_feature_loss"])
    sym_weight = float(train["symmetric_feature_loss_weight"])
    symloss = "symloss_on" if sym_path and sym_weight > 0.0 else "symloss_off"
    lognorm_on = bool(model.get("post_log_transform", True))
    dataset = _dataset_tag(cfg)
    if dataset == "c12":
        prefix = f"gen_96_run_{run:03d}"
    elif dataset == "ngc":
        prefix = "gen_96_run_2_ngc"
    elif dataset == "chengdu":
        prefix = "gen_96_run_3_chengdu"
    else:
        raise AssertionError(dataset)
    parts = [
        prefix,
        f"ms{_float_tag(model['mask_size_scaling'])}",
        pred_type,
        f"h{int(model['predictor_hidden'])}",
        f"predln_{'on' if bool(model['predictor_layernorm']) else 'off'}",
        f"perscalenorm_{'on' if bool(model['scaleaware_norm_per_scale']) else 'off'}",
        symloss,
    ]
    if not lognorm_on:
        parts.append("lognorm_off")
    if int(train["epochs"]) != 5:
        parts.append(f"ep{int(train['epochs'])}")
    return "_".join(parts)


def main() -> None:
    paths = sorted(glob(os.path.join(OUT_DIR, "gen_96_run_*.json")))
    for path in paths:
        basename = os.path.basename(path)
        run = int(basename.split("_")[3])
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dst = os.path.join(OUT_DIR, f"{canonical_config_name(run, cfg)}.json")
        if os.path.abspath(path) == os.path.abspath(dst):
            print(f"unchanged {os.path.relpath(path, ROOT)}")
            continue
        if os.path.exists(dst):
            raise FileExistsError(f"Refusing to overwrite existing config: {dst}")
        os.rename(path, dst)
        print(f"renamed {os.path.relpath(path, ROOT)} -> {os.path.relpath(dst, ROOT)}")


if __name__ == "__main__":
    main()
