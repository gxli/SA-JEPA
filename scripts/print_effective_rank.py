#!/usr/bin/env python3
"""Session diagnostics — rank, energy, spread loss summary.

CLI:
    python scripts/print_effective_rank.py sessions/gen_148_*

API:
    from scripts.print_effective_rank import rank_summary
    rows = rank_summary(["sessions/gen_148_run_001", ...])
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import List, Tuple


# ── readers ────────────────────────────────────────────────────

def _first_present(d: dict, keys: tuple[str, ...], default="NA"):
    for key in keys:
        if key in d:
            return d.get(key)
    return default


def _fmt_cfg_value(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt_cfg_value(v) for v in value) + "]"
    return str(value)


def _count_cfg_list(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(len(value))
    return "NA"


def _read_model_inputs(session_dir: str) -> dict:
    cfg_path = os.path.join(session_dir, "config_used.json")
    if not os.path.exists(cfg_path):
        return _missing_model_inputs()
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        m = cfg.get("model", {})
        t = cfg.get("train", {})
        spread = t.get("spread_regularizer", {})
        if not isinstance(spread, dict):
            spread = {}
        mask_scale = _first_present(m, ("mask_size_scaling", "mask_scale_factor", "mask_scale"))
        mask_box = _first_present(m, ("mask_size_manual", "mask_size", "mask_footprint_px", "mask_box_size"))
        return {
            "mode": str(m.get("mode", "NA")),
            "ms": _fmt_cfg_value(mask_scale),
            "mbox": _fmt_cfg_value(mask_box),
            "sampling": str(m.get("target_sampling_mode", "NA")),
            "l2": _fmt_bool(m.get("normalize_loss_l2", False)),
            "psn": _fmt_bool(m.get("scaleaware_norm_per_scale", False)),
            "fin": _fmt_bool(m.get("scaleaware_final_norm", False)),
            "sigtype": str(spread.get("type", "NA")),
            "spread_mode": str(spread.get("spatial_mode", "pooled")),
            "spread_w": str(spread.get("weight", "NA")),
            "spread_t": str(spread.get("target_std", "NA")),
            "vicvar_w": str(t.get("vicreg_var_weight", t.get("experimental_losses", {}).get("vicreg_var_weight", "0"))),
            "viccov_w": str(t.get("vicreg_cov_weight", t.get("experimental_losses", {}).get("vicreg_cov_weight", "0"))),
            "symw": str(t.get("symmetry_loss_weight", "NA")),
            "depth": str(m.get("encoder_depth", "NA")),
            "dilations": _fmt_cfg_value(_first_present(m, ("convnext_layer_dilations", "dilations"), "None")),
            "hardcap": str(m.get("mask_box_hardcap", "—")),
            "pred_hidden": str(m.get("predictor_hidden", "NA")),
            "cdd_scales": _count_cfg_list(m.get("sigmas")),
        }
    except Exception:
        return _missing_model_inputs()


def _fmt_bool(value) -> str:
    return "1" if bool(value) else "0"


def _missing_model_inputs() -> dict:
    return {k: "NA" for k in (
        "mode", "ms", "mbox", "sampling", "l2", "psn", "fin", "sigtype", "spread_mode", "spread_w", "spread_t",
        "vicvar_w", "viccov_w", "symw", "depth", "dilations", "hardcap", "pred_hidden", "cdd_scales",
    )}


def _read_effective_rank(session_dir: str) -> str:
    txt = os.path.join(session_dir, "effective_rank.txt")
    if os.path.exists(txt):
        try:
            with open(txt, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    rr = os.path.join(session_dir, "run_results.csv")
    if os.path.exists(rr):
        try:
            last = ""
            with open(rr, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    v = str(row.get("effective_rank", "")).strip()
                    if v:
                        last = v
            return last
        except Exception:
            return ""
    return ""


def _read_loss_ratios(session_dir: str) -> dict[str, str]:
    path = os.path.join(session_dir, "metrics.csv")
    if not os.path.exists(path):
        return {}
    try:
        epoch_sums: dict[str, dict[int, float]] = {}
        epoch_counts: dict[str, dict[int, int]] = {}
        keys = [
            "loss_total",
            "sim",
            "loss_prediction",
            "loss_spread",
            "weighted_spread",
            "loss_vicreg_var",
            "loss_vicreg_cov",
            "weighted_vicreg_var",
            "weighted_vicreg_cov",
        ]
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep = int(row.get("epoch", -1))
                if ep < 0:
                    continue
                for k in keys:
                    v = row.get(k, row.get({"loss_spread": "loss_sigreg", "weighted_spread": "weighted_sigreg"}.get(k, k), ""))
                    if v and v.strip():
                        try:
                            fv = float(v)
                        except (ValueError, TypeError):
                            continue
                        epoch_sums.setdefault(k, {}).setdefault(ep, 0.0)
                        epoch_counts.setdefault(k, {}).setdefault(ep, 0)
                        epoch_sums[k][ep] += fv
                        epoch_counts[k][ep] += 1
        result = {}
        for k in keys:
            if k in epoch_sums:
                eps = sorted(epoch_sums[k].keys())
                fa = epoch_sums[k][eps[0]] / max(1, epoch_counts[k][eps[0]])
                la = epoch_sums[k][eps[-1]] / max(1, epoch_counts[k][eps[-1]])
                result[f"{k}_last"] = str(la)
            if k in epoch_sums and len(epoch_sums[k]) >= 2:
                if fa > 1e-20:
                    result[k] = str(la / fa)
        return result
    except Exception:
        return {}


def _read_rank_diag(session_dir: str) -> dict:
    path = os.path.join(session_dir, "rank_diagnostics.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _diag_get(diag: dict, branch: str, key: str) -> str:
    try:
        v = diag.get(branch, {}).get(key, "")
        if isinstance(v, (int, float)):
            return f"{float(v):.6f}"
        return str(v)
    except Exception:
        return ""


def _fmt_float(v: str, width: int = 8, digits: int = 4) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return f"{'-':>{width}}"
        return f"{x:>{width}.{digits}f}"
    except Exception:
        return f"{'-':>{width}}"


def _is_nonzero_value(value: str) -> bool:
    try:
        x = float(str(value).strip())
        return math.isfinite(x) and abs(x) > 1e-12
    except Exception:
        return False


# ── public API ─────────────────────────────────────────────────

def rank_summary(session_dirs: List[str], prefix: str = "") -> List[Tuple[str, ...]]:
    """Return list of (name, mode, ms, mbox, sampling, l2, psn, fin,
    spread_type, spread_spatial, spread_w, spread_t, vicvar_w, viccov_w, symw, depth, dilations, hardcap, cdd_scales, energy,
    sim_r, hinge_r, sig_r, vicv_r, vicc_r, wvicv_r, wvicc_r, erank, context, predictor, target, top1,
    pred_part, target_part, part_ratio, dead_frac, dead_ch, total_last, pred_last, spread_last) tuples.

    Usage:
        from scripts.print_effective_rank import rank_summary
        rows = rank_summary(["sessions/gen_148_run_001", ...])
    """
    rows: List[Tuple[str, ...]] = []
    for path in sorted(session_dirs):
        name = os.path.basename(path.rstrip("/"))
        if prefix and not name.startswith(prefix):
            continue
        rank = _read_effective_rank(path)
        inputs = _read_model_inputs(path)
        diag = _read_rank_diag(path)
        c_er = _diag_get(diag, "context", "erank")
        p_er = _diag_get(diag, "pred", "erank")
        g_er = _diag_get(diag, "gt", "erank")
        p_t1 = _diag_get(diag, "pred", "top1_energy")
        p_pr = _diag_get(diag, "pred", "manifold_size")
        g_pr = _diag_get(diag, "gt", "manifold_size")
        pr_match = ""
        try:
            pr_match = f"{float(diag.get('volume_match_ratio', 'nan')):.4f}"
        except Exception:
            pr_match = ""
        energy = _fmt_float(str(diag.get("energy", "")), 9, 4)
        ratios = _read_loss_ratios(path)
        sim_r = _fmt_float(ratios.get("sim", ""), 7, 4)
        total_last = _fmt_float(ratios.get("loss_total_last", ""), 7, 4)
        pred_last = _fmt_float(ratios.get("loss_prediction_last", ""), 7, 4)
        spread_last = _fmt_float(ratios.get("loss_spread_last", ""), 7, 4)
        hinge_r = _fmt_float(ratios.get("loss_spread", ""), 7, 4)
        sig_r = _fmt_float(ratios.get("weighted_spread", ""), 7, 4)
        vicv_r = _fmt_float(ratios.get("loss_vicreg_var", ""), 7, 4)
        vicc_r = _fmt_float(ratios.get("loss_vicreg_cov", ""), 7, 4)
        wvicv_r = _fmt_float(ratios.get("weighted_vicreg_var", ""), 7, 4)
        wvicc_r = _fmt_float(ratios.get("weighted_vicreg_cov", ""), 7, 4)
        p_dead = _diag_get(diag, "pred", "dead_channel_fraction")
        p_dead_n = diag.get("pred", {}).get("dead_channel_count", diag.get("pred", {}).get("num_dead_channels", 0))
        try:
            p_dead_str = str(int(p_dead_n))
        except Exception:
            p_dead_str = "-"
        rows.append(
            (name,
             inputs.get("mode", "NA"), inputs.get("ms", "NA"), inputs.get("mbox", "NA"), inputs.get("sampling", "NA"),
             inputs.get("l2", "NA"), inputs.get("psn", "NA"), inputs.get("fin", "NA"),
             inputs.get("sigtype", "NA"), inputs.get("spread_mode", "pooled"), inputs.get("spread_w", "NA"), inputs.get("spread_t", "NA"),
             inputs.get("vicvar_w", "NA"), inputs.get("viccov_w", "NA"),
             inputs.get("symw", "NA"), inputs.get("depth", "NA"),
             inputs.get("dilations", "NA"), inputs.get("hardcap", "NA"),
             inputs.get("cdd_scales", "NA"),
             energy, sim_r, hinge_r, sig_r, vicv_r, vicc_r, wvicv_r, wvicc_r,
             rank, c_er, p_er, g_er, p_t1, p_pr, g_pr, pr_match, p_dead, p_dead_str,
             total_last, pred_last, spread_last))
    return sorted(rows, key=lambda x: x[0])


def print_rank_table(session_dirs: List[str], prefix: str = "") -> None:
    """Print a formatted rank-summary table to stdout."""
    rows = rank_summary(session_dirs, prefix=prefix)
    print("Sampled Embedding Diagnostics (sorted by session filename A-Z)")
    if not rows:
        print("No matching sessions found.")
        return
    include_vicreg = any(_is_nonzero_value(row[12]) or _is_nonzero_value(row[13]) for row in rows)
    include_sim_ratio = any(str(row[5]).strip() == "1" for row in rows)
    include_spread_ratio = any(_is_nonzero_value(row[10]) for row in rows)
    setup_columns = [
        ("session", 0, "str", "<"),
        ("status", -1, "status", "<"),
        ("mask_scale", 2, "str", ">"),
        ("mask_box", 3, "str", ">"),
        ("sampling", 4, "str", "<"),
        ("cdd_n", 18, "str", ">"),
        ("dil", 16, "str", ">"),
        ("hardcap", 17, "str", ">"),
        ("spread_type", 8, "str", ">"),
        ("spread_sp", 9, "short", ">"),
        ("spread_w", 10, "float2", ">"),
    ]
    vicreg_columns = [
        ("vicvar_w", 12, "float2", ">"),
        ("viccov_w", 13, "float2", ">"),
    ] if include_vicreg else []
    diagnostic_columns = [
        ("energy", 19, "raw", ">"),
        ("total", 37, "raw", ">"),
        ("pred", 38, "raw", ">"),
        ("spread", 39, "raw", ">"),
    ]
    ratio_columns = []
    if include_sim_ratio:
        ratio_columns.append(("sim_r", 20, "raw", ">"))
    if include_spread_ratio:
        ratio_columns.append(("hinge_r", 21, "raw", ">"))
    vicreg_ratio_columns = [
        ("vicv_r", 23, "raw", ">"),
        ("vicc_r", 24, "raw", ">"),
        ("wvv_r", 25, "raw", ">"),
        ("wvc_r", 26, "raw", ">"),
    ] if include_vicreg else []
    representation_columns = [
        ("context_effrank", 28, "float4", ">"),
        ("predictor_effrank", 29, "float4", ">"),
        ("target_effrank", 27, "float4", ">"),
        ("pred_top1", 31, "float3", ">"),
        ("target_part", 33, "float2", ">"),
        ("part_ratio", 34, "float4", ">"),
        ("dead_frac", 35, "float3", ">"),
    ]
    columns = setup_columns + vicreg_columns + diagnostic_columns + ratio_columns + vicreg_ratio_columns + representation_columns

    def row_status(row: tuple[str, ...]) -> str:
        if str(row[27]).strip():
            return "ok"
        if any(str(row[idx]).strip() for idx in (28, 29, 30, 31, 32, 33, 34, 35)):
            return "partial_diag"
        return "missing_diag"

    def format_cell(row: tuple[str, ...], idx: int, kind: str) -> str:
        if kind == "status":
            return row_status(row)
        value = row[idx]
        if kind == "short":
            return str(value)[:6]
        if kind == "float2":
            return _fmt_float(value, 0, 2).strip()
        if kind == "float3":
            return _fmt_float(value, 0, 3).strip()
        if kind == "float4":
            return _fmt_float(value, 0, 4).strip()
        return str(value).strip()

    widths = []
    for name, idx, kind, _align in columns:
        cells = [format_cell(row, idx, kind) for row in rows]
        widths.append(max(len(name), *(len(cell) for cell in cells)))

    header = " ".join(f"{name:{align}{width}}" for (name, _idx, _kind, align), width in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = [
            format_cell(row, idx, kind)
            for _name, idx, kind, _align in columns
        ]
        print(" ".join(f"{cell:{align}{width}}" for cell, (_name, _idx, _kind, align), width in zip(cells, columns, widths)))
    n_total = len(rows)
    n_rank = sum(1 for r in rows if r[27] != "")
    print("-" * len(header))
    print(f"sessions={n_total} with_rank={n_rank} missing_rank={n_total - n_rank}")


# ── CLI ────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/print_effective_rank.py <sessions_dir_or_session_dir...> [prefix]", file=sys.stderr)
        return 2
    args = sys.argv[1:]
    prefix = ""
    if len(args) >= 2 and ("*" not in args[-1]) and ("/" not in args[-1]) and (not os.path.isdir(args[-1])):
        prefix = args[-1]
        args = args[:-1]
    session_dirs: List[str] = []
    for a in args:
        if not os.path.isdir(a):
            continue
        if os.path.isfile(os.path.join(a, "config_used.json")):
            session_dirs.append(a)
        else:
            for name in sorted(os.listdir(a)):
                p = os.path.join(a, name)
                if os.path.isdir(p):
                    session_dirs.append(p)
    print_rank_table(session_dirs, prefix=prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
