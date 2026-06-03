#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import List, Tuple


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
        return {
            "mode": str(m.get("mode", "NA")),
            "ms": str(m.get("mask_size_scaling", m.get("mask_size_scaling_range", "NA"))),
            "mbox": str(m.get("mask_box_size", m.get("mask_box_size_range", "NA"))),
            "l2": _fmt_bool(m.get("normalize_loss_l2", False)),
            "psn": _fmt_bool(m.get("scaleaware_norm_per_scale", False)),
            "fin": _fmt_bool(m.get("scaleaware_final_norm", False)),
            "spread_w": str(spread.get("weight", t.get("sigreg_weight", "NA"))),
            "sigtype": str(spread.get("type", "NA")),
            "symw": str(t.get("symmetric_feature_loss_weight", "NA")),
            "depth": str(m.get("encoder_depth", "NA")),
        }
    except Exception:
        return _missing_model_inputs()


def _fmt_bool(value) -> str:
    return "1" if bool(value) else "0"


def _missing_model_inputs() -> dict:
    return {k: "NA" for k in (
        "mode", "ms", "mbox", "l2", "psn", "fin", "spread_w", "sigtype", "symw", "depth",
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
        values = diag.get(branch, {})
        v = values.get(key, "")
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


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/print_effective_rank.py <sessions_dir_or_session_dir...> [prefix]", file=sys.stderr)
        return 2

    args = sys.argv[1:]
    prefix = ""
    if len(args) >= 2 and ("*" not in args[-1]) and ("/" not in args[-1]) and (not os.path.isdir(args[-1])):
        prefix = args[-1]
        args = args[:-1]
    if not args:
        print("No session path provided", file=sys.stderr)
        return 2

    session_dirs: List[str] = []
    for a in args:
        if not os.path.isdir(a):
            continue
        if os.path.isfile(os.path.join(a, "config_used.json")):
            session_dirs.append(a)
            continue
        for name in sorted(os.listdir(a)):
            p = os.path.join(a, name)
            if os.path.isdir(p):
                session_dirs.append(p)

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
        p_dead = _diag_get(diag, "pred", "dead_channel_fraction")
        p_dead_n = diag.get("pred", {}).get("dead_channel_count",
                    diag.get("pred", {}).get("num_dead_channels", 0))
        try:
            p_dead_str = str(int(p_dead_n))
        except Exception:
            p_dead_str = "-"
        rows.append(
            (name,
             inputs["mode"], inputs["ms"], inputs["mbox"],
             inputs["l2"], inputs["psn"], inputs["fin"],
             inputs["spread_w"], inputs["symw"], inputs["depth"],
             rank, c_er, p_er, g_er, p_t1, p_pr, g_pr, pr_match, p_dead, p_dead_str)
        )

    rows_sorted = sorted(rows, key=lambda x: x[0])

    print("Effective Rank Summary (sorted by session filename A-Z)")
    session_w = max(len("session"), *(len(row[0]) for row in rows_sorted))
    header = (
        f"{'session':<{session_w}} {'mode':<9} {'mask_scale':>9} {'mask_box':>9} "
        f"{'l2_norm':>7} {'psnorm':>6} {'final_norm':>10} {'sigreg':>7} {'sym_loss':>9} {'depth':>6} "
        f"{'erank':>8} {'context':>9} {'predictor':>10} {'target':>9} "
        f"{'top1':>7} {'pred_part':>10} {'target_part':>11} {'part_ratio':>10} {'dead_frac':>10} {'dead_ch':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows_sorted:
        (s, mode, ms, mbox, l2, psn, fin, sigw, symw, d,
         rk, c_er, p_er, g_er, p_t1, p_pr, g_pr, pr_match, p_dead, p_dead_n) = row
        print(
            f"{s:<{session_w}} {mode:<9} {ms:>9} {mbox:>9} "
            f"{l2:>7} {psn:>6} {fin:>10} {_fmt_float(sigw,7,2)} {_fmt_float(symw,9,4)} {d:>6} "
            f"{_fmt_float(rk,8,4)} {_fmt_float(c_er,9,4)} {_fmt_float(p_er,10,4)} {_fmt_float(g_er,9,4)} "
            f"{_fmt_float(p_t1,7,3)} {_fmt_float(p_pr,10,2)} {_fmt_float(g_pr,11,2)} {_fmt_float(pr_match,10,4)} "
            f"{_fmt_float(p_dead,10,3)} {p_dead_n:>7}"
        )

    n_total = len(rows_sorted)
    n_rank = sum(1 for r in rows_sorted if r[10] != "")
    print("-" * len(header))
    print(f"sessions={n_total} with_rank={n_rank} missing_rank={n_total - n_rank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
