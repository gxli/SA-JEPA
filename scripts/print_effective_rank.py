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
        return {
            "mode": "NA",
            "mask_fraction": "NA",
            "mask_size": "NA",
        }
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        m = cfg.get("model", {})
        return {
            "mode": str(m.get("mode", "NA")),
            "mask_fraction": str(m.get("mask_fraction", "NA")),
            "mask_size": str(m.get("mask_size", "NA")),
        }
    except Exception:
        return {
            "mode": "NA",
            "mask_fraction": "NA",
            "mask_size": "NA",
        }


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


def _parse_rank(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("-inf")


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


def _fmt_float(v: str, width: int = 9, digits: int = 4) -> str:
    try:
        x = float(v)
        if not math.isfinite(x):
            return f"{'-':>{width}}"
        return f"{x:>{width}.{digits}f}"
    except Exception:
        return f"{'-':>{width}}"


def _clip(s: str, width: int) -> str:
    s = str(s)
    if len(s) <= width:
        return s.ljust(width)
    if width <= 1:
        return s[:width]
    return (s[: width - 1] + "…")


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
        p_t4 = _diag_get(diag, "pred", "top4_energy")
        p_t8 = _diag_get(diag, "pred", "top8_energy")
        p_pr = _diag_get(diag, "pred", "participation_rank")
        g_pr = _diag_get(diag, "gt", "participation_rank")
        p_dead = _diag_get(diag, "pred", "dead_channel_fraction")
        p_dead_n_raw = diag.get("pred", {}).get("num_dead_channels", "")
        try:
            p_dead_n = str(int(p_dead_n_raw))
        except Exception:
            p_dead_n = ""
        ratio_er = ""
        ratio_pr = ""
        try:
            ratio_er = f"{float(diag.get('pred_gt_erank_ratio', 'nan')):.6f}"
        except Exception:
            ratio_er = ""
        try:
            ratio_pr = f"{float(diag.get('pred_gt_participation_ratio', 'nan')):.6f}"
        except Exception:
            ratio_pr = ""
        rows.append(
            (
                name,
                inputs["mode"],
                inputs["mask_fraction"],
                inputs["mask_size"],
                rank,
                c_er,
                p_er,
                g_er,
                p_t1,
                p_t4,
                p_t8,
                p_pr,
                g_pr,
                p_dead,
                p_dead_n,
                ratio_er,
                ratio_pr,
            )
        )

    rows_sorted = sorted(rows, key=lambda x: x[0])

    print("Effective Rank Summary (sorted by session filename A->Z)")
    session_w = 62
    print(
        f"{'session':<{session_w}} {'mode':<8} {'mfrac':>6} {'msize':>6} "
        f"{'erank':>9} {'ctx_er':>9} {'pred_er':>9} {'gt_er':>9} "
        f"{'t1':>7} {'t4':>7} {'t8':>7} {'pred_pr':>9} {'gt_pr':>9} "
        f"{'dead_f':>8} {'dead_n':>6} {'er_ratio':>9} {'pr_ratio':>9}"
    )
    print("-" * 214)
    for row in rows_sorted:
        (
            s,
            mode,
            mf,
            ms,
            rk,
            c_er,
            p_er,
            g_er,
            p_t1,
            p_t4,
            p_t8,
            p_pr,
            g_pr,
            p_dead,
            p_dead_n,
            ratio_er,
            ratio_pr,
        ) = row
        dead_n_str = p_dead_n if p_dead_n != "" else "-"
        print(
            f"{_clip(s, session_w)} {mode:<8} {_fmt_float(mf,6,2)} {_fmt_float(ms,6,2)} "
            f"{_fmt_float(rk,9,4)} {_fmt_float(c_er,9,4)} {_fmt_float(p_er,9,4)} {_fmt_float(g_er,9,4)} "
            f"{_fmt_float(p_t1,7,3)} {_fmt_float(p_t4,7,3)} {_fmt_float(p_t8,7,3)} {_fmt_float(p_pr,9,4)} {_fmt_float(g_pr,9,4)} "
            f"{_fmt_float(p_dead,8,3)} {dead_n_str:>6} {_fmt_float(ratio_er,9,4)} {_fmt_float(ratio_pr,9,4)}"
        )

    n_total = len(rows_sorted)
    n_rank = sum(1 for r in rows_sorted if r[4] != "")
    print("-" * 214)
    print(f"sessions={n_total} with_rank={n_rank} missing_rank={n_total - n_rank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
