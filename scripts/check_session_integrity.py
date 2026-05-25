#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


REQUIRED_BRANCH_FILES = (
    "{b}_pca_xyz.npy",
    "{b}_umap_x.npy",
    "{b}_umap_y.npy",
    "{b}_umap_z.npy",
    "{b}_spatial_shape.npy",
)
OPTIONAL_CONTEXT_FILES = (
    "context_pca_xyz.npy",
    "context_umap_x.npy",
    "context_umap_y.npy",
    "context_umap_z.npy",
    "context_spatial_shape.npy",
)
REQUIRED_INFERENCE_KEYS = ("x_clean", "pred_map")


@dataclass
class SessionReport:
    name: str
    path: str
    ok: bool
    issues: list[str]
    warnings: list[str]


def _check_branch_artifacts(results_dir: str, branch: str) -> list[str]:
    missing: list[str] = []
    for tmpl in REQUIRED_BRANCH_FILES:
        fn = tmpl.format(b=branch)
        p = os.path.join(results_dir, fn)
        if not os.path.exists(p):
            missing.append(f"missing_file:{fn}")
    return missing


def _check_inference_outputs(session_dir: str) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    inf_path = os.path.join(session_dir, "inference_outputs.pt")
    if not os.path.exists(inf_path):
        issues.append("missing_file:inference_outputs.pt")
        return issues, warnings
    if torch is None:
        warnings.append("torch_unavailable:skipped_inference_key_validation")
        return issues, warnings
    try:
        outputs: Any = torch.load(inf_path, map_location="cpu")
    except Exception as e:
        issues.append(f"invalid_inference_outputs:{type(e).__name__}")
        return issues, warnings
    if not isinstance(outputs, dict):
        issues.append(f"invalid_inference_outputs:type={type(outputs).__name__}")
        return issues, warnings
    for k in REQUIRED_INFERENCE_KEYS:
        if k not in outputs:
            issues.append(f"missing_inference_key:{k}")
    return issues, warnings


def check_session(session_dir: str) -> SessionReport:
    path = os.path.abspath(session_dir)
    name = os.path.basename(path.rstrip(os.sep))
    issues: list[str] = []
    warnings: list[str] = []

    if not os.path.isdir(path):
        return SessionReport(name=name, path=path, ok=False, issues=["not_a_directory"], warnings=[])

    results_dir = os.path.join(path, "results")
    if not os.path.isdir(results_dir):
        issues.append("missing_dir:results")
    else:
        issues.extend(_check_branch_artifacts(results_dir, "predict"))
        issues.extend(_check_branch_artifacts(results_dir, "target"))
        missing_ctx = [fn for fn in OPTIONAL_CONTEXT_FILES if not os.path.exists(os.path.join(results_dir, fn))]
        if missing_ctx:
            warnings.append("context_fallback_to_predict")

    inf_issues, inf_warnings = _check_inference_outputs(path)
    issues.extend(inf_issues)
    warnings.extend(inf_warnings)

    return SessionReport(name=name, path=path, ok=(len(issues) == 0), issues=issues, warnings=warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dashboard-critical session integrity.")
    parser.add_argument("sessions", nargs="+", help="Session directories, e.g. sessions/*")
    args = parser.parse_args()

    input_paths = [os.path.abspath(s) for s in args.sessions]
    skipped_non_dirs = [p for p in input_paths if not os.path.isdir(p)]
    reports = [check_session(s) for s in input_paths if os.path.isdir(s)]
    reports.sort(key=lambda r: r.name)

    ok_n = 0
    warn_n = 0
    fail_n = 0

    for p in skipped_non_dirs:
        print(f"[SKIP] {os.path.basename(p)}")
        print("  warnings=not_a_directory")

    for r in reports:
        status = "OK" if r.ok else "FAIL"
        if r.ok:
            ok_n += 1
        else:
            fail_n += 1
        if r.warnings:
            warn_n += 1
            status = f"{status}+WARN"
        print(f"[{status}] {r.name}")
        if r.issues:
            print("  issues=" + "; ".join(r.issues))
        if r.warnings:
            print("  warnings=" + "; ".join(sorted(set(r.warnings))))

    print(
        "summary "
        f"total={len(input_paths)} checked={len(reports)} skipped={len(skipped_non_dirs)} "
        f"ok={ok_n} fail={fail_n} warn={warn_n}"
    )
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
