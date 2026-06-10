#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" != "--force" ]]; then
  echo "run_all_session_plots is deprecated for the main pipeline and is disabled by default."
  echo "Use ./session_to_dash.sh instead, or pass --force to run this legacy script intentionally."
  exit 0
fi

SESSIONS_DIR="$ROOT_DIR/sessions"
PLOT_SCRIPT="$ROOT_DIR/scripts/session_overview_4panel.py"
FALLBACK_DASH_SCRIPT="$ROOT_DIR/scripts/session_to_dash.py"
GLOBAL_PLOTS_DIR="$ROOT_DIR/results/plots/session_dashboards"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "Missing plot script: $PLOT_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$FALLBACK_DASH_SCRIPT" ]]; then
  echo "Missing fallback dash script: $FALLBACK_DASH_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$SESSIONS_DIR" ]]; then
  echo "No sessions directory: $SESSIONS_DIR" >&2
  exit 1
fi

shopt -s nullglob
session_dirs=("$SESSIONS_DIR"/*)

if [[ ${#session_dirs[@]} -eq 0 ]]; then
  echo "No session folders found under: $SESSIONS_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$GLOBAL_PLOTS_DIR"
export MPLCONFIGDIR="${ROOT_DIR}/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

for session_dir in "${session_dirs[@]}"; do
  [[ -d "$session_dir" ]] || continue

  if [[ ! -f "$session_dir/resolved_config.json" && ! -f "$session_dir/config_used.json" ]]; then
    echo "Skipping (not a session): $session_dir"
    continue
  fi

  required=(
    "$session_dir/results/latent_vectors_full.npy"
    "$session_dir/results/umap_x.npy"
    "$session_dir/results/umap_y.npy"
    "$session_dir/results/umap_z.npy"
  )
  missing=0
  for rf in "${required[@]}"; do
    if [[ ! -f "$rf" ]]; then
      missing=1
      break
    fi
  done
  out_dir="$session_dir/results/plots"
  mkdir -p "$out_dir"
  session_id="$(basename "$session_dir")"

  echo "========================================"
  echo "Generating plots for: $session_dir"

  generated_any=0
  if [[ "$missing" -eq 0 ]]; then
    python3 "$PLOT_SCRIPT" --session-dir "$session_dir" --out-dir "$out_dir" --inference all
    generated_any=1
  else
    if [[ -f "$session_dir/inference_outputs.pt" ]]; then
      # Fallback path: produce dashboard review artifact only for current session.
      python3 - <<PY >/dev/null
import sys
from pathlib import Path
sys.path.insert(0, str(Path("$ROOT_DIR")))
from scripts.session_to_dash import compute_dash_data, plot_dash
session_dir = "$session_dir"
compute_dash_data(session_dir, overwrite=False)
plot_dash(session_dir, overwrite=False)
PY
      if [[ -f "$session_dir/dashboard.png" ]]; then
        cp -f "$session_dir/dashboard.png" "$out_dir/dashboard.png"
        generated_any=1
      fi
    fi
  fi

  if [[ "$generated_any" -eq 0 ]]; then
    echo "Skipping (no plottable artifacts): $session_dir"
    echo "========================================"
    echo
    continue
  fi

  shopt -s nullglob
  for f in "$out_dir"/*.html; do
    base="$(basename "$f")"
    merged_name="${session_id}__${base}"
    cp -f "$f" "$GLOBAL_PLOTS_DIR/$merged_name"
  done
  for f in "$out_dir"/*.png; do
    base="$(basename "$f")"
    merged_name="${session_id}__${base}"
    cp -f "$f" "$GLOBAL_PLOTS_DIR/$merged_name"
  done
  echo "Saved plots to: $out_dir"
  echo "Merged plots to: $GLOBAL_PLOTS_DIR"
  echo "========================================"
  echo
done
