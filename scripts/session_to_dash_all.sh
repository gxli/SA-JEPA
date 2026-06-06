#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSIONS_DIR="$ROOT_DIR/sessions"
OUT_DIR="$ROOT_DIR/results/dashboard"
ONE_DASH_DIR="$ROOT_DIR/results/one_dashboards"
DASH_SCRIPT="$ROOT_DIR/scripts/session_to_dash.py"
export MPLCONFIGDIR="$ROOT_DIR/.mplconfig"

mkdir -p "$OUT_DIR"
mkdir -p "$ONE_DASH_DIR"
mkdir -p "$MPLCONFIGDIR"
# Clean legacy flattened exports that break relative links.
find "$OUT_DIR" -maxdepth 1 -type f -name "*__*" -delete
find "$ONE_DASH_DIR" -maxdepth 1 -type f -name "*.html" -delete

if [[ ! -d "$SESSIONS_DIR" ]]; then
  echo "No sessions directory found: $SESSIONS_DIR" >&2
  exit 1
fi

if [[ ! -f "$DASH_SCRIPT" ]]; then
  echo "Missing dashboard script: $DASH_SCRIPT" >&2
  exit 1
fi

processed=0
skipped=0
copied=0
index_items=()
one_dash_items=()

shopt -s nullglob
for session_dir in "$SESSIONS_DIR"/*; do
  [[ -d "$session_dir" ]] || continue
  session_name="$(basename "$session_dir")"

  echo "process_session=$session_name"

  # If reusable artifacts already exist, export them even without inference_outputs.pt.
  has_existing_artifacts=0
  if [[ -f "$session_dir/dashboard.html" || -f "$session_dir/latent_overview_4panel.html" ]]; then
    has_existing_artifacts=1
  fi

  # Smart regenerate:
  # Recompute only when inference outputs exist and outputs are missing/stale.
  regen=0
  if [[ -f "$session_dir/inference_outputs.pt" ]]; then
    if [[ ! -f "$session_dir/dash_data.npz" || ! -f "$session_dir/dashboard.html" ]]; then
      regen=1
    elif [[ "$DASH_SCRIPT" -nt "$session_dir/dash_data.npz" || "$DASH_SCRIPT" -nt "$session_dir/dashboard.html" ]]; then
      regen=1
    fi
  elif [[ "$has_existing_artifacts" -eq 0 ]]; then
    echo "skip_no_inference_or_artifacts=$session_name"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$regen" -eq 1 ]]; then
    echo "regen_dash=$session_name"
    python3 - <<PY >/dev/null
import sys
from pathlib import Path
sys.path.insert(0, str(Path("$ROOT_DIR")))
from scripts.session_to_dash import compute_dash_data, plot_dash
session_dir = "$session_dir"
compute_dash_data(session_dir, overwrite=True)
plot_dash(session_dir, overwrite=True)
PY
  else
    echo "reuse_dash=$session_name"
  fi

  # Preserve per-session structure so dashboard.html relative links remain valid.
  session_out="$OUT_DIR/$session_name"
  mkdir -p "$session_out/results"
  found_any=0

  if [[ -f "$session_dir/dashboard.html" ]]; then
    cp -f "$session_dir/dashboard.html" "$session_out/dashboard.html"
    copied=$((copied + 1))
    found_any=1
  fi
  if [[ -f "$session_dir/latent_overview_4panel.html" ]]; then
    cp -f "$session_dir/latent_overview_4panel.html" "$session_out/latent_overview_4panel.html"
    copied=$((copied + 1))
    found_any=1
  fi
  # Optional richer plotly session overview pages produced by session_overview_4panel.
  overview_artifacts=(
    "$session_dir/results/plots/session_overview_4panel.html"
    "$session_dir/results/plots/session_overview_4panel_context.html"
    "$session_dir/results/plots/session_overview_4panel_predict.html"
    "$session_dir/results/plots/session_overview_4panel_target.html"
  )
  for of in "${overview_artifacts[@]}"; do
    if [[ -f "$of" ]]; then
      cp -f "$of" "$session_out/$(basename "$of")"
      copied=$((copied + 1))
      found_any=1
    fi
  done

  if [[ "$found_any" -eq 0 ]]; then
    echo "skip_no_dashboard_artifacts=$session_name"
    skipped=$((skipped + 1))
    continue
  fi

  session_file_count=$(find "$session_out" -type f | wc -l | tr -d ' ')
  echo "done_session=$session_name exported_files=$session_file_count"

  if [[ -f "$session_out/dashboard.html" ]]; then
    index_items+=("<li><a href=\"./$session_name/dashboard.html\">$session_name</a></li>")
  elif [[ -f "$session_out/latent_overview_4panel.html" ]]; then
    index_items+=("<li><a href=\"./$session_name/latent_overview_4panel.html\">$session_name (latent overview)</a></li>")
  fi

  # One session, one canonical dashboard.
  # IMPORTANT: do not choose dashboard.html here because it has relative deps.
  canonical_src=""
  if [[ -f "$session_out/session_overview_4panel.html" ]]; then
    canonical_src="$session_out/session_overview_4panel.html"
  elif [[ -f "$session_out/latent_overview_4panel.html" ]]; then
    canonical_src="$session_out/latent_overview_4panel.html"
  fi
  if [[ -n "$canonical_src" ]]; then
    cp -f "$canonical_src" "$ONE_DASH_DIR/${session_name}.html"
    one_dash_items+=("<li><a href=\"./${session_name}.html\">$session_name</a></li>")
  fi

  processed=$((processed + 1))
done

{
  echo "<!doctype html>"
  echo "<html><head><meta charset=\"utf-8\"><title>Session Dashboards</title></head><body>"
  echo "<h1>Session Dashboards</h1>"
  echo "<ul>"
  for item in "${index_items[@]}"; do
    echo "$item"
  done
  echo "</ul>"
  echo "</body></html>"
} > "$OUT_DIR/index.html"

# Also provide a stable root dashboard.html entrypoint that forwards to index.
{
  echo "<!doctype html>"
  echo "<html><head><meta charset=\"utf-8\">"
  echo "<meta http-equiv=\"refresh\" content=\"0; url=./index.html\">"
  echo "<title>Dashboard Index</title></head>"
  echo "<body><p>Open <a href=\"./index.html\">Session Dashboards</a>.</p></body></html>"
} > "$OUT_DIR/dashboard.html"

{
  echo "<!doctype html>"
  echo "<html><head><meta charset=\"utf-8\"><title>One Session One Dash</title></head><body>"
  echo "<h1>One Session One Dashboard</h1>"
  echo "<ul>"
  for item in "${one_dash_items[@]}"; do
    echo "$item"
  done
  echo "</ul>"
  echo "</body></html>"
} > "$ONE_DASH_DIR/index.html"

echo "dashboard_summary processed=$processed skipped=$skipped copied=$copied out_dir=$OUT_DIR one_dash_dir=$ONE_DASH_DIR"
