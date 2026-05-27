#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSIONS_DIR="$ROOT_DIR/sessions"
CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/configs/experiments}"
STAGE="plot"                       # compute | plot | all
OVERWRITE="false"                  # default: skip existing
EXPORT_DIR="$ROOT_DIR/results/dashboard"
RESET="false"

# Optional switches first.
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --reset)
      RESET="true"
      ;;
    --overwrite)
      OVERWRITE="true"
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  set -- "${POSITIONAL[@]}"
else
  set --
fi

# Positional overrides for compatibility:
# 1: sessions_dir, 2: stage, 3: overwrite(true|false), 4: export_dir, 5: reset(true|false)
if [[ $# -ge 1 ]]; then SESSIONS_DIR="$1"; fi
if [[ $# -ge 2 ]]; then STAGE="$2"; fi
if [[ $# -ge 3 ]]; then OVERWRITE="$3"; fi
if [[ $# -ge 4 ]]; then EXPORT_DIR="$4"; fi
if [[ $# -ge 5 ]]; then RESET="$5"; fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -d "$SESSIONS_DIR" ]]; then
  echo "sessions_dir_not_found=$SESSIONS_DIR" >&2
  exit 1
fi

ARGS=(--sessions-dir "$SESSIONS_DIR" --stage "$STAGE" --export-dir "$EXPORT_DIR")
if [[ "$OVERWRITE" == "true" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "$RESET" == "true" ]]; then
  ARGS+=(--reset)
fi

echo "run_session_to_dash sessions_dir=$SESSIONS_DIR stage=$STAGE overwrite=$OVERWRITE export_dir=$EXPORT_DIR reset=$RESET"
export SESSION_DASH_CONFIG_DIR="$CONFIG_DIR"
python3 "$ROOT_DIR/scripts/session_to_dash.py" "${ARGS[@]}"
