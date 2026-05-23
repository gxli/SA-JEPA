#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
SESSIONS_DIR="${ROOT_DIR}/sessions"
FORCE_REINFERENCE="${FORCE_REINFERENCE:-0}"
CONFIG_GLOB="${CONFIG_GLOB:-*.json}"
OVERRIDE_DIR="${ROOT_DIR}/.run_all_overrides"

mkdir -p "${SESSIONS_DIR}"
mkdir -p "${OVERRIDE_DIR}"

shopt -s nullglob
configs=("${CONFIG_DIR}"/${CONFIG_GLOB})
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No config files found in ${CONFIG_DIR} matching ${CONFIG_GLOB}" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  printf "\nRunning config: %s\n" "${cfg}"
  # Use a stable override path so config_name (derived from basename) stays
  # identical to the original config and sessions resume correctly.
  tmp_cfg="${OVERRIDE_DIR}/$(basename "${cfg}")"
  python3 - "${cfg}" "${tmp_cfg}" "${FORCE_REINFERENCE}" <<'PY'
import json
import os
import sys

src, dst, force_flag = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)
train = cfg.setdefault("train", {})
train["force_recompute_inference"] = bool(int(force_flag))
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
  python3 "${ROOT_DIR}/main.py" --config "${tmp_cfg}" --sessions-dir "${SESSIONS_DIR}"
done

echo "All sessions complete."
echo "Post-training inference artifacts are saved in each session directory."
echo "Use ./session_to_dash.sh to build dashboards into results/dashboard."
