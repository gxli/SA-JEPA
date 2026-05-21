#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
SESSIONS_DIR="${ROOT_DIR}/sessions"
FORCE_REINFERENCE="${FORCE_REINFERENCE:-1}"
CONFIG_GLOB="${CONFIG_GLOB:-*.json}"

mkdir -p "${SESSIONS_DIR}"

shopt -s nullglob
configs=("${CONFIG_DIR}"/${CONFIG_GLOB})
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No config files found in ${CONFIG_DIR} matching ${CONFIG_GLOB}" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  echo "Running config: ${cfg}"
  if [[ "${FORCE_REINFERENCE}" == "1" ]]; then
    tmp_cfg="$(mktemp "${TMPDIR:-/tmp}/run_all_cfg.XXXXXX.json")"
    python3 - "${cfg}" "${tmp_cfg}" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)
train = cfg.setdefault("train", {})
train["force_recompute_inference"] = True
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
    python3 "${ROOT_DIR}/main.py" --config "${tmp_cfg}" --sessions-dir "${SESSIONS_DIR}"
    rm -f "${tmp_cfg}"
  else
    python3 "${ROOT_DIR}/main.py" --config "${cfg}" --sessions-dir "${SESSIONS_DIR}"
  fi
done

echo "All sessions complete."
echo "Post-training inference artifacts are saved in each session directory."
echo "Use ./session_to_dash.sh to build dashboards into results/dashboard."
