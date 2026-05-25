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
  config_name="$(basename "${cfg}")"
  config_name="${config_name%.json}"
  session_dir="${SESSIONS_DIR}/${config_name}"
  model_ckpt="${session_dir}/model_last.pt"
  inference_pt="${session_dir}/inference_outputs.pt"
  epoch_summary_csv="${session_dir}/epoch_summary.csv"

  # Skip as early as possible, but only when training actually completed.
  # Require:
  #  1) model checkpoint exists
  #  2) inference outputs exist
  #  3) epoch_summary indicates max(epoch) >= configured train.epochs
  if [[ "${FORCE_REINFERENCE}" != "1" && -f "${model_ckpt}" && -f "${inference_pt}" && -f "${epoch_summary_csv}" ]]; then
    skip_ok="$(
      python3 - "${cfg}" "${epoch_summary_csv}" <<'PY'
import csv, json, sys
cfg_path, epoch_csv = sys.argv[1], sys.argv[2]
try:
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    target_epochs = int(cfg.get("train", {}).get("epochs", 0))
except Exception:
    print("0")
    raise SystemExit(0)
if target_epochs <= 0:
    print("0")
    raise SystemExit(0)
max_epoch = 0
try:
    with open(epoch_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                e = int(float(row.get("epoch", "0")))
            except Exception:
                continue
            if e > max_epoch:
                max_epoch = e
except Exception:
    print("0")
    raise SystemExit(0)
print("1" if max_epoch >= target_epochs else "0")
PY
    )"
    if [[ "${skip_ok}" == "1" ]]; then
      echo "skip_complete_session config=${config_name} reason=epochs_and_inference_complete"
      continue
    else
      echo "resume_session config=${config_name} reason=incomplete_epochs_or_missing_summary"
    fi
  fi

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
