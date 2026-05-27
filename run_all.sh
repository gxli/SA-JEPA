#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
SESSIONS_DIR="${ROOT_DIR}/sessions"
FORCE_REINFERENCE="${FORCE_REINFERENCE:-0}"
CONFIG_GLOB="${CONFIG_GLOB:-experiments/*.json}"
CONFIG_SEARCH_ROOT="${CONFIG_SEARCH_ROOT:-${CONFIG_DIR}}"
OVERRIDE_DIR="${ROOT_DIR}/.run_all_overrides"

mkdir -p "${SESSIONS_DIR}"
mkdir -p "${OVERRIDE_DIR}"

mapfile -t configs < <(find "${CONFIG_SEARCH_ROOT}" -type f -path "${CONFIG_SEARCH_ROOT}/${CONFIG_GLOB}" | sort)
if [[ ${#configs[@]} -eq 0 && "${CONFIG_GLOB}" == "experiments/*.json" ]]; then
  # Backward-compat for legacy misspelled folder name.
  mapfile -t configs < <(find "${CONFIG_SEARCH_ROOT}" -type f -path "${CONFIG_SEARCH_ROOT}/exeriments/*.json" | sort)
fi
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No config files found in ${CONFIG_SEARCH_ROOT} matching ${CONFIG_GLOB}" >&2
  echo "Hint: default expects configs under ${CONFIG_DIR}/experiments/*.json (or legacy ${CONFIG_DIR}/exeriments/*.json)." >&2
  echo "You can override with CONFIG_SEARCH_ROOT and/or CONFIG_GLOB." >&2
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
import os

cfg_path, epoch_csv = os.path.abspath(sys.argv[1]), sys.argv[2]

def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_with_base(path, seen=None):
    seen = seen or set()
    if path in seen:
        return {}
    seen.add(path)
    cfg = json.load(open(path, "r", encoding="utf-8"))
    base_ref = cfg.pop("base_config", None)
    if base_ref is None:
        seen.remove(path)
        return cfg
    if not os.path.isabs(base_ref):
        base_ref = os.path.abspath(os.path.join(os.path.dirname(path), base_ref))
    base_cfg = load_with_base(base_ref, seen)
    seen.remove(path)
    return deep_merge(base_cfg, cfg)

try:
    cfg = load_with_base(cfg_path)
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
# Preserve base-config inheritance when writing to override dir by
# converting relative base_config paths to absolute paths.
base_ref = cfg.get("base_config")
if isinstance(base_ref, str) and base_ref:
    if not os.path.isabs(base_ref):
        cfg["base_config"] = os.path.abspath(os.path.join(os.path.dirname(src), base_ref))
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
