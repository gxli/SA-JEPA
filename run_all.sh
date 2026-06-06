#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
SESSIONS_DIR="${SESSIONS_DIR:-${ROOT_DIR}/sessions}"
FORCE_REINFERENCE="${FORCE_REINFERENCE:-0}"
CONFIG_GLOB="${CONFIG_GLOB:-experiments/*.json}"
CONFIG_SEARCH_ROOT="${CONFIG_SEARCH_ROOT:-${CONFIG_DIR}}"
FAIL_LOG="${SESSIONS_DIR}/.run_all_failures.log"

mkdir -p "${SESSIONS_DIR}"

mapfile -t configs < <(find "${CONFIG_SEARCH_ROOT}" -type f -path "${CONFIG_SEARCH_ROOT}/${CONFIG_GLOB}" | sort)
if [[ ${#configs[@]} -eq 0 && "${CONFIG_GLOB}" == "experiments/*.json" ]]; then
  mapfile -t configs < <(find "${CONFIG_SEARCH_ROOT}" -type f -path "${CONFIG_SEARCH_ROOT}/exeriments/*.json" | sort)
fi
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No config files found in ${CONFIG_SEARCH_ROOT} matching ${CONFIG_GLOB}" >&2
  echo "Hint: default expects configs under ${CONFIG_DIR}/experiments/*.json (or legacy ${CONFIG_DIR}/exeriments/*.json)." >&2
  echo "You can override with CONFIG_SEARCH_ROOT and/or CONFIG_GLOB." >&2
  exit 1
fi

echo "Found ${#configs[@]} config(s).  Sessions dir: ${SESSIONS_DIR}"
> "${FAIL_LOG}"

total=${#configs[@]}
skipped=0
failed=0
ok=0

# ── helper: check if a config's training is already complete ───────────
_check_complete() {
  local cfg_path="$1" epoch_csv="$2"
  python3 - "$cfg_path" "$epoch_csv" 2>/dev/null <<'PY'
import csv, json, sys, os

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
}

# ═══════════════════════════════════════════════════════════════════════
for cfg in "${configs[@]}"; do
  config_name="$(basename "${cfg}")"
  config_name="${config_name%.json}"
  printf '\n━━━ [%s] ━━━\n' "${config_name}"

  session_dir="${SESSIONS_DIR}/${config_name}"

  # ── skip if already complete ────────────────────────────────────
  skip_ok="0"
  if [[ "${FORCE_REINFERENCE}" != "1" ]]; then
    model_ckpt="${session_dir}/model_last.pt"
    inference_pt="${session_dir}/inference_outputs.pt"
    epoch_summary_csv="${session_dir}/epoch_summary.csv"
    if [[ -f "${model_ckpt}" && -f "${inference_pt}" && -f "${epoch_summary_csv}" ]]; then
      skip_ok="$(_check_complete "${cfg}" "${epoch_summary_csv}" || echo "0")"
    fi
  fi

  if [[ "${skip_ok}" == "1" ]]; then
    echo "  SKIP  (epochs complete + inference exists)"
    ((skipped++)) || true
    continue
  fi

  # ── run main.py (isolated — failure does NOT kill the loop) ─────
  run_args=(python3 "${ROOT_DIR}/main.py" --config "${cfg}" --sessions-dir "${SESSIONS_DIR}")
  if [[ "${FORCE_REINFERENCE}" == "1" ]]; then
    run_args+=(--recompute-inference)
  fi
  echo "  RUN   ${run_args[*]}"
  if "${run_args[@]}"; then
    echo "  OK    ${config_name}"
    ((ok++)) || true
  else
    rc=$?
    echo "  FAIL  ${config_name}  (exit code ${rc})" >&2
    echo "[$(date -Iseconds)] ${config_name}: main.py exit ${rc}" >> "${FAIL_LOG}"
    ((failed++)) || true
  fi
done

# ── summary ────────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  total:   ${total}"
echo "  ok:      ${ok}"
echo "  skipped: ${skipped}"
echo "  failed:  ${failed}"
if [[ "${failed}" -gt 0 ]]; then
  echo "  failures logged → ${FAIL_LOG}"
fi
echo "═══════════════════════════════════════════════════════════════"
echo "Post-training inference artifacts are saved in each session directory."
echo "Use ./session_to_dash.sh to build dashboards into results/dashboard."

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
