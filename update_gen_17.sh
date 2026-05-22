#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SESSIONS_DIR="${1:-sessions}"
RECOMPUTE="${RECOMPUTE_INFERENCE:-1}"  # 1=recompute inference, 0=reuse existing

shopt -s nullglob
configs=(configs/gen_17*.json)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configs matched: configs/gen_17*.json" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  echo "update_effective_rank config=$cfg sessions_dir=$SESSIONS_DIR recompute_inference=$RECOMPUTE"
  if [[ "$RECOMPUTE" == "1" ]]; then
    python main.py --config "$cfg" --sessions-dir "$SESSIONS_DIR" --update-effective-rank --recompute-inference
  else
    python main.py --config "$cfg" --sessions-dir "$SESSIONS_DIR" --update-effective-rank
  fi
done

./scripts/collect_effective_rank.sh "$SESSIONS_DIR" gen_17 "$SESSIONS_DIR/effective_rank_gen_17.csv"
echo "done_update_gen_17 sessions_dir=$SESSIONS_DIR"
