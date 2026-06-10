#!/usr/bin/env bash
set -euo pipefail

SESSIONS_DIR="${1:-sessions}"
SORT_MODE="${2:-name}"  # name | rank

{
  echo -e "session\tmask_fraction\teffective_rank"
  shopt -s nullglob
  for d in "$SESSIONS_DIR"/gen_17*; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    rank_file="$d/effective_rank.txt"
    cfg_file="$d/config_used.json"
    rank=""
    mask_fraction="NA"
    if [[ -f "$rank_file" ]]; then
      rank="$(tr -d '\r\n' < "$rank_file")"
    fi
    if [[ -f "$cfg_file" ]]; then
      mask_fraction="$(
        python - "$cfg_file" <<'PY'
import json,sys
p=sys.argv[1]
try:
    d=json.load(open(p,'r',encoding='utf-8'))
    v=d.get('model',{}).get('mask_fraction','NA')
    print(v)
except Exception:
    print('NA')
PY
      )"
      mask_fraction="$(printf "%s" "$mask_fraction" | tr -d '\r\n')"
    fi
    echo -e "$name\t$mask_fraction\t$rank"
  done
} | {
  IFS= read -r header
  if [[ "$SORT_MODE" == "rank" ]]; then
    {
      printf "%s\n" "$header"
      cat | awk -F'\t' 'NF>=3 && $3!=""' | sort -t$'\t' -k3,3gr
    } | column -t -s $'\t'
  else
    {
      printf "%s\n" "$header"
      cat | sort
    } | column -t -s $'\t'
  fi
}
