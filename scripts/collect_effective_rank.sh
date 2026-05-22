#!/usr/bin/env bash
set -euo pipefail

SESSIONS_DIR="${1:-sessions}"
PREFIX="${2:-gen_117}"
OUT_CSV="${3:-$SESSIONS_DIR/effective_rank_${PREFIX}.csv}"

mkdir -p "$(dirname "$OUT_CSV")"
echo "session,effective_rank" > "$OUT_CSV"

shopt -s nullglob
for d in "$SESSIONS_DIR"/"$PREFIX"*; do
  [[ -d "$d" ]] || continue
  name="$(basename "$d")"
  f="$d/effective_rank.txt"
  if [[ -f "$f" ]]; then
    rank="$(tr -d '\r\n' < "$f")"
    echo "$name,$rank" >> "$OUT_CSV"
  else
    echo "$name," >> "$OUT_CSV"
  fi
done

echo "saved=$OUT_CSV"
