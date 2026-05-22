#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  set -- "sessions"
fi

echo "session,effective_rank"

emit_session() {
  local d="$1"
  local name f rank
  [[ -d "$d" ]] || return 0
  name="$(basename "$d")"
  f="$d/effective_rank.txt"
  if [[ -f "$f" ]]; then
    rank="$(tr -d '\r\n' < "$f")"
    echo "$name,$rank"
  else
    echo "$name,"
  fi
}

for arg in "$@"; do
  # If arg is already a session dir, emit directly.
  if [[ -d "$arg" && -f "$arg/config_used.json" ]]; then
    emit_session "$arg"
    continue
  fi
  # Otherwise treat arg as a root dir and scan one level of session dirs.
  if [[ -d "$arg" ]]; then
    shopt -s nullglob
    for d in "$arg"/*; do
      [[ -d "$d" ]] || continue
      emit_session "$d"
    done
  fi
done
