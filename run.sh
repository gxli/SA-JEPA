#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
SESSIONS_DIR="${ROOT_DIR}/sessions"

mkdir -p "${SESSIONS_DIR}"

shopt -s nullglob
configs=("${CONFIG_DIR}"/*.json)
if [ ${#configs[@]} -eq 0 ]; then
  echo "No config files found in ${CONFIG_DIR}"
  exit 1
fi

for cfg in "${configs[@]}"; do
  echo "Running config: ${cfg}"
  python3 "${ROOT_DIR}/main.py" --config "${cfg}" --sessions-dir "${SESSIONS_DIR}"
done

echo "All sessions complete."
