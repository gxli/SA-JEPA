#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
SESSIONS_DIR="${ROOT_DIR}/sessions"
PLOTS_RUNNER="${ROOT_DIR}/run_all_session_plots.sh"
DASH_DIR="${ROOT_DIR}/results/dashboards"

mkdir -p "${SESSIONS_DIR}"
mkdir -p "${DASH_DIR}"

shopt -s nullglob
configs=("${CONFIG_DIR}"/*.json)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No config files found in ${CONFIG_DIR}" >&2
  exit 1
fi

for cfg in "${configs[@]}"; do
  echo "Running config: ${cfg}"
  python3 "${ROOT_DIR}/main.py" --config "${cfg}" --sessions-dir "${SESSIONS_DIR}"
done

if [[ -x "${PLOTS_RUNNER}" ]]; then
  bash "${PLOTS_RUNNER}"
else
  echo "Missing executable plot runner: ${PLOTS_RUNNER}" >&2
  exit 1
fi

# Collect all dashboard-like HTMLs into results/dashboards for clear review.
for sdir in "${SESSIONS_DIR}"/*; do
  [[ -d "${sdir}" ]] || continue
  sid="$(basename "${sdir}")"

  for f in \
    "${sdir}/masking_demo_all.html" \
    "${sdir}/latent_overview_4panel.html" \
    "${sdir}/results/plots/"*.html \
    "${sdir}/plots/"*.html; do
    [[ -f "${f}" ]] || continue
    base="$(basename "${f}")"
    cp -f "${f}" "${DASH_DIR}/${sid}__${base}"
  done
done

echo "All sessions complete."
echo "Dashboards copied to: ${DASH_DIR}"
