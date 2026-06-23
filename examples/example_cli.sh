#!/usr/bin/env bash
# CLI-driven example: identical to quickstart but run from terminal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

sajepa-train --config configs/examples/mhd_example.yaml

echo ""
echo "Done."
echo "  session:          sessions/mhd_example"
echo "  dashboard:        sessions/mhd_example/dashboard.html"
echo "  interactive_umap: sessions/mhd_example/results/interactive_umap_predict.html"
