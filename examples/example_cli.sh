#!/usr/bin/env bash
# CLI-driven example: identical to quickstart but run from terminal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

sajepa-train --config configs/examples/mhd_example.yaml --name example_cli

echo ""
echo "Done."
echo "  session:          sessions/example_cli"
echo "  dashboard:        sessions/example_cli/dashboard.html"
echo "  interactive_umap: sessions/example_cli/results/interactive_umap_predict.html"
