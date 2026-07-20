#!/usr/bin/env bash
# Sync root code → submission_sandbox/submit/ and sanitize author traces.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/submission_sandbox/submit"

echo "=== Cleaning destination ==="
rm -rf "$DEST"
mkdir -p "$DEST"

echo "=== Copying source files ==="
# Core code
rsync -a --exclude '__pycache__/' --exclude '*.pyc' --exclude '.git' \
    "$ROOT/src" "$ROOT/sajepa" "$ROOT/scripts" "$ROOT/tests" "$ROOT/examples" \
    "$DEST/"

# Configs (skip local_configs, experiments, inference)
rsync -a --exclude 'local_configs/' --exclude 'experiments/' --exclude 'inference/' --exclude 'README.md' \
    "$ROOT/configs/" "$DEST/configs/"

# Figures
if [ -d "$ROOT/figures" ]; then
    rsync -a "$ROOT/figures/" "$DEST/figures/"
fi

# Project files
cp "$ROOT/pyproject.toml" "$ROOT/requirements.txt" "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/MANIFEST.in" "$DEST/"

# Data + sessions placeholders (data is gitignored; ship empty dirs with README)
mkdir -p "$DEST/data" "$DEST/sessions"
cat > "$DEST/data/README.md" << 'EOF'
# Data Directory

Place your `.npy` or `.fits` files here.  Example configs expect:

- `C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5.npy` (MHD 2D)
- `C12_Beta20_256_0060-rho.npy_slice.npy_sm_0.5_gaussian_contaminated.npy`
- `orion_cut.npy_sm.npy`

See `configs/README.md` and `examples/` for usage.
EOF

echo "=== Sanitizing author traces ==="

# pyproject.toml
sed -i '' 's/Guang-Xing Li/Anonymous Author/g' "$DEST/pyproject.toml"
sed -i '' 's/ligx\.ngc7293@gmail\.com/anonymous@example.com/g' "$DEST/pyproject.toml"

# LICENSE
sed -i '' 's/Guang-Xing Li/Anonymous Author/g' "$DEST/LICENSE"

# README.md
sed -i '' 's/gxli\.ai@proton\.me/anonymous@example.com/g' "$DEST/README.md"
sed -i '' 's/ligx\.ngc7293@gmail\.com/anonymous@example.com/g' "$DEST/README.md"
sed -i '' 's/github\.com\/gxli\/SA-JEPA/github.com\/anonymous\/SA-JEPA/g' "$DEST/README.md"
sed -i '' 's/Li, Guang-Xing/Anonymous, Author/g' "$DEST/README.md"
sed -i '' 's/Guang-Xing Li/Anonymous Author/g' "$DEST/README.md"

echo "=== Done ==="
echo "Submission ready: $DEST"
ls -la "$DEST/"
