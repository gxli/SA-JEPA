#!/usr/bin/env bash
# Generate movie PNGs and MP4s for all sessions with movie_frames/
# Usage: ./run_movie.sh [session_name1 session_name2 ...]
#   No args: process all sessions with movie_frames/
#   With args: process only named sessions

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ROOT}/configs_bk/movie_default.json"
MOVIE_SCRIPT="${ROOT}/scripts/session_to_movie.py"

if [ $# -gt 0 ]; then
    SESSIONS=("$@")
else
    SESSIONS=()
    for d in "${ROOT}"/sessions/*/; do
        if [ -d "${d}movie_frames" ]; then
            SESSIONS+=("$(basename "$d")")
        fi
    done
fi

if [ ${#SESSIONS[@]} -eq 0 ]; then
    echo "No sessions with movie_frames/ found. Set train.movie_dump_every_epoch=true to enable."
    exit 1
fi

echo "=== Movie generation for ${#SESSIONS[@]} session(s) ==="
echo ""

# Detect UMAP backend
UMAP_BACKEND="CPU (umap-learn)"
python3 -c "from cuml.manifold import UMAP; print('cuml OK')" 2>/dev/null && UMAP_BACKEND="GPU (cuml)"
echo "UMAP backend: ${UMAP_BACKEND}"
echo ""

for name in "${SESSIONS[@]}"; do
    # Support both full paths and bare session names
    if [ -d "$name" ]; then
        session_dir="$(cd "$name" && pwd)"
    elif [ -d "${ROOT}/sessions/${name}" ]; then
        session_dir="${ROOT}/sessions/${name}"
    else
        echo "SKIP: session not found: $name"
        continue
    fi
    session_name="$(basename "$session_dir")"
    cp "$CONFIG" "${session_dir}/movie_config.json"
    echo "--- ${session_name} ---"
    python3 "$MOVIE_SCRIPT" "$session_dir" --make-mp4 --fps 10 --force
done

echo ""
echo "=== Done ==="
echo "PNGs: ${ROOT}/results/movie_png/<session_name>/"
echo "MP4s: ${ROOT}/results/movie_png/<session_name>/<session_name>.mp4"
