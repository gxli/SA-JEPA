#!/usr/bin/env bash
# Generate movie PNGs and MP4s for all sessions with movie_frames/
# Usage: ./run_movie.sh [session_name1 session_name2 ...]
#   No args: process all sessions with movie_frames/
#   With args: process only named sessions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${ROOT}/configs/movie/movie_default.json"
MOVIE_SCRIPT="${ROOT}/scripts/session_to_movie.py"

if [ $# -gt 0 ]; then
    # Explicit session list
    SESSIONS=("$@")
else
    # Auto-discover sessions with movie_frames/
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
for name in "${SESSIONS[@]}"; do
    session_dir="${ROOT}/sessions/${name}"
    if [ ! -d "$session_dir" ]; then
        echo "SKIP: session not found: $session_dir"
        continue
    fi
    echo ""
    echo "--- ${name} ---"
    python3 "$MOVIE_SCRIPT" "$session_dir" --config "$CONFIG" --make-mp4
done

echo ""
echo "=== Done ==="
echo "PNGs: ${ROOT}/results/movie_png/<session_name>/"
echo "MP4s: ${ROOT}/results/movie_png/<session_name>/<session_name>.mp4"
