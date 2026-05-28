#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 <session_dir> [--make-mp4] [--fps 5]"
    exit 1
}

SESSION_DIR=""
MAKE_MP4=""
FPS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --make-mp4)
            MAKE_MP4="--make-mp4"
            shift
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -z "$SESSION_DIR" ]]; then
                SESSION_DIR="$1"
                shift
            else
                echo "Unexpected argument: $1"
                usage
            fi
            ;;
    esac
done

if [[ -z "$SESSION_DIR" ]]; then
    usage
fi

if [[ -n "$MAKE_MP4" ]]; then
    python "$SCRIPT_DIR/scripts/session_to_movie.py" "$SESSION_DIR" --make-mp4 --fps "$FPS"
else
    python "$SCRIPT_DIR/scripts/session_to_movie.py" "$SESSION_DIR" --fps "$FPS"
fi
