#!/usr/bin/env bash
# If invoked via `sh sync_v2.sh ...`, re-exec under bash so arrays/strict mode behave correctly.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Mirror full relative path under ~/proj
BASE_LOCAL_ROOT="$HOME/proj"
if [[ "$PWD" != "$BASE_LOCAL_ROOT"* ]]; then
    echo "ERROR: Current directory must be under $BASE_LOCAL_ROOT"
    exit 1
fi
REL_PATH="${PWD#"$BASE_LOCAL_ROOT"/}"
REMOTE="gxli@100.73.221.104"

# Resolve remote HOME explicitly (avoid '~' path ambiguity across shells/hosts).
REMOTE_HOME="$(ssh "$REMOTE" 'printf %s "$HOME"')"
if [[ -z "${REMOTE_HOME}" ]]; then
    echo "ERROR: Could not resolve remote HOME for $REMOTE"
    exit 1
fi

# Remote destination mirrors local path relative to local ~/proj
REMOTE_DEST="${REMOTE_HOME}/proj/${REL_PATH}/"
REMOTE_OUTPUTS_DEST="${REMOTE_HOME}/proj/${REL_PATH}/outputs/"
LOCAL_RESULTS_DIR="result_local"

RSYNC_RSH='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes'
RSYNC_COMMON=(
    -avz
    --checksum
    --partial
    --timeout=120
    --contimeout=20
    -e "$RSYNC_RSH"
)

RSYNC_RESUME_FLAG=""
if rsync --help 2>/dev/null | grep -q -- '--append-verify'; then
    RSYNC_RESUME_FLAG="--append-verify"
elif rsync --help 2>/dev/null | grep -q -- '--append'; then
    RSYNC_RESUME_FLAG="--append"
fi

run_rsync_retry() {
    # Retries transient rsync/ssh failures (e.g., status 11 over unstable link).
    local attempts=3
    local n=1
    while true; do
        if rsync "$@"; then
            return 0
        fi
        rc=$?
        if [[ "${rc}" -eq 0 ]]; then
            rc=1
        fi
        if [[ $n -ge $attempts ]]; then
            echo "rsync failed after ${attempts} attempts (exit ${rc})"
            return "$rc"
        fi
        echo "rsync attempt ${n}/${attempts} failed (exit ${rc}), retrying..."
        sleep $((2 * n))
        n=$((n + 1))
    done
}

verify_remote_file_md5() {
    local local_file="$1"
    local remote_file="$2"
    if [[ ! -f "$local_file" ]]; then
        echo "verify skipped (missing local): $local_file"
        return 0
    fi
    local local_md5
    local_md5="$(md5 -q "$local_file" 2>/dev/null || md5sum "$local_file" | awk '{print $1}')"
    local remote_md5
    remote_md5="$(ssh "$REMOTE" "if command -v md5sum >/dev/null 2>&1; then md5sum '$remote_file' | awk '{print \$1}'; elif command -v md5 >/dev/null 2>&1; then md5 -q '$remote_file'; else echo ''; fi" || true)"
    if [[ -z "$remote_md5" ]]; then
        echo "verify failed (no remote hash): $remote_file"
        return 1
    fi
    if [[ "$local_md5" != "$remote_md5" ]]; then
        echo "verify mismatch:"
        echo "  local : $local_file $local_md5"
        echo "  remote: $remote_file $remote_md5"
        return 1
    fi
    echo "verify ok: $local_file"
}

copy_and_verify_file() {
    local local_file="$1"
    local remote_file="$2"
    if verify_remote_file_md5 "$local_file" "$remote_file"; then
        return 0
    fi
    echo "forcing copy via scp: $local_file -> $remote_file"
    ssh "$REMOTE" "mkdir -p \"$(dirname "$remote_file")\""
    scp -q -p "$local_file" "$REMOTE:$remote_file"
    verify_remote_file_md5 "$local_file" "$remote_file"
}

force_sync_critical_files() {
    local remote_root="$1"
    local files=(
        "src/train.py"
        "src/models/build_jepa.py"
        "scripts/session_overview_4panel.py"
    )
    for f in "${files[@]}"; do
        if [[ -f "$f" ]]; then
            run_rsync_retry "${RSYNC_COMMON[@]}" \
                ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
                "$f" "$REMOTE:${remote_root}/$f"
        fi
    done
}

usage() {
    echo "----------------------------------------------------------------"
    echo "Sync Script for: $REL_PATH"
    echo "----------------------------------------------------------------"
    echo "Usage: $0 {push|push-preview|pull|pull-plots|pull_plots|pull-plot|pull_plot|pull-all} [refresh]"
    echo ""
    echo "Commands:"
    echo "  push        -> Local to Remote mirror APPLY (ignores sessions/, results/, result_local/)"
    echo "  push-preview-> Local to Remote mirror PREVIEW (dry-run, ignores sessions/, results/, result_local/)"
    echo "  pull        -> Remote outputs/ to Local $LOCAL_RESULTS_DIR/ (all files)"
    echo "  pull-plots  -> Remote results/ to Local $LOCAL_RESULTS_DIR/ (only .html/.png/.jpg/.pdf/.svg)"
    echo "                 add 'refresh' to copy only when remote file is newer"
    echo "  pull_plots  -> Alias of pull-plots"
    echo "  pull-plot   -> Alias of pull-plots"
    echo "  pull_plot   -> Alias of pull-plots"
    echo "  pull-all    -> Remote project to Local current directory (legacy full pull)"
    echo "----------------------------------------------------------------"
    exit 1
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
fi

REFRESH_ONLY=0
if [[ "${2:-}" == "refresh" ]]; then
    REFRESH_ONLY=1
elif [[ -n "${2:-}" ]]; then
    usage
fi

case "$1" in
    push)
        echo "Applying safe mirror push to $REMOTE:$REMOTE_DEST (preserve excluded dirs)"
        ssh "$REMOTE" "mkdir -p \"$REMOTE_HOME/proj/$REL_PATH\""
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
            --delete \
            --exclude '.git' \
            --exclude 'experiments/' \
            --exclude 'result_local/' \
            --exclude 'results/' \
            --exclude 'sessions/' \
            --exclude 'outputs/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            ./ "$REMOTE:$REMOTE_DEST"
        # Force-sync critical code paths explicitly (guards against edge-case filter drift).
        force_sync_critical_files "$REMOTE_HOME/proj/$REL_PATH"
        # Verify critical code files landed correctly.
        copy_and_verify_file "src/train.py" "$REMOTE_HOME/proj/$REL_PATH/src/train.py"
        copy_and_verify_file "src/models/build_jepa.py" "$REMOTE_HOME/proj/$REL_PATH/src/models/build_jepa.py"
        copy_and_verify_file "scripts/session_overview_4panel.py" "$REMOTE_HOME/proj/$REL_PATH/scripts/session_overview_4panel.py"
        ;;
    push-preview)
        echo "Previewing safe mirror push (dry-run) to $REMOTE:$REMOTE_DEST (preserve excluded dirs)"
        ssh "$REMOTE" "mkdir -p \"$REMOTE_HOME/proj/$REL_PATH\""
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
            --dry-run \
            --delete \
            --exclude '.git' \
            --exclude 'experiments/' \
            --exclude 'result_local/' \
            --exclude 'results/' \
            --exclude 'sessions/' \
            --exclude 'outputs/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            ./ "$REMOTE:$REMOTE_DEST"
        ;;
    pull)
        echo "Pulling remote outputs from $REMOTE:$REMOTE_OUTPUTS_DEST to ./$LOCAL_RESULTS_DIR/"
        mkdir -p "$LOCAL_RESULTS_DIR"
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
            "$REMOTE:$REMOTE_OUTPUTS_DEST" "$LOCAL_RESULTS_DIR"/
        ;;
    pull-plots|pull_plots|pull-plot|pull_plot)
        if [[ "$REFRESH_ONLY" -eq 1 ]]; then
            echo "Refreshing plot files from remote results/ to ./$LOCAL_RESULTS_DIR/ (only if remote is newer)"
        else
            echo "Pulling plot files (.html and images) from remote results/ to ./$LOCAL_RESULTS_DIR/"
        fi
        mkdir -p "$LOCAL_RESULTS_DIR"
        d="$REMOTE_HOME/proj/$REL_PATH/results/"
        local_dst="$LOCAL_RESULTS_DIR/results/"
        echo "  trying: $REMOTE:$d -> ./$local_dst"
        if ssh "$REMOTE" "test -d \"$d\""; then
            mkdir -p "$local_dst"
            EXTRA_REFRESH_ARGS=()
            if [[ "$REFRESH_ONLY" -eq 1 ]]; then
                EXTRA_REFRESH_ARGS+=(--update)
            fi
            run_rsync_retry "${RSYNC_COMMON[@]}" \
                ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
                "${EXTRA_REFRESH_ARGS[@]}" \
                --include='*/' \
                --include='*.html' --include='*.htm' \
                --include='*.png'  --include='*.jpg' \
                --include='*.jpeg' --include='*.pdf' \
                --include='*.svg' \
                --exclude='*' "$REMOTE:$d" "$local_dst"
        else
            echo "    skipped (missing): $d"
        fi
        ;;
    pull-all)
        echo "Pulling all remote contents from $REMOTE:$REMOTE_DEST to current directory"
        run_rsync_retry "${RSYNC_COMMON[@]}" \
            ${RSYNC_RESUME_FLAG:+$RSYNC_RESUME_FLAG} \
            --exclude '.git' "$REMOTE:$REMOTE_DEST" ./
        ;;
    *)
        usage
        ;;
esac
