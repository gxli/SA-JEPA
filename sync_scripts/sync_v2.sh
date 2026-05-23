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
LOCAL_RESULTS_DIR="result_local"

# Default version suffix for remote directory (enables versioned parallel workspaces).
SUFFIX=""

RSYNC_RSH='ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes'
RSYNC_COMMON=(
    -avz
    --checksum
    --partial
    --timeout=120
    --contimeout=20
    -e "$RSYNC_RSH"
)

# For pull-plots we intentionally skip checksum work and never overwrite existing local files.
RSYNC_PLOT_PULL_COMMON=(
    -avz
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

RSYNC_MKPATH_FLAG=""
if rsync --help 2>/dev/null | grep -q -- '--mkpath'; then
    RSYNC_MKPATH_FLAG="--mkpath"
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

collect_changed_code_files() {
    local remote_root="$1"
    local -a changed=()
    local line path
    # Use rsync dry-run itemization to collect only changed .py/.sh files.
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        # Skip rsync metadata lines.
        [[ "$line" == "sending incremental file list" ]] && continue
        [[ "$line" == sent\ * ]] && continue
        [[ "$line" == total\ size\ * ]] && continue
        # Itemized output starts with change flags then a space then path.
        # Example: >f..t...... src/train.py
        path="${line#* }"
        [[ -z "$path" || "$path" == "$line" ]] && continue
        case "$path" in
            *.py|*.sh)
                if [[ -f "$path" ]]; then
                    changed+=("$path")
                fi
                ;;
        esac
    done < <(
        rsync "${RSYNC_COMMON[@]}" \
            --dry-run \
            --itemize-changes \
            --delete \
            --exclude '.git' \
            --exclude 'experiments/' \
            --exclude 'result_local/' \
            --exclude 'results/' \
            --exclude 'sessions/' \
            --exclude 'outputs/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            ./ "$REMOTE:$remote_root" 2>/dev/null || true
    )
    if [[ ${#changed[@]} -gt 0 ]]; then
        printf '%s\n' "${changed[@]}"
    fi
}

usage() {
    echo "----------------------------------------------------------------"
    echo "Sync Script for: $REL_PATH"
    echo "----------------------------------------------------------------"
    echo "Usage: $0 {push|push-preview|pull|pull-plots|pull_plots|pull-plot|pull_plot|pull-all} [refresh] [suffix]"
    echo ""
    echo "Arguments:"
    echo "  suffix  -> Remote directory version suffix (e.g. _v2)."
    echo "             Remote path becomes .../proj/<project>_v2/"
    echo ""
    echo "Commands:"
    echo "  push        -> Local to Remote mirror APPLY (ignores sessions/, results/, result_local/)"
    echo "  push-preview-> Local to Remote mirror PREVIEW (dry-run, ignores sessions/, results/, result_local/)"
    echo "  pull        -> Remote outputs/ to Local $LOCAL_RESULTS_DIR/ (all files)"
    echo "  pull-plots  -> Remote results/ to Local $LOCAL_RESULTS_DIR/ (only .html/.png/.jpg/.pdf/.svg)"
    echo "                 skips files that already exist locally (no checksum)"
    echo "  pull_plots  -> Alias of pull-plots"
    echo "  pull-plot   -> Alias of pull-plots"
    echo "  pull_plot   -> Alias of pull-plots"
    echo "  pull-all    -> Remote project to Local current directory (legacy full pull)"
    echo "----------------------------------------------------------------"
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    usage
fi

REFRESH_ONLY=0
# Parse optional 2nd arg: refresh
if [[ "${2:-}" == "refresh" ]]; then
    REFRESH_ONLY=1
elif [[ -n "${2:-}" ]]; then
    SUFFIX="$2"
fi

# Parse optional 3rd arg: suffix (only valid when 2nd is refresh)
if [[ -n "${3:-}" ]]; then
    if [[ "$REFRESH_ONLY" -eq 0 ]]; then
        usage
    fi
    SUFFIX="$3"
fi

# Resolve remote HOME explicitly (avoid '~' path ambiguity across shells/hosts).
REMOTE_HOME="$(ssh "$REMOTE" 'printf %s "$HOME"')"
if [[ -z "${REMOTE_HOME}" ]]; then
    echo "ERROR: Could not resolve remote HOME for $REMOTE"
    exit 1
fi

# Remote destination with version suffix
VERSIONED_REL_PATH="${REL_PATH}${SUFFIX}"
REMOTE_DEST="${REMOTE_HOME}/proj/${VERSIONED_REL_PATH}/"
REMOTE_OUTPUTS_DEST="${REMOTE_HOME}/proj/${VERSIONED_REL_PATH}/outputs/"

case "$1" in
    push)
        echo "Applying safe mirror push to $REMOTE:$REMOTE_DEST (preserve excluded dirs)"
        ssh "$REMOTE" "mkdir -p \"$REMOTE_HOME/proj/$VERSIONED_REL_PATH\""
        changed_code_files=()
        while IFS= read -r _f; do
            [[ -n "$_f" ]] && changed_code_files+=("$_f")
        done < <(collect_changed_code_files "$REMOTE_HOME/proj/$VERSIONED_REL_PATH")
        if [[ ${#changed_code_files[@]} -eq 0 ]]; then
            echo "changed_code_files=0"
        else
            echo "changed_code_files=${#changed_code_files[@]}"
        fi
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
        # Verify only changed python/shell files landed correctly.
        if [[ ${#changed_code_files[@]} -eq 0 ]]; then
            echo "verify skipped: no changed .py/.sh files detected"
        else
            for f in "${changed_code_files[@]}"; do
                copy_and_verify_file "$f" "$REMOTE_HOME/proj/$VERSIONED_REL_PATH/$f"
            done
        fi
        ;;
    push-preview)
        echo "Previewing safe mirror push (dry-run) to $REMOTE:$REMOTE_DEST (preserve excluded dirs)"
        ssh "$REMOTE" "mkdir -p \"$REMOTE_HOME/proj/$VERSIONED_REL_PATH\""
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
            echo "pull-plots refresh mode: existing local files are still skipped (--ignore-existing)"
        fi
        echo "Pulling plot files (.html and images) from remote results/ to ./$LOCAL_RESULTS_DIR/ (skip existing local files)"
        mkdir -p "$LOCAL_RESULTS_DIR"
        d="$REMOTE_HOME/proj/$VERSIONED_REL_PATH/results/"
        local_dst="$LOCAL_RESULTS_DIR/results/"
        echo "  trying: $REMOTE:$d -> ./$local_dst"
        if ssh "$REMOTE" "test -d \"$d\""; then
            mkdir -p "$local_dst"
            # Precreate common nested plot destinations to avoid move_file errors on some rsync builds.
            mkdir -p "$local_dst/dashboard" "$local_dst/plots/session_dashboards" "$local_dst/dashboards" "$local_dst/plots"
            # Resume flags (--append/--append-verify) can interact badly with filtered tree pulls.
            # Use plain transfer for pull-plots stability.
            run_rsync_retry "${RSYNC_PLOT_PULL_COMMON[@]}" \
                ${RSYNC_MKPATH_FLAG:+$RSYNC_MKPATH_FLAG} \
                --ignore-existing \
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
