#!/usr/bin/env bash
# Download the sampled PyQCD comparison data listed in data4pyqcd_list-v20261025.txt.
#
# Usage:
#   bash download_data4pyqcd-v20261026.sh
#   bash download_data4pyqcd-v20261026.sh --dry-run
#   bash download_data4pyqcd-v20261026.sh --yes --skip-existing
#   bash download_data4pyqcd-v20261026.sh --jobs 8 --yes --skip-existing
#   bash download_data4pyqcd-v20261026.sh --dest-root /data/pyqcd
#   bash download_data4pyqcd-v20261026.sh --dest-root / --yes --skip-existing
#
# Default destination:
#   /public/...
#   /nexdata/...
#
# The remote absolute path is preserved exactly beneath --dest-root.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
LIST_FILE="${SCRIPT_DIR}/data4pyqcd_list-v20261025.txt"
DEST_ROOT="/"

SSH_HOST="${PYQCD_SSH_HOST:-222.200.137.16}"
SSH_PORT="${PYQCD_SSH_PORT:-10023}"
SSH_USER="${PYQCD_SSH_USER:-zhangxin}"
SSH_TARGET="${SSH_USER}@${SSH_HOST}"

# Guard against downloading a different or truncated sample accidentally.
EXPECTED_ENTRIES="${EXPECTED_ENTRIES:-101}"
EXPECTED_TOTAL_BYTES="${EXPECTED_TOTAL_BYTES:-1958277360128}"

DRY_RUN=false
SKIP_CONFIRM=false
SKIP_EXISTING=false
PARALLEL_JOBS=${PYQCD_DOWNLOAD_JOBS:-4}

usage() {
    cat <<'EOF'
Usage: download_data4pyqcd-v20261026.sh [OPTIONS]

Options:
  --dry-run              Validate and print transfers without network access
  --yes, -y              Do not prompt before each file or directory
  --skip-existing        Skip completed entries; verify unmarked files by SHA-256
  --resume               Alias for --skip-existing
  --dest-root PATH       Preserve /public and /nexdata below PATH (default: /)
  --list FILE            Use a different sample list
  --jobs N, -j N         Run up to N transfers concurrently (default: 4; 1 = sequential)
  --help, -h             Show this help

Environment:
  PYQCD_SSH_HOST         Remote host (default: 222.200.137.16)
  PYQCD_SSH_PORT         Remote SSH port (default: 10023)
  PYQCD_SSH_USER         Remote user (default: zhangxin)
  PYQCD_DOWNLOAD_JOBS    Default concurrency (default: 4)
  EXPECTED_ENTRIES       Expected list entry count (default: 101)
  EXPECTED_TOTAL_BYTES   Expected sampled bytes (default: 1958277360128)
EOF
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes|-y)
            SKIP_CONFIRM=true
            shift
            ;;
        --skip-existing|--resume)
            SKIP_EXISTING=true
            shift
            ;;
        --dest-root)
            (($# >= 2)) || { printf 'ERROR: --dest-root requires a path\n' >&2; exit 2; }
            DEST_ROOT=$2
            shift 2
            ;;
        --list)
            (($# >= 2)) || { printf 'ERROR: --list requires a file\n' >&2; exit 2; }
            LIST_FILE=$2
            shift 2
            ;;
        --jobs|-j)
            (($# >= 2)) || { printf 'ERROR: --jobs requires a positive integer\n' >&2; exit 2; }
            PARALLEL_JOBS=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: --jobs requires a positive integer: %s\n' "$PARALLEL_JOBS" >&2
    exit 2
}

if ((PARALLEL_JOBS > 1)) && { ((BASH_VERSINFO[0] < 4)) || ((BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3)); }; then
    printf 'WARN: bash >= 4.3 is required for parallel transfers; using --jobs 1\n' >&2
    PARALLEL_JOBS=1
fi

# Per-entry prompts and concurrent transfer logs would interleave on a terminal.
if ((PARALLEL_JOBS > 1)) && ! $DRY_RUN && ! $SKIP_CONFIRM && [[ -t 0 ]]; then
    printf 'NOTE: interactive confirmation requires --yes for parallel transfers; using --jobs 1\n' >&2
    PARALLEL_JOBS=1
fi

[[ -f "$LIST_FILE" ]] || { printf 'ERROR: list not found: %s\n' "$LIST_FILE" >&2; exit 1; }

LIST_PATHS=()
LIST_BYTES=()
TOTAL_BYTES=0
declare -A SEEN_PATHS=()

while IFS=$'\t' read -r size path; do
    [[ -n "$size" && -n "$path" ]] || continue
    [[ "$size" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid size in list: %s\n' "$size" >&2; exit 1; }
    [[ "$path" == /public/* || "$path" == /nexdata/* ]] || {
        printf 'ERROR: unsupported remote root: %s\n' "$path" >&2
        exit 1
    }
    [[ -z "${SEEN_PATHS[$path]:-}" ]] || { printf 'ERROR: duplicate list path: %s\n' "$path" >&2; exit 1; }
    SEEN_PATHS["$path"]=1
    LIST_PATHS+=("$path")
    LIST_BYTES+=("$size")
    TOTAL_BYTES=$((TOTAL_BYTES + size))
done < <(
    awk -F'|' '
        $1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
            size=$1; path=$5
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", size)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", path)
            print size "\t" path
        }
    ' "$LIST_FILE"
)

ENTRY_COUNT=${#LIST_PATHS[@]}
if ((ENTRY_COUNT != EXPECTED_ENTRIES)); then
    printf 'ERROR: expected %s entries, found %s\n' "$EXPECTED_ENTRIES" "$ENTRY_COUNT" >&2
    exit 1
fi
if ((TOTAL_BYTES != EXPECTED_TOTAL_BYTES)); then
    printf 'ERROR: expected %s bytes, found %s\n' "$EXPECTED_TOTAL_BYTES" "$TOTAL_BYTES" >&2
    exit 1
fi

DEST_ROOT=${DEST_ROOT%/}
[[ -n "$DEST_ROOT" ]] || DEST_ROOT=/
STATE_DIR="${DEST_ROOT%/}/.download-state/data4pyqcd-v20261026"
[[ "$DEST_ROOT" == / ]] && STATE_DIR="/.download-state/data4pyqcd-v20261026"

map_remote_to_local() {
    local remote_path=$1
    if [[ "$DEST_ROOT" == / ]]; then
        printf '%s\n' "$remote_path"
    else
        printf '%s%s\n' "$DEST_ROOT" "$remote_path"
    fi
}

guess_remote_kind() {
    case "$1" in
        *.lime|*.npy|*.npz|*.json|*.tsv|*.txt|*.pdf|*.png) printf 'file\n' ;;
        *) printf 'dir\n' ;;
    esac
}

quote_remote_path() {
    local path=$1
    [[ "$path" != *"'"* ]] || { printf 'ERROR: unsupported quote in remote path: %s\n' "$path" >&2; return 1; }
    printf "'%s'\n" "$path"
}

remote_kind() {
    local remote_path quoted
    remote_path=$1
    quoted=$(quote_remote_path "$remote_path") || return 1
    ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" \
        "if [ -d $quoted ]; then echo dir; elif [ -f $quoted ]; then echo file; else echo missing; fi"
}

remote_file_size() {
    local remote_path quoted size
    remote_path=$1
    quoted=$(quote_remote_path "$remote_path") || return 1
    size=$(ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" "stat -c%s $quoted") || return 1
    [[ "$size" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$size"
}

remote_file_sha256() {
    local remote_path quoted checksum
    remote_path=$1
    quoted=$(quote_remote_path "$remote_path") || return 1
    checksum=$(ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" "sha256sum $quoted" | awk '{print $1}') || return 1
    [[ "$checksum" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s\n' "$checksum"
}

list_state_file() {
    local remote_path=$1 key
    key=$(printf '%s' "$remote_path" | sha256sum | awk '{print $1}')
    printf '%s/%s.complete\n' "$STATE_DIR" "$key"
}

already_complete() {
    local remote_path=$1 local_path=$2 expected=$3 kind=$4
    local state marker_size marker_path remote_size local_size remote_checksum local_checksum
    state=$(list_state_file "$remote_path")
    if [[ -e "$state" && -e "$local_path" ]]; then
        if IFS=$'\t' read -r marker_size marker_path < "$state" &&
            [[ "$marker_size" == "$expected" && "$marker_path" == "$remote_path" ]]; then
            return 0
        fi
    fi

    # The list stores allocated bytes, so unmarked files require a checksum
    # against the remote source instead of trusting the list or preallocated size.
    [[ "$kind" == file && "$DRY_RUN" == false && -f "$local_path" ]] || return 1
    printf '  VERIFY: existing file lacks a completion marker; comparing SHA-256...\n'
    remote_size=$(remote_file_size "$remote_path") || return 1
    local_size=$(stat -c%s "$local_path" 2>/dev/null) || return 1
    [[ "$remote_size" == "$local_size" ]] || return 1
    remote_checksum=$(remote_file_sha256 "$remote_path") || return 1
    local_checksum=$(sha256sum "$local_path" | awk '{print $1}') || return 1
    [[ "$remote_checksum" == "$local_checksum" ]]
}

mark_complete() {
    local remote_path=$1 expected=$2 state tmp
    state=$(list_state_file "$remote_path")
    tmp="${state}.tmp.$$"
    mkdir -p "$(dirname "$state")"
    printf '%s\t%s\n' "$expected" "$remote_path" > "$tmp"
    mv -f "$tmp" "$state"
}

confirm_transfer() {
    local kind=$1 remote_path=$2
    local answer
    if $SKIP_CONFIRM; then
        return 0
    fi
    read -r -p "Download ${kind} ${remote_path}? [Y/n] " answer
    answer=${answer:-Y}
    [[ "$answer" =~ ^[Yy]$ ]]
}

download_file() {
    local remote_path=$1 local_path=$2
    if [[ "$TRANSFER_TOOL" == rsync ]]; then
        rsync -avP --partial -e "ssh ${SSH_COMMON_OPTS[*]}" \
            "${SSH_TARGET}:${remote_path}" "$local_path"
    else
        scp "${SCP_OPTS[@]}" "${SSH_TARGET}:${remote_path}" "$local_path"
    fi
}

download_dir() {
    local remote_path=$1 local_path=$2
    if [[ "$TRANSFER_TOOL" == rsync ]]; then
        rsync -avP --partial --partial-dir=.rsync-partial -e "ssh ${SSH_COMMON_OPTS[*]}" \
            "${SSH_TARGET}:${remote_path}/" "${local_path}/"
    else
        # Copy the directory itself into its parent; scp's SFTP mode rejects a trailing "/.".
        scp -r "${SCP_OPTS[@]}" "${SSH_TARGET}:${remote_path}" "$(dirname "$local_path")/"
    fi
}

execute_transfer() {
    local index=$1 kind=$2 remote_path=$3 local_path=$4 rc=0
    if [[ "$kind" == file ]]; then
        download_file "$remote_path" "$local_path" || rc=$?
    else
        download_dir "$remote_path" "$local_path" || rc=$?
    fi
    if ((rc == 0)); then
        mark_complete "$remote_path" "${LIST_BYTES[$index]}" || rc=$?
    fi
    return "$rc"
}

launch_transfer() {
    local index=$1 kind=$2 remote_path=$3 local_path=$4
    local log_file="${JOB_LOG_DIR}/${index}.log"
    local status_file="${STATUS_DIR}/${index}.status"
    (
        set +e
        execute_transfer "$index" "$kind" "$remote_path" "$local_path" >"$log_file" 2>&1
        rc=$?
        if ((rc == 0)); then
            printf 'done\n' >"$status_file"
            printf '  DONE   %s\n' "$remote_path"
        else
            printf 'failed\n' >"$status_file"
            printf '  FAILED %s (log: %s)\n' "$remote_path" "$log_file" >&2
        fi
        exit 0
    ) &
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
}

cleanup_ssh() {
    if [[ -n "${SSH_CONTROL_PATH:-}" && -S "$SSH_CONTROL_PATH" ]]; then
        ssh "${SSH_COMMON_OPTS[@]}" -O exit "$SSH_TARGET" >/dev/null 2>&1 || true
    fi
    if [[ -n "${SSH_CONTROL_DIR:-}" ]]; then
        rm -rf "$SSH_CONTROL_DIR"
    fi
    if [[ -n "${WORK_DIR:-}" ]] && ! ${KEEP_WORK_DIR:-false}; then
        rm -rf "$WORK_DIR"
    fi
}

TRANSFER_TOOL=none
if command -v rsync >/dev/null 2>&1; then
    TRANSFER_TOOL=rsync
elif command -v scp >/dev/null 2>&1; then
    TRANSFER_TOOL=scp
elif ! $DRY_RUN; then
    printf 'ERROR: neither rsync nor scp is installed\n' >&2
    exit 1
fi

printf '%s\n' '============================================================================='
printf '%s\n' ' PyQCD sampled data download'
printf '%s\n' '============================================================================='
printf ' List:       %s\n' "$LIST_FILE"
printf ' Entries:    %d\n' "$ENTRY_COUNT"
printf ' Bytes:      %d\n' "$TOTAL_BYTES"
printf ' Destination:%s\n' "$DEST_ROOT"
printf ' SSH:        %s:%s (%s)\n' "$SSH_HOST" "$SSH_PORT" "$SSH_USER"
printf ' Tool:       %s\n' "$TRANSFER_TOOL"
printf ' Jobs:       %d\n' "$PARALLEL_JOBS"
printf ' Mode:       %s%s\n' \
    "$($DRY_RUN && printf DRY-RUN || printf LIVE)" \
    "$($SKIP_EXISTING && printf ' + skip-existing' || true)"
printf '%s\n' '============================================================================='

if ! $DRY_RUN; then
    SSH_CONTROL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pyqcd-download.XXXXXX")
    SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/sock-%r@%h:%p"
    SSH_COMMON_OPTS=(
        -p "$SSH_PORT"
        -o ControlMaster=auto
        -o "ControlPath=${SSH_CONTROL_PATH}"
        -o ControlPersist=600
    )
    SCP_OPTS=(
        -P "$SSH_PORT"
        -o ControlMaster=auto
        -o "ControlPath=${SSH_CONTROL_PATH}"
        -o ControlPersist=600
    )
    trap cleanup_ssh EXIT

    printf 'Establishing SSH master connection...\n'
    if ! ssh "${SSH_COMMON_OPTS[@]}" -o ConnectTimeout=10 "$SSH_TARGET" true; then
        printf 'ERROR: cannot connect to %s:%s as %s\n' "$SSH_HOST" "$SSH_PORT" "$SSH_USER" >&2
        exit 1
    fi
    mkdir -p "$STATE_DIR"
    WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pyqcd-download-work.XXXXXX")
    STATUS_DIR="${WORK_DIR}/status"
    JOB_LOG_DIR="${WORK_DIR}/logs"
    mkdir -p "$STATUS_DIR" "$JOB_LOG_DIR"
    printf ' Concurrent job logs: %s\n' "$JOB_LOG_DIR"
else
    SSH_COMMON_OPTS=()
    SCP_OPTS=()
    WORK_DIR=""
    STATUS_DIR=""
    JOB_LOG_DIR=""
fi

COUNT_SKIPPED=0
COUNT_DONE=0
COUNT_FAILED=0
RUNNING_JOBS=0
KEEP_WORK_DIR=false
LAUNCHED_INDICES=()
FAILED_LOG_INDICES=()

for i in "${!LIST_PATHS[@]}"; do
    remote_path=${LIST_PATHS[$i]}
    expected=${LIST_BYTES[$i]}
    local_path=$(map_remote_to_local "$remote_path")

    if $DRY_RUN; then
        kind=$(guess_remote_kind "$remote_path")
    else
        if ! kind=$(remote_kind "$remote_path"); then
            printf 'FAILED: cannot inspect remote path: %s\n' "$remote_path" >&2
            COUNT_FAILED=$((COUNT_FAILED + 1))
            continue
        fi
        if [[ "$kind" == missing ]]; then
            printf 'FAILED: remote path missing: %s\n' "$remote_path" >&2
            COUNT_FAILED=$((COUNT_FAILED + 1))
            continue
        fi
    fi

    printf '\n[%d/%d] %s (%s, %s bytes)\n' "$((i + 1))" "$ENTRY_COUNT" "$remote_path" "$kind" "$expected"
    printf '  Local: %s\n' "$local_path"

    if $SKIP_EXISTING && already_complete "$remote_path" "$local_path" "$expected" "$kind"; then
        if ! $DRY_RUN; then
            mark_complete "$remote_path" "$expected"
        fi
        printf '  SKIP: completed entry verified\n'
        COUNT_SKIPPED=$((COUNT_SKIPPED + 1))
        continue
    fi

    if ! confirm_transfer "$kind" "$remote_path"; then
        printf '  SKIP: declined by user\n'
        COUNT_SKIPPED=$((COUNT_SKIPPED + 1))
        continue
    fi

    if $DRY_RUN; then
        printf '  DRY-RUN: no transfer performed\n'
        continue
    fi

    mkdir -p "$(dirname "$local_path")"
    if [[ "$kind" != file && "$kind" != dir ]]; then
        printf 'FAILED: unsupported remote type: %s\n' "$kind" >&2
        COUNT_FAILED=$((COUNT_FAILED + 1))
    elif ((PARALLEL_JOBS == 1)); then
        if execute_transfer "$i" "$kind" "$remote_path" "$local_path"; then
            COUNT_DONE=$((COUNT_DONE + 1))
            printf '  DONE\n'
        else
            COUNT_FAILED=$((COUNT_FAILED + 1))
            printf '  FAILED\n' >&2
        fi
    else
        launch_transfer "$i" "$kind" "$remote_path" "$local_path"
        LAUNCHED_INDICES+=("$i")
        while ((RUNNING_JOBS >= PARALLEL_JOBS)); do
            wait -n || true
            RUNNING_JOBS=$((RUNNING_JOBS - 1))
        done
    fi
done

if ((PARALLEL_JOBS > 1)); then
    wait || true
    RUNNING_JOBS=0
    for i in "${LAUNCHED_INDICES[@]}"; do
        status_file="${STATUS_DIR}/${i}.status"
        if [[ -f "$status_file" && "$(< "$status_file")" == done ]]; then
            COUNT_DONE=$((COUNT_DONE + 1))
        else
            COUNT_FAILED=$((COUNT_FAILED + 1))
            FAILED_LOG_INDICES+=("$i")
        fi
    done
fi

printf '\n%s\n' '============================================================================='
printf ' Skipped: %d\n' "$COUNT_SKIPPED"
printf ' Done:    %d\n' "$COUNT_DONE"
printf ' Failed:  %d\n' "$COUNT_FAILED"
printf '%s\n' '============================================================================='

if ((${#FAILED_LOG_INDICES[@]} > 0)); then
    KEEP_WORK_DIR=true
    printf '\nFailure logs (last 20 lines):\n' >&2
    for i in "${FAILED_LOG_INDICES[@]}"; do
        printf -- '--- %s (job log: %s) ---\n' "${LIST_PATHS[$i]}" "${JOB_LOG_DIR}/${i}.log" >&2
        tail -n 20 "${JOB_LOG_DIR}/${i}.log" >&2 || true
    done
    printf 'Full job logs kept in: %s\n' "$JOB_LOG_DIR" >&2
fi

if $DRY_RUN; then
    printf 'Dry run complete; remove --dry-run to transfer data.\n'
elif ((COUNT_FAILED == 0)); then
    printf 'All selected entries downloaded successfully.\n'
else
    printf 'Some entries failed; rerun with --skip-existing after fixing the cause.\n' >&2
    exit 1
fi
