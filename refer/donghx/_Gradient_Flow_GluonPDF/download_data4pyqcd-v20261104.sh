#!/usr/bin/env bash
# Download the sampled PyQCD comparison data listed in data4pyqcd_list-v20261025.txt.
#
# This is the v20261104 rewrite of the v20261026 script. The sample list, guards
# and destination layout are unchanged; the transport now walks each source once,
# streams files through a bounded worker pool with per-file resume and retry, and
# shows aggregate progress and total transfer rate.
#
# Usage:
#   bash download_data4pyqcd-v20261104.sh --dry-run
#   bash download_data4pyqcd-v20261104.sh --preflight
#   bash download_data4pyqcd-v20261104.sh --yes
#   bash download_data4pyqcd-v20261104.sh --jobs 8 --yes
#   bash download_data4pyqcd-v20261104.sh --jobs 8 --discovery-jobs 4 --yes
#   bash download_data4pyqcd-v20261104.sh --dest-root /data/pyqcd --yes
#
# Default destination: /public/... and /nexdata/... below --dest-root.

set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_NAME=${BASH_SOURCE[0]##*/}
readonly ENGINE="${SCRIPT_DIR}/download_engine_v20261104.sh"

LIST_FILE="${SCRIPT_DIR}/data4pyqcd_list-v20261025.txt"
DEST_ROOT="/"

# Engine flags default to false here so the front-end can read them before the
# engine is sourced under `set -u`.
DL_DRY_RUN=false
DL_PREFLIGHT_ONLY=false
DL_CHECKSUM=false
DL_SOURCE_LOCAL=false
DL_MISSING_POLICY="fail"
DL_JOBS=""
DL_DISCOVERY_JOBS=""
DL_RETRIES=""
DL_PROGRESS=""

# Guard against downloading a different or truncated sample accidentally.
EXPECTED_ENTRIES="${EXPECTED_ENTRIES:-101}"
EXPECTED_TOTAL_BYTES="${EXPECTED_TOTAL_BYTES:-1958277360128}"

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Options:
  --dry-run              Validate and print the plan without network access
  --preflight            Probe every listed source path, then exit
  --yes, -y              Do not ask for a final confirmation
  --skip-existing        Accepted for compatibility; completed files are always skipped
  --resume               Alias for --skip-existing
  --verify               Re-hash files whose size already matches before skipping
  --allow-missing        Transfer available entries instead of aborting on missing ones
  --dest-root PATH       Preserve /public and /nexdata below PATH (default: /)
  --list FILE            Use a different sample list
  --jobs N, -j N         Concurrent file transfers (default: 4)
  --discovery-jobs N     Concurrent source-tree probes (default: 3)
  --retries N            Attempts per file before giving up (default: 6)
  --progress             Force the live progress panel
  --no-progress          Disable the live progress panel
  --help, -h             Show this help

Environment:
  PYQCD_SSH_HOST         Remote host (default: 222.200.137.16)
  PYQCD_SSH_PORT         Remote SSH port (default: 10023)
  PYQCD_SSH_USER         Remote user (default: zhangxin)
  PYQCD_DOWNLOAD_JOBS    Default transfer concurrency (default: 4)
  PYQCD_DOWNLOAD_DISCOVERY_JOBS  Default probe concurrency (default: 3)
  PYQCD_DOWNLOAD_RETRIES Attempts per file (default: 6)
  PYQCD_DOWNLOAD_PROGRESS  Progress mode: auto, on, or off (default: auto)
  PYQCD_SOURCE_LOCAL     Set to 1 to read the source paths on this host
  EXPECTED_ENTRIES       Expected list entry count (default: 101)
  EXPECTED_TOTAL_BYTES   Expected sampled bytes (default: 1958277360128)

The list stores allocated bytes (du -sB1 -x), so it is used as a declared budget
and a guard against a truncated list. Per-file completion is verified against the
real remote size taken from the source scan, and transfers resume from the exact
byte offset of an interrupted file.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --dry-run)          DL_DRY_RUN=true; shift ;;
        --preflight)        DL_PREFLIGHT_ONLY=true; shift ;;
        --yes|-y)           DL_ASSUME_YES=true; shift ;;
        --skip-existing|--resume) shift ;;
        --verify)           DL_CHECKSUM=true; shift ;;
        --allow-missing)    DL_MISSING_POLICY=skip; shift ;;
        --dest-root)
            (($# >= 2)) || die "--dest-root requires a path"
            DEST_ROOT=$2; shift 2 ;;
        --list)
            (($# >= 2)) || die "--list requires a file"
            LIST_FILE=$2; shift 2 ;;
        --jobs|-j)
            (($# >= 2)) || die "--jobs requires a positive integer"
            DL_JOBS=$2; shift 2 ;;
        --discovery-jobs)
            (($# >= 2)) || die "--discovery-jobs requires a positive integer"
            DL_DISCOVERY_JOBS=$2; shift 2 ;;
        --retries)
            (($# >= 2)) || die "--retries requires a non-negative integer"
            DL_RETRIES=$2; shift 2 ;;
        --progress)         DL_PROGRESS=on; shift ;;
        --no-progress)      DL_PROGRESS=off; shift ;;
        --help|-h)          usage; exit 0 ;;
        *)                  die "unknown argument: $1" ;;
    esac
done

[[ -f "$ENGINE" ]] || die "shared engine not found next to this script: $ENGINE"
[[ -f "$LIST_FILE" ]] || die "list not found: $LIST_FILE"

DL_JOBS="${DL_JOBS:-${PYQCD_DOWNLOAD_JOBS:-4}}"
DL_DISCOVERY_JOBS="${DL_DISCOVERY_JOBS:-${PYQCD_DOWNLOAD_DISCOVERY_JOBS:-3}}"
DL_RETRIES="${DL_RETRIES:-${PYQCD_DOWNLOAD_RETRIES:-6}}"
DL_PROGRESS="${DL_PROGRESS:-${PYQCD_DOWNLOAD_PROGRESS:-auto}}"
DL_SSH_HOST="${PYQCD_SSH_HOST:-222.200.137.16}"
DL_SSH_PORT="${PYQCD_SSH_PORT:-10023}"
DL_SSH_USER="${PYQCD_SSH_USER:-zhangxin}"
if [[ "${PYQCD_SOURCE_LOCAL:-0}" == 1 ]]; then DL_SOURCE_LOCAL=true; fi
DL_DEST_ROOT="$DEST_ROOT"
DL_TITLE="PyQCD sampled data download  ($(basename "$LIST_FILE"))"
DL_FAILED_LIST="${SCRIPT_DIR}/.download-state/v20261104-data4pyqcd-failed.txt"

# ---------------------------------------------------------------------------
# Parse and validate the sample list
# ---------------------------------------------------------------------------
ENTRY_REMOTE=()
ENTRY_LABEL=()
ENTRY_KIND=()
LIST_BYTES=()
TOTAL_BYTES=0
declare -A SEEN_PATHS=()

while IFS=$'\t' read -r size path; do
    [[ -n "$size" && -n "$path" ]] || continue
    [[ "$size" =~ ^[0-9]+$ ]] || die "invalid size in list: $size"
    [[ "$path" == /public/* || "$path" == /nexdata/* ]] || die "unsupported remote root: $path"
    [[ -z "${SEEN_PATHS[$path]:-}" ]] || die "duplicate list path: $path"
    SEEN_PATHS["$path"]=1
    LIST_BYTES+=("$size")
    TOTAL_BYTES=$(( TOTAL_BYTES + size ))
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

ENTRY_COUNT=${#LIST_BYTES[@]}
(( ENTRY_COUNT == EXPECTED_ENTRIES )) ||
    die "expected $EXPECTED_ENTRIES entries, found $ENTRY_COUNT"
(( TOTAL_BYTES == EXPECTED_TOTAL_BYTES )) ||
    die "expected $EXPECTED_TOTAL_BYTES bytes, found $TOTAL_BYTES"

# A path is treated as a directory unless its suffix is unambiguously a file;
# the source scan decides for real, this only labels the plan.
while IFS=$'\t' read -r size path; do
    case "$path" in
        *.lime|*.npy|*.npz|*.json|*.tsv|*.txt|*.pdf|*.png|*.csv|*.dat|*.h5|*.root)
            kind=file ;;
        *)  kind=dir ;;
    esac
    ENTRY_REMOTE+=("$path")
    ENTRY_LABEL+=("$(basename "$path")")
    ENTRY_KIND+=("$kind")
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

DL_BYTES_HINT="$TOTAL_BYTES"

printf '%s\n' '============================================================================='
printf ' PyQCD sampled data download plan\n'
printf '%s\n' '============================================================================='
printf ' List:        %s\n' "$LIST_FILE"
printf ' Entries:     %d\n' "$ENTRY_COUNT"
printf ' List bytes:  %d (%s allocated)\n' "$TOTAL_BYTES" "$(printf '%d.%02d TiB' \
    $(( TOTAL_BYTES / 1099511627776 )) $(( (TOTAL_BYTES % 1099511627776) * 100 / 1099511627776 )))"
printf ' Destination: %s\n' "$DEST_ROOT"
printf '%s\n' '============================================================================='

if $DL_DRY_RUN; then
    printf '\nPlanned sources:\n'
    for i in "${!ENTRY_REMOTE[@]}"; do
        printf '  [%04d] %-8s %12s  %s\n' "$((i + 1))" "${ENTRY_KIND[$i]}" \
            "${LIST_BYTES[$i]}" "${ENTRY_REMOTE[$i]}"
    done
fi

# shellcheck source=download_engine_v20261104.sh
source "$ENGINE"
dl_main
