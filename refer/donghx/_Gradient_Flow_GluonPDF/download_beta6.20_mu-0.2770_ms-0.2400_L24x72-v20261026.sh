#!/usr/bin/env bash
# Download the required core data for 50 L24x72 configurations.
#
# Usage:
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --dry-run
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --preflight
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --source-local --preflight
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --source-local --dest-root /data/lqcd --yes
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --yes
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --yes --skip-existing
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --jobs 8 --yes --skip-existing
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --jobs 8 --progress --yes
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh --dest-root /data/lqcd --yes
#
# Default configuration set:
#   The ten PyQCD configurations 6250,6450,...,8050, followed by 40 additional
#   configurations 8250,8450,...,16050.  The total is exactly 50 configurations.
#
# Default data groups (for every configuration):
#   gauge configuration, eigensystem, and light perambulator.
#
# Optional auxiliary groups are never included unless explicitly requested:
#   --with-smear, --with-momsmear, --with-vvv-vdv, --with-2pt, --with-ope.
#
# The user-provided capacity estimate is about 600 GB.  Measured conf6250
# inputs already give a much larger lower bound for gauge + eigensystem + light
# perambulator alone; the script prints both values to avoid a false capacity claim.

set -euo pipefail

readonly ENSEMBLE="beta6.20_mu-0.2770_ms-0.2400_L24x72"
readonly EXPECTED_DEFAULT_CONF_COUNT=50
readonly CAPACITY_HINT_GB=600
readonly MEASURED_CORE_BYTES_PER_CONF=$((573444096 + 4777578496 + 13271052288))
readonly MEASURED_CORE_GB_50=$((MEASURED_CORE_BYTES_PER_CONF * 50 / 1000000000))

readonly SSH_HOST="${PYQCD_SSH_HOST:-222.200.137.16}"
readonly SSH_PORT="${PYQCD_SSH_PORT:-10023}"
readonly SSH_USER="${PYQCD_SSH_USER:-zhangxin}"
readonly SSH_TARGET="${SSH_USER}@${SSH_HOST}"

readonly PREVIOUS_CONF_IDS=(
    6250 6450 6650 6850 7050
    7250 7450 7650 7850 8050
)

readonly PUBLIC_BASE="${PUBLIC_BASE:-/public}"
readonly DONGHX_BASE="${DONGHX_BASE:-${PUBLIC_BASE}/group/lqcd/donghx}"
readonly EIGENSYSTEM_BASE="${EIGENSYSTEM_BASE:-${PUBLIC_BASE}/group/lqcd/eigensystem/${ENSEMBLE}}"
readonly PERAMBULATOR_BASE="${PERAMBULATOR_BASE:-${PUBLIC_BASE}/group/lqcd/perambulators/${ENSEMBLE}/light}"
readonly GAUGE_CONFIG_BASE="${GAUGE_CONFIG_BASE:-${PUBLIC_BASE}/group/lqcd/configurations/CLOVER/${ENSEMBLE}}"

readonly HYP_SMEAR_BASE="${HYP_SMEAR_BASE:-${DONGHX_BASE}/Hpysmear_${ENSEMBLE}}"
readonly VVV_BASE="${VVV_BASE:-${DONGHX_BASE}/VVV}"
readonly TWOPT_RESULT_BASE="${TWOPT_RESULT_BASE:-${DONGHX_BASE}/2pt_Result/${ENSEMBLE}}"
readonly TMD_OPE_BASE="${TMD_OPE_BASE:-${DONGHX_BASE}/Ope_Gluon/TMD_Ope_Gluon/Result/${ENSEMBLE}}"
readonly HPY_OPE_BASE="${HPY_OPE_BASE:-${DONGHX_BASE}/Ope_Gluon/Result_hpy_4D_10times/L24x72}"

# The momentum-smeared distillation inputs used by the historical 2pt jobs.
readonly MOMSMEAR_EIGEN_BASE="${MOMSMEAR_EIGEN_BASE:-${PUBLIC_BASE}/home/zhongsy/Distillation/eigenvector_chromaHYP3D3_mom+2}"
readonly MOMSMEAR_PERAM_BASE="${MOMSMEAR_PERAM_BASE:-${PUBLIC_BASE}/home/zhongsy/Distillation/peramb_mom_smear_eigen/results_chromaHYP3D3_mom+2}"

DEST_ROOT="${DEST_ROOT:-/}"
DRY_RUN=false
SKIP_CONFIRM=false
SKIP_EXISTING=false
PARALLEL_JOBS=${PYQCD_DOWNLOAD_JOBS:-4}
PROGRESS_MODE=${PYQCD_DOWNLOAD_PROGRESS:-auto}
PREFLIGHT_ONLY=false
NO_PREFLIGHT=false
ALLOW_MISSING=false
SUMMARY_ONLY=false
LIST_CONF_IDS=false
CONF_IDS_FILE=""
SOURCE_LOCAL=false
WITH_SMEAR=false
WITH_MOMSMEAR=false
WITH_VVV_VDV=false
WITH_2PT=false
WITH_OPE=false
WITH_ALL_AUXILIARY=false

usage() {
    cat <<'EOF'
Usage: download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261026.sh [OPTIONS]

Options:
  --dry-run              Build and summarize the transfer plan without network access
  --preflight            Check every remote source path, then exit
  --yes, -y              Skip the final transfer confirmation
  --skip-existing        Skip files/directories whose remote and local manifests match
  --resume               Alias for --skip-existing
  --allow-missing        Download available entries instead of aborting on missing sources
  --no-preflight         Do not probe all remote paths before a live transfer
  --source-local         Read source paths directly on this host; do not use SSH
  --dest-root PATH       Preserve remote absolute paths below PATH (default: /)
  --conf-ids-file FILE   Use one numeric configuration ID per line instead of defaults
  --list-conf-ids        Print the selected configuration IDs and exit
  --summary-only         In dry-run mode, print category counts without every path
  --jobs N, -j N         Run up to N transfers concurrently (default: 4; 1 = sequential)
  --progress             Force byte progress and transfer-speed display
  --no-progress          Disable byte progress and transfer-speed display
  --with-smear           Also request four HYP-smear gauge trees per configuration
  --with-momsmear        Also request momentum-smeared eigenvectors/perambulators
  --with-vvv-vdv         Also request VVV/VdV caches for mom_smear=0 and -2
  --with-2pt             Also request the seven 2pt result trees per configuration
  --with-ope             Also request TMD OPE and HYP OPE in x/y/z directions
  --with-all-auxiliary   Enable every optional auxiliary group above
  --help, -h             Show this help

Environment:
  PYQCD_SSH_HOST         Remote host (default: 222.200.137.16)
  PYQCD_SSH_PORT         Remote SSH port (default: 10023)
  PYQCD_SSH_USER         Remote user (default: zhangxin)
  PYQCD_DOWNLOAD_JOBS    Default concurrency (default: 4)
  PYQCD_DOWNLOAD_PROGRESS  Progress mode: auto, on, or off (default: auto)
  PYQCD_SOURCE_LOCAL     Set to 1 to use local source paths without SSH
  PUBLIC_BASE            Remote public root (default: /public)
  DONGHX_BASE            Donghx data root
  EIGENSYSTEM_BASE       Eigensystem root
  PERAMBULATOR_BASE      Light perambulator root
  GAUGE_CONFIG_BASE      Gauge configuration root
  HYP_SMEAR_BASE         HYP-smear root
  VVV_BASE               VVV/VdV cache root
  TWOPT_RESULT_BASE      2pt result root
  TMD_OPE_BASE           TMD OPE result root
  HPY_OPE_BASE           HYP OPE result root
  MOMSMEAR_EIGEN_BASE    Momentum-smeared eigenvector root
  MOMSMEAR_PERAM_BASE    Momentum-smeared perambulator root
  DEST_ROOT              Destination root (same as --dest-root)

The default configuration list contains the previous ten configurations
6250,6450,...,8050 and extends the same 200-spacing pattern to 16050.
Only gauge, eigensystem, and light perambulator are downloaded by default.
Directory transfers prefer rsync for resumability and otherwise use an ssh tar stream.
With --source-local, source paths are read directly on this host and SSH is skipped.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

log_info()  { printf '[INFO]  %s\n' "$*"; }
log_warn()  { printf '[WARN]  %s\n' "$*" >&2; }
log_error() { printf '[ERROR] %s\n' "$*" >&2; }
log_done()  { printf '[DONE]  %s\n' "$*"; }
log_skip()  { printf '[SKIP]  %s\n' "$*"; }
log_step()  { printf '\n==== %s ====\n' "$*"; }

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --preflight)
            PREFLIGHT_ONLY=true
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
        --allow-missing)
            ALLOW_MISSING=true
            shift
            ;;
        --no-preflight)
            NO_PREFLIGHT=true
            shift
            ;;
        --source-local)
            SOURCE_LOCAL=true
            shift
            ;;
        --dest-root)
            (($# >= 2)) || die "--dest-root requires a path"
            DEST_ROOT=$2
            shift 2
            ;;
        --conf-ids-file)
            (($# >= 2)) || die "--conf-ids-file requires a file"
            CONF_IDS_FILE=$2
            shift 2
            ;;
        --list-conf-ids)
            LIST_CONF_IDS=true
            shift
            ;;
        --summary-only)
            SUMMARY_ONLY=true
            shift
            ;;
        --jobs|-j)
            (($# >= 2)) || die "--jobs requires a positive integer"
            PARALLEL_JOBS=$2
            shift 2
            ;;
        --progress)
            PROGRESS_MODE=on
            shift
            ;;
        --no-progress)
            PROGRESS_MODE=off
            shift
            ;;
        --with-smear)
            WITH_SMEAR=true
            shift
            ;;
        --with-momsmear)
            WITH_MOMSMEAR=true
            shift
            ;;
        --with-vvv-vdv)
            WITH_VVV_VDV=true
            shift
            ;;
        --with-2pt)
            WITH_2PT=true
            shift
            ;;
        --with-ope)
            WITH_OPE=true
            shift
            ;;
        --with-all-auxiliary)
            WITH_ALL_AUXILIARY=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [[ "${PYQCD_SOURCE_LOCAL:-0}" == 1 ]]; then
    SOURCE_LOCAL=true
fi

[[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]] ||
    die "--jobs requires a positive integer: $PARALLEL_JOBS"

case "$PROGRESS_MODE" in
    auto|on|off) ;;
    *) die "PYQCD_DOWNLOAD_PROGRESS must be auto, on, or off: $PROGRESS_MODE" ;;
esac

PROGRESS_ENABLED=false
PROGRESS_TTY=false
if [[ "$PROGRESS_MODE" == on ]] || { [[ "$PROGRESS_MODE" == auto ]] && [[ -t 2 ]]; }; then
    PROGRESS_ENABLED=true
    if [[ -t 2 ]]; then
        PROGRESS_TTY=true
    fi
fi

if ((PARALLEL_JOBS > 1)) && { ((BASH_VERSINFO[0] < 4)) || ((BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3)); }; then
    log_warn "bash >= 4.3 is required for parallel transfers; using --jobs 1"
    PARALLEL_JOBS=1
fi

if $WITH_ALL_AUXILIARY; then
    WITH_SMEAR=true
    WITH_MOMSMEAR=true
    WITH_VVV_VDV=true
    WITH_2PT=true
    WITH_OPE=true
fi

if $PREFLIGHT_ONLY && $DRY_RUN; then
    die "--preflight and --dry-run cannot be combined"
fi
if $PREFLIGHT_ONLY && $NO_PREFLIGHT; then
    die "--preflight and --no-preflight cannot be combined"
fi

DEST_ROOT=${DEST_ROOT%/}
[[ -n "$DEST_ROOT" ]] || DEST_ROOT=/

if [[ -n "$CONF_IDS_FILE" ]]; then
    [[ -e "$CONF_IDS_FILE" ]] || die "configuration list not found: $CONF_IDS_FILE"
    mapfile -t CONF_IDS < <(
        sed -e 's/#.*$//' -e 's/[[:space:]]//g' "$CONF_IDS_FILE" | awk 'NF'
    )
else
    CONF_IDS=("${PREVIOUS_CONF_IDS[@]}")
    while ((${#CONF_IDS[@]} < EXPECTED_DEFAULT_CONF_COUNT)); do
        last=${CONF_IDS[${#CONF_IDS[@]}-1]}
        CONF_IDS+=("$((last + 200))")
    done
fi

((${#CONF_IDS[@]} > 0)) || die "configuration list is empty"

declare -A SEEN_CONF=()
for conf_id in "${CONF_IDS[@]}"; do
    [[ "$conf_id" =~ ^[0-9]+$ ]] || die "invalid configuration ID: $conf_id"
    [[ -z "${SEEN_CONF[$conf_id]:-}" ]] || die "duplicate configuration ID: $conf_id"
    SEEN_CONF["$conf_id"]=1
done

if [[ -z "$CONF_IDS_FILE" ]] && ((${#CONF_IDS[@]} != EXPECTED_DEFAULT_CONF_COUNT)); then
    die "default configuration count is ${#CONF_IDS[@]}, expected ${EXPECTED_DEFAULT_CONF_COUNT}"
fi

if $LIST_CONF_IDS; then
    printf '%s\n' "${CONF_IDS[@]}"
    exit 0
fi

JOB_CATEGORIES=()
JOB_SOURCES=()
JOB_TYPES=()

add_job() {
    local category=$1 source=$2 type=$3
    [[ "$type" == file || "$type" == dir ]] || die "invalid job type: $type"
    JOB_CATEGORIES+=("$category")
    JOB_SOURCES+=("$source")
    JOB_TYPES+=("$type")
}

for conf_id in "${CONF_IDS[@]}"; do
    add_job gauge \
        "${GAUGE_CONFIG_BASE}/${ENSEMBLE}_cfg_${conf_id}.lime" file
    add_job eigensystem \
        "${EIGENSYSTEM_BASE}/${conf_id}" dir
    add_job perambulator-light \
        "${PERAMBULATOR_BASE}/${conf_id}" dir

    if $WITH_SMEAR; then
        for smear_size in hpy_3D_1times hpy_3D_3times hpy_3D_5times hpy_4D_10times_new; do
            add_job "hyp-smear:${smear_size}" \
                "${HYP_SMEAR_BASE}/${smear_size}/${ENSEMBLE}_cfg_${conf_id}.lime.contents" dir
        done
    fi

    if $WITH_MOMSMEAR; then
        add_job momsmear-eigenvector \
            "${MOMSMEAR_EIGEN_BASE}/${conf_id}_eigvec.npy" file
        add_job momsmear-perambulator \
            "${MOMSMEAR_PERAM_BASE}/${conf_id}" dir
    fi

    if $WITH_VVV_VDV; then
        add_job vvv-vdv:mom0 \
            "${VVV_BASE}/${ENSEMBLE}_momsmear0_Liu/${conf_id}" dir
        add_job vvv-vdv:mom2 \
            "${VVV_BASE}/${ENSEMBLE}_momsmear2/${conf_id}" dir
    fi

    if $WITH_2PT; then
        for twopt_channel in \
            momsmear-2x momsmear-2y momsmear-2z \
            momsmear2x momsmear2z \
            momsmear0_Cg5 momsmear0_Cg5g4; do
            add_job "2pt:${twopt_channel}" \
                "${TWOPT_RESULT_BASE}/${twopt_channel}/${conf_id}" dir
        done
    fi

    if $WITH_OPE; then
        for ope_dir in xdir ydir zdir; do
            add_job "tmd-ope:${ope_dir}" \
                "${TMD_OPE_BASE}/${ope_dir}/${conf_id}" dir
            add_job "hyp-ope:${ope_dir}" \
                "${HPY_OPE_BASE}/${ope_dir}/${conf_id}" dir
        done
    fi
done

declare -A CATEGORY_COUNTS=()
for category in "${JOB_CATEGORIES[@]}"; do
    CATEGORY_COUNTS["$category"]=$(( ${CATEGORY_COUNTS["$category"]:-0} + 1 ))
done

map_remote_to_local() {
    local remote_path=$1
    if [[ "$DEST_ROOT" == / ]]; then
        printf '%s\n' "$remote_path"
    else
        printf '%s%s\n' "$DEST_ROOT" "$remote_path"
    fi
}

quote_remote_path() {
    local path=$1
    [[ "$path" != *"'"* ]] || die "unsupported quote in remote path: $path"
    printf "'%s'" "$path"
}

path_bytes() {
    local path=$1
    if [[ -f "$path" ]]; then
        stat -c%s -- "$path" 2>/dev/null || printf '0\n'
    elif [[ -d "$path" ]]; then
        du -sb -- "$path" 2>/dev/null | awk 'NR == 1 { print $1 }'
    else
        printf '0\n'
    fi
}

format_bytes() {
    local bytes=${1:-0} unit value fraction suffix
    if ((bytes >= 1099511627776)); then
        unit=1099511627776
        suffix=TiB
    elif ((bytes >= 1073741824)); then
        unit=1073741824
        suffix=GiB
    elif ((bytes >= 1048576)); then
        unit=1048576
        suffix=MiB
    elif ((bytes >= 1024)); then
        unit=1024
        suffix=KiB
    else
        printf '%d B\n' "$bytes"
        return
    fi
    value=$((bytes / unit))
    fraction=$(((bytes % unit) * 100 / unit))
    printf '%d.%02d %s\n' "$value" "$fraction" "$suffix"
}

format_speed() {
    printf '%s/s\n' "$(format_bytes "${1:-0}")"
}

time_ms() {
    local milliseconds
    milliseconds=$(date +%s%3N 2>/dev/null)
    if [[ "$milliseconds" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$milliseconds"
    else
        printf '%s000\n' "$(date +%s)"
    fi
}

progress_bar_string() {
    local percent_tenths=${1:-0} width=${2:-28} filled empty
    ((percent_tenths < 0)) && percent_tenths=0
    ((percent_tenths > 1000)) && percent_tenths=1000
    filled=$((percent_tenths * width / 1000))
    empty=$((width - filled))
    printf -v filled '%*s' "$filled" ''
    filled=${filled// /#}
    printf -v empty '%*s' "$empty" ''
    empty=${empty// /-}
    printf '[%s%s]\n' "$filled" "$empty"
}

progress_current_bytes() {
    local index expected value total=0 local_path
    for index in "${TRANSFER_INDICES[@]}"; do
        if ((${JOB_FINAL_BYTES[$index]:--1} >= 0)); then
            value=${JOB_FINAL_BYTES[$index]}
        elif ((${JOB_STARTED[$index]:-0} == 1)); then
            local_path=$(map_remote_to_local "${JOB_SOURCES[$index]}")
            value=$(path_bytes "$local_path")
        else
            value=${JOB_BASE_BYTES[$index]:-0}
        fi
        expected=${REMOTE_SIZES[$index]:--1}
        if ((expected >= 0 && value > expected)); then
            value=$expected
        fi
        total=$((total + value))
    done
    printf '%d\n' "$total"
}

progress_clear_line() {
    if $PROGRESS_TTY && ${PROGRESS_LINE_ACTIVE:-false}; then
        printf '\r\033[K' >&2
        PROGRESS_LINE_ACTIVE=false
    fi
}

render_progress() {
    local final=${1:-false} force=${2:-false}
    $PROGRESS_ENABLED || return 0

    local now
    now=$(time_ms)
    if ! $final && ! $force && ((now - PROGRESS_LAST_SAMPLE_MS < 1000)); then
        return 0
    fi
    PROGRESS_LAST_SAMPLE_MS=$now

    local current transferred elapsed_ms speed percent_tenths percent_whole percent_fraction
    local bar total_display queue_count active_count line
    current=$(progress_current_bytes)
    elapsed_ms=$((now - PROGRESS_START_MS))
    ((elapsed_ms > 0)) || elapsed_ms=1
    transferred=$((current - PROGRESS_START_BYTES))
    ((transferred >= 0)) || transferred=0
    speed=$((transferred * 1000 / elapsed_ms))
    queue_count=$((${#TRANSFER_INDICES[@]} - TRANSFER_STARTED_COUNT))
    active_count=${#ACTIVE_PIDS[@]}

    if $PROGRESS_TOTAL_KNOWN; then
        if ((${PROGRESS_TOTAL_BYTES:-0} == 0)); then
            percent_tenths=1000
        else
            if ((current > PROGRESS_TOTAL_BYTES)); then
                current=$PROGRESS_TOTAL_BYTES
            fi
            percent_tenths=$((current * 1000 / PROGRESS_TOTAL_BYTES))
        fi
        percent_whole=$((percent_tenths / 10))
        percent_fraction=$((percent_tenths % 10))
        bar=$(progress_bar_string "$percent_tenths")
        total_display=$(format_bytes "$PROGRESS_TOTAL_BYTES")
        line=$(printf '[PROGRESS] %s %d.%d%% %s/%s %s active=%d queued=%d' \
            "$bar" "$percent_whole" "$percent_fraction" \
            "$(format_bytes "$current")" "$total_display" "$(format_speed "$speed")" \
            "$active_count" "$queue_count")
    else
        percent_tenths=-1
        line=$(printf '[PROGRESS] [------------------------------] --%% %s %s active=%d queued=%d' \
            "$(format_bytes "$current")" "$(format_speed "$speed")" \
            "$active_count" "$queue_count")
    fi

    if ! $PROGRESS_TTY && ! $final && ! $force; then
        if ((now - PROGRESS_LAST_RENDER_MS < 5000)); then
            if ((percent_tenths < 0 || percent_tenths < PROGRESS_LAST_PERCENT_TENTHS + 100)); then
                return 0
            fi
        fi
    fi

    if $PROGRESS_TTY; then
        printf '\r%s\033[K' "$line" >&2
        PROGRESS_LINE_ACTIVE=true
        if $final; then
            printf '\n' >&2
            PROGRESS_LINE_ACTIVE=false
        fi
    else
        printf '%s\n' "$line" >&2
    fi
    PROGRESS_LAST_RENDER_MS=$now
    PROGRESS_LAST_PERCENT_TENTHS=$percent_tenths
}

execute_transfer() {
    local index=$1 job_type=$2 local_path=$3 rc=0
    if [[ "$job_type" == file ]]; then
        transfer_file "$index" "$local_path" || rc=$?
    else
        transfer_dir "$index" "$local_path" || rc=$?
    fi
    return "$rc"
}

launch_transfer() {
    local index=$1 job_type=$2 remote_path=$3 local_path=$4
    local log_file="${JOB_LOG_DIR}/${index}.log"
    local status_file="${STATUS_DIR}/${index}.status"
    (
        set +e
        execute_transfer "$index" "$job_type" "$local_path" >"$log_file" 2>&1
        rc=$?
        if ((rc == 0)); then
            printf 'done\n' >"$status_file"
        else
            printf 'failed\n' >"$status_file"
        fi
        exit 0
    ) &
    LAST_TRANSFER_PID=$!
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
}

print_transfer_job() {
    local index=$1 remote_path=${JOB_SOURCES[$1]} local_path=$2
    progress_clear_line
    printf '\n[%d/%d] %s\n' "$((index + 1))" "${#JOB_SOURCES[@]}" "${JOB_CATEGORIES[$index]}"
    if $SOURCE_LOCAL; then
        printf '  Source: %s\n' "$remote_path"
    else
        printf '  Remote: %s:%s\n' "$SSH_TARGET" "$remote_path"
    fi
    printf '  Local:  %s\n' "$local_path"
}

launch_indexed_transfer() {
    local index=$1 job_type=${JOB_TYPES[$1]} remote_path=${JOB_SOURCES[$1]} local_path
    local_path=$(map_remote_to_local "$remote_path")
    print_transfer_job "$index" "$local_path"
    JOB_STARTED[$index]=1
    TRANSFER_STARTED_COUNT=$((TRANSFER_STARTED_COUNT + 1))
    launch_transfer "$index" "$job_type" "$remote_path" "$local_path"
    ACTIVE_PIDS+=("$LAST_TRANSFER_PID")
    ACTIVE_INDICES+=("$index")
}

reap_finished_transfers() {
    local position pid index status expected final_bytes
    local remaining_pids=()
    local remaining_indices=()

    for position in "${!ACTIVE_PIDS[@]}"; do
        pid=${ACTIVE_PIDS[$position]}
        index=${ACTIVE_INDICES[$position]}
        if kill -0 "$pid" 2>/dev/null; then
            remaining_pids+=("$pid")
            remaining_indices+=("$index")
            continue
        fi

        wait "$pid" || true
        RUNNING_JOBS=$((RUNNING_JOBS - 1))
        status=""
        if [[ -f "${STATUS_DIR}/${index}.status" ]]; then
            status=$(< "${STATUS_DIR}/${index}.status")
        fi

        progress_clear_line
        if [[ "$status" == done ]]; then
            expected=${REMOTE_SIZES[$index]:--1}
            if ((expected >= 0)); then
                final_bytes=$expected
            else
                final_bytes=$(path_bytes "$(map_remote_to_local "${JOB_SOURCES[$index]}")")
                [[ -n "$final_bytes" ]] || final_bytes=0
            fi
            JOB_FINAL_BYTES[$index]=$final_bytes
            if [[ "${JOB_TYPES[$index]}" == file ]]; then
                log_done "${JOB_SOURCES[$index]}"
            else
                log_done "${JOB_SOURCES[$index]}/"
            fi
            COUNT_DONE=$((COUNT_DONE + 1))
        else
            if [[ "${JOB_TYPES[$index]}" == file ]]; then
                log_error "failed: ${JOB_SOURCES[$index]}"
            else
                log_error "failed: ${JOB_SOURCES[$index]}/"
            fi
            COUNT_FAILED=$((COUNT_FAILED + 1))
            FAILED_LOG_INDICES+=("$index")
        fi
    done

    ACTIVE_PIDS=()
    ACTIVE_INDICES=()
    if ((${#remaining_pids[@]} > 0)); then
        ACTIVE_PIDS=("${remaining_pids[@]}")
        ACTIVE_INDICES=("${remaining_indices[@]}")
    fi
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

init_ssh() {
    trap cleanup_ssh EXIT

    if $SOURCE_LOCAL; then
        log_info "Using source paths directly on this host; SSH setup skipped"
        return 0
    fi

    command -v ssh >/dev/null 2>&1 || die "ssh is not installed"

    SSH_CONTROL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pyqcd-download.XXXXXX")
    SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/sock-%r@%h:%p"
    # Keep SSH interactive because the source host requires password plus dynamic verification code.
    SSH_COMMON_OPTS=(
        -p "$SSH_PORT"
        -o ControlMaster=auto
        -o "ControlPath=${SSH_CONTROL_PATH}"
        -o ControlPersist=600
        -o StrictHostKeyChecking=yes
        -o ConnectTimeout=15
        -o ConnectionAttempts=1
        -o ServerAliveInterval=15
        -o ServerAliveCountMax=4
    )
    SCP_OPTS=(
        -P "$SSH_PORT"
        -o ControlMaster=auto
        -o "ControlPath=${SSH_CONTROL_PATH}"
        -o ControlPersist=600
        -o StrictHostKeyChecking=yes
        -o ConnectTimeout=15
        -o ConnectionAttempts=1
        -o ServerAliveInterval=15
        -o ServerAliveCountMax=4
    )

    log_info "Establishing SSH master connection to ${SSH_TARGET}:${SSH_PORT} ..."
    ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" true ||
        die "cannot connect to ${SSH_TARGET}:${SSH_PORT}; check password/verification-code authentication and known_hosts, or use --source-local"
    log_done "SSH master connection established"
}

build_remote_probe_command() {
    local start=$1 end=$2 command="export LC_ALL=C LANG=C; set +e; "
    local i path quoted
    for ((i = start; i < end; i++)); do
        path=${JOB_SOURCES[$i]}
        quoted=$(quote_remote_path "$path")
        command+="if [ -f $quoted ]; then size=\$(stat -c%s $quoted 2>/dev/null); printf 'F\\t%s\\t%s\\n' $i \"\${size:-0}\"; "
        if $PROGRESS_ENABLED; then
            command+="elif [ -d $quoted ]; then size=\$(du -sb $quoted 2>/dev/null | cut -f1); [ -n \"\$size\" ] || size=-1; printf 'D\\t%s\\t%s\\n' $i \"\$size\"; "
        else
            command+="elif [ -d $quoted ]; then printf 'D\\t%s\\t0\\n' $i; "
        fi
        command+="else printf 'M\\t%s\\t-1\\n' $i; fi; "
    done
    printf '%s\n' "$command"
}

REMOTE_STATES=()
REMOTE_SIZES=()
MISSING_JOBS=()
PREFLIGHT_FAILED=false

run_preflight() {
    if $SOURCE_LOCAL; then
        log_step "Local source preflight"
    else
        log_step "Remote source preflight"
    fi
    local batch_size=64 start end command output state index size expected
    local missing_count=0 wrong_type_count=0

    REMOTE_STATES=()
    REMOTE_SIZES=()
    MISSING_JOBS=()
    for i in "${!JOB_SOURCES[@]}"; do
        REMOTE_STATES[$i]="?"
        REMOTE_SIZES[$i]=-1
    done

    if $SOURCE_LOCAL; then
        for i in "${!JOB_SOURCES[@]}"; do
            path=${JOB_SOURCES[$i]}
            expected=${JOB_TYPES[$i]}
            if [[ -f "$path" ]]; then
                state=F
                size=$(stat -c%s "$path" 2>/dev/null || printf '0')
            elif [[ -d "$path" ]]; then
                state=D
                if $PROGRESS_ENABLED; then
                    size=$(du -sb -- "$path" 2>/dev/null | awk 'NR == 1 { print $1 }')
                    [[ -n "$size" ]] || size=-1
                else
                    size=0
                fi
            else
                state=M
                size=-1
            fi
            REMOTE_STATES[$i]=$state
            REMOTE_SIZES[$i]=$size

            if [[ "$state" == M ]]; then
                log_error "missing: ${JOB_CATEGORIES[$i]} $path"
                MISSING_JOBS+=("$i")
                missing_count=$((missing_count + 1))
            elif [[ "$state" == F && "$expected" != file ]] ||
                 [[ "$state" == D && "$expected" != dir ]]; then
                log_error "wrong source type: ${JOB_CATEGORIES[$i]} $path"
                MISSING_JOBS+=("$i")
                wrong_type_count=$((wrong_type_count + 1))
            fi
        done
    else
        for ((start = 0; start < ${#JOB_SOURCES[@]}; start += batch_size)); do
            end=$((start + batch_size))
            ((end <= ${#JOB_SOURCES[@]})) || end=${#JOB_SOURCES[@]}
            command=$(build_remote_probe_command "$start" "$end")
            output=$(ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" "$command") ||
                die "remote preflight command failed for entries $((start + 1))..$end"

            while IFS=$'\t' read -r state index size; do
                [[ -n "$state" && -n "$index" ]] || continue
                expected=${JOB_TYPES[$index]}
                REMOTE_STATES[$index]=$state
                REMOTE_SIZES[$index]=${size:--1}

                if [[ "$state" == M ]]; then
                    log_error "missing: ${JOB_CATEGORIES[$index]} ${JOB_SOURCES[$index]}"
                    MISSING_JOBS+=("$index")
                    missing_count=$((missing_count + 1))
                elif [[ "$state" == F && "$expected" != file ]] ||
                     [[ "$state" == D && "$expected" != dir ]]; then
                    log_error "wrong source type: ${JOB_CATEGORIES[$index]} ${JOB_SOURCES[$index]}"
                    MISSING_JOBS+=("$index")
                    wrong_type_count=$((wrong_type_count + 1))
                fi
            done <<< "$output"
        done
    fi

    for i in "${!JOB_SOURCES[@]}"; do
        if [[ "${REMOTE_STATES[$i]:-?}" == '?' ]]; then
            log_error "source was not inspected: ${JOB_SOURCES[$i]}"
            MISSING_JOBS+=("$i")
            missing_count=$((missing_count + 1))
        fi
    done

    if ((${#MISSING_JOBS[@]} > 0)); then
        log_error "preflight found ${#MISSING_JOBS[@]} unavailable or mismatched source(s)"
        log_error "missing=${missing_count}, wrong_type=${wrong_type_count}"
        return 1
    fi

    log_done "preflight passed for ${#JOB_SOURCES[@]} source paths"
}

file_already_downloaded() {
    local index=$1 relocal=$2 remote_size local_size
    [[ -f "$relocal" ]] || return 1
    remote_size=${REMOTE_SIZES[$index]:-}
    if [[ ! "$remote_size" =~ ^[0-9]+$ ]]; then
        if $SOURCE_LOCAL; then
            remote_size=$(stat -c%s "${JOB_SOURCES[$index]}" 2>/dev/null) || return 1
        else
            remote_size=$(ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" \
                "stat -c%s $(quote_remote_path "${JOB_SOURCES[$index]}")" 2>/dev/null) || return 1
        fi
    fi
    local_size=$(stat -c%s "$relocal" 2>/dev/null) || return 1
    [[ "$remote_size" == "$local_size" ]]
}

dir_already_downloaded() {
    local index=$1 reldir=$2 remote_manifest remote_size rel_path local_file local_size
    [[ -d "$reldir" ]] || return 1
    if $SOURCE_LOCAL; then
        remote_manifest=$(cd "${JOB_SOURCES[$index]}" 2>/dev/null &&
            find . -type f -printf '%s\t%P\n' 2>/dev/null) || return 1
    else
        remote_manifest=$(ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" \
            "cd $(quote_remote_path "${JOB_SOURCES[$index]}") 2>/dev/null && find . -type f -printf '%s\\t%P\\n'" 2>/dev/null) || return 1
    fi
    [[ -n "$remote_manifest" ]] || return 1

    while IFS=$'\t' read -r remote_size rel_path; do
        [[ -n "$remote_size" && -n "$rel_path" ]] || continue
        local_file="${reldir}/${rel_path}"
        [[ -f "$local_file" ]] || return 1
        local_size=$(stat -c%s "$local_file" 2>/dev/null) || return 1
        [[ "$remote_size" == "$local_size" ]] || return 1
    done <<< "$remote_manifest"
}

transfer_file() {
    local index=$1 remote_path=${JOB_SOURCES[$1]} local_path=$2
    if $SOURCE_LOCAL; then
        if [[ "$remote_path" == "$local_path" ]]; then
            log_info "source already at destination: $remote_path"
            return 0
        fi
        mkdir -p "$(dirname "$local_path")"
        if [[ "$TRANSFER_TOOL" == rsync ]]; then
            rsync -avP --inplace --partial "$remote_path" "$local_path"
        else
            cp -a -- "$remote_path" "$local_path"
        fi
        return
    fi

    mkdir -p "$(dirname "$local_path")"
    if [[ "$TRANSFER_TOOL" == rsync ]]; then
        rsync -avP --inplace --partial -e "ssh ${SSH_COMMON_OPTS[*]}" \
            "${SSH_TARGET}:${remote_path}" "$local_path"
    else
        scp "${SCP_OPTS[@]}" "${SSH_TARGET}:${remote_path}" "$local_path"
    fi
}

transfer_dir() {
    local index=$1 remote_path=${JOB_SOURCES[$1]} local_path=$2
    if $SOURCE_LOCAL; then
        if [[ "$remote_path" == "$local_path" ]]; then
            log_info "source already at destination: $remote_path"
            return 0
        fi
        if [[ "$local_path/" == "$remote_path/"* ]]; then
            log_error "destination is inside local source: $local_path"
            return 1
        fi
        mkdir -p "$local_path"
        if [[ "$TRANSFER_TOOL" == rsync ]]; then
            rsync -avP --partial --partial-dir=.rsync-partial \
                "${remote_path}/" "${local_path}/"
        else
            cp -a -- "${remote_path}/." "${local_path}/"
        fi
        return
    fi

    mkdir -p "$local_path"
    if [[ "$TRANSFER_TOOL" == rsync ]]; then
        rsync -avP --partial --partial-dir=.rsync-partial -e "ssh ${SSH_COMMON_OPTS[*]}" \
            "${SSH_TARGET}:${remote_path}/" "$local_path/"
    else
        command -v tar >/dev/null 2>&1 || die "tar is required for the scp directory fallback"
        ssh "${SSH_COMMON_OPTS[@]}" "$SSH_TARGET" \
            "export LC_ALL=C LANG=C; tar -C $(quote_remote_path "$remote_path") -cf - ." |
            tar -C "$local_path" -xf -
    fi
}

confirm_transfer_plan() {
    local answer
    $SKIP_CONFIRM && return 0
    printf 'Proceed with %d transfer entries? [y/N] ' "${#JOB_SOURCES[@]}"
    read -r answer
    [[ "$answer" =~ ^[Yy]$ ]]
}

TRANSFER_TOOL=none
if command -v rsync >/dev/null 2>&1; then
    TRANSFER_TOOL=rsync
elif $SOURCE_LOCAL && command -v cp >/dev/null 2>&1; then
    TRANSFER_TOOL=cp
elif command -v scp >/dev/null 2>&1; then
    TRANSFER_TOOL=scp+tar
elif ! $DRY_RUN; then
    die "neither rsync nor scp is installed"
fi

# Fail early: a missing tar inside a background transfer would only kill one job.
if [[ "$TRANSFER_TOOL" == scp+tar ]] && ! $DRY_RUN; then
    command -v tar >/dev/null 2>&1 || die "tar is required for the scp directory fallback"
fi

AUXILIARY_GROUPS=()
if $WITH_SMEAR; then AUXILIARY_GROUPS+=(smear); fi
if $WITH_MOMSMEAR; then AUXILIARY_GROUPS+=(momsmear); fi
if $WITH_VVV_VDV; then AUXILIARY_GROUPS+=(vvv-vdv); fi
if $WITH_2PT; then AUXILIARY_GROUPS+=(2pt); fi
if $WITH_OPE; then AUXILIARY_GROUPS+=(ope); fi
if ((${#AUXILIARY_GROUPS[@]} == 0)); then
    AUXILIARY_GROUPS=(none)
fi

printf '%s\n' '============================================================================='
printf ' L24x72 v20261026 data download\n'
printf '%s\n' '============================================================================='
printf ' Configurations: %d (%s ... %s)\n' \
    "${#CONF_IDS[@]}" "${CONF_IDS[0]}" "${CONF_IDS[${#CONF_IDS[@]}-1]}"
printf ' Auxiliary:      %s\n' "${AUXILIARY_GROUPS[*]}"
printf ' Transfer jobs:  %d\n' "${#JOB_SOURCES[@]}"
printf ' User estimate:  about %d GB\n' "$CAPACITY_HINT_GB"
printf ' Measured core:  at least %d GB for gauge + eigensystem + light perambulators\n' \
    "$MEASURED_CORE_GB_50"
printf ' Destination:    %s\n' "$DEST_ROOT"
if $SOURCE_LOCAL; then
    printf ' Source:         local filesystem\n'
else
    printf ' SSH:            %s:%s (%s)\n' "$SSH_HOST" "$SSH_PORT" "$SSH_USER"
fi
printf ' Tool:           %s\n' "$TRANSFER_TOOL"
printf ' Jobs:           %d\n' "$PARALLEL_JOBS"
printf ' Progress:       %s\n' "$($PROGRESS_ENABLED && printf 'bytes + speed' || printf disabled)"
printf ' Mode:           %s%s\n' \
    "$($DRY_RUN && printf DRY-RUN || printf LIVE)" \
    "$($SKIP_EXISTING && printf ' + skip-existing' || true)"
printf '%s\n' '============================================================================='

printf '\nTransfer entries by category:\n'
for category in $(printf '%s\n' "${!CATEGORY_COUNTS[@]}" | sort); do
    printf '  %-28s %d\n' "$category" "${CATEGORY_COUNTS[$category]}"
done

if $DRY_RUN; then
    if ! $SUMMARY_ONLY; then
        printf '\nPlanned sources:\n'
        for i in "${!JOB_SOURCES[@]}"; do
            printf '  [%04d] %-28s %s\n' "$((i + 1))" "${JOB_CATEGORIES[$i]}" "${JOB_SOURCES[$i]}"
        done
    fi
    log_info "Dry run complete; no SSH connection or transfer was attempted."
    exit 0
fi

init_ssh

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pyqcd-download-work.XXXXXX")
STATUS_DIR="${WORK_DIR}/status"
JOB_LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$STATUS_DIR" "$JOB_LOG_DIR"
printf ' Concurrent job logs: %s\n' "$JOB_LOG_DIR"

if ! $NO_PREFLIGHT; then
    if ! run_preflight; then
        PREFLIGHT_FAILED=true
        if $ALLOW_MISSING; then
            log_warn "continuing because --allow-missing was requested"
        else
            die "source inventory is incomplete; no files were transferred"
        fi
    fi
else
    log_warn "remote preflight disabled"
fi

if $PREFLIGHT_ONLY; then
    if $PREFLIGHT_FAILED; then
        die "preflight-only run found an incomplete source inventory"
    fi
    log_done "preflight-only run complete"
    exit 0
fi

if ! confirm_transfer_plan; then
    log_warn "transfer cancelled by user"
    exit 0
fi

declare -A SKIP_SET=()
for index in "${MISSING_JOBS[@]:-}"; do
    [[ -n "$index" ]] && SKIP_SET["$index"]=1
done

COUNT_SKIPPED=0
COUNT_DONE=0
COUNT_FAILED=0
RUNNING_JOBS=0
KEEP_WORK_DIR=false
FAILED_LOG_INDICES=()
TRANSFER_INDICES=()
ACTIVE_PIDS=()
ACTIVE_INDICES=()
TRANSFER_STARTED_COUNT=0

for i in "${!JOB_SOURCES[@]}"; do
    if [[ -n "${SKIP_SET[$i]:-}" ]]; then
        log_skip "missing source: ${JOB_CATEGORIES[$i]} ${JOB_SOURCES[$i]}"
        COUNT_SKIPPED=$((COUNT_SKIPPED + 1))
        continue
    fi

    job_type=${JOB_TYPES[$i]}
    local_path=$(map_remote_to_local "${JOB_SOURCES[$i]}")

    if $SKIP_EXISTING; then
        if [[ "$job_type" == file ]] && file_already_downloaded "$i" "$local_path"; then
            log_skip "size match: ${JOB_SOURCES[$i]}"
            COUNT_SKIPPED=$((COUNT_SKIPPED + 1))
            continue
        elif [[ "$job_type" == dir ]] && dir_already_downloaded "$i" "$local_path"; then
            log_skip "remote/local manifest match: ${JOB_SOURCES[$i]}"
            COUNT_SKIPPED=$((COUNT_SKIPPED + 1))
            continue
        fi
    fi
    TRANSFER_INDICES+=("$i")
done

for i in "${TRANSFER_INDICES[@]}"; do
    JOB_BASE_BYTES[$i]=0
    JOB_FINAL_BYTES[$i]=-1
    JOB_STARTED[$i]=0
done

if $PROGRESS_ENABLED && ((${#TRANSFER_INDICES[@]} > 0)); then
    command -v du >/dev/null 2>&1 || die "du is required for byte progress display"
    PROGRESS_TOTAL_BYTES=0
    PROGRESS_TOTAL_KNOWN=true
    PROGRESS_START_BYTES=0
    for i in "${TRANSFER_INDICES[@]}"; do
        expected=${REMOTE_SIZES[$i]:--1}
        local_path=$(map_remote_to_local "${JOB_SOURCES[$i]}")
        base_bytes=$(path_bytes "$local_path")
        [[ -n "$base_bytes" ]] || base_bytes=0
        if ((expected >= 0)); then
            ((base_bytes <= expected)) || base_bytes=$expected
            PROGRESS_TOTAL_BYTES=$((PROGRESS_TOTAL_BYTES + expected))
        else
            PROGRESS_TOTAL_KNOWN=false
        fi
        JOB_BASE_BYTES[$i]=$base_bytes
        PROGRESS_START_BYTES=$((PROGRESS_START_BYTES + base_bytes))
    done
    PROGRESS_START_MS=$(time_ms)
    PROGRESS_LAST_SAMPLE_MS=$((PROGRESS_START_MS - 1000))
    PROGRESS_LAST_RENDER_MS=$((PROGRESS_START_MS - 5000))
    PROGRESS_LAST_PERCENT_TENTHS=-100
    PROGRESS_LINE_ACTIVE=false
    render_progress false true
fi

TRANSFER_CURSOR=0
while ((TRANSFER_CURSOR < ${#TRANSFER_INDICES[@]} || RUNNING_JOBS > 0)); do
    while ((RUNNING_JOBS < PARALLEL_JOBS && TRANSFER_CURSOR < ${#TRANSFER_INDICES[@]})); do
        launch_indexed_transfer "${TRANSFER_INDICES[$TRANSFER_CURSOR]}"
        TRANSFER_CURSOR=$((TRANSFER_CURSOR + 1))
    done
    render_progress
    if ((RUNNING_JOBS > 0)); then
        sleep 0.2
        reap_finished_transfers
    fi
done

if $PROGRESS_ENABLED && ((${#TRANSFER_INDICES[@]} > 0)); then
    render_progress true true
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
        printf -- '--- %s (job log: %s) ---\n' "${JOB_SOURCES[$i]}" "${JOB_LOG_DIR}/${i}.log" >&2
        tail -n 20 "${JOB_LOG_DIR}/${i}.log" >&2 || true
    done
    printf 'Full job logs kept in: %s\n' "$JOB_LOG_DIR" >&2
fi

if ((COUNT_FAILED > 0)); then
    log_error "some transfers failed; rerun with --skip-existing after fixing the cause"
    exit 1
fi

log_done "requested data download completed"
