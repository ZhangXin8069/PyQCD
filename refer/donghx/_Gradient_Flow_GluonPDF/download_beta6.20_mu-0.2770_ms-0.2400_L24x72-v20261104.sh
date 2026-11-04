#!/usr/bin/env bash
# Download the required core data for the 50-configuration L24x72 ensemble.
#
# This is the v20261104 rewrite of the v20261026 script. The plan is unchanged,
# the transport is not: instead of one rsync/scp process per directory it walks
# the source once, builds a file-level manifest, and streams independent files
# through a bounded worker pool with resumable, retrying transfers and a live
# aggregate progress panel.
#
# Usage:
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261104.sh --dry-run
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261104.sh --preflight
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261104.sh --source-local --preflight
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261104.sh --yes
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261104.sh --jobs 8 --yes
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261104.sh --jobs 8 --discovery-jobs 4 --yes
#   bash download_beta6.20_mu-0.2770_ms-0.2400_L24x72-v20261104.sh --dest-root /data/lqcd --yes
#
# Default data groups (for every configuration):
#   gauge configuration, eigensystem, and light perambulator.
#
# Optional auxiliary groups are never included unless explicitly requested:
#   --with-smear, --with-momsmear, --with-vvv-vdv, --with-2pt, --with-ope.

set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly SCRIPT_NAME=${BASH_SOURCE[0]##*/}
readonly ENGINE="${SCRIPT_DIR}/download_engine_v20261104.sh"

readonly ENSEMBLE="beta6.20_mu-0.2770_ms-0.2400_L24x72"
readonly EXPECTED_DEFAULT_CONF_COUNT=50
readonly CAPACITY_HINT_GB=600
readonly MEASURED_CORE_BYTES_PER_CONF=$((573444096 + 4777578496 + 13271052288))
readonly MEASURED_CORE_GB_50=$((MEASURED_CORE_BYTES_PER_CONF * 50 / 1000000000))

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
CONF_IDS_FILE=""
LIST_CONF_IDS=false
SUMMARY_ONLY=false
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
WITH_SMEAR=false
WITH_MOMSMEAR=false
WITH_VVV_VDV=false
WITH_2PT=false
WITH_OPE=false
WITH_ALL_AUXILIARY=false

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Options:
  --dry-run              Print the transfer plan without network access
  --preflight            Probe every remote source path, then exit
  --yes, -y              Do not ask for a final confirmation
  --skip-existing        Accepted for compatibility; completed files are always skipped
  --resume               Alias for --skip-existing
  --verify               Re-hash files whose size already matches before skipping
  --allow-missing        Download available entries instead of aborting on missing sources
  --no-preflight         Accepted for compatibility; the source scan always runs
  --source-local         Read source paths directly on this host; do not use SSH
  --dest-root PATH       Preserve remote absolute paths below PATH (default: /)
  --conf-ids-file FILE   Use one numeric configuration ID per line instead of defaults
  --list-conf-ids        Print the selected configuration IDs and exit
  --summary-only         With --dry-run, print the category summary only
  --jobs N, -j N         Concurrent file transfers (default: 4)
  --discovery-jobs N     Concurrent source-tree probes (default: 3)
  --retries N            Attempts per file before giving up (default: 6)
  --progress             Force the live progress panel
  --no-progress          Disable the live progress panel
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
  PYQCD_DOWNLOAD_JOBS    Default transfer concurrency (default: 4)
  PYQCD_DOWNLOAD_DISCOVERY_JOBS  Default probe concurrency (default: 3)
  PYQCD_DOWNLOAD_RETRIES Attempts per file (default: 6)
  PYQCD_DOWNLOAD_PROGRESS  Progress mode: auto, on, or off (default: auto)
  PYQCD_SOURCE_LOCAL     Set to 1 to use local source paths without SSH
  PUBLIC_BASE            Remote public root (default: /public)
  DEST_ROOT              Destination root (same as --dest-root)

Only gauge, eigensystem, and light perambulator are downloaded by default.
Transfers resume from the exact byte offset of an interrupted file, so rerunning
the same command after a network drop continues where it stopped.
A single SSH connection is reused for all streams; keepalive probes follow the
~/configure convention (ServerAliveInterval 30, ServerAliveCountMax 3).
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --dry-run)          DL_DRY_RUN=true; shift ;;
        --preflight)        DL_PREFLIGHT_ONLY=true; shift ;;
        --yes|-y)           DL_ASSUME_YES=true; shift ;;
        --skip-existing|--resume) shift ;;
        --verify)           DL_CHECKSUM=true; shift ;;
        --allow-missing)    DL_MISSING_POLICY=skip; shift ;;
        --no-preflight)     shift ;;
        --source-local)     DL_SOURCE_LOCAL=true; shift ;;
        --dest-root)
            (($# >= 2)) || die "--dest-root requires a path"
            DEST_ROOT=$2; shift 2 ;;
        --conf-ids-file)
            (($# >= 2)) || die "--conf-ids-file requires a file"
            CONF_IDS_FILE=$2; shift 2 ;;
        --list-conf-ids)    LIST_CONF_IDS=true; shift ;;
        --summary-only)     SUMMARY_ONLY=true; shift ;;
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
        --with-smear)       WITH_SMEAR=true; shift ;;
        --with-momsmear)    WITH_MOMSMEAR=true; shift ;;
        --with-vvv-vdv)     WITH_VVV_VDV=true; shift ;;
        --with-2pt)         WITH_2PT=true; shift ;;
        --with-ope)         WITH_OPE=true; shift ;;
        --with-all-auxiliary) WITH_ALL_AUXILIARY=true; shift ;;
        --help|-h)          usage; exit 0 ;;
        *)                  die "unknown argument: $1" ;;
    esac
done

[[ -f "$ENGINE" ]] || die "shared engine not found next to this script: $ENGINE"

if [[ "${PYQCD_SOURCE_LOCAL:-0}" == 1 ]]; then DL_SOURCE_LOCAL=true; fi

DL_JOBS="${DL_JOBS:-${PYQCD_DOWNLOAD_JOBS:-4}}"
DL_DISCOVERY_JOBS="${DL_DISCOVERY_JOBS:-${PYQCD_DOWNLOAD_DISCOVERY_JOBS:-3}}"
DL_RETRIES="${DL_RETRIES:-${PYQCD_DOWNLOAD_RETRIES:-6}}"
DL_PROGRESS="${DL_PROGRESS:-${PYQCD_DOWNLOAD_PROGRESS:-auto}}"
DL_SSH_HOST="${PYQCD_SSH_HOST:-222.200.137.16}"
DL_SSH_PORT="${PYQCD_SSH_PORT:-10023}"
DL_SSH_USER="${PYQCD_SSH_USER:-zhangxin}"
DL_DEST_ROOT="$DEST_ROOT"
DL_TITLE="L24x72 ensemble data download  (${ENSEMBLE})"
DL_FAILED_LIST="${SCRIPT_DIR}/.download-state/v20261104-beta6.20_L24x72-failed.txt"

if $WITH_ALL_AUXILIARY; then
    WITH_SMEAR=true; WITH_MOMSMEAR=true; WITH_VVV_VDV=true; WITH_2PT=true; WITH_OPE=true
fi

if [[ -n "$CONF_IDS_FILE" ]]; then
    [[ -e "$CONF_IDS_FILE" ]] || die "configuration list not found: $CONF_IDS_FILE"
    mapfile -t CONF_IDS < <(sed -e 's/#.*$//' -e 's/[[:space:]]//g' "$CONF_IDS_FILE" | awk 'NF')
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

# ---------------------------------------------------------------------------
# Build the source list
# ---------------------------------------------------------------------------
ENTRY_REMOTE=()
ENTRY_LABEL=()
ENTRY_KIND=()
declare -A CATEGORY_COUNTS=()

add_entry() {
    local category=$1 remote=$2 kind=$3
    [[ "$kind" == file || "$kind" == dir ]] || die "invalid entry kind: $kind"
    ENTRY_REMOTE+=("$remote")
    ENTRY_LABEL+=("$category")
    ENTRY_KIND+=("$kind")
    CATEGORY_COUNTS["$category"]=$(( ${CATEGORY_COUNTS["$category"]:-0} + 1 ))
}

for conf_id in "${CONF_IDS[@]}"; do
    add_entry gauge "${GAUGE_CONFIG_BASE}/${ENSEMBLE}_cfg_${conf_id}.lime" file
    add_entry eigensystem "${EIGENSYSTEM_BASE}/${conf_id}" dir
    add_entry perambulator-light "${PERAMBULATOR_BASE}/${conf_id}" dir

    if $WITH_SMEAR; then
        for smear_size in hpy_3D_1times hpy_3D_3times hpy_3D_5times hpy_4D_10times_new; do
            add_entry "hyp-smear:${smear_size}" \
                "${HYP_SMEAR_BASE}/${smear_size}/${ENSEMBLE}_cfg_${conf_id}.lime.contents" dir
        done
    fi

    if $WITH_MOMSMEAR; then
        add_entry momsmear-eigenvector "${MOMSMEAR_EIGEN_BASE}/${conf_id}_eigvec.npy" file
        add_entry momsmear-perambulator "${MOMSMEAR_PERAM_BASE}/${conf_id}" dir
    fi

    if $WITH_VVV_VDV; then
        add_entry vvv-vdv:mom0 "${VVV_BASE}/${ENSEMBLE}_momsmear0_Liu/${conf_id}" dir
        add_entry vvv-vdv:mom2 "${VVV_BASE}/${ENSEMBLE}_momsmear2/${conf_id}" dir
    fi

    if $WITH_2PT; then
        for twopt_channel in \
            momsmear-2x momsmear-2y momsmear-2z \
            momsmear2x momsmear2z \
            momsmear0_Cg5 momsmear0_Cg5g4; do
            add_entry "2pt:${twopt_channel}" "${TWOPT_RESULT_BASE}/${twopt_channel}/${conf_id}" dir
        done
    fi

    if $WITH_OPE; then
        for ope_dir in xdir ydir zdir; do
            add_entry "tmd-ope:${ope_dir}" "${TMD_OPE_BASE}/${ope_dir}/${conf_id}" dir
            add_entry "hyp-ope:${ope_dir}" "${HPY_OPE_BASE}/${ope_dir}/${conf_id}" dir
        done
    fi
done

AUXILIARY_GROUPS=()
$WITH_SMEAR && AUXILIARY_GROUPS+=(smear)
$WITH_MOMSMEAR && AUXILIARY_GROUPS+=(momsmear)
$WITH_VVV_VDV && AUXILIARY_GROUPS+=(vvv-vdv)
$WITH_2PT && AUXILIARY_GROUPS+=(2pt)
$WITH_OPE && AUXILIARY_GROUPS+=(ope)
((${#AUXILIARY_GROUPS[@]})) || AUXILIARY_GROUPS=(none)

# ---------------------------------------------------------------------------
# Report the plan, then hand over to the shared engine
# ---------------------------------------------------------------------------
printf '%s\n' '============================================================================='
printf ' L24x72 v20261104 data download plan\n'
printf '%s\n' '============================================================================='
printf ' Configurations: %d (%s ... %s)\n' \
    "${#CONF_IDS[@]}" "${CONF_IDS[0]}" "${CONF_IDS[${#CONF_IDS[@]}-1]}"
printf ' Auxiliary:      %s\n' "${AUXILIARY_GROUPS[*]}"
printf ' Source entries: %d\n' "${#ENTRY_REMOTE[@]}"
printf ' User estimate:  about %d GB\n' "$CAPACITY_HINT_GB"
printf ' Measured core:  at least %d GB for gauge + eigensystem + light perambulators\n' \
    "$MEASURED_CORE_GB_50"
printf '%s\n' '============================================================================='

printf '\nSource entries by category:\n'
for category in $(printf '%s\n' "${!CATEGORY_COUNTS[@]}" | sort); do
    printf '  %-28s %d\n' "$category" "${CATEGORY_COUNTS[$category]}"
done

if $DL_DRY_RUN && ! $SUMMARY_ONLY; then
    printf '\nPlanned sources:\n'
    for i in "${!ENTRY_REMOTE[@]}"; do
        printf '  [%04d] %-30s %s\n' "$((i + 1))" "${ENTRY_LABEL[$i]}" "${ENTRY_REMOTE[$i]}"
    done
fi

# shellcheck source=download_engine_v20261104.sh
source "$ENGINE"
dl_main
