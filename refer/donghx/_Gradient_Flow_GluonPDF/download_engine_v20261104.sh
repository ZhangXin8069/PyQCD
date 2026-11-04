#!/usr/bin/env bash
# Shared transfer engine for the v20261104 download front-ends.
#
# Why this exists: the source host authenticates with password + a dynamic
# verification code, so every connection is expensive and cannot be replayed
# unattended. The engine therefore opens exactly one SSH master connection and
# multiplexes every transfer channel over it, which is also what makes the
# keepalive settings from ~/configure (30 s interval, 3 misses) meaningful.
#
# Stability
#   * one ControlMaster socket, ControlPersist 3600, TCPKeepAlive on
#   * per-file resume from the exact byte offset already on disk
#   * exponential-backoff retries that never discard received bytes
#   * socket liveness check before each retry, reconnect only when needed
#
# Throughput
#   * one walk per source entry builds a file-level manifest
#   * a bounded worker pool streams independent files in parallel
#   * largest-file-first ordering keeps the pool busy to the end
#   * no rsync required: only ssh, tail, stat, find
#
# Visibility
#   * live panel: aggregate bar, aggregate rate, ETA, per-worker rows
#   * one-line [PROGRESS] reports when stderr is not a terminal
#
# A front-end fills ENTRY_REMOTE / ENTRY_LABEL / ENTRY_KIND and calls dl_main.
# Requires bash >= 4.0 and GNU coreutils; the remote needs GNU find and tail.

DL_ENGINE_VERSION="20261104.1"

DL_DEST_ROOT="${DL_DEST_ROOT:-/}"
DL_JOBS="${DL_JOBS:-4}"
DL_RETRIES="${DL_RETRIES:-6}"
DL_RETRY_BASE="${DL_RETRY_BASE:-3}"
DL_RETRY_MAX="${DL_RETRY_MAX:-60}"
DL_PROGRESS="${DL_PROGRESS:-auto}"
DL_DISCOVERY_JOBS="${DL_DISCOVERY_JOBS:-3}"
DL_MISSING_POLICY="${DL_MISSING_POLICY:-fail}"
DL_CHECKSUM="${DL_CHECKSUM:-false}"
DL_TITLE="${DL_TITLE:-download}"
DL_MODE_LABEL="${DL_MODE_LABEL:-LIVE}"
DL_PLAIN_INTERVAL_MS="${DL_PLAIN_INTERVAL_MS:-30000}"

DL_DRY_RUN="${DL_DRY_RUN:-false}"
DL_PREFLIGHT_ONLY="${DL_PREFLIGHT_ONLY:-false}"
DL_ASSUME_YES="${DL_ASSUME_YES:-false}"
DL_SOURCE_LOCAL="${DL_SOURCE_LOCAL:-false}"
DL_BYTES_HINT="${DL_BYTES_HINT:-}"

DL_SSH_HOST="${DL_SSH_HOST:-222.200.137.16}"
DL_SSH_PORT="${DL_SSH_PORT:-10023}"
DL_SSH_USER="${DL_SSH_USER:-zhangxin}"
DL_SSH_TARGET="${DL_SSH_USER}@${DL_SSH_HOST}"
DL_SSH_KEEPALIVE_INTERVAL="${DL_SSH_KEEPALIVE_INTERVAL:-30}"
DL_SSH_KEEPALIVE_COUNT="${DL_SSH_KEEPALIVE_COUNT:-3}"

if [[ -t 2 && -z "${NO_COLOR:-}" ]]; then
    DL_C_RESET=$'\033[0m'; DL_C_DIM=$'\033[2m'; DL_C_BOLD=$'\033[1m'
    DL_C_RED=$'\033[31m'; DL_C_GREEN=$'\033[32m'; DL_C_YELLOW=$'\033[33m'
    DL_C_BLUE=$'\033[34m'; DL_C_CYAN=$'\033[36m'; DL_C_GREY=$'\033[90m'
else
    DL_C_RESET=''; DL_C_DIM=''; DL_C_BOLD=''
    DL_C_RED=''; DL_C_GREEN=''; DL_C_YELLOW=''
    DL_C_BLUE=''; DL_C_CYAN=''; DL_C_GREY=''
fi

# ---------------------------------------------------------------------------
# Message helpers. dl_ui_log is defined with the panel code below; these
# wrappers only forward to it.
# ---------------------------------------------------------------------------
dl_info()  { dl_ui_log "${DL_C_BLUE}[INFO]${DL_C_RESET}  $*"; }
dl_warn()  { dl_ui_log "${DL_C_YELLOW}[WARN]${DL_C_RESET}  $*"; }
dl_error() { dl_ui_log "${DL_C_RED}[ERROR]${DL_C_RESET} $*"; }
dl_done()  { dl_ui_log "${DL_C_GREEN}[DONE]${DL_C_RESET}  $*"; }
dl_skip()  { dl_ui_log "${DL_C_GREY}[SKIP]${DL_C_RESET}  $*"; }

dl_die() {
    dl_ui_log "${DL_C_RED}[ERROR]${DL_C_RESET} $*"
    exit 1
}

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
dl_human() {
    local bytes=${1:-0}
    if (( bytes < 0 )); then bytes=0; fi
    if (( bytes >= 1099511627776 )); then
        printf '%d.%02d TiB' $(( bytes / 1099511627776 )) $(( (bytes % 1099511627776) * 100 / 1099511627776 ))
    elif (( bytes >= 1073741824 )); then
        printf '%d.%02d GiB' $(( bytes / 1073741824 )) $(( (bytes % 1073741824) * 100 / 1073741824 ))
    elif (( bytes >= 1048576 )); then
        printf '%d.%01d MiB' $(( bytes / 1048576 )) $(( (bytes % 1048576) * 10 / 1048576 ))
    elif (( bytes >= 1024 )); then
        printf '%d KiB' $(( bytes / 1024 ))
    else
        printf '%d B' "$bytes"
    fi
}

dl_rate() { printf '%s/s' "$(dl_human "${1:-0}")"; }

dl_eta() {
    local secs=${1:--1}
    if (( secs < 0 )); then
        printf -- '--'
    elif (( secs >= 86400 )); then
        printf '%dd%02dh' $(( secs / 86400 )) $(( secs % 86400 / 3600 ))
    elif (( secs >= 3600 )); then
        printf '%dh%02dm' $(( secs / 3600 )) $(( secs % 3600 / 60 ))
    elif (( secs >= 60 )); then
        printf '%dm%02ds' $(( secs / 60 )) $(( secs % 60 ))
    else
        printf '%ds' "$secs"
    fi
}

dl_bar() {
    local permille=${1:-0} width=${2:-30} filled empty
    if (( permille < 0 )); then permille=0; fi
    if (( permille > 1000 )); then permille=1000; fi
    filled=$(( permille * width / 1000 ))
    empty=$(( width - filled ))
    printf -v filled '%*s' "$filled" ''; filled=${filled// /#}
    printf -v empty '%*s' "$empty" ''; empty=${empty// /-}
    printf '%s%s' "$filled" "$empty"
}

dl_now_ms() {
    local ms
    ms=$(date +%s%3N 2>/dev/null)
    if [[ "$ms" =~ ^[0-9]+$ ]]; then
        printf '%s' "$ms"
    else
        printf '%s000' "$(date +%s)"
    fi
}

# Reject quotes instead of building nested escaping; every path here is a plain
# cluster path, and a hard error is better than a subtly wrong remote command.
dl_sq() {
    local value=$1
    if [[ "$value" == *"'"* ]]; then
        dl_die "unsupported quote in remote path: $value"
    fi
    printf "'%s'" "$value"
}

dl_fit() {
    local text=$1 width=$2
    if (( width <= 3 )); then
        printf '%s' "${text:0:width}"
    elif (( ${#text} <= width )); then
        printf '%s' "$text"
    else
        printf '%s...' "${text:0:width-3}"
    fi
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DL_UI_TTY=false
DL_UI_PANEL_LINES=0
DL_UI_COLS=100

dl_ui_init() {
    DL_UI_TTY=false
    if [[ "$DL_PROGRESS" != off && -t 2 ]]; then
        DL_UI_TTY=true
        local cols
        cols=$(tput cols 2>/dev/null || true)
        if [[ "$cols" =~ ^[0-9]+$ ]]; then DL_UI_COLS=$cols; fi
        if (( DL_UI_COLS > 150 )); then DL_UI_COLS=150; fi
        if (( DL_UI_COLS < 72 )); then DL_UI_COLS=72; fi
        printf '\033[?25l' >&2
    fi
    return 0
}

dl_ui_fini() {
    if ! $DL_UI_TTY; then return 0; fi
    if (( DL_UI_PANEL_LINES > 0 )); then
        printf '\033[%dA\033[J' "$DL_UI_PANEL_LINES" >&2
        DL_UI_PANEL_LINES=0
    fi
    printf '\033[?25h' >&2
    return 0
}

# Emit a line above the live panel.
dl_ui_log() {
    if $DL_UI_TTY && (( DL_UI_PANEL_LINES > 0 )); then
        printf '\033[%dA\033[J' "$DL_UI_PANEL_LINES" >&2
        DL_UI_PANEL_LINES=0
    fi
    printf '%s\n' "$*" >&2
}

dl_ui_panel() {
    if ! $DL_UI_TTY; then return 0; fi

    local width=$DL_UI_COLS
    local total=${DL_TOTAL_BYTES:-0} done=${DL_DONE_BYTES:-0}
    local active=${#DL_ACTIVE_TASK[@]}
    local queued=$(( DL_RUN_TASKS - DL_FINISHED_TASKS - active ))
    if (( queued < 0 )); then queued=0; fi

    local permille=0
    if (( total > 0 )); then
        permille=$(( done * 1000 / total ))
        if (( permille > 1000 )); then permille=1000; fi
    elif (( DL_RUN_TASKS > 0 && DL_FINISHED_TASKS >= DL_RUN_TASKS )); then
        permille=1000
    fi

    local speed=${DL_SMOOTH_RATE:-0} eta=-1
    if (( total > 0 )); then
        if (( speed > 0 && total > done )); then
            eta=$(( (total - done) / speed ))
        elif (( total <= done )); then
            eta=0
        fi
    fi

    local -a lines=()
    local source_desc
    if $DL_SOURCE_LOCAL; then
        source_desc="local filesystem"
    else
        source_desc="${DL_SSH_TARGET}:${DL_SSH_PORT}"
    fi

    lines+=("${DL_C_BOLD}${DL_TITLE}${DL_C_RESET} ${DL_C_DIM}engine ${DL_ENGINE_VERSION}${DL_C_RESET} ${DL_C_CYAN}${DL_MODE_LABEL}${DL_C_RESET} jobs=${DL_JOBS} probes=${DL_DISCOVERY_JOBS} src=${source_desc}")
    lines+=("$(printf '[%s] %3d.%d%%  %s / %s' "$(dl_bar "$permille" 30)" \
        $(( permille / 10 )) $(( permille % 10 )) "$(dl_human "$done")" "$(dl_human "$total")")")
    lines+=("$(printf 'rate %-14s eta %-9s files %d/%d  active %d  queued %d  failed %d' \
        "$(dl_rate "$speed")" "$(dl_eta "$eta")" \
        "$DL_FINISHED_TASKS" "$DL_RUN_TASKS" "$active" "$queued" "${#DL_FAILED_TASKS[@]}")")

    local label_width=$(( width - 42 ))
    if (( label_width < 20 )); then label_width=20; fi
    local shown=0 slot idx cur size
    for slot in "${!DL_ACTIVE_TASK[@]}"; do
        if (( shown >= 8 )); then break; fi
        idx=${DL_ACTIVE_TASK[$slot]:-}
        if [[ -z "$idx" ]]; then continue; fi
        cur=${DL_TASK_CUR[$idx]:-0}
        size=${DL_TASK_SIZE[$idx]:-0}
        lines+=("${DL_C_DIM}$(printf '  %-4s %-*s %15s %12s' "${DL_TASK_ID[$idx]}" \
            "$label_width" "$(dl_fit "${DL_TASK_DISPLAY[$idx]}" "$label_width")" \
            "$(dl_human "$cur")/$(dl_human "$size")" "$(dl_rate "${DL_TASK_RATE[$idx]:-0}")")${DL_C_RESET}")
        shown=$(( shown + 1 ))
    done
    if (( active > shown )); then
        lines+=("${DL_C_DIM}  (+$(( active - shown )) more in flight)${DL_C_RESET}")
    fi

    if (( DL_UI_PANEL_LINES > 0 )); then
        printf '\033[%dA' "$DL_UI_PANEL_LINES" >&2
    fi
    local out
    for out in "${lines[@]}"; do
        printf '%s\033[K\n' "$out" >&2
    done
    DL_UI_PANEL_LINES=${#lines[@]}
    return 0
}

# ---------------------------------------------------------------------------
# SSH transport
# ---------------------------------------------------------------------------
DL_SSH_CONTROL_DIR=""
DL_SSH_CONTROL_PATH=""
DL_SSH_OPTS=()
DL_SSH_CHAN_OPTS=()

dl_ssh_init() {
    if $DL_SOURCE_LOCAL; then return 0; fi
    if ! command -v ssh >/dev/null 2>&1; then dl_die "ssh is not installed"; fi
    DL_SSH_CONTROL_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pyqcd-dl.XXXXXX") || dl_die "cannot create a control socket directory"
    DL_SSH_CONTROL_PATH="${DL_SSH_CONTROL_DIR}/sock-%r@%h:%p"
    DL_SSH_OPTS=(
        -p "$DL_SSH_PORT"
        -o ControlMaster=auto
        -o "ControlPath=${DL_SSH_CONTROL_PATH}"
        -o ControlPersist=3600
        -o "ServerAliveInterval=${DL_SSH_KEEPALIVE_INTERVAL}"
        -o "ServerAliveCountMax=${DL_SSH_KEEPALIVE_COUNT}"
        -o StrictHostKeyChecking=yes
        -o ConnectTimeout=20
        -o TCPKeepAlive=yes
    )
    # Transfer channels must never prompt: with a live master they multiplex,
    # and without one they fail immediately so the retry counter decides.
    DL_SSH_CHAN_OPTS=("${DL_SSH_OPTS[@]}" -o BatchMode=yes)
    dl_info "Establishing SSH master connection to ${DL_SSH_TARGET}:${DL_SSH_PORT}"
    if ! ssh "${DL_SSH_OPTS[@]}" "$DL_SSH_TARGET" true; then
        dl_die "cannot connect to ${DL_SSH_TARGET}:${DL_SSH_PORT}; check password/verification code and known_hosts, or use --source-local"
    fi
    dl_done "SSH master connection established; all channels multiplex over it"
    return 0
}

dl_ssh_alive() {
    if $DL_SOURCE_LOCAL; then return 0; fi
    if [[ -z "$DL_SSH_CONTROL_PATH" ]]; then return 1; fi
    ssh "${DL_SSH_OPTS[@]}" -O check "$DL_SSH_TARGET" >/dev/null 2>&1
}

# The master is the only connection that can prompt for a verification code, so
# only retry establishing it from an interactive session.
dl_ssh_recover() {
    if $DL_SOURCE_LOCAL; then return 0; fi
    if dl_ssh_alive; then return 0; fi
    if [[ ! -t 0 ]]; then return 1; fi
    dl_warn "SSH master connection lost; re-authenticating (password + verification code)"
    ssh "${DL_SSH_OPTS[@]}" "$DL_SSH_TARGET" true >/dev/null 2>&1
}

dl_remote_run() {
    if $DL_SOURCE_LOCAL; then
        bash -c "$1"
    else
        ssh "${DL_SSH_CHAN_OPTS[@]}" "$DL_SSH_TARGET" "$1"
    fi
}

# Stream the remote file to stdout starting at 1-based byte $2.
dl_remote_stream() {
    local path=$1 start=${2:-1}
    local cmd="tail -c +${start} -- $(dl_sq "$path")"
    if $DL_SOURCE_LOCAL; then
        bash -c "$cmd"
    else
        ssh "${DL_SSH_CHAN_OPTS[@]}" "$DL_SSH_TARGET" "$cmd"
    fi
}

dl_remote_checksum() {
    dl_remote_run "sha256sum -- $(dl_sq "$1")" | awk '{print $1}'
}

dl_cleanup() {
    dl_ui_fini
    if ! $DL_SOURCE_LOCAL; then
        if [[ -n "${DL_SSH_CONTROL_PATH:-}" && -S "${DL_SSH_CONTROL_PATH:-}" ]]; then
            ssh "${DL_SSH_OPTS[@]}" -O exit "$DL_SSH_TARGET" >/dev/null 2>&1 || true
        fi
        if [[ -n "${DL_SSH_CONTROL_DIR:-}" && -d "${DL_SSH_CONTROL_DIR:-}" ]]; then
            rm -rf "$DL_SSH_CONTROL_DIR"
        fi
    fi
    if [[ -n "${DL_WORK_DIR:-}" && -d "${DL_WORK_DIR:-}" ]] && ! ${DL_WORK_DIR_KEEP:-false}; then
        rm -rf "$DL_WORK_DIR"
    fi
    return 0
}

dl_map_local() {
    if [[ "$DL_DEST_ROOT" == / ]]; then
        printf '%s' "$1"
    else
        printf '%s%s' "${DL_DEST_ROOT%/}" "$1"
    fi
}

dl_local_size() {
    local size
    if [[ ! -f "$1" ]]; then
        printf '0'
        return 0
    fi
    size=$(stat -c%s -- "$1" 2>/dev/null || printf '0')
    if [[ ! "$size" =~ ^[0-9]+$ ]]; then size=0; fi
    printf '%s' "$size"
}

# ---------------------------------------------------------------------------
# Discovery: one round trip per source entry.
#   META_DIR/<i>.meta : "state<TAB>bytes<TAB>files", state in F|D|M|X
#   MANI_DIR/<i>.tsv  : "f<TAB>bytes<TAB>relpath" for directory contents
# ---------------------------------------------------------------------------
dl_discover_entry() {
    set +e
    local idx=$1 path=${ENTRY_REMOTE[$1]}
    local meta="$DL_META_DIR/$idx.meta" mani="$DL_MANI_DIR/$idx.tsv"
    local raw="$DL_MANI_DIR/$idx.raw" rc=0 first

    if $DL_SOURCE_LOCAL; then
        if [[ -f "$path" ]]; then
            printf 'F\t%s\t1\n' "$(stat -c%s -- "$path")" > "$meta"
            return 0
        fi
        if [[ ! -d "$path" ]]; then
            printf 'M\t0\t0\n' > "$meta"
            return 0
        fi
        if ! find "$path" -type f -printf 'f\t%s\t%P\n' > "$mani" 2>/dev/null; then
            printf 'X\t0\t0\n' > "$meta"
            return 0
        fi
        { awk -F'\t' '$1=="f"{b+=$2;n++} END{printf "D\t%.0f\t%.0f\n", b+0, n+0}' "$mani"; } > "$meta"
        return 0
    fi

    local quoted
    quoted=$(dl_sq "$path")
    if ! dl_remote_run "export LC_ALL=C
if [ -f $quoted ]; then
  printf 'F\t%s\t1\n' \"\$(stat -c%s -- $quoted)\"
elif [ -d $quoted ]; then
  printf 'D\t0\t0\n'
  exec find $quoted -type f -printf 'f\t%s\t%P\n'
else
  printf 'M\t0\t0\n'
fi" > "$raw" 2>/dev/null; then
        rc=$?
    fi
    if (( rc != 0 )); then
        printf 'X\t0\t0\n' > "$meta"
        return 0
    fi

    first=""
    IFS= read -r first < "$raw" || true
    if [[ "$first" == D$'\t'* ]]; then
        if (( $(wc -l < "$raw") > 1 )); then
            tail -n +2 "$raw" > "$mani"
        else
            : > "$mani"
        fi
        { awk -F'\t' '$1=="f"{b+=$2;n++} END{printf "D\t%.0f\t%.0f\n", b+0, n+0}' "$mani"; } > "$meta"
    else
        printf '%s\n' "${first:-M	0	0}" > "$meta"
    fi
    rm -f "$raw"
    return 0
}

dl_discover_all() {
    local total=${#ENTRY_REMOTE[@]}
    local next=0 running=0 completed=0
    local -a pid=() slot=()
    local i idx state bytes files

    while (( next < total || running > 0 )); do
        while (( running < DL_DISCOVERY_JOBS && next < total )); do
            ( dl_discover_entry "$next" ) &
            pid+=("$!")
            slot+=("$next")
            next=$(( next + 1 ))
            running=$(( running + 1 ))
        done

        for i in "${!pid[@]}"; do
            if [[ -z "${pid[$i]:-}" ]]; then continue; fi
            if kill -0 "${pid[$i]}" 2>/dev/null; then continue; fi
            wait "${pid[$i]}" 2>/dev/null || true
            idx=${slot[$i]}
            pid[$i]=""
            running=$(( running - 1 ))
            completed=$(( completed + 1 ))
            if [[ ! -f "$DL_META_DIR/$idx.meta" ]]; then
                printf 'X\t0\t0\n' > "$DL_META_DIR/$idx.meta"
            fi
            state="X"; bytes=0; files=0
            IFS=$'\t' read -r state bytes files < "$DL_META_DIR/$idx.meta" || true
            ENTRY_STATE[$idx]=${state:-X}
            ENTRY_BYTES[$idx]=${bytes:-0}
            ENTRY_FILES[$idx]=${files:-0}
            if $DL_UI_TTY; then
                printf '\r\033[K  scanning %d/%d source entries ...' "$completed" "$total" >&2
            fi
        done

        if (( running > 0 )); then sleep 0.15; fi
    done
    if $DL_UI_TTY; then printf '\r\033[K' >&2; fi
    return 0
}

# ---------------------------------------------------------------------------
# Task construction: one task per regular file, largest first
# ---------------------------------------------------------------------------
dl_build_tasks() {
    DL_TASK_REMOTE=()
    DL_TASK_LOCAL=()
    DL_TASK_SIZE=()
    DL_TASK_DISPLAY=()
    DL_TASK_BASE=()
    DL_PENDING=()

    local idx type bytes rel display base
    local -a file_entries=()

    for idx in "${!ENTRY_REMOTE[@]}"; do
        case "${ENTRY_STATE[$idx]:-M}" in
            F) file_entries+=("$idx") ;;
            D)
                local mani="$DL_MANI_DIR/$idx.tsv" base_local
                base_local=$(dl_map_local "${ENTRY_REMOTE[$idx]}")
                if [[ ! -f "$mani" ]]; then continue; fi
                while IFS=$'\t' read -r type bytes rel; do
                    if [[ "$type" != f || -z "$rel" ]]; then continue; fi
                    DL_TASK_REMOTE+=("${ENTRY_REMOTE[$idx]}/${rel}")
                    DL_TASK_LOCAL+=("${base_local}/${rel}")
                    DL_TASK_SIZE+=("${bytes:-0}")
                    DL_TASK_DISPLAY+=("${ENTRY_LABEL[$idx]}/${rel}")
                done < "$mani"
                ;;
        esac
    done

    for idx in "${file_entries[@]}"; do
        DL_TASK_REMOTE+=("${ENTRY_REMOTE[$idx]}")
        DL_TASK_LOCAL+=("$(dl_map_local "${ENTRY_REMOTE[$idx]}")")
        DL_TASK_SIZE+=("${ENTRY_BYTES[$idx]:-0}")
        DL_TASK_DISPLAY+=("${ENTRY_LABEL[$idx]}")
    done

    # Directories with no regular files still need to exist locally.
    for idx in "${!ENTRY_REMOTE[@]}"; do
        if [[ "${ENTRY_STATE[$idx]:-M}" != D ]]; then continue; fi
        local d
        d=$(dl_map_local "${ENTRY_REMOTE[$idx]}")
        if [[ ! -d "$d" ]]; then mkdir -p "$d" 2>/dev/null || true; fi
    done

    if (( ${#DL_TASK_SIZE[@]} > 1 )); then
        local -a sorted r=() l=() s=() dp=()
        mapfile -t sorted < <(
            for i in "${!DL_TASK_SIZE[@]}"; do
                printf '%s\t%s\n' "${DL_TASK_SIZE[$i]}" "$i"
            done | sort -k1,1nr -k2,2n | cut -f2
        )
        for i in "${sorted[@]}"; do
            r+=("${DL_TASK_REMOTE[$i]}")
            l+=("${DL_TASK_LOCAL[$i]}")
            s+=("${DL_TASK_SIZE[$i]}")
            dp+=("${DL_TASK_DISPLAY[$i]}")
        done
        DL_TASK_REMOTE=("${r[@]}")
        DL_TASK_LOCAL=("${l[@]}")
        DL_TASK_SIZE=("${s[@]}")
        DL_TASK_DISPLAY=("${dp[@]}")
    fi

    DL_TOTAL_TASKS=${#DL_TASK_REMOTE[@]}
    DL_TASK_ID=()
    DL_TOTAL_BYTES=0
    local i missing_file
    for i in "${!DL_TASK_REMOTE[@]}"; do
        DL_TASK_ID+=("$(printf '%04x' "$i")")
        base=$(dl_local_size "${DL_TASK_LOCAL[$i]}")
        if (( base > DL_TASK_SIZE[i] )); then base=0; fi
        DL_TASK_BASE[i]=$base
        missing_file=false
        if [[ ! -f "${DL_TASK_LOCAL[$i]}" ]]; then missing_file=true; fi
        # A zero-byte source still has to produce a local file.
        if $missing_file || (( base < DL_TASK_SIZE[i] )); then
            DL_PENDING+=("$i")
            DL_TOTAL_BYTES=$(( DL_TOTAL_BYTES + DL_TASK_SIZE[i] - base ))
        fi
    done
    return 0
}

# ---------------------------------------------------------------------------
# One file: resume from the local size, retry with backoff
# ---------------------------------------------------------------------------
dl_task_transfer() {
    set +e
    local idx=$1
    local remote=${DL_TASK_REMOTE[$idx]} localpath=${DL_TASK_LOCAL[$idx]}
    local size=${DL_TASK_SIZE[$idx]} id=${DL_TASK_ID[$idx]}
    local attempt=1 cur rc delay rsize rsum lsum

    if $DL_SOURCE_LOCAL && [[ "$remote" == "$localpath" ]]; then
        return 0
    fi

    mkdir -p "$(dirname -- "$localpath")" || return 1

    while :; do
        cur=$(dl_local_size "$localpath")

        if (( cur > size )); then
            printf '%s local copy is larger than the source; restarting\n' "$id"
            : > "$localpath" || return 1
            cur=0
        fi

        if (( cur == size )); then
            if [[ ! -f "$localpath" ]]; then
                : > "$localpath" || return 1
            fi
            if ! $DL_CHECKSUM; then
                return 0
            fi
            rsum=$(dl_remote_checksum "$remote" 2>/dev/null || true)
            lsum=$(sha256sum -- "$localpath" 2>/dev/null | awk '{print $1}')
            if [[ -n "$rsum" && "$rsum" == "$lsum" ]]; then
                return 0
            fi
            printf '%s checksum mismatch; restarting\n' "$id"
            : > "$localpath"
            cur=0
        fi

        if (( attempt > 1 )); then
            printf '%s retry %d/%d from %s of %s\n' \
                "$id" "$attempt" "$DL_RETRIES" "$(dl_human "$cur")" "$(dl_human "$size")"
        fi

        rc=0
        dl_remote_stream "$remote" $(( cur + 1 )) >> "$localpath" || rc=$?
        cur=$(dl_local_size "$localpath")

        if (( rc == 0 )); then
            if (( cur == size )); then
                return 0
            fi
            # The manifest may be older than the file; accept the streamed
            # length when the source really is that long.
            rsize=$(dl_remote_run "stat -c%s -- $(dl_sq "$remote")" 2>/dev/null || true)
            if [[ "$rsize" =~ ^[0-9]+$ ]] && (( rsize == cur )); then
                return 0
            fi
            printf '%s short stream: %s of %s\n' "$id" "$(dl_human "$cur")" "$(dl_human "$size")"
        else
            printf '%s interrupted at %s of %s (rc=%d)\n' \
                "$id" "$(dl_human "$cur")" "$(dl_human "$size")" "$rc"
        fi

        if (( attempt >= DL_RETRIES )); then
            printf '%s giving up after %d attempt(s)\n' "$id" "$attempt"
            return 1
        fi

        if ! dl_ssh_alive; then
            if dl_ssh_recover; then
                :
            elif [[ -t 0 ]]; then
                printf '%s giving up: SSH master connection is gone\n' "$id"
                return 1
            else
                # Non-interactive: a dead master cannot be re-authenticated
                # here, so let the remaining attempts decide.
                printf '%s master socket looks dead; retrying anyway\n' "$id"
            fi
        fi

        delay=$(( DL_RETRY_BASE * (1 << (attempt - 1)) ))
        if (( delay > DL_RETRY_MAX )); then delay=$DL_RETRY_MAX; fi
        sleep "$delay"
        attempt=$(( attempt + 1 ))
    done
}

# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------
dl_worker() {
    set +e
    local slot=$1 idx=$2 rc=0
    printf '%s\n' "$idx" > "$DL_SLOT_DIR/$slot"
    dl_task_transfer "$idx" > "$DL_MSG_DIR/$slot.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$DL_RESULT_DIR/$idx" 2>/dev/null || true
    rm -f "$DL_SLOT_DIR/$slot" 2>/dev/null || true
    return 0
}

dl_flush_messages() {
    local slot f lines line
    for slot in "${!DL_MSG_SEEN[@]}"; do
        f="$DL_MSG_DIR/$slot.log"
        if [[ ! -f "$f" ]]; then continue; fi
        lines=$(wc -l < "$f" 2>/dev/null || printf '0')
        if (( lines <= DL_MSG_SEEN[$slot] )); then continue; fi
        while IFS= read -r line; do
            dl_ui_log "${DL_C_DIM}${line}${DL_C_RESET}"
        done < <(tail -n +"$(( DL_MSG_SEEN[$slot] + 1 ))" "$f")
        DL_MSG_SEEN[$slot]=$lines
    done
    return 0
}

dl_progress_measure() {
    local slot idx cur
    DL_ACTIVE_BYTES=0
    for slot in "${!DL_ACTIVE_TASK[@]}"; do
        idx=${DL_ACTIVE_TASK[$slot]:-}
        if [[ -z "$idx" ]]; then continue; fi
        cur=$(dl_local_size "${DL_TASK_LOCAL[$idx]}")
        if (( cur > DL_TASK_SIZE[$idx] )); then cur=${DL_TASK_SIZE[$idx]}; fi
        DL_TASK_CUR[$idx]=$cur
        DL_ACTIVE_BYTES=$(( DL_ACTIVE_BYTES + cur - ${DL_TASK_BASE[$idx]:-0} ))
    done
    DL_DONE_BYTES=$(( DL_FINISHED_BYTES + DL_ACTIVE_BYTES ))
    return 0
}

dl_rate_update() {
    local now=$1 dt_ms delta instant slot idx cur prev d
    dt_ms=$(( now - DL_LAST_SAMPLE_MS ))
    if (( dt_ms <= 0 )); then dt_ms=1; fi
    delta=$(( DL_DONE_BYTES - DL_SAMPLE_BYTES ))
    if (( delta < 0 )); then delta=0; fi
    instant=$(( delta * 1000 / dt_ms ))

    for slot in "${!DL_ACTIVE_TASK[@]}"; do
        idx=${DL_ACTIVE_TASK[$slot]:-}
        if [[ -z "$idx" ]]; then continue; fi
        cur=${DL_TASK_CUR[$idx]:-0}
        prev=${DL_TASK_PREV[$idx]:-0}
        d=$(( cur - prev ))
        if (( d < 0 )); then d=0; fi
        DL_TASK_RATE[$idx]=$(( d * 1000 / dt_ms ))
        DL_TASK_PREV[$idx]=$cur
    done

    if (( DL_SMOOTH_RATE > 0 )); then
        DL_SMOOTH_RATE=$(( (DL_SMOOTH_RATE * 3 + instant) / 4 ))
    else
        DL_SMOOTH_RATE=$instant
    fi
    DL_SAMPLE_BYTES=$DL_DONE_BYTES
    DL_LAST_SAMPLE_MS=$now
    return 0
}

dl_run_tasks() {
    local total=${#DL_PENDING[@]}
    local next=0 rc idx slot now
    local -a slot_pid=()
    local last_plain=0 permille eta moved

    DL_FINISHED_BYTES=0
    DL_FINISHED_TASKS=0
    DL_FAILED_TASKS=()
    DL_ACTIVE_TASK=()
    DL_SMOOTH_RATE=0
    DL_SAMPLE_BYTES=0
    DL_DONE_BYTES=0
    DL_ACTIVE_BYTES=0
    DL_TASK_CUR=()
    DL_TASK_PREV=()
    DL_TASK_RATE=()
    DL_MSG_SEEN=()

    rm -f "$DL_MSG_DIR"/*.log "$DL_RESULT_DIR"/* "$DL_SLOT_DIR"/* 2>/dev/null || true

    DL_RUN_TASKS=$total
    dl_info "Transferring $total file(s), $(dl_human "$DL_TOTAL_BYTES") to fetch"

    now=$(dl_now_ms)
    DL_LAST_SAMPLE_MS=$now
    DL_LAST_RENDER_MS=$(( now - 700 ))

    while (( next < total || ${#DL_ACTIVE_TASK[@]} > 0 )); do
        for (( slot = 0; slot < DL_JOBS; slot++ )); do
            if [[ -n "${slot_pid[$slot]:-}" ]]; then continue; fi
            if (( next >= total )); then break; fi
            idx=${DL_PENDING[$next]:-}
            if [[ -z "$idx" ]]; then break; fi
            next=$(( next + 1 ))
            DL_ACTIVE_TASK[$slot]=$idx
            DL_TASK_PREV[$idx]=$(dl_local_size "${DL_TASK_LOCAL[$idx]}")
            DL_MSG_SEEN[$slot]=0
            ( dl_worker "$slot" "$idx" ) &
            slot_pid[$slot]=$!
        done

        for slot in "${!slot_pid[@]}"; do
            if [[ -z "${slot_pid[$slot]:-}" ]]; then continue; fi
            if kill -0 "${slot_pid[$slot]}" 2>/dev/null; then continue; fi
            wait "${slot_pid[$slot]}" 2>/dev/null || true
            idx=${DL_ACTIVE_TASK[$slot]:-}
            unset 'DL_ACTIVE_TASK[slot]'
            slot_pid[$slot]=""
            if [[ -z "$idx" ]]; then continue; fi

            rc=1
            if [[ -f "$DL_RESULT_DIR/$idx" ]]; then rc=$(< "$DL_RESULT_DIR/$idx"); fi
            DL_FINISHED_TASKS=$(( DL_FINISHED_TASKS + 1 ))
            moved=$(( DL_TASK_SIZE[$idx] - ${DL_TASK_BASE[$idx]:-0} ))
            if [[ "$rc" == 0 ]]; then
                DL_FINISHED_BYTES=$(( DL_FINISHED_BYTES + moved ))
                dl_ui_log "${DL_C_GREEN}[OK]${DL_C_RESET}    $(dl_human "$moved")  ${DL_TASK_DISPLAY[$idx]}"
            else
                DL_FAILED_TASKS+=("$idx")
                dl_ui_log "${DL_C_RED}[FAIL]${DL_C_RESET}  ${DL_TASK_DISPLAY[$idx]}"
            fi
        done

        now=$(dl_now_ms)
        dl_flush_messages
        dl_progress_measure
        if (( now - DL_LAST_SAMPLE_MS >= 500 )); then dl_rate_update "$now"; fi

        if $DL_UI_TTY; then
            if (( now - DL_LAST_RENDER_MS >= 700 )); then
                DL_LAST_RENDER_MS=$now
                dl_ui_panel
            fi
        elif (( now - last_plain >= DL_PLAIN_INTERVAL_MS )); then
            last_plain=$now
            permille=0
            eta=-1
            if (( DL_TOTAL_BYTES > 0 )); then
                permille=$(( DL_DONE_BYTES * 1000 / DL_TOTAL_BYTES ))
                if (( permille > 1000 )); then permille=1000; fi
                if (( DL_SMOOTH_RATE > 0 && DL_TOTAL_BYTES > DL_DONE_BYTES )); then
                    eta=$(( (DL_TOTAL_BYTES - DL_DONE_BYTES) / DL_SMOOTH_RATE ))
                fi
            fi
            dl_ui_log "$(printf '[PROGRESS] %d.%d%% %s/%s %s eta %s files %d/%d active %d failed %d' \
                $(( permille / 10 )) $(( permille % 10 )) \
                "$(dl_human "$DL_DONE_BYTES")" "$(dl_human "$DL_TOTAL_BYTES")" \
                "$(dl_rate "$DL_SMOOTH_RATE")" "$(dl_eta "$eta")" \
                "$DL_FINISHED_TASKS" "$DL_RUN_TASKS" \
                "${#DL_ACTIVE_TASK[@]}" "${#DL_FAILED_TASKS[@]}")"
        fi

        if (( ${#DL_ACTIVE_TASK[@]} > 0 )); then sleep 0.25; fi
    done

    dl_flush_messages
    DL_DONE_BYTES=$DL_FINISHED_BYTES
    # One last sample so the closing panel shows a real aggregate rate.
    if (( ${#DL_ACTIVE_TASK[@]} == 0 )); then
        DL_ACTIVE_TASK=()
        dl_rate_update "$(dl_now_ms)"
        DL_DONE_BYTES=$DL_FINISHED_BYTES
    fi
    dl_ui_panel
    return 0
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
dl_main() {
    if [[ "$DL_DEST_ROOT" != /* ]]; then dl_die "destination root must be absolute: $DL_DEST_ROOT"; fi
    DL_DEST_ROOT="${DL_DEST_ROOT%/}"
    if [[ -z "$DL_DEST_ROOT" ]]; then DL_DEST_ROOT=/; fi
    if [[ ! "$DL_JOBS" =~ ^[1-9][0-9]*$ ]]; then dl_die "--jobs requires a positive integer: $DL_JOBS"; fi
    if [[ ! "$DL_DISCOVERY_JOBS" =~ ^[1-9][0-9]*$ ]]; then dl_die "--discovery-jobs requires a positive integer: $DL_DISCOVERY_JOBS"; fi
    if [[ ! "$DL_RETRIES" =~ ^[0-9]+$ ]]; then dl_die "--retries requires a non-negative integer: $DL_RETRIES"; fi
    case "$DL_PROGRESS" in
        auto|on|off) ;;
        *) dl_die "progress mode must be auto, on or off: $DL_PROGRESS" ;;
    esac
    if (( BASH_VERSINFO[0] < 4 )); then dl_die "bash >= 4.0 is required"; fi
    if (( ${#ENTRY_REMOTE[@]} == 0 )); then dl_die "no source entries were defined"; fi

    ENTRY_STATE=(); ENTRY_BYTES=(); ENTRY_FILES=()
    DL_TASK_REMOTE=(); DL_TASK_LOCAL=(); DL_TASK_SIZE=(); DL_TASK_DISPLAY=(); DL_TASK_ID=()
    DL_TASK_BASE=(); DL_PENDING=(); DL_TASK_CUR=(); DL_TASK_PREV=(); DL_TASK_RATE=()
    DL_MSG_SEEN=()
    DL_TOTAL_BYTES=0; DL_DONE_BYTES=0; DL_FINISHED_BYTES=0; DL_ACTIVE_BYTES=0
    DL_TOTAL_TASKS=0; DL_RUN_TASKS=0; DL_FINISHED_TASKS=0; DL_FAILED_TASKS=(); DL_ACTIVE_TASK=()
    DL_SMOOTH_RATE=0; DL_SAMPLE_BYTES=0
    DL_LAST_SAMPLE_MS=0; DL_LAST_RENDER_MS=0; DL_WORK_DIR=""; DL_WORK_DIR_KEEP=false

    trap 'dl_cleanup' EXIT
    trap 'dl_ui_fini; exit 130' INT
    trap 'dl_ui_fini; exit 143' TERM

    dl_ui_init

    printf '%s\n' '=============================================================================' >&2
    printf ' %s\n' "$DL_TITLE" >&2
    printf '%s\n' '=============================================================================' >&2
    printf ' entries:     %d\n' "${#ENTRY_REMOTE[@]}" >&2
    printf ' destination: %s\n' "$DL_DEST_ROOT" >&2
    if $DL_SOURCE_LOCAL; then
        printf ' source:      local filesystem\n' >&2
    else
        printf ' source:      %s:%s (user %s)\n' "$DL_SSH_HOST" "$DL_SSH_PORT" "$DL_SSH_USER" >&2
    fi
    printf ' concurrency: %d transfer(s) over one SSH master, %d discovery probe(s)\n' \
        "$DL_JOBS" "$DL_DISCOVERY_JOBS" >&2
    printf ' resilience:  %d attempt(s) per file, backoff %ds..%ds, resume from local offset\n' \
        "$DL_RETRIES" "$DL_RETRY_BASE" "$DL_RETRY_MAX" >&2
    printf ' keepalive:   ServerAliveInterval %ds x %d\n' \
        "$DL_SSH_KEEPALIVE_INTERVAL" "$DL_SSH_KEEPALIVE_COUNT" >&2
    printf ' mode:        %s\n' "$DL_MODE_LABEL" >&2
    printf '%s\n' '=============================================================================' >&2

    if $DL_DRY_RUN; then
        local i
        for i in "${!ENTRY_REMOTE[@]}"; do
            printf '  [%04d] %-8s %s\n' "$(( i + 1 ))" "${ENTRY_KIND[$i]:-?}" "${ENTRY_REMOTE[$i]}" >&2
        done
        dl_info "Dry run complete; no SSH connection or transfer was attempted"
        return 0
    fi

    dl_ssh_init

    DL_WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pyqcd-dl-work.XXXXXX") || dl_die "cannot create a work directory"
    DL_META_DIR="$DL_WORK_DIR/meta"
    DL_MANI_DIR="$DL_WORK_DIR/manifests"
    DL_MSG_DIR="$DL_WORK_DIR/messages"
    DL_RESULT_DIR="$DL_WORK_DIR/results"
    DL_SLOT_DIR="$DL_WORK_DIR/slots"
    mkdir -p "$DL_META_DIR" "$DL_MANI_DIR" "$DL_MSG_DIR" "$DL_RESULT_DIR" "$DL_SLOT_DIR"

    local scan_start scan_elapsed
    scan_start=$(dl_now_ms)
    dl_info "Walking ${#ENTRY_REMOTE[@]} source entries"
    dl_discover_all
    scan_elapsed=$(( ($(dl_now_ms) - scan_start) / 1000 ))
    dl_info "Source walk finished in ${scan_elapsed}s"

    local -a missing=() unreadable=()
    local idx
    for idx in "${!ENTRY_REMOTE[@]}"; do
        case "${ENTRY_STATE[$idx]:-M}" in
            F|D) ;;
            M) missing+=("$idx") ;;
            *) unreadable+=("$idx") ;;
        esac
    done
    for idx in "${missing[@]}"; do dl_error "missing source:    ${ENTRY_LABEL[$idx]}  ${ENTRY_REMOTE[$idx]}"; done
    for idx in "${unreadable[@]}"; do dl_error "unreadable source: ${ENTRY_LABEL[$idx]}  ${ENTRY_REMOTE[$idx]}"; done

    if (( ${#missing[@]} + ${#unreadable[@]} > 0 )); then
        if [[ "$DL_MISSING_POLICY" == skip ]]; then
            dl_warn "$(( ${#missing[@]} + ${#unreadable[@]} )) entr(y/ies) unavailable; continuing because --allow-missing was given"
        else
            dl_die "source inventory is incomplete; nothing was transferred (use --allow-missing to skip)"
        fi
    fi

    DL_EXPECTED_BYTES=0
    DL_EXPECTED_FILES=0
    for idx in "${!ENTRY_REMOTE[@]}"; do
        case "${ENTRY_STATE[$idx]:-M}" in
            F|D)
                DL_EXPECTED_BYTES=$(( DL_EXPECTED_BYTES + ${ENTRY_BYTES[$idx]:-0} ))
                DL_EXPECTED_FILES=$(( DL_EXPECTED_FILES + ${ENTRY_FILES[$idx]:-0} ))
                ;;
        esac
    done
    dl_done "$(printf 'inventory: %d readable entries, %d file(s), %s' \
        "$(( ${#ENTRY_REMOTE[@]} - ${#missing[@]} - ${#unreadable[@]} ))" \
        "$DL_EXPECTED_FILES" "$(dl_human "$DL_EXPECTED_BYTES")")"
    if [[ -n "$DL_BYTES_HINT" && "$DL_BYTES_HINT" =~ ^[0-9]+$ ]] && (( DL_BYTES_HINT > 0 )); then
        dl_info "list declares $(dl_human "$DL_BYTES_HINT") of allocated bytes; the walk measures $(dl_human "$DL_EXPECTED_BYTES") of file bytes"
    fi

    if $DL_PREFLIGHT_ONLY; then
        dl_done "preflight-only run complete"
        return 0
    fi

    dl_build_tasks

    local completed_tasks=$(( DL_TOTAL_TASKS - ${#DL_PENDING[@]} ))
    if (( completed_tasks > 0 )); then
        dl_skip "$completed_tasks of $DL_TOTAL_TASKS file(s) already at the expected size"
    fi

    if (( ${#DL_PENDING[@]} == 0 )); then
        dl_done "requested data download completed (nothing to transfer)"
        return 0
    fi

    if ! $DL_ASSUME_YES; then
        local answer
        printf 'Proceed with %d file transfer(s), %s to fetch? [Y/n] ' \
            "${#DL_PENDING[@]}" "$(dl_human "$DL_TOTAL_BYTES")" >&2
        read -r answer || answer=""
        answer=${answer:-Y}
        if [[ ! "$answer" =~ ^[Yy] ]]; then
            dl_warn "cancelled by user"
            return 0
        fi
    fi

    dl_run_tasks
    dl_ui_fini

    local ok=$(( DL_FINISHED_TASKS - ${#DL_FAILED_TASKS[@]} ))
    printf '\n%s\n' '=============================================================================' >&2
    printf ' files:   %d/%d transferred successfully\n' "$ok" "${#DL_PENDING[@]}" >&2
    printf ' bytes:   %s moved in this run\n' "$(dl_human "$DL_FINISHED_BYTES")" >&2
    printf ' failed:  %d\n' "${#DL_FAILED_TASKS[@]}" >&2
    printf '%s\n' '=============================================================================' >&2

    if (( ${#DL_FAILED_TASKS[@]} > 0 )); then
        local keep="$DL_WORK_DIR/keep"
        mkdir -p "$keep"
        cp -f "$DL_MSG_DIR"/*.log "$keep"/ 2>/dev/null || true
        DL_WORK_DIR_KEEP=true
        if [[ -n "${DL_FAILED_LIST:-}" ]]; then
            mkdir -p "$(dirname -- "$DL_FAILED_LIST")" 2>/dev/null || true
            : > "$DL_FAILED_LIST"
            for idx in "${DL_FAILED_TASKS[@]}"; do
                printf '%s\n' "${DL_TASK_REMOTE[$idx]}" >> "$DL_FAILED_LIST"
            done
            dl_error "failed paths listed in $DL_FAILED_LIST"
        fi
        dl_error "per-job logs kept in $keep"
        dl_error "rerun the same command to resume from the current byte offsets"
        return 1
    fi

    dl_done "requested data download completed"
    return 0
}
