#!/bin/bash
# batch_baryon_3pt.sh — Batch run LQCDMaster local3pt tasks
#
# Usage:  ./batch_baryon_3pt.sh [--start N] [--end N] [--test]
#
# Options:
#   --start N   Start from task N (default 1)
#   --end N     End at task N (default 19)
#   --test      Add --test flag (generate code only, skip SLURM submission)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TEST_FLAG=""
START=1
END=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START="$2"; shift 2 ;;
        --end)   END="$2";   shift 2 ;;
        --test)  TEST_FLAG="--test"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="examples/3pt/result_ds_baryon/${TIMESTAMP}"

echo "============================================"
echo " LQCDMaster baryon_3pt Batch Test"
echo " Time:    $(date)"
echo " Range:   task_${START}.txt ~ task_${END}.txt"
echo " Output:  ${BASE_DIR}/task_n/"
echo " Mode:    $([ -n "$TEST_FLAG" ] && echo '--test (generate only)' || echo 'full run')"
echo "============================================"
echo

for i in $(seq "$START" "$END"); do
    TASK_FILE="examples/3pt/task_baryon/task_${i}.txt"
    RUN_DIR="${BASE_DIR}/task_${i}"

    if [ ! -f "$TASK_FILE" ]; then
        echo "[SKIP] ${TASK_FILE} not found"
        continue
    fi

    echo "----------------------------------------"
    echo "[$(date +%H:%M:%S)] Running task_${i}"
    echo "  task:   ${TASK_FILE}"
    echo "  output: ${RUN_DIR}"
    echo "----------------------------------------"

    LOG_DIR="${RUN_DIR}/logs"
    mkdir -p "$LOG_DIR"

    python3 run.py \
        --task "$TASK_FILE" \
        --run-dir "$RUN_DIR" \
        --non-interactive \
        $TEST_FLAG > "$LOG_DIR/stdout.log" 2> "$LOG_DIR/stderr.log" || true

    EXIT_CODE=$?
    echo "[$(date +%H:%M:%S)] task_${i} exit code: ${EXIT_CODE}" >> "${BASE_DIR}/batch.log"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[OK] task_${i} finished (exit code: ${EXIT_CODE})"
    else
        echo "[FAIL] task_${i} failed (exit code: ${EXIT_CODE})"
        echo "  stderr (last 20 lines):" >> "${BASE_DIR}/batch.log"
        tail -20 "$LOG_DIR/stderr.log" | sed 's/^/    /' >> "${BASE_DIR}/batch.log"
    fi
    echo

    sleep 5

done

echo "============================================"
echo " All done"
echo " Tasks:  ${START} ~ ${END}"
echo " Output: ${BASE_DIR}/"
echo " Time:   $(date)"
echo "============================================"
