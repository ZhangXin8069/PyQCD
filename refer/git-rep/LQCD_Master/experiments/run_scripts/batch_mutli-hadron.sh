#!/bin/bash
# batch_test_local2pt.sh — Batch run LQCDMaster local2pt tasks
#
# Usage:  ./batch_test_local2pt.sh [--start N] [--end N] [--test]
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
END=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START="$2"; shift 2 ;;
        --end)   END="$2";   shift 2 ;;
        --test)  TEST_FLAG="--test"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="examples/multi-hadron/result/${TIMESTAMP}"

echo "============================================"
echo " LQCDMaster local2pt Batch Test"
echo " Time:    $(date)"
echo " Range:   task_${START}.txt ~ task_${END}.txt"
echo " Output:  ${BASE_DIR}/task_n/"
echo " Mode:    $([ -n "$TEST_FLAG" ] && echo '--test (generate only)' || echo 'full run')"
echo "============================================"
echo

for i in $(seq "$START" "$END"); do
    TASK_FILE="examples/multi-hadron/task/task_${i}.txt"
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

    python3 run.py \
        --task "$TASK_FILE" \
        --run-dir "$RUN_DIR" \
        --non-interactive \
        $TEST_FLAG

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[OK] task_${i} finished (exit code: ${EXIT_CODE})"
    else
        echo "[FAIL] task_${i} failed (exit code: ${EXIT_CODE})"
    fi
    echo

done

echo "============================================"
echo " All done"
echo " Tasks:  ${START} ~ ${END}"
echo " Output: ${BASE_DIR}/"
echo " Time:   $(date)"
echo "============================================"
