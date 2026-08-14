#!/usr/bin/env bash
# test1 — 调用 pyqcd 包复现 docker-v20260805 全量蒸馏 GPU 管线（test12 形式）
# 用法: bash examples/test1/run-local.sh [--dry-run] [--conf-ids 6250] [--skip-4pt]
set -u
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
DRY=0
EXTRA=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) EXTRA+=("$a") ;;
  esac
done

VDIR="${TEST1_VDIR:-$SCRIPT_DIR/v202608140630}"
mkdir -p "$VDIR"
export TEST1_OUTDIR="$VDIR"
TS=$(date +%Y-%m-%d-%H-%M-%S)
LOG="$VDIR/run-local-$TS.log"

run() {
  echo "\$ $*"
  [ "$DRY" = 1 ] && return 0
  timeout 86400 "$@" || { echo "[warn] $* 失败(exit=$?)" | tee -a "$LOG"; }
}

echo "=== test1 run-local [$TS] ===" | tee "$LOG"
echo "版本目录: $VDIR" | tee -a "$LOG"

# 1) env 自检
run python "$SCRIPT_DIR/main.py" env

# 2) 完整管线（单步失败仅记录并继续）
if [ ${#EXTRA[@]} -gt 0 ]; then
  run python "$SCRIPT_DIR/main.py" pipeline "${EXTRA[@]}"
else
  run python "$SCRIPT_DIR/main.py" pipeline
fi

# 3) verify 数值一致性 vs 基线（需 run_dir；自动找最新 output_*）
LATEST=$(ls -1d "$VDIR"/output_* 2>/dev/null | sort | tail -1)
if [ -n "$LATEST" ]; then
  RD=$(basename "$LATEST")
  run python "$SCRIPT_DIR/main.py" verify --run-dir "$RD"
  run python "$SCRIPT_DIR/main.py" collect --run-dir "$RD"
  run python "$SCRIPT_DIR/main.py" report --run-dir "$RD"
else
  echo "[warn] 未找到 output_* 目录，跳过 verify/collect/report" | tee -a "$LOG"
fi

echo "=== test1 run-local done [$TS] ===" | tee -a "$LOG"
