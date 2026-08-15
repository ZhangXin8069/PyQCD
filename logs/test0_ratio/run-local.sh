#!/usr/bin/env bash
# test0_ratio —— 02_ratio 功能一键测试（test12 风格）。
# 流程：env → makedata → run → verify → check → collect。
set -uo pipefail

REPO="${HOME}/PyQCD"
WORK="$REPO/logs/test0_ratio"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)

VDIR="$WORK/v$(date +%Y%m%d%H%M)"
if [ -e "$VDIR" ]; then VDIR="$VDIR-$(date +%S)"; fi
mkdir -p "$VDIR"
export TEST0_OUTDIR="$VDIR"
DATA_DIR="${TEST0_DATA_DIR:-$WORK/input}"
LOG_FILE="$VDIR/run-local-$TS.log"

DRY="${DRY:-${1:-}}"
[ "$DRY" = "--dry-run" ] && DRY=1 || DRY=0

step() { echo; echo "===== $1 ====="; }
run() {
  local tmo="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] timeout $tmo $*"; return 0; fi
  echo ">>> timeout $tmo $*"
  timeout "$tmo" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[warn] 上一步失败 rc=$rc（继续）" >&2; fi
  return $rc
}

cd "$REPO" || exit 1
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== test0_ratio local runner start $(date) ==="
echo "版本目录: $VDIR"

step "Step 0: 环境自检"
run 120 python "$MAIN" env

step "Step 1: 生成合成数据（含 ground-truth）"
run 120 python "$MAIN" makedata --outdir "$DATA_DIR"

step "Step 2a: compute 全链（2pt+OPE → ratio，管道验证）"
run 600 python "$MAIN" run --data-root "$DATA_DIR" --outdir "$VDIR" \
    --parts "1,1" --tag compute

step "Step 2b: fit+plot 全链（预生成解析 ratio → fit → 图）"
run 600 python "$MAIN" run --data-root "$DATA_DIR" --outdir "$VDIR" \
    --parts "2,3" --ratio-source "$DATA_DIR/fit_ratio.npy" --tag fit

step "Step 3: 断言验证"
run 300 python "$MAIN" verify --run-dir "$VDIR" --data-root "$DATA_DIR"

step "Step 4: 断言门"
run 120 python "$MAIN" check --run-dir "$VDIR" --label "test0_ratio check"

step "Step 5: 产物清单"
run 120 python "$MAIN" collect --run-dir "$VDIR"

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
echo "版本目录: $VDIR"
