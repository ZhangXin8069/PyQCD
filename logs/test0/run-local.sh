#!/usr/bin/env bash
# test0 —— ana_3dir 三方向差异分析一键测试（参考 test12 run-local.sh）。
#
# 流程：env → makedata → run → verify → check → collect
# 特点：每次运行创建版本目录 logs/test0/v<YYYYMMDDHHMM>/（同分钟重跑加 -<SS>），
#       全部产物（json/png/env.json/运行日志）落在版本目录，互不覆盖。
#
# 用法：
#   bash logs/test0/run-local.sh            # 实际执行
#   bash logs/test0/run-local.sh --dry-run  # 只打印命令
set -uo pipefail

REPO="${HOME}/PyQCD"
WORK="$REPO/logs/test0"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)

# ---- 版本目录：v<YYYYMMDDHHMM>；同分钟重跑加 -<SS> 防覆盖 ----
VDIR="$WORK/v$(date +%Y%m%d%H%M)"
if [ -e "$VDIR" ]; then VDIR="$VDIR-$(date +%S)"; fi
mkdir -p "$VDIR"
export TEST0_OUTDIR="$VDIR"
DATA_DIR="${TEST0_DATA_DIR:-$WORK/input}"
LOG_FILE="$VDIR/run-local-$TS.log"

DRY="${DRY:-${1:-}}"
[ "$DRY" = "--dry-run" ] && DRY=1 || DRY=0

step() { echo; echo "===== $1 ====="; }
run() { # $1=timeout_s  "$@"=command
  local tmo="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] timeout $tmo $*"; return 0; fi
  echo ">>> timeout $tmo $*"
  timeout "$tmo" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[warn] 上一步失败 rc=$rc（继续）" >&2
  fi
  return $rc
}

# ---- 环境与完整输出归档 ----
cd "$REPO" || exit 1
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== test0 local runner start $(date) ==="
echo "版本目录: $VDIR"
echo "数据目录: $DATA_DIR"

step "Step 0: 环境自检"
run 120 python "$MAIN" env

step "Step 1: 生成合成数据（含 ground-truth）"
run 120 python "$MAIN" makedata --outdir "$DATA_DIR"

step "Step 2: 分析 + 作图（调用 pyqcd.analysis.analyze_3dir）"
run 300 python "$MAIN" run --data-root "$DATA_DIR" --outdir "$VDIR"

step "Step 3: 断言验证"
run 300 python "$MAIN" verify --run-dir "$VDIR" --data-root "$DATA_DIR"

step "Step 4: 断言门"
run 120 python "$MAIN" check --run-dir "$VDIR" --label "test0 check"

step "Step 5: 产物清单"
run 120 python "$MAIN" collect --run-dir "$VDIR"

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
echo "版本目录: $VDIR （run-local-*.log + env.json + test0_verify.json + png + summary）"
echo "跨环境比对：各环境各跑一次本脚本，目录 v* 下同名产物可直接 diff/叠图。"
