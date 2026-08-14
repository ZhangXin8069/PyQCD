#!/usr/bin/env bash
# test0 —— docker-v20260805 全量蒸馏 GPU 管线一致性测试（test12 形式）
# 每次运行创建版本目录 logs/test0/v<YYYYMMDDHHMM>/，全部产物（中间数据/图表/
# 报告/env.json/运行日志）落在该目录（互不覆盖，跨环境可横向比对）。
#
# 流程：Step 0 环境自检 → Step 1 env → Step 2 pipeline 全量 10 组态
#       → Step 3 verify（数值一致性 vs 基线）→ Step 4 collect → Step 5 report
# 特点：每步 timeout 防卡壳；单步失败仅记录并继续（GRID 保持轻量）。
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
LOG_FILE="$VDIR/run-local-$TS.log"

DRY="${DRY:-${1:-}}"
[ "$DRY" = "--dry-run" ] && DRY=1 || DRY=0

step() { echo; echo "===== $1 ====="; }
run() { # $1=timeout_s  "$@"=command；失败仅记录（继续）
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
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

step "Step 0: 环境自检"
run 300 python -c "import numpy, matplotlib; print('numpy', numpy.__version__)"
run 300 python -c "import cupy; print('cupy', cupy.__version__, 'CUDA', cupy.cuda.runtime.runtimeGetVersion())"

step "Step 1: env —— 数据/GPU 自检 + env.json"
run 300 python "$MAIN" env

step "Step 2: pipeline —— 全量 10 组态（vertex→2pt/3pt/4pt/OPE→analysis→plots→report）"
run 86400 python "$MAIN" pipeline

step "Step 3: verify —— 数值一致性 vs 基线 output_20260802_120104（rtol=1e-3）"
run 3600 python "$MAIN" verify

step "Step 4: collect —— 汇总 test0_results.json"
run 300 python "$MAIN" collect --run-dir "$(ls -1t "$VDIR"/output_* 2>/dev/null | head -1 | xargs -r basename)"

step "Step 5: report —— LaTeX 物理报告（physics_report.tex → pdf）"
RUNDIR="$(ls -1t "$VDIR"/output_* 2>/dev/null | head -1 | xargs -r basename)"
if [ -n "$RUNDIR" ]; then
  run 1800 python "$MAIN" report --run-dir "$RUNDIR"
else
  echo "[warn] 无管线输出目录，跳过 report"
fi

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
echo "版本目录: $VDIR （run-local-*.log + env.json + output_*/ + test0_verify.json + test0_results.json）"
echo "一致性结论: $(python3 -c "import json,glob; f=glob.glob('$VDIR/test0_verify.json'); print(json.load(open(f[0])) if f else '未生成')" 2>/dev/null)"
