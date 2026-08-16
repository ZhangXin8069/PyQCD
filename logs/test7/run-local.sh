#!/usr/bin/env bash
# test7 —— pyqcd 全功能真实数据实战一键脚本（服务器正式工作版）。
# 服务器规范：
#   - 启动即 source /public/home/zhangxin/mgmt04-env.sh（存在时；本地缺省跳过）
#   - GPU（NV-V100-32GB）探测简报
#   - 正式版 100 组态（6250 起、间隔 200 → 6250..26050），输入数据带检查机制
#   - 实时进度日志：全部输出 tee 落盘（tee 无缓冲），python 侧时间戳+flush，
#     便于实时调控（tail -f 日志 / Ctrl+C / ~stop）
#   - --server：nohup 后台运行（服务器规范），日志 run-server-<TS>.log
# 流程：env → makedata（含检查）→ run（全链）→ verify → check → collect。
set -uo pipefail

SERVER_ENV=/public/home/zhangxin/mgmt04-env.sh
REPO="${HOME}/PyQCD"
WORK="$REPO/logs/test7"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)

VDIR="$WORK/v$(date +%Y%m%d%H%M)"
if [ -e "$VDIR" ]; then VDIR="$VDIR-$(date +%S)"; fi
mkdir -p "$VDIR"
export test7_OUTDIR="$VDIR"
DATA_DIR="${test7_DATA_DIR:-$WORK/input}"
LOG_FILE="$VDIR/run-local-$TS.log"

# 服务器规范：环境启动（存在才 source，本地开发机缺省跳过）
if [ -f "$SERVER_ENV" ]; then
  echo "[env] source $SERVER_ENV"
  # shellcheck disable=SC1090
  source "$SERVER_ENV" || { echo "[fatal] source $SERVER_ENV 失败" >&2; exit 1; }
else
  echo "[env] 未找到 $SERVER_ENV（本地模式，跳过环境启动）"
fi

# 服务器规范：GPU 探测简报（V100-32GB）
GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,driver_version \
                      --format=csv,noheader 2>/dev/null | head -1)
echo "[gpu] ${GPU_LINE:-nvidia-smi 不可用（无 GPU 或未加载驱动）}"

# 数据源提示（正式数据源 /public/group/lqcd：本地 10 组态 / 服务器 100+ 组态）
if [ -n "${test7_BASELINE:-}" ] || [ -n "${TEST7_BASELINE:-}" ]; then
  echo "[env] 数据源: ${test7_BASELINE:-$TEST7_BASELINE}"
else
  echo "[env] 数据源: 正式数据源 /public/group/lqcd（本地与服务器路径一致）"
fi

# --server：nohup 后台运行（正式跑；日志实时落盘，tail -f 调控）
if [ "${1:-}" = "--server" ]; then
  nohup bash "$0" --inner >"$WORK/run-server-$TS.log" 2>&1 &
  echo "test7 后台运行中 PID=$!（日志: $WORK/run-server-$TS.log）"
  echo "调控: tail -f $WORK/run-server-$TS.log  /  kill $! 停止"
  exit 0
fi

DRY="${DRY:-}"
[ "$DRY" = "1" ] || [ "$DRY" = "--dry-run" ] || [ "${1:-}" = "--dry-run" ] && DRY=1 || DRY=0

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
echo "=== test7 server runner start $(date) ==="
echo "版本目录: $VDIR"

step "Step 0: 环境自检（含 GPU 探测 + 数据源 100 组态预检）"
run 180 python "$MAIN" env

step "Step 1: 检查 + 自行计算 corr/ops + 整理（收缩管线：单组态约 20 分钟，10 组态约 3h，100 组态数小时至 1 天+）"
run "${test7_MAKEDATA_TIMEOUT:-172800}" python "$MAIN" makedata --outdir "$DATA_DIR"

step "Step 2: 全功能实战（02_ratio→03_ana_ratio→04_energy(P2/P0)→06_fh→05_ana3dir）"
run 28800 python "$MAIN" run --data-root "$DATA_DIR" --outdir "$VDIR"

step "Step 3: 断言验证（产物存在 + 物理合理性 meff≈1.12 GeV）"
run 600 python "$MAIN" verify --run-dir "$VDIR" --data-root "$DATA_DIR"

step "Step 4: 断言门"
run 120 python "$MAIN" check --run-dir "$VDIR" --label "test7 check"

step "Step 5: 产物清单"
run 120 python "$MAIN" collect --run-dir "$VDIR"

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
echo "版本目录: $VDIR"
