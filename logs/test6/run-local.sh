#!/usr/bin/env bash
# test6 —— 用 pyqcd 独立复现 refer/huangcl/04_proton_energy/code_proton_energy.py
# 全链：三方向 corr2 → 平台拟合 → 7 图 + 报告 → 数值比对（vs refer 实跑真值）+ 物理断言。
# 用法：bash logs/test6/run-local.sh
set -uo pipefail

REPO="${HOME}/PyQCD"
WORK="$REPO/logs/test6"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$WORK/run-local-$TS.log"

cd "$WORK" || exit 1
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== test6 local runner start $(date) ==="
echo "工作目录: $WORK"

step() { echo; echo "===== $1 ====="; }
run() {
  local tmo="$1"; shift
  echo ">>> timeout $tmo $*"
  timeout "$tmo" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[warn] 上一步失败 rc=$rc（继续）" >&2; fi
  return $rc
}

step "Step 0: 环境自检"
run 120 python -c "import numpy, matplotlib, gvar, lsqfit; import pyqcd; print('deps OK')"

step "Step 1: 全链（corr2 → E0 拟合 → 7 图 + 报告）"
run 1800 python "$MAIN"

step "Step 2: 数值比对（vs refer 实跑真值 .ref_run/）"
run 300 python "$WORK/verify_04_repro.py" || { echo "[FAIL] 数值比对未通过"; exit 1; }

step "Step 3: 物理断言"
run 300 python - <<'EOF'
import os, sys
import numpy as np
base = "1_result/L24x72/Pz6"
ok = True
def check(cond, msg):
    global ok
    print(("PASS" if cond else "FAIL") + ": " + msg)
    ok &= cond

c = {d: np.load(f"{base}/corr2_{d}.npy") for d in ("x", "y", "z", "ave")}
check(all(v.shape == (879, 20) and np.isrealobj(v) for v in c.values()),
      "corr2_x/y/z/ave shape=(879,20) 实数")
for png in ("eff_mass.png", "sem_comparison.png", "eff_mass_GeV.png",
            "eff_mass_fit_dirs.png", "corr2_raw.png", "meff_corr.png",
            "meff_hist.png"):
    check(os.path.getsize(f"{base}/{png}") > 10_000, f"{png} 存在且非空")
check(os.path.getsize(f"{base}/2_fit_report.txt") > 500, "2_fit_report.txt 存在")
fit = np.load(f"{base}/1_fit_data.npz")
e0 = {d: fit[f"E0_{d}"] for d in ("x", "y", "z", "ave")}
sem = {d: v.std(0) * np.sqrt(v.shape[0] - 1) for d, v in e0.items()}
check(all(abs(v.mean() * 1.871 - 3.14) < 0.35 for v in e0.values()),
      "E0(GeV) 与色散预期 3.14 GeV 偏差 < 0.35 GeV")
check(max(e0[d].mean() for d in e0) - min(e0[d].mean() for d in e0) < 0.1,
      "三方向+ave E0 互差 < 0.1 a^-1")
meff = {d: np.log(np.abs(c[d]) / np.abs(np.roll(c[d], -1, axis=1))).mean(0)
        for d in ("x", "y", "z", "ave")}
check(all(1.3 <= meff[d][3:8].mean() <= 1.8 for d in meff),
      "meff 平台(t=3..7) aE 落在 Pz6 预期 [1.3, 1.8]")
print("物理断言: " + ("全部 PASS" if ok else "存在 FAIL"))
sys.exit(0 if ok else 1)
EOF
[ $? -ne 0 ] && exit 1

step "Step 4: 产物清单"
run 60 bash -c "ls -la 1_result/L24x72/Pz6/ | awk '{print \$5, \$9}'"

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
