#!/usr/bin/env python3
"""
04_proton_energy 复现数值比对
==============================

比对对象：
- 参考真值: .ref_run/1_result/L24x72/Pz6/（refer 脚本原样实跑产物）
- 复现产物: 1_result/L24x72/Pz6/（本驱动调用 pyqcd 的产物）

判据（预授权通过标准）：各 dt 点 |Δ| ≤ 3×SEM；并报告逐位/相对差异上限。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REF = BASE / ".ref_run" / "1_result" / "L24x72" / "Pz6"
OUR = BASE / "1_result" / "L24x72" / "Pz6"
NAMES = ["corr2_x", "corr2_y", "corr2_z", "corr2_ave"]
TOL_SIGMA = 3.0


def load(p: Path, name: str) -> np.ndarray:
    return np.load(p / f"{name}.npy")


def main() -> int:
    lines = ["=" * 78,
             "  04_proton_energy 复现比对报告（refer 实跑 vs pyqcd 驱动）",
             "=" * 78]
    lines.append(f"  参考: {REF}")
    lines.append(f"  复现: {OUR}")
    lines.append(f"  判据: 各 dt 点 |Δ| <= {TOL_SIGMA}×SEM\n")

    ok_all = True
    for name in NAMES:
        ref = load(REF, name)
        our = load(OUR, name)
        same_shape = ref.shape == our.shape and ref.dtype == our.dtype
        lines.append(f"---- {name}: shape ref={ref.shape} our={our.shape} "
                     f"dtype={ref.dtype} 形状一致={same_shape}")
        if not same_shape:
            ok_all = False
            continue

        diff = our - ref
        abs_diff = np.abs(diff)
        max_ad = abs_diff.max()
        max_rd = (abs_diff / np.maximum(np.abs(ref), 1e-300)).max()
        sem_ = our.std(0)
        ratio = np.where(sem_ > 0, abs_diff / sem_, 0.0)
        lines.append(f"  最大 |Δ|      = {max_ad:.6e}")
        lines.append(f"  最大相对 |Δ|  = {max_rd:.6e}")
        lines.append(f"  最大 |Δ|/SEM  = {ratio.max():.6f}  (dt={ratio.argmax()})")
        bad = np.where(ratio > TOL_SIGMA)[0]
        if bad.size == 0:
            lines.append(f"  PASS: 全部 dt 点 |Δ| <= {TOL_SIGMA}×SEM\n")
        else:
            ok_all = False
            lines.append(f"  FAIL: dt={bad.tolist()} 超出容差\n")

    # eff mass 曲线（与脚本一致：log 比值，NaN 模式需一致）
    lines.append("---- eff mass 曲线（mean 与 NaN 模式）----")
    for label, k in [("xdir", "x"), ("ydir", "y"), ("zdir", "z"), ("ave", "ave")]:
        r, o = load(REF, f"corr2_{k}"), load(OUR, f"corr2_{k}")
        mr = np.log(r / np.roll(r, shift=-1, axis=1)).mean(0)
        mo = np.log(o / np.roll(o, shift=-1, axis=1)).mean(0)
        nan_r, nan_o = np.isnan(mr), np.isnan(mo)
        both_fin = np.isfinite(mr) & np.isfinite(mo)
        d = np.abs(mr[both_fin] - mo[both_fin]) if both_fin.any() else 0.0
        dmax = d.max() if np.ndim(d) else d
        pat_ok = np.array_equal(nan_r, nan_o)
        lines.append(f"  {label}: NaN 模式一致={pat_ok}  "
                     f"有限点最大 |Δmeff|={dmax:.3e}  meff[3:6]={mo[3:6]}")
        ok_all &= pat_ok

    lines.append("=" * 78)
    lines.append(f"总判定: {'PASS' if ok_all else 'FAIL'}")
    report = "\n".join(lines) + "\n"
    print(report)
    out = OUR / "verify_report.txt"
    out.write_text(report)
    print(f"报告已写入: {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
