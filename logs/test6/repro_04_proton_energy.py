#!/usr/bin/env python3
"""
04_proton_energy 独立复现（驱动层，调用本库 pyqcd）
=====================================================

目标：调用本库 pyqcd 独立复现 refer/huangcl/04_proton_energy/code_proton_energy.py
      （输入参数与输入数据路径保持一致，不 import refer/，不照抄示例代码）。

- 统计核心（重采样 / 标准误）与图表（误差棒 / 散点 / 峰值内存）全部调用 pyqcd；
- z 方向直接调用 pyqcd.analysis.compute_corr2（其内部路径恰好与 refer 的
  momsmear2z + Px0Py0Pz6 一致）；x/y 方向 pyqcd 无方向感知接口，
  驱动层按数据目录约定（momsmear{momP}{dir} + 动量置换）加载后复用 pyqcd.resample；
- 行为对齐说明：refer 脚本以 `resample(_corr2_ave, jack, Nsample)` 位置传参调用
  （98_tools 签名 resample(corr, Nsample, jackknife)），实际执行的是 jackknife
  重采样（Nsample 被忽略），复现以实测行为为准 → 全部方向 jackknife=True；
- 输出约定与 refer 一致：<cwd>/1_result/L24x72/Pz6/
  {corr2_x,corr2_y,corr2_z,corr2_ave}.npy + eff_mass.png + sem_comparison.png。
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqcd.analysis import (  # noqa: E402
    EnergyParams,
    compute_corr2,
    resample,
    sem,
    plot_errbar,
    plot_scatter,
    get_peak_memory_gb,
)

# ===== 输入参数（与 refer 一致） =====
DATA_ROOT = "/public/group/lqcd/donghx/2pt_Result"
CONF_SHORT = "L24x72"
CONF_NAME = "beta6.20_mu-0.2770_ms-0.2400_L24x72"
CONF_IDS = [x for x in range(4050, 48001, 50) if x != 12300]
NT = 72
NX = 24
MOMP = 2
PX, PY, PZ = 0, 0, 6
NSAMPLE = 3000
DT_MAX = 20
# =====================================

# 画图参数（refer _plotpa_config[6]）
XLIM = [-0.5, 15.5]
YLIM = [0.9, 1.4]
SEM_YLIM = [-0.01, 0.1]
X_OFFSET = 0.2


def dir_momentum(d: str):
    """方向 → 动量分量（与 refer 一致：x/y/z 依次置换 Pz/Px/Py）。"""
    if d == "x":
        return (PZ, PX, PY)
    if d == "y":
        return (PY, PZ, PX)
    if d == "z":
        return (PX, PY, PZ)
    raise ValueError(f"unknown dir: {d}")


def load_dir_corr2(d: str) -> np.ndarray:
    """读 2pt 切片 → 平移不变相对时间 → ti 平均 → pyqcd jackknife 重采样。
    数据路径约定与 refer 完全一致；统计核心调用 pyqcd.resample。"""
    Px, Py, Pz = dir_momentum(d)
    n_conf = len(CONF_IDS)
    _corr = np.zeros((n_conf, NT, NT), dtype=complex)
    for i, conf_id in enumerate(CONF_IDS):
        _corr[i] = np.load(os.path.join(
            DATA_ROOT, CONF_NAME, f"momsmear{MOMP}{d}", str(conf_id),
            f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}"
            f"_eginphase2_Cg5g4_nopol_ss_conf{conf_id}.npy"))
    _corr2_rel = np.zeros((n_conf, NT, DT_MAX), dtype=complex)
    for ti in range(NT):
        _corr2_rel[:, ti, :] = np.roll(_corr[:, :, ti], shift=-ti, axis=1)[:, :DT_MAX]
    _corr2_ave = _corr2_rel.mean(1)
    del _corr, _corr2_rel
    gc.collect()
    return resample(_corr2_ave, jackknife=True, Nsample=NSAMPLE).real


def compute_all_corr2() -> dict:
    """三方向 corr2 + 平均。z 方向调用 pyqcd.compute_corr2，x/y 走驱动层。"""
    params = EnergyParams(
        conf_short=CONF_SHORT, conf_name=CONF_NAME, conf_ids=CONF_IDS,
        Nt=NT, Nx=NX, Px=PX, Py=PY, Pz=PZ, Nsample=NSAMPLE, dt_max=DT_MAX)
    result = {}
    for d in ("x", "y", "z"):
        t0 = time.perf_counter()
        print(f"==================== compute_corr2 ({d}) start ====================")
        if d == "z":
            result[d] = compute_corr2(DATA_ROOT, params, jack=True)
        else:
            result[d] = load_dir_corr2(d)
        print(f"corr2 ({d}) shape: {result[d].shape}")
        print(f"corr2 ({d}) time: {time.perf_counter() - t0:.2f}s")
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"==================== compute_corr2 ({d}) end ====================")
    result["ave"] = (result["x"] + result["y"] + result["z"]) / 3.0
    return result


def plot_all(corr2: dict, out_dir: str):
    """eff_mass.png + sem_comparison.png（pyqcd.plot_errbar / plot_scatter）。"""
    t0 = time.perf_counter()
    x_vals = np.arange(DT_MAX)
    effmass_data = {}
    sem_data = {}
    for label, _corr2 in [("xdir", corr2["x"]), ("ydir", corr2["y"]),
                          ("zdir", corr2["z"]), ("ave", corr2["ave"])]:
        mass = np.log(_corr2 / np.roll(_corr2, shift=-1, axis=1))
        err = sem(mass, jackknife=False)
        effmass_data[label] = (mass.mean(0), err)
        sem_data[label] = err

    title = (f"{CONF_SHORT}, P=({PX},{PY},{PZ}), "
             f"Nconf={len(CONF_IDS)}, Nsample={NSAMPLE}")

    plot_errbar(x_vals, effmass_data,
                save_path=os.path.join(out_dir, "eff_mass.png"),
                xlabel="t/a", ylabel="aE", xlim=XLIM, ylim=YLIM,
                x_offset=X_OFFSET, title=title)

    t_max_sem = 15
    x_sem = np.arange(t_max_sem)
    sem_scatter = {k: v[:t_max_sem] for k, v in sem_data.items()}
    plot_scatter(x_sem, sem_scatter,
                 save_path=os.path.join(out_dir, "sem_comparison.png"),
                 xlabel="t/a", ylabel="SEM(aE)", xlim=XLIM, ylim=SEM_YLIM,
                 x_offset=X_OFFSET, title=title)
    print(f"plot time: {time.perf_counter() - t0:.2f}s\n")


def main():
    print("jackknife: True (refer 位置参数语义实际执行 jackknife 重采样)")
    print("Nconf:", len(CONF_IDS))
    print("Nsample:", NSAMPLE)
    print("conf_short:", CONF_SHORT)
    out_dir = os.path.join(os.getcwd(), "1_result", CONF_SHORT, f"Pz{PZ}")
    print("result base:", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Part 1: compute corr2 ----
    t0 = time.perf_counter()
    corr2 = compute_all_corr2()
    for name in ("x", "y", "z", "ave"):
        np.save(os.path.join(out_dir, f"corr2_{name}.npy"), corr2[name])
    print(f"corr2 arrays saved to {out_dir}")
    print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
    print(f"corr2 time: {time.perf_counter() - t0:.2f}s\n")

    # ---- Part 2: fit（refer 为占位符 pass，不涉及）----

    # ---- Part 3: plot ----
    plot_all(corr2, out_dir)
    print("job finish")


if __name__ == "__main__":
    main()
