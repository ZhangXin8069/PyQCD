#!/usr/bin/env python3
"""
test6 —— 用 pyqcd 独立复现 refer/huangcl/04_proton_energy/code_proton_energy.py
=================================================================================

功能对齐（输入参数与数据路径与原脚本一致）：

- Part 1 compute：读三方向 (x/y/z) momsmear2 2pt 切片 → 平移不变相对时间 →
  ti 平均 → jackknife 重采样 → corr2_x/y/z/ave.npy（(Nsample, dt_max)）。
- Part 2 fit：平台拟合 C(t) = c0·e^{−E0·t}·(1 + c1·e^{−dE·t})
  （lsqfit 逐样本，svdcut=1e-6）→ 1_fit_data.npz + 2_fit_report.txt。
- Part 3 plot：eff_mass.png（aE，x/y/z/ave 对比）+ sem_comparison.png；
  扩展图（参考 refer/huangcl/code.py 拟合色带与 05_ana_3dir 差异分析）：
  eff_mass_GeV.png / eff_mass_fit_dirs.png / corr2_raw.png /
  meff_corr.png / meff_hist.png。

统计（sem/resample/cov_mat）、图表（plot_errbar/plot_scatter）与报告
（fit_report_lines/make_summary_table）全部复用 pyqcd.analysis 既有模块，
不 import 也不照抄 refer/ 任何脚本。

用法：
    python main.py [--debug] [--parts S-E] [--conf-ids a,b,...]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = str(Path(__file__).resolve().parent.parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pyqcd.analysis._disconnected import cov_mat, resample, sem  # noqa: E402
from pyqcd.analysis._fitter import (  # noqa: E402
    fit_report_lines,
    make_summary_table,
)
from pyqcd.analysis._plots import (  # noqa: E402
    DEFAULT_PLOT_COLORS,
    get_peak_memory_gb,
    plot_errbar,
    plot_scatter,
)

_DATA_ROOT = "/public/group/lqcd/donghx/2pt_Result"


# ===== 参数 dataclass =====

@dataclass
class SampleParams:
    conf_short: str
    conf_name: str
    conf_ids: list
    Nt: int
    Nx: int
    momP: int
    Px: int
    Py: int
    Pz: int
    Nsample: int
    dt_max: int

    @property
    def Nconf(self):
        return len(self.conf_ids)

    @property
    def unit(self):
        """格点能 → GeV。"""
        return 0.197 / 0.1053


@dataclass
class FitParams:
    p0: dict = field(default_factory=lambda: {
        "c0": 0.6, "c1": 0.6, "E0": 1.5, "dE": 0.4})
    dt_start: int = 3
    dt_end: int = 7


@dataclass
class PlotParams:
    xlim: list
    ylim: list
    sem_ylim: list
    x_offset: float = 0.2


# ===== 参数块（与 code_proton_energy.py 一致）=====

def make_params(conf_ids, debug=False):
    sampa = SampleParams(
        conf_short="L24x72",
        conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72",
        conf_ids=conf_ids,
        Nt=72,
        Nx=24,
        momP=2,
        Px=0,
        Py=0,
        Pz=6,
        Nsample=3000,
        dt_max=20,
    )
    plotpa = PlotParams(
        xlim=[-0.5, 15.5],
        ylim=[1.3, 1.8],
        sem_ylim=[-0.01, 0.1],
        x_offset=0.2,
    )
    if debug:
        sampa.conf_ids = sampa.conf_ids[:5]
    return sampa, plotpa


# ===== Part 1: corr2 =====

def momentum_for(sampa: SampleParams, dir: str):
    """按方向置换动量分量（与 ratio 代码一致）。"""
    if dir == "x":
        return sampa.Pz, sampa.Px, sampa.Py
    if dir == "y":
        return sampa.Py, sampa.Pz, sampa.Px
    if dir == "z":
        return sampa.Px, sampa.Py, sampa.Pz
    raise ValueError(f"unknown dir {dir}")


def raw_path(sampa: SampleParams, data_root: str, conf_id: int, dir: str) -> str:
    Px, Py, Pz = momentum_for(sampa, dir)
    return os.path.join(
        data_root, sampa.conf_name, f"momsmear{sampa.momP}{dir}", str(conf_id),
        f"twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase2_Cg5g4_nopol_ss_"
        f"conf{conf_id}.npy")


def compute_corr2(sampa: SampleParams, data_root: str, dir: str,
                  jack: bool, verbose=True) -> np.ndarray:
    """单方向 2pt → 相对时间 → ti 平均 → jackknife → (Nsample, dt_max)。"""
    if verbose:
        print(f"==================== compute_corr2 ({dir}) start ====================")

    _corr = np.zeros((sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)
    for i, conf_id in enumerate(sampa.conf_ids):
        _corr[i] = np.load(raw_path(sampa, data_root, conf_id, dir))
        if verbose and (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{sampa.Nconf}")
    if verbose:
        print("load finish")
        print("2pt shape:", _corr.shape)

    _corr2_rel = np.zeros((sampa.Nconf, sampa.Nt, sampa.dt_max),
                          dtype=complex)
    for ti in range(sampa.Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :sampa.dt_max]

    _corr2_ave = _corr2_rel.mean(1)
    corr2 = resample(_corr2_ave, jack, sampa.Nsample).real

    del _corr, _corr2_rel, _corr2_ave
    gc.collect()

    if verbose:
        print(f"corr2 ({dir}) shape: {corr2.shape}")
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"==================== compute_corr2 ({dir}) end ====================")
    return corr2


def run_corr2(sampa: SampleParams, data_root: str, out_dir: str, jack: bool):
    """三方向 + 平均 corr2，保存 corr2_x/y/z/ave.npy。"""
    corr2 = {}
    for _dir in ("x", "y", "z"):
        corr2[_dir] = compute_corr2(sampa, data_root, _dir, jack)
    corr2["ave"] = (corr2["x"] + corr2["y"] + corr2["z"]) / 3.0

    for _dir in ("x", "y", "z", "ave"):
        np.save(os.path.join(out_dir, f"corr2_{_dir}.npy"), corr2[_dir])
    return corr2


# ===== Part 2: 平台拟合 =====

def energy_model(x, p):
    """C(t) = c0·e^{−E0·t}·(1 + c1·e^{−dE·t})。"""
    dt = np.asarray(x, dtype=np.float64)
    return (p["c0"] * np.exp(-p["E0"] * dt)
            * (1 + p["c1"] * np.exp(-p["dE"] * dt)))


def do_fit(corr2: np.ndarray, fitpa: FitParams, sampa: SampleParams,
           jack: bool, verbose=True) -> dict:
    """逐样本 lsqfit 平台拟合 → {c0, c1, E0, dE, chi2} 数组。"""
    import gvar as gv
    import lsqfit

    n_sample = corr2.shape[0]
    x_coor = list(range(fitpa.dt_start, fitpa.dt_end + 1))
    Ndata = len(x_coor)
    names = list(fitpa.p0.keys())

    sub_sample = np.zeros((n_sample, Ndata))
    for i, dt in enumerate(x_coor):
        sub_sample[:, i] = corr2[:, dt]
    cov, cond = cov_mat(sub_sample, jack)

    result = {name: np.zeros(n_sample) for name in names}
    result["chi2"] = np.zeros(n_sample)
    for _id in range(n_sample):
        y_coor = gv.gvar(sub_sample[_id], cov)
        _fit = lsqfit.nonlinear_fit(data=(x_coor, y_coor), p0=fitpa.p0,
                                    fcn=energy_model, svdcut=1e-6)
        for name in names:
            result[name][_id] = _fit.pmean[name]
        result["chi2"][_id] = _fit.chi2 / _fit.dof

    if verbose:
        for name in names:
            print(f"{name} = {result[name].mean():.3g} "
                  f"+- {sem(result[name], jack):.3g}")
        print(f"chi2 = {result['chi2'].mean():.3g}")

    return result, cond


def run_fit(corr2: dict, fitpa: FitParams, sampa: SampleParams,
            out_dir: str, jack: bool, verbose=True):
    """对 x/y/z/ave 逐一拟合，写 1_fit_data.npz + 2_fit_report.txt。"""
    if verbose:
        print("==================== do_fit start ====================")
    report = fit_report_lines(
        f"Fit Report  : {sampa.conf_short}", {
            "dt range": f"[{fitpa.dt_start}, {fitpa.dt_end}]",
            "Nsample": corr2["ave"].shape[0],
            "jackknife": jack,
        })
    fits, rows = {}, []
    for _dir in ("x", "y", "z", "ave"):
        res, cond = do_fit(corr2[_dir], fitpa, sampa, jack, verbose=verbose)
        fits[_dir] = res
        report.append("-" * 72)
        report.append(f"dir = {_dir}, condition number = {cond:.3g}")
        report.append("")
        report.append(f"  {_dir}  c0={res['c0'].mean():.3g}("
                      f"{sem(res['c0'], jack) * 1e3:.0f})  "
                      f"E0={res['E0'].mean():.3g}("
                      f"{sem(res['E0'], jack) * 1e3:.0f})  "
                      f"chi2/dof={res['chi2'].mean():.2g}")
        report.append("")
        rows.append([f"{_dir}",
                     f"{res['E0'].mean():.3f}({sem(res['E0'], jack) * 1e3:.0f})",
                     f"{res['E0'].mean() * sampa.unit:.3f}("
                     f"{sem(res['E0'], jack) * sampa.unit * 1e3:.0f})",
                     f"{res['c0'].mean():.3f}({sem(res['c0'], jack) * 1e3:.0f})",
                     f"{res['chi2'].mean():.2g}"])

    report.append("=" * 72)
    report.append("  Summary Table (E0 in lattice & GeV)")
    report.append("=" * 72)
    report.append(make_summary_table(
        ["dir", "E0(a^-1)", "E0(GeV)", "c0", "chi2/dof"], rows))
    report.append("")

    with open(os.path.join(out_dir, "2_fit_report.txt"), "w") as f:
        f.write("\n".join(report))
    np.savez(os.path.join(out_dir, "1_fit_data.npz"),
             **{f"{k}_{d}": v for d, r in fits.items()
                for k, v in r.items()})
    if verbose:
        print(f"report saved to {out_dir}/2_fit_report.txt")
        print(f"fit result saved to {out_dir}/1_fit_data.npz")
        print("==================== do_fit end ====================")
    return fits


# ===== Part 3: 图 =====

def eff_mass(corr2: dict, sampa: SampleParams):
    """meff(t) = log(|C(t)|/|C(t+1)|)，末点卷绕无物理意义。"""
    mass = {}
    for _dir, _c in corr2.items():
        mass[_dir] = np.log(np.abs(_c) / np.abs(np.roll(_c, -1, axis=1)))
    return mass


def plot_part3(corr2: dict, mass: dict, fits: dict, fitpa: FitParams,
               sampa: SampleParams, plotpa: PlotParams, out_dir: str,
               jack: bool, verbose=True):
    if verbose:
        print("==================== plot start ====================")
    time0 = time.perf_counter()
    t0 = time.perf_counter()

    x_vals = np.arange(sampa.dt_max)
    eff_data = {_dir: (mass[_dir].mean(0), sem(mass[_dir], jack))
                for _dir in ("x", "y", "z", "ave")}

    title = (f"{sampa.conf_short}, P=({sampa.Px},{sampa.Py},{sampa.Pz}), "
             f"Nconf={sampa.Nconf}, Nsample={sampa.Nsample}")

    # 图1: eff mass 对比（原版，aE 单位）
    plot_errbar(x_vals, eff_data,
                save_path=os.path.join(out_dir, "eff_mass.png"),
                xlabel="t/a", ylabel="aE",
                xlim=plotpa.xlim, ylim=plotpa.ylim,
                x_offset=plotpa.x_offset, title=title)
    print(f"eff_mass.png done ({time.perf_counter() - t0:.1f}s)")

    # 图2: SEM 对比散点图 t=0~14（原版）
    t_max_sem = 15
    sem_scatter = {_dir: sem(mass[_dir], jack)[:t_max_sem]
                   for _dir in ("x", "y", "z", "ave")}
    plot_scatter(np.arange(t_max_sem), sem_scatter,
                 save_path=os.path.join(out_dir, "sem_comparison.png"),
                 xlabel="t/a", ylabel="SEM(aE)",
                 xlim=plotpa.xlim, ylim=plotpa.sem_ylim,
                 x_offset=plotpa.x_offset, title=title)
    print(f"sem_comparison.png done ({time.perf_counter() - t0:.1f}s)")

    # 图3: ave 有效质量（GeV）+ E0 拟合色带（参考 code.py plot_corr2）
    res = fits["ave"]
    E0, chi2 = res["E0"].mean(0), res["chi2"].mean(0)
    E0_err = sem(res["E0"], jack)
    band = [E0 * sampa.unit - E0_err * sampa.unit,
            E0 * sampa.unit + E0_err * sampa.unit]
    plot_errbar(x_vals,
                {"ave": (mass["ave"].mean(0) * sampa.unit,
                         sem(mass["ave"], jack) * sampa.unit)},
                save_path=os.path.join(out_dir, "eff_mass_GeV.png"),
                xlabel="t/a", ylabel="eff mass (GeV)",
                xlim=plotpa.xlim, ylim=[2.0, 4.2],
                title=(f"{title}\nE0={E0 * sampa.unit:.3f}"
                       f"({E0_err * sampa.unit * 1e3:.0f}) GeV, "
                       f"chi2/dof={chi2:.2f}"),
                show_band=True,
                band_x=np.array([fitpa.dt_start, fitpa.dt_end]),
                band_y_down=[band[0], band[0]], band_y_up=[band[1], band[1]],
                band_label="Fit E0")
    print(f"eff_mass_GeV.png done ({time.perf_counter() - t0:.1f}s)")

    # 图4: 各方向有效质量（GeV）+ 各自拟合色带（2x2）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    colors = DEFAULT_PLOT_COLORS
    for k, (_dir, ax) in enumerate(zip(("x", "y", "z", "ave"), axes.flat)):
        _mass = mass[_dir]
        _res = fits[_dir]
        _E0 = _res["E0"].mean(0) * sampa.unit
        _E0_err = sem(_res["E0"], jack) * sampa.unit
        ax.errorbar(x_vals, _mass.mean(0) * sampa.unit,
                    yerr=sem(_mass, jack) * sampa.unit,
                    fmt="x", color=colors[k], ecolor=colors[k],
                    capsize=0, markersize=6, label=f"{_dir} dir")
        ax.fill_between([fitpa.dt_start, fitpa.dt_end],
                        [_E0 - _E0_err] * 2, [_E0 + _E0_err] * 2,
                        color="gray", alpha=0.35, linewidth=0,
                        label=f"Fit E0={_E0:.3f}")
        ax.set_xlim(plotpa.xlim); ax.set_ylim([2.0, 4.2])
        ax.set_xlabel("t/a"); ax.set_ylabel("eff mass (GeV)")
        ax.set_title(f"{_dir} dir")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eff_mass_fit_dirs.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"eff_mass_fit_dirs.png done ({time.perf_counter() - t0:.1f}s)")

    # 图5: log|corr2| vs t（4 组对比）
    logc = {_dir: (np.log(np.abs(_c)).mean(0), sem(np.log(np.abs(_c)), jack))
            for _dir, _c in corr2.items()}
    plot_errbar(x_vals, logc,
                save_path=os.path.join(out_dir, "corr2_raw.png"),
                xlabel="t/a", ylabel="log|C(t)|",
                xlim=[-0.5, 15.5],
                x_offset=plotpa.x_offset, title=title)
    print(f"corr2_raw.png done ({time.perf_counter() - t0:.1f}s)")

    # 图6: 三方向 meff 相关系数矩阵（平台中点 dt，参考 05_ana_3dir）
    dt_ref = (fitpa.dt_start + fitpa.dt_end) // 2
    arr = np.column_stack([mass[_d][:, dt_ref] for _d in ("x", "y", "z")])
    cov, cond = cov_mat(arr, jack)
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(("x", "y", "z")); ax.set_yticklabels(("x", "y", "z"))
    ax.set_title(f"meff corr matrix @ t={dt_ref}\ncond={cond:.1f}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "meff_corr.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"meff_corr.png done ({time.perf_counter() - t0:.1f}s)")

    # 图7: 各方向 meff 直方图（平台 dt 处，mean±sem 标注，参考 05_ana_3dir）
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    for k, (_dir, ax) in enumerate(zip(("x", "y", "z", "ave"), axes.flat)):
        vals = mass[_dir][:, dt_ref]
        ax.hist(vals, bins=min(30, sampa.Nsample), color=colors[k], alpha=0.7)
        ax.axvline(vals.mean(), color="black", ls="--", lw=1)
        ax.set_title(f"{_dir} dir: {vals.mean():.3f}"
                     f"({sem(vals, jack) * 1e3:.0f})")
        ax.set_xlabel("aE")
    fig.suptitle(f"meff @ t={dt_ref} (mean±sem)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "meff_hist.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"meff_hist.png done ({time.perf_counter() - t0:.1f}s)")

    print(f"plot done, time: {time.perf_counter() - time0:.1f}s")
    if verbose:
        print("==================== plot end ====================")


# ===== main =====

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true",
                        help="前 5 个组态，输出到 0_debug/")
    parser.add_argument("--parts", default="1-3", help="起始-结束部分 (1=corr2,2=fit,3=plot)")
    parser.add_argument("--data-root", default=_DATA_ROOT)
    parser.add_argument("--conf-ids", default="",
                        help="覆盖 conf_ids（逗号分隔）")
    args = parser.parse_args()

    parts = [int(x) for x in args.parts.split("-")]
    jack = True

    if args.conf_ids:
        conf_ids = [int(x) for x in args.conf_ids.split(",")]
    else:
        conf_ids = [x for x in range(4050, 48001, 50) if x != 12300]

    sampa, plotpa = make_params(conf_ids, debug=args.debug)
    fitpa = FitParams()
    base = "0_debug" if args.debug else "1_result"
    out_dir = os.path.join(os.getcwd(), base, sampa.conf_short, f"Pz{sampa.Pz}")
    os.makedirs(out_dir, exist_ok=True)

    print("jackknife:", jack)
    print("Nconf:", sampa.Nconf)
    print("Nsample:", sampa.Nsample)
    print("conf_short:", sampa.conf_short)
    print("result base:", out_dir)

    corr2 = {}
    if parts[0] <= 1:
        time0 = time.perf_counter()
        corr2 = run_corr2(sampa, args.data_root, out_dir, jack)
        print(f"corr2 arrays saved to {out_dir}")
        print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
        print(f"corr2 time: {time.perf_counter() - time0:.2f}s\n")
        if parts[1] == 1:
            print("job finish"); return
    else:
        print("===== skip compute corr2, loading from file =====")
        for _dir in ("x", "y", "z", "ave"):
            corr2[_dir] = np.load(os.path.join(out_dir, f"corr2_{_dir}.npy"))
        print(f"corr2 loaded, shape: {corr2['ave'].shape}")

    fits = {}
    if parts[0] <= 2:
        time0 = time.perf_counter()
        fits = run_fit(corr2, fitpa, sampa, out_dir, jack)
        print(f"fit time: {time.perf_counter() - time0:.2f}s\n")
        if parts[1] == 2:
            print("fit finish"); return
    else:
        print("===== skip do_fit, loading fit result from file =====")
        data = np.load(os.path.join(out_dir, "1_fit_data.npz"))
        fits = {_dir: {k: data[f"{k}_{_dir}"]
                       for k in (*fitpa.p0.keys(), "chi2")}
                for _dir in ("x", "y", "z", "ave")}

    if parts[0] <= 3:
        mass = eff_mass(corr2, sampa)
        plot_part3(corr2, mass, fits, fitpa, sampa, plotpa, out_dir, jack)

    print("job finish")


if __name__ == "__main__":
    main()
