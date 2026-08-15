"""
02_ratio 功能：3pt/2pt 比值 R(z) 计算 + 逐 z 多参数拟合 + 画图（独立实现）
============================================================================

功能对齐 refer/huangcl/02_ratio/code.py：

- Part 1 compute：读 2pt 切片（twopt_slice_pp_*）与 3 个 OPE（ops_mu0_nu1 /
  ops_mu3_nu0 / ops_mu3_nu1）→ 组合 O = −O30 − O31 + 2·O01 → 平移不变相对
  时间构造 C3 = C2×OPE（不相连因子化）→ jackknife/bootstrap 重采样 →
  真空扣除 ratio = ⟨(C3 − C2·⟨OPE⟩)/C2⟩_ti → 保存 ratio_Pz*_Nsam*_dtmax*.npy。
- Part 2 fit：模型 R = c0 + c1·e^{−dE·dtau} + c1·e^{−dE·(dt−dtau)}，逐 z、
  逐样本 lsqfit 拟合（prior 优先）→ 0_fit_data.npz + 1_fit_report.txt。
- Part 3 plot：ratio.png（逐 z 子图 + Fit c0 色带）、c0.png、chi2.png。

数据读取路径模板可注入（data_root 下按 conf_id 组织），核心计算自包含。
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field

import numpy as np

from ._disconnected import sem, resample, cov_mat, model_ratio
from ._fitter import (FitParams, calc_chi2_dof, fit, fit_report_lines,
                      make_summary_table)


@dataclass
class SampleParams2pt:
    """样本与系综参数。"""

    conf_short: str
    conf_name: str
    conf_ids: list
    Nt: int
    Nx: int
    Px: int
    Py: int
    Pz: int
    Nsample: int
    dt_max: int

    @property
    def Nconf(self):
        return len(self.conf_ids)


@dataclass
class PlotParamsRatio:
    """ratio/c0/chi2 画图参数。"""

    plot_z: int = 2
    dt_list: list = field(default_factory=lambda: list(range(6, 13)))
    z_list: list = field(default_factory=lambda: list(range(0, 24, 4)))
    xlim: list = field(default_factory=lambda: [-7, 7])
    ylim: list = field(default_factory=lambda: [-0.1, 1.0])
    c0_ylim: list = field(default_factory=lambda: [-0.2, 1.0])
    colors: list = field(default_factory=lambda: [
        "#b3d9ff", "#c7e9c0", "#fdd49e", "#d4b9da", "#99d8c9",
        "#6baed6", "#41ab5d", "#ef6548", "#8856a7", "#08306b",
    ])


def ope_combine(ope01, ope30, ope31):
    """组合胶子 OPE：O = −O30 − O31 + 2·O01，转置到 (conf, tau, z)。"""
    return (-ope30 - ope31 + 2 * ope01).transpose(0, 2, 1)


def load_raw(data_root, conf_id, sampa: SampleParams2pt, tdir=(0, 1),
             dz=None):
    """读取单个组态的 2pt 切片与 OPE。

    数据布局（参考集群约定，可参数化）:
        {data_root}/{conf_name}/momsmear2z/{conf_id}/twopt_slice_pp_*_{conf_id}.npy
        {data_root}/{conf_short}/zdir/{conf_id}/ops_mu{a}_nu{b}_dz{dz}_conf{id}.npz["ops"]
    """
    dz = dz or sampa.Nx
    mom = f"Px{sampa.Px}Py{sampa.Py}Pz{sampa.Pz}"
    corr_path = os.path.join(
        data_root, sampa.conf_name, "momsmear2z", str(conf_id),
        f"twopt_slice_pp_{mom}_eginphase2_Cg5g4_nopol_ss_conf{conf_id}.npy")
    ope_dir = os.path.join(data_root, sampa.conf_short, "zdir", str(conf_id))
    mu1, nu1 = tdir
    ope01 = np.load(os.path.join(
        ope_dir, f"ops_mu{mu1}_nu{nu1}_dz{dz}_conf{conf_id}.npz"))["ops"]
    ope30 = np.load(os.path.join(
        ope_dir, f"ops_mu3_nu{mu1}_dz{dz}_conf{conf_id}.npz"))["ops"]
    ope31 = np.load(os.path.join(
        ope_dir, f"ops_mu3_nu{nu1}_dz{dz}_conf{conf_id}.npz"))["ops"]
    return np.load(corr_path), ope01, ope30, ope31


def compute_ratio(data_root, sampa: SampleParams2pt, jack: bool,
                  verbose=True) -> np.ndarray:
    """加载 2pt + OPE 计算 ratio 数组。

    Returns
    -------
    ratio : (Nsample, dt_max, dt_max, Nx)，real。
    """
    if verbose:
        print("==================== compute ratio start ====================")

    _corr = np.zeros((sampa.Nconf, sampa.Nt, sampa.Nt), dtype=complex)
    _ope_01 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_30 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)
    _ope_31 = np.zeros((sampa.Nconf, sampa.Nx, sampa.Nt), dtype=complex)

    for i, conf_id in enumerate(sampa.conf_ids):
        c, o01, o30, o31 = load_raw(data_root, conf_id, sampa)
        _corr[i] = c
        _ope_01[i] = o01
        _ope_30[i] = o30
        _ope_31[i] = o31
    if verbose:
        print("load finish")
        print("2pt shape:", _corr.shape)
        print("ope shape:", _ope_01.shape)

    _ope = ope_combine(_ope_01, _ope_30, _ope_31)   # (Nconf, tau, z)

    _corr2_rel = np.zeros((sampa.Nconf, sampa.Nt, sampa.dt_max), dtype=complex)
    _ope_rel = np.zeros((sampa.Nconf, sampa.Nt, sampa.dt_max, sampa.Nx),
                        dtype=complex)
    for ti in range(sampa.Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :sampa.dt_max]
        ope_shift = np.roll(_ope, shift=-ti, axis=1)
        _ope_rel[:, ti, :, :] = ope_shift[:, :sampa.dt_max, :]

    _corr3 = np.zeros((sampa.Nconf, sampa.Nt, sampa.dt_max,
                       sampa.dt_max, sampa.Nx), dtype=complex)
    for _dt in range(sampa.dt_max):
        for _dtau in range(_dt + 1):
            _corr3[:, :, _dt, _dtau, :] = (
                _ope_rel[:, :, _dtau, :] * _corr2_rel[:, :, _dt][:, :, None])

    del _corr, _ope, _ope_30, _ope_31, _ope_01
    gc.collect()

    corr2 = resample(_corr2_rel, jack, sampa.Nsample)
    ope = resample(_ope_rel, jack, sampa.Nsample)
    corr3 = resample(_corr3, jack, sampa.Nsample)

    del _corr2_rel, _ope_rel, _corr3
    gc.collect()

    corr3_disc = corr3 - corr2[:, :, :, None, None] * ope[:, :, None, :, :]
    eps = 1e-30   # 除零保护（与 _disconnected.py 一致）
    ratio = np.mean(corr3_disc / (corr2[:, :, :, None, None] + eps), axis=1)
    ratio = ratio.real
    if verbose:
        print("ratio shape:", ratio.shape)
        print("==================== compute ratio end ====================")
    return ratio


def ratio_file_name(sampa: SampleParams2pt) -> str:
    """ratio 保存文件名（参考约定，含 Pz/Nsample/dt_max）。"""
    return (f"ratio_Pz{sampa.Pz}_Nsam{sampa.Nsample}"
            f"_dtmax{sampa.dt_max}.npy")


def fit_dir_name(sampa: SampleParams2pt, fitpa: FitParams, tag: str = "fit") -> str:
    """拟合目录名：fit_Pz*_Nsam*_dtmax*_tsep{}_{}_nex{}。"""
    return (f"{tag}_Pz{sampa.Pz}_Nsam{sampa.Nsample}_dtmax{sampa.dt_max}"
            f"_tsep{fitpa.dt_start}_{fitpa.dt_end}_nex{fitpa.nex}")


def fit_x_coor(fitpa: FitParams) -> list:
    """拟合坐标 (dt, dtau) 列表：dtau ∈ [nex, dt−nex]。"""
    return [(dt, dtau)
            for dt in range(fitpa.dt_start, fitpa.dt_end + 1)
            for dtau in range(fitpa.nex, dt - fitpa.nex + 1)]


def do_fit_and_report(ratio, fitpa: FitParams, sampa: SampleParams2pt,
                      fit_dir: str, jack: bool, verbose=True) -> dict:
    """逐 z、逐样本拟合，输出 0_fit_data.npz + 1_fit_report.txt。

    Returns
    -------
    fit_result : {c0, c1, dE, chi2}，各 (Nsample, Nx)。
    """
    if verbose:
        print("==================== do_fit start ====================")

    x_coor = fit_x_coor(fitpa)
    Ndata = len(x_coor)
    param_names = list(fitpa.p0.keys())

    all_fit_result = {name: np.zeros((sampa.Nsample, sampa.Nx))
                      for name in param_names + ["chi2"]}
    all_cond = np.zeros(sampa.Nx)

    report_lines = fit_report_lines(
        f"Fit Report: {sampa.conf_short}", {
            "t_sep range": f"[{fitpa.dt_start}, {fitpa.dt_end}]",
            "nex": fitpa.nex,
            "Nsample": sampa.Nsample,
            "jackknife": jack,
        })

    for _z in range(sampa.Nx):
        t0_fit = time.perf_counter()
        sub_sample = np.zeros((sampa.Nsample, Ndata))
        for i, (dt, dtau) in enumerate(x_coor):
            sub_sample[:, i] = ratio[:, dt, dtau, _z]

        _fit_result, _cov, _cond, _last_fit = fit(
            sub_sample, x_coor, model_ratio, fitpa, jack)
        for name in param_names + ["chi2"]:
            all_fit_result[name][:, _z] = _fit_result[name]
        all_cond[_z] = _cond

        if verbose:
            print(f"z={_z}")
            for name in param_names:
                print(f"{name} = {all_fit_result[name][:, _z].mean():.3g} "
                      f"+- {sem(all_fit_result[name][:, _z], jack):.3g}")
            print(f"chi2 = {all_fit_result['chi2'][:, _z].mean():.3g}")
            print(f"fit z = {_z}, time: {time.perf_counter() - t0_fit:.2f}s\n")

        report_lines.append(f"z = {_z}")
        report_lines.append("-" * 72)
        report_lines.append(f"condition number = {_cond:.3g}")
        report_lines.append("")
        if _last_fit is not None:
            report_lines.append(_last_fit.format(maxline=True))
        report_lines.append("")

    # ---- Summary Table ----
    report_lines.append("=" * 72)
    report_lines.append("  Summary Table")
    report_lines.append("=" * 72)
    summary_rows = []
    for _z in range(sampa.Nx):
        row = [str(_z)]
        for name in param_names:
            mean = all_fit_result[name][:, _z].mean()
            err = sem(all_fit_result[name][:, _z], jack)
            row.append(f"{mean:.3f}({err * 1e3:.0f})")
        row.append(f"{all_fit_result['chi2'][:, _z].mean():.2g}")
        summary_rows.append(row)
    report_lines.append(make_summary_table(
        ["z"] + param_names + ["chi2/dof"], summary_rows))
    report_lines.append("")

    with open(os.path.join(fit_dir, "1_fit_report.txt"), "w") as f:
        f.write("\n".join(report_lines))
    np.savez(os.path.join(fit_dir, "0_fit_data.npz"), **all_fit_result)
    if verbose:
        print(f"report saved to {fit_dir}/1_fit_report.txt")
        print(f"fit result saved to {fit_dir}/0_fit_data.npz")
        print("==================== do_fit end ====================")
    return all_fit_result


def plot_ratio_fits(ratio, fit_result, sampa: SampleParams2pt,
                    fitpa: FitParams, plotpa: PlotParamsRatio,
                    out_dir: str, jack: bool, verbose=True) -> list:
    """绘制 ratio.png（逐 z 子图 + Fit c0 色带）、c0.png、chi2.png。

    Returns
    -------
    list: 保存的图片路径列表。
    """
    from ._plots import plot_single_chi2, plot_single_errbar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if verbose:
        print("==================== plot_ratio start ====================")
    time0 = time.perf_counter()

    para_c0 = fit_result["c0"]
    chi2 = fit_result["chi2"]
    ratio_mean = ratio.mean(0)
    ratio_err = sem(ratio, jack)
    c0_mean = para_c0.mean(0)
    c0_err = sem(para_c0, jack)
    chi2_mean = chi2.mean(0)

    n_z = len(plotpa.z_list)
    n_cols = 3
    n_rows = int(np.ceil(n_z / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 4 * n_rows),
                             sharex=True, sharey=True, dpi=150)
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, _z in enumerate(plotpa.z_list):
        ax = axes_flat[idx]
        band_down = c0_mean[_z] - c0_err[_z]
        band_up = c0_mean[_z] + c0_err[_z]

        for i, dt in enumerate(plotpa.dt_list):
            tau_vals = np.arange(0, dt + 1)
            x_vals = tau_vals - dt / 2.0
            color = plotpa.colors[i % len(plotpa.colors)]
            ax.errorbar(x_vals, ratio_mean[dt, tau_vals, _z],
                        yerr=ratio_err[dt, tau_vals, _z],
                        fmt="x", color=color, ecolor=color, capsize=0,
                        markersize=7, markeredgewidth=1.8, linewidth=1.2,
                        zorder=3, label=f"dt={dt}")

        x_band = np.array(plotpa.xlim)
        ax.fill_between(x_band, [band_down, band_down], [band_up, band_up],
                        color="gray", alpha=0.35, linewidth=0, zorder=1,
                        label="Fit c0")

        c0_str = f"{para_c0[:, _z].mean():.3f}({sem(para_c0[:, _z], jack) * 1e3:.0f})"
        ax.set_title(f"z={_z}, c0={c0_str}, chi2={chi2_mean[_z]:.2f}",
                     fontsize=12)
        ax.set_xlim(plotpa.xlim[0], plotpa.xlim[1])
        ax.set_ylim(plotpa.ylim[0], plotpa.ylim[1])
        ax.set_box_aspect(3 / 4)
        ax.legend(loc="upper right", fontsize=8)

    for idx in range(n_z, n_rows * n_cols):
        axes_flat[idx].set_visible(False)

    fig.supxlabel("t_ins - t_sep/2", fontsize=16)
    fig.supylabel("C3 / C2", fontsize=16)
    fig.suptitle(
        f"Unpolarized, P({sampa.Px},{sampa.Py},{sampa.Pz}), "
        f"Nconf={sampa.Nconf}, Nsample = {sampa.Nsample}\n"
        f"fit: tsep = [{fitpa.dt_start}, {fitpa.dt_end}], nex = {fitpa.nex}",
        fontsize=13, y=1.02)
    plt.tight_layout()
    saved = [os.path.join(out_dir, "ratio.png")]
    fig.savefig(saved[0], bbox_inches="tight")
    plt.close(fig)

    # ---- c0 vs z ----
    z_vals = np.arange(sampa.Nx)
    title = (f"Unpolarized, P({sampa.Px},{sampa.Py},{sampa.Pz}), "
             f"Nconf={sampa.Nconf}, Nsample = {sampa.Nsample}\n"
             f"fit: tsep = [{fitpa.dt_start}, {fitpa.dt_end}], nex = {fitpa.nex}")
    saved.append(os.path.join(out_dir, "c0.png"))
    plot_single_errbar(z_vals, c0_mean, c0_err, saved[-1],
                       xlabel="z", ylabel="c0",
                       xlim=[-0.5, sampa.Nx - 0.5], ylim=plotpa.c0_ylim,
                       title=title, label="c0")

    # ---- chi2 vs z ----
    saved.append(os.path.join(out_dir, "chi2.png"))
    plot_single_chi2(z_vals, chi2_mean, saved[-1],
                     xlabel="z", title=title, label="chi2/dof")

    if verbose:
        print(f"plot saved to {out_dir}/ratio.png")
        print(f"plot saved to {out_dir}/c0.png")
        print(f"plot saved to {out_dir}/chi2.png")
        print(f"plot done, time: {time.perf_counter() - time0:.2f}s")
        print("==================== plot_ratio end ====================")
    return saved


def run_ratio2pt(data_root, out_root, sampa: SampleParams2pt,
                 fitpa_list, plotpa: PlotParamsRatio, jack: bool = True,
                 parts=(1, 3), verbose=True) -> dict:
    """02_ratio 全链：compute → fit（多窗口）→ plot。

    parts: (start, end)，1=ratio 计算，2=拟合，3=画图；start>1 时从中间产物读取。
    """
    result = {"saved": []}
    ratio_dir = os.path.join(out_root, sampa.conf_short)
    os.makedirs(ratio_dir, exist_ok=True)
    ratio_path = os.path.join(ratio_dir, ratio_file_name(sampa))

    if parts[0] <= 1:
        time0 = time.perf_counter()
        ratio = compute_ratio(data_root, sampa, jack, verbose=verbose)
        np.save(ratio_path, ratio)
        if verbose:
            from ._plots import get_peak_memory_gb
            print(f"ratio saved to {ratio_path}")
            print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
            print(f"spend time: {time.perf_counter() - time0:.2f}s\n")
    else:
        ratio = np.load(ratio_path)
        if verbose:
            print(f"===== skip compute ratio, loading ratio from file =====")
            print(f"ratio loaded, shape: {ratio.shape}")

    result["ratio"] = ratio

    for _fitpa in fitpa_list:
        _fit_dir = os.path.join(ratio_dir, fit_dir_name(sampa, _fitpa))
        os.makedirs(_fit_dir, exist_ok=True)
        if parts[0] <= 2:
            time0 = time.perf_counter()
            fit_result = do_fit_and_report(ratio, _fitpa, sampa, _fit_dir,
                                           jack, verbose=verbose)
            if verbose:
                print(f"spend time: {time.perf_counter() - time0:.2f}s\n")
        else:
            if verbose:
                print(f"===== skip do_fit, loading from file =====")
            fit_result = dict(np.load(os.path.join(_fit_dir, "0_fit_data.npz")))
        result.setdefault("fit_results", {})[fit_dir_name(sampa, _fitpa)] = fit_result

        if parts[0] <= 3:
            result["saved"] += plot_ratio_fits(
                ratio, fit_result, sampa, _fitpa, plotpa, _fit_dir, jack,
                verbose=verbose)
    return result
