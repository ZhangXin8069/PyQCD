"""
04_proton_energy 功能：质子有效能量提取 + 平台拟合 + eff_mass 图（独立实现）
============================================================================

功能对齐 refer/huangcl/04_proton_energy/code.py：

- Part 1 compute：读 2pt 切片 → 平移不变相对时间 → ti 平均 → 重采样 →
  0_corr2.npy（(Nsample, dt_max+1)）。
- Part 2 fit：模型 C(t) = c0·e^{−E0·t}·(1 + c1·e^{−dE·t})，
  lsqfit 逐样本拟合（p0，svdcut=1e-6）→ 1_fit_data.npz + 2_fit_report.txt。
- Part 3 plot：eff_mass.png —— meff(t) = log(C(t)/C(t+1))·unit 误差棒
  + E0 平台色带（unit = 0.197/a GeV，a 为格距 fm）。

统计/拟合/图表全部复用 pyqcd.analysis 既有模块。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from ._disconnected import sem, resample, cov_mat
from ._fitter import fit_report_lines, make_summary_table


@dataclass
class EnergyParams:
    """04 参数。

    dir: 方向（'z' 默认；'x'/'y' 时按 dir_momentum 置换动量并读
         momsmear{Pz}{dir} 子目录——test6 三方向能量链整合项，
         消除驱动层自行置换的重复实现）。
    """

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
    a: float = 0.1053            # 格距 (fm)
    fm_to_GeV: float = 0.197
    dir: str = 'z'
    p0: dict = field(default_factory=lambda: {
        "c0": 0.6, "c1": 0.6, "E0": 1.5, "dE": 0.4})
    dt_start: int = 6
    dt_end: int = 12
    xlim: list = field(default_factory=lambda: [2.5, 19.5])
    ylim: list = field(default_factory=lambda: [0.5, 2.0])

    @property
    def Nconf(self):
        return len(self.conf_ids)

    @property
    def unit(self):
        """格点能 → GeV。"""
        return self.fm_to_GeV / self.a

    @property
    def mom_tag(self):
        """当前方向的动量标签 (Px,Py,Pz)（dir_momentum 置换）。"""
        from ._bare_matrix import dir_momentum
        return dir_momentum(self.Px, self.Py, self.Pz, self.dir)


def load_raw_corr(data_root, conf_id, params: EnergyParams) -> np.ndarray:
    """读取单组态 2pt 切片 (Nt, Nt)（方向感知：momsmear{Pz}{dir}）。"""
    Px, Py, Pz = params.mom_tag
    mom = f"Px{Px}Py{Py}Pz{Pz}"
    return np.load(os.path.join(
        data_root, params.conf_name, f"momsmear{params.Pz}{params.dir}",
        str(conf_id),
        f"twopt_slice_pp_{mom}_eginphase2_Cg5g4_nopol_ss_conf{conf_id}.npy"))


def compute_corr2(data_root, params: EnergyParams, jack: bool,
                  verbose=True) -> np.ndarray:
    """读 2pt → 相对时间 → ti 平均 → 重采样 → corr2 (Nsample, dt_max+1)。"""
    if verbose:
        print("==================== compute corr2 start ====================")

    _corr = np.zeros((params.Nconf, params.Nt, params.Nt), dtype=complex)
    for i, conf_id in enumerate(params.conf_ids):
        _corr[i] = load_raw_corr(data_root, conf_id, params)
    if verbose:
        print("load finish")
        print("2pt shape:", _corr.shape)

    _corr2_rel = np.zeros((params.Nconf, params.Nt, params.dt_max),
                          dtype=complex)
    for ti in range(params.Nt):
        corr2_shift = np.roll(_corr[:, :, ti], shift=-ti, axis=1)
        _corr2_rel[:, ti, :] = corr2_shift[:, :params.dt_max]

    _corr2_ave = _corr2_rel.mean(1)[:, :params.dt_max + 1]
    corr2 = resample(_corr2_ave, jack, params.Nsample).real

    if verbose:
        print("==================== compute corr2 end ====================")
    return corr2


def energy_model(x, p):
    """C(t) = c0·e^{−E0·t}·(1 + c1·e^{−dE·t})。"""
    dt = np.asarray(x, dtype=np.float64)
    return p["c0"] * np.exp(-p["E0"] * dt) * (1 + p["c1"] * np.exp(-p["dE"] * dt))


def do_fit(corr2: np.ndarray, params: EnergyParams, out_dir: str, jack: bool,
           verbose=True) -> dict:
    """逐样本 lsqfit 拟合（p0，svdcut=1e-6）→ 1_fit_data.npz + 2_fit_report.txt。"""
    if verbose:
        print("==================== do_fit start ====================")

    import gvar as gv
    import lsqfit

    x_coor = list(range(params.dt_start, params.dt_end + 1))
    Ndata = len(x_coor)
    param_names = list(params.p0.keys())

    fit_result = {name: np.zeros(params.Nsample) for name in param_names}
    fit_result["chi2"] = np.zeros(params.Nsample)

    report_lines = fit_report_lines(
        f"Fit Report  : {params.conf_short}", {
            "dt range": f"[{params.dt_start}, {params.dt_end}]",
            "Nsample": params.Nsample,
            "jackknife": jack,
        })

    sub_sample = np.zeros((params.Nsample, Ndata))
    for i, dt in enumerate(x_coor):
        sub_sample[:, i] = corr2[:, dt]

    cov, cond = cov_mat(sub_sample, jack)

    for _id in range(params.Nsample):
        y_coor = gv.gvar(sub_sample[_id], cov)
        _fit = lsqfit.nonlinear_fit(data=(x_coor, y_coor), p0=params.p0,
                                    fcn=energy_model, svdcut=1e-6)
        for name in param_names:
            fit_result[name][_id] = _fit.pmean[name]
        fit_result["chi2"][_id] = _fit.chi2 / _fit.dof

    if verbose:
        for name in param_names:
            print(f"{name} = {fit_result[name].mean():.3g} "
                  f"+- {sem(fit_result[name], jack):.3g}")
        print(f"chi2 = {fit_result['chi2'].mean():.3g}")
        print(f"fit time: ...")

    report_lines.append("-" * 72)
    report_lines.append(f"condition number = {cond:.3g}")
    report_lines.append("")
    report_lines.append(_fit.format(maxline=True))
    report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("  Summary Table")
    report_lines.append("=" * 72)
    row = []
    for name in param_names:
        row.append(f"{fit_result[name].mean():.3f}"
                   f"({sem(fit_result[name], jack) * 1e3:.0f})")
    row.append(f"{fit_result['chi2'].mean():.2g}")
    report_lines.append(make_summary_table(param_names + ["chi2/dof"], [row]))
    report_lines.append("")

    with open(os.path.join(out_dir, "2_fit_report.txt"), "w") as f:
        f.write("\n".join(report_lines))
    np.savez(os.path.join(out_dir, "1_fit_data.npz"), **fit_result)
    if verbose:
        print(f"report saved to {out_dir}/2_fit_report.txt")
        print(f"fit result saved to {out_dir}/1_fit_data.npz")
        print("==================== do_fit end ====================")
    return fit_result


def plot_eff_mass(corr2: np.ndarray, fit_result: dict, params: EnergyParams,
                  out_dir: str, jack: bool, verbose=True) -> str:
    """eff_mass.png：meff(t)·unit 误差棒 + E0 平台色带。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ._plots import DEFAULT_PLOT_COLORS

    if verbose:
        print("==================== plot_corr2 start ====================")
    time0 = time.perf_counter()

    para_E0 = fit_result["E0"] * params.unit
    chi2 = fit_result["chi2"]

    mass = np.log(corr2 / np.roll(corr2, shift=-1, axis=1)) * params.unit
    mass_mean = mass.mean(0)
    mass_err = sem(mass, jack)

    E0_mean = para_E0.mean(0)
    E0_err = sem(para_E0, jack)
    chi2_mean = chi2.mean(0)

    band_down = E0_mean - E0_err
    band_up = E0_mean + E0_err

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    color = DEFAULT_PLOT_COLORS[0]
    x_vals = np.arange(params.dt_max)
    ax.errorbar(x_vals, mass_mean, yerr=mass_err,
                fmt="x", color=color, ecolor=color, capsize=0,
                markersize=7, markeredgewidth=1.8, linewidth=1.2, zorder=3)

    x_band = np.array([params.dt_start, params.dt_end])
    ax.fill_between(x_band, [band_down, band_down], [band_up, band_up],
                    color="gray", alpha=0.35, linewidth=0, zorder=1,
                    label="Fit E0")

    E0_str = f"{E0_mean:.3f}({E0_err * 1e3:.0f})"
    Px, Py, Pz = params.mom_tag
    ax.set_title(
        f"P=({Px},{Py},{Pz}) [{params.dir}], E0={E0_str}, "
        f"chi2={chi2_mean:.2f}", fontsize=12)
    ax.set_xlim(params.xlim[0], params.xlim[1])
    ax.set_ylim(params.ylim[0], params.ylim[1])
    ax.set_box_aspect(3 / 4)
    ax.legend(loc="upper right", fontsize=8)

    fig.supxlabel("t/a", fontsize=16)
    fig.supylabel("eff mass (GeV)", fontsize=16)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "eff_mass.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"plot saved to {save_path}")
        print(f"plot done, time: {time.perf_counter() - time0:.2f}s")
        print("==================== plot_corr2 end ====================")
    return save_path


def run_energy(data_root, out_root, params: EnergyParams, jack: bool = True,
               parts=(1, 3), verbose=True) -> dict:
    """04 全链：compute corr2 → fit → eff_mass 图（方向感知）。

    out_dir：dir='z' 时保持 `_Pz{Pz}`（向后兼容）；非 z 方向追加
    `_{dir}` 后缀以区分三方向产物。
    """
    suffix = f"_{params.dir}" if params.dir != 'z' else ""
    out_dir = os.path.join(out_root, params.conf_short,
                           f"_Pz{params.Pz}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    result = {}

    if parts[0] <= 1:
        time0 = time.perf_counter()
        corr2 = compute_corr2(data_root, params, jack, verbose=verbose)
        np.save(os.path.join(out_dir, "0_corr2.npy"), corr2)
        if verbose:
            print(f"corr2 saved to {out_dir}/0_corr2.npy")
            from ._plots import get_peak_memory_gb
            print(f"Peak Memory: {get_peak_memory_gb():.3f} GB")
            print(f"spend time: {time.perf_counter() - time0:.2f}s\n")
    else:
        corr2 = np.load(os.path.join(out_dir, "0_corr2.npy"))
        if verbose:
            print("===== skip compute corr2, loading corr2 from file =====")
            print(f"corr2 loaded, shape: {corr2.shape}")
    result["corr2"] = corr2

    if parts[0] <= 2:
        time0 = time.perf_counter()
        fit_result = do_fit(corr2, params, out_dir, jack, verbose=verbose)
        if verbose:
            print(f"spend time: {time.perf_counter() - time0:.2f}s\n")
    else:
        fit_result = dict(np.load(os.path.join(out_dir, "1_fit_data.npz")))
    result["fit"] = fit_result

    if parts[0] <= 3:
        result["saved"] = [plot_eff_mass(corr2, fit_result, params, out_dir,
                                         jack, verbose=verbose)]
    return result
