"""
06_FH_bare_matele 功能：FH 变换 + 常数拟合 + 全套图（独立实现）
===============================================================

功能对齐 refer/huangcl/06_FH_bare_matele/code_FH_bare_matele.py：

- Part 1 FH：读取 6 个方向（pos_x/pos_y/pos_z/neg_x/neg_y/neg_z）的 ratio
  并平均 → FH 变换 FH(t) = Σ_{τ=nex}^{t+1−nex} R(t+1,τ) − Σ_{τ=nex}^{t−nex} R(t,τ)
  → 保存 FH_nex{n}.npy → 画 FH 图（多 nex 对比，z{iz}.png）。
- Part 2 fit：模型 FH(t) = c0（常数），逐 z 拟合 → fit_nex{N}/dt{}_{}/
  report_dt{}_{}.txt + fit_dt{}_{}.npz。
- Part 3 plot：每窗口参数 vs z（c0.png/chi2.png）、多窗口对比图
  （参数+chi2.png）、bestfit FH + c0 色带图（bestfit/）。

数据布局（参考约定）: <data_root>/<conf>/P{4}/{pos_x,...}/ratio.npy。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from ._disconnected import sem
from ._fitter import (FitParams, fit, fit_report_lines, make_summary_table)
from ._plots import plot_errbar, plot_scatter


@dataclass
class FHParams:
    """06 参数。"""

    conf_short: str
    P: int                       # 动量大小
    nexmax: int = 2              # FH 时 τ 两端去掉点数循环上限
    ave_dirs: list = field(default_factory=lambda: [
        "pos_x", "pos_y", "pos_z", "neg_x", "neg_y", "neg_z"])
    z_list: list = field(default_factory=lambda: list(range(8)))
    z_step: int = 3
    xoffset: float = 0.2
    fh_xlim: list = field(default_factory=lambda: [2.5, 11.5])
    fh_ylim: list = field(default_factory=lambda: [-0.1, 1.1])
    para_xlim: list = field(default_factory=lambda: [-0.5, 24.5])
    param_ylim: dict = field(default_factory=lambda: {
        "c0": [-0.1, 1.0], "c1": [-0.5, 0.5],
        "c2": [-0.1, 0.1], "dE": [0.0, 1.0]})


def load_one_ratio(data_root: str, conf_short: str, P: int, dir: str) -> np.ndarray:
    """读取单方向 ratio (Nsample, dt, dtau, z)。"""
    path = os.path.join(data_root, conf_short, f"P{P}", dir, "ratio.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ratio file not found: {path}")
    return np.load(path)


def compute_fh(ratio: np.ndarray, save_path: str, nex: int = 0,
               verbose=True) -> np.ndarray:
    """FH 变换：FH(t) = Σ_{τ=nex}^{t+1−nex} R(t+1,τ) − Σ_{τ=nex}^{t−nex} R(t,τ)。

    Returns
    -------
    fh : (Nsample, dt, z)
    """
    if verbose:
        print(f"  computing FH (nex={nex}) ")

    Nsample, dtmax, _, Nz = ratio.shape
    temp = np.zeros((Nsample, dtmax, Nz))
    for dt in range(2 * nex, dtmax):
        temp[:, dt] = ratio[:, dt, nex:dt - nex + 1, :].sum(axis=1)
    fh = temp - np.roll(temp, 1, axis=1)

    if verbose:
        print(f"    FH shape: (Nsample, dt, z) = {fh.shape}")
    np.save(save_path, fh)
    if verbose:
        print(f"    FH saved to: {save_path}")
    return fh


def fh_model(t, p):
    """FH(t) ≈ c0（常数模型）。"""
    return p["c0"] * np.ones_like(np.asarray(t, dtype=np.float64))


def plot_fh(all_fh: dict, save_dir: str, params: FHParams,
            c0_data: dict = None, band_t_range: tuple = None,
            verbose=True) -> list:
    """画 FH 图：每个 z 一张，多 nex 对比 errorbar；可选 c0 色带。"""
    saved = []
    _first_nex = next(iter(all_fh.keys()))
    _, dt_max, Nz = all_fh[_first_nex].shape
    t_vals = np.arange(dt_max)

    for _iz in params.z_list:
        if _iz >= Nz:
            if verbose:
                print(f"    Warning: z={_iz} exceeds Nz={Nz}, skipping")
            continue

        _data = {}
        for _nex, _fh in sorted(all_fh.items()):
            _data[f"nex={_nex}"] = (_fh.mean(0)[:, _iz],
                                    sem(_fh, jackknife=False)[:, _iz])

        _show_band = False
        _band_x = _band_down = _band_up = _band_label = None
        if c0_data is not None and _iz in c0_data:
            c0_mean, c0_err = c0_data[_iz]
            _show_band = True
            if band_t_range is not None:
                _band_x = np.arange(band_t_range[0], band_t_range[1] + 1,
                                    dtype=float)
            else:
                _band_x = t_vals
            _band_down = np.full_like(_band_x, c0_mean - c0_err, dtype=float)
            _band_up = np.full_like(_band_x, c0_mean + c0_err, dtype=float)
            _band_label = f"c0 = {c0_mean:.3f} ± {c0_err:.3f}"

        sp = os.path.join(save_dir, f"z{_iz}.png")
        plot_errbar(t_vals, _data, sp,
                    xlabel="t", ylabel="FH",
                    xlim=params.fh_xlim, ylim=params.fh_ylim,
                    x_offset=params.xoffset,
                    title=f"{params.conf_short}, P={params.P}, z={_iz}",
                    show_band=_show_band, band_x=_band_x,
                    band_y_down=_band_down, band_y_up=_band_up,
                    band_label=_band_label)
        saved.append(sp)
    return saved


def do_fit_and_report(fh: np.ndarray, save_dir: str, fitpa: FitParams,
                      params: FHParams, debug: bool = False,
                      debugsample: int = 20, verbose=True):
    """对 FH 数据逐 z 拟合（常数模型），输出报告与 npz。"""
    Nsample, _, Nz = fh.shape
    dt_start, dt_end = fitpa.dt_start, fitpa.dt_end
    param_names = list(fitpa.p0.keys())
    window_tag = f"dt{dt_start}_{dt_end}"
    t_vals = np.arange(dt_start, dt_end + 1, dtype=int)

    Nfit = min(debugsample, Nsample) if debug else Nsample
    if verbose:
        print(f"\n    fitting window: t = [{dt_start}, {dt_end}], "
              f"Nfit = {Nfit}/{Nsample}")

    all_fit_result = {name: np.zeros((Nfit, Nz))
                      for name in param_names + ["chi2"]}
    all_cond = np.zeros(Nz)

    lines = fit_report_lines(
        f"Fit Report, {window_tag}, nex={fitpa.nex}", {
            "model": "FH(t) = c0",
            "fitpa": f"p0={fitpa.p0}, t=[{dt_start},{dt_end}]",
        })

    for _iz in range(Nz):
        _tz = time.perf_counter()
        y_data = fh[:, t_vals, _iz]

        _fit_result, _cov, _cond, _last_fit = fit(
            y_coor=y_data, x_coor=t_vals, model=fh_model, fitpa=fitpa,
            jackknife=False, debug=debug, debugNfit=debugsample)
        if verbose:
            print(f"z = {_iz}, time = {time.perf_counter() - _tz:.2f}s")

        for name in param_names + ["chi2"]:
            all_fit_result[name][:, _iz] = _fit_result[name][:Nfit]
        all_cond[_iz] = _cond

        lines.append(f"  z = {_iz}: condition number = {_cond:.3g}")
        if _last_fit is not None:
            lines.append(_last_fit.format(maxline=True))
        lines.append("")

    lines.append("=" * 72)
    lines.append(f"  Summary Table, {window_tag}, nex={fitpa.nex}")
    lines.append("=" * 72)
    summary_rows = []
    for _iz in range(Nz):
        row = [str(_iz)]
        for name in param_names:
            mean = all_fit_result[name][:, _iz].mean()
            err = sem(all_fit_result[name][:, _iz], False)
            row.append(f"{mean:.3f}({err * 1e3:.0f})")
        row.append(f"{all_fit_result['chi2'][:, _iz].mean():.2g}")
        summary_rows.append(row)
    lines.append(make_summary_table(["z"] + param_names + ["chi2/dof"],
                                    summary_rows))
    lines.append("")

    with open(os.path.join(save_dir, f"report_{window_tag}.txt"), "w") as f:
        f.write("\n".join(lines))
    np.savez(os.path.join(save_dir, f"fit_{window_tag}.npz"),
             **all_fit_result)
    if verbose:
        print(f"    report saved to: {save_dir}/report_{window_tag}.txt")
        print(f"    fit result saved to: {save_dir}/fit_{window_tag}.npz")


def plot_para(all_fit_result: dict, save_dir: str, fitpa: FitParams,
              params: FHParams, verbose=True) -> list:
    """单窗口参数 vs z：参数 errorbar 图 + chi2 散点图。"""
    saved = []
    param_names = list(fitpa.p0.keys())
    Nz = all_fit_result[param_names[0]].shape[1]
    z_vals = np.arange(Nz)

    for _name in param_names:
        _arr = all_fit_result[_name]
        sp = os.path.join(save_dir, f"{_name}.png")
        plot_errbar(z_vals, {_name: (_arr.mean(0), sem(_arr, jackknife=False))},
                    sp, xlabel="z", ylabel=_name,
                    xlim=params.para_xlim, ylim=params.param_ylim.get(_name),
                    title=f"{params.conf_short}, P={params.P}, "
                          f"fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}]",
                    x_offset=0.3, figsize=(10, 6), dpi=150)
        saved.append(sp)

    _chi2_arr = all_fit_result["chi2"]
    sp = os.path.join(save_dir, "chi2.png")
    plot_scatter(z_vals, {"chi2/dof": _chi2_arr.mean(0)}, sp,
                 xlabel="z", ylabel="chi2/dof",
                 xlim=params.para_xlim, ylim=[0, 2],
                 title=f"{params.conf_short}, P={params.P}, "
                       f"fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}]",
                 x_offset=0.3, figsize=(10, 6), dpi=150,
                 show_hline=True, hline_y=1.0, hline_label="chi2/dof=1")
    saved.append(sp)
    return saved


def plot_para_cmp(fit_data: dict, save_dir: str, params: FHParams,
                  fitpa_list, verbose=True) -> list:
    """多窗口参数对比图：不同拟合窗口叠加（z 按 z_step 抽样）。"""
    saved = []
    param_names = list(fitpa_list[0].p0.keys())

    _first_tag = next(iter(fit_data.keys()))
    Nz = fit_data[_first_tag][param_names[0]].shape[1]
    z_vals = np.arange(Nz)[::params.z_step]

    for _name in param_names:
        _data = {}
        for _fitpa in fitpa_list:
            _tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
            _arr = fit_data[_tag][_name]
            _data[f"dt: {_fitpa.dt_start}~{_fitpa.dt_end}"] = (
                _arr.mean(0)[::params.z_step],
                sem(_arr, jackknife=False)[::params.z_step])
        sp = os.path.join(save_dir, f"{_name}.png")
        plot_errbar(z_vals, _data, sp, xlabel="z", ylabel=_name,
                    xlim=params.para_xlim, ylim=params.param_ylim.get(_name),
                    title=f"{params.conf_short}, P={params.P}, {_name}",
                    x_offset=0.3, figsize=(10, 6), dpi=150)
        saved.append(sp)

    _chi2_data = {}
    for _fitpa in fitpa_list:
        _tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
        _chi2_data[f"dt: {_fitpa.dt_start}~{_fitpa.dt_end}"] = (
            fit_data[_tag]["chi2"].mean(0)[::params.z_step])
    sp = os.path.join(save_dir, "chi2.png")
    plot_scatter(z_vals, _chi2_data, sp, xlabel="z", ylabel="chi2/dof",
                 xlim=params.para_xlim, ylim=[0, 2],
                 title=f"{params.conf_short}, P={params.P}, chi2/dof",
                 x_offset=0.3, figsize=(10, 6), dpi=150,
                 show_hline=True, hline_y=1.0, hline_label="chi2/dof=1")
    saved.append(sp)
    return saved


def run_fh(data_root, out_root, params: FHParams, fitpa_list,
           bestfit_params: dict = None, parts=(1, 3), verbose=True) -> dict:
    """06 全链：FH → fit（多窗口）→ plot（含 bestfit 色带图）。"""
    out_dir = os.path.join(out_root, params.conf_short, f"P{params.P}")
    os.makedirs(os.path.join(out_dir, "fh"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bestfit"), exist_ok=True)
    result = {"saved": []}

    # ---- Part 1: FH ----
    if parts[0] <= 1:
        _t0 = time.perf_counter()
        _ratios = [load_one_ratio(data_root, params.conf_short, params.P, d)
                   for d in params.ave_dirs]
        ratio_ave = np.mean(_ratios, axis=0)
        if verbose:
            print(f"ratio_ave shape: {ratio_ave.shape}")

        fh_dir = os.path.join(out_dir, "fh")
        all_fh = {}
        for _nex in range(params.nexmax + 1):
            _fh_path = os.path.join(fh_dir, f"FH_nex{_nex}.npy")
            fh = compute_fh(ratio_ave, _fh_path, nex=_nex, verbose=verbose)
            all_fh[_nex] = fh
        result["saved"] += plot_fh(all_fh, fh_dir, params, verbose=verbose)
        result["fh"] = all_fh
        if verbose:
            print(f"    FH time: {time.perf_counter() - _t0:.2f}s\n")

    # ---- Part 2: fit ----
    _nex_fit = fitpa_list[0].nex
    fh = np.load(os.path.join(out_dir, "fh", f"FH_nex{_nex_fit}.npy"))
    if verbose:
        print(f"  loading FH (nex={_nex_fit}) shape: {fh.shape}")

    _fit_dir = os.path.join(out_dir, f"fit_nex{_nex_fit}")
    os.makedirs(_fit_dir, exist_ok=True)
    for _fitpa in fitpa_list:
        _tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
        _window_dir = os.path.join(_fit_dir, _tag)
        os.makedirs(_window_dir, exist_ok=True)
        if parts[0] <= 2:
            do_fit_and_report(fh, _window_dir, _fitpa, params,
                              verbose=verbose)
        else:
            if verbose:
                print(f"  ==== skip fit ({_tag}) ====")

    # ---- Part 3: plot ----
    if parts[0] <= 3:
        fit_data = {}
        for _fitpa in fitpa_list:
            _tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
            fit_data[_tag] = np.load(os.path.join(
                _fit_dir, _tag, f"fit_{_tag}.npz"))

        for _fitpa in fitpa_list:
            _tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
            result["saved"] += plot_para(fit_data[_tag],
                                         os.path.join(_fit_dir, _tag),
                                         _fitpa, params, verbose=verbose)
        result["saved"] += plot_para_cmp(fit_data, _fit_dir, params,
                                         fitpa_list, verbose=verbose)

        # bestfit: FH + c0 色带
        if bestfit_params:
            _bf = bestfit_params
            _bf_tag = f"dt{_bf['dt_start']}_{_bf['dt_end']}"
            _bf_fh = np.load(os.path.join(
                out_dir, "fh", f"FH_nex{_bf['nex']}.npy"))
            _bf_c0 = np.load(os.path.join(
                _fit_dir, _bf_tag, f"fit_{_bf_tag}.npz"))["c0"]
            _c0_data = {}
            for _iz in params.z_list:
                if _iz < _bf_c0.shape[1]:
                    _c0_data[_iz] = (_bf_c0[:, _iz].mean(),
                                     sem(_bf_c0[:, _iz], jackknife=False))
            result["saved"] += plot_fh(
                {_bf["nex"]: _bf_fh}, os.path.join(out_dir, "bestfit"),
                params, c0_data=_c0_data,
                band_t_range=(_bf["dt_start"], _bf["dt_end"]),
                verbose=verbose)
    return result
