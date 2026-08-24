#!/public/home/huangcl/.venv/bin/python
"""
code_FH_bare_matele.py

从 08_zch_ratio 的结果目录读取 zch_ratio.py 生成的 ratio 数据:
  part 1: FH  — 读取 ratio → FH 变换 → 画 FH 图
  part 2: fit — 读取 FH 结果 → 做 fit → 保存 fit 结果
  part 3: plot — 读取 fit 结果 → 画图

目录结构:
  1_result/{conf_short}/P{P}/
    ratio.npy                    # zch_ratio.py 生成的 ratio 数据
    fh/                          # FH 结果
    fit_nex{N}/                  # 拟合结果 (N = nex 值)
      dt{start}_{end}/           # 每个拟合窗口一个子目录
        report_*.txt, fit_*.npz, fit_z*.png
      param_*_vs_z.png           # 参数随 z 变化对比图

"""

import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np
import argparse
import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
import gvar as gv
import lsqfit

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import (  # noqa: E402
    sem, resample, get_peak_memory_gb, DEFAULT_PLOT_COLORS,
    plot_errbar, plot_scatter, plot_hist,
    fit, FitParams,
)


# ===== 独立开关，方便调试时修改 =====
debug = True  # 在登录节点跑, 结果输出到 0_debug 文件夹
debugsample = 100  # debug 时保留的样本数, 仅在 debug=True 时生效
# ===================================


# ===== 定义 dataclass =====

@dataclass
class ReadParams:
    """读取 ratio 所需的参数"""
    conf_short: str
    P: int          # 动量大小 (正值), 直接在 py 文件中修改
    nex: int = 0    # FH 时 τ 方向两端各去掉的点数
    nexmax: int = 0  # FH 时循环的最大 nex 值 (nex=0..nexmax)


@dataclass
class PlotParams:
    """画图参数"""
    z_list: list[int]       # 要画的 z 列表, 如 [0,1,2,...,9]
    z_step: int = 3         # 画参数对比图时 z 的步长, 如 3 表示取 0,3,6,...
    xoffset: float = 0.2    # 多条曲线时横坐标错开量

    # ---- FH 图参数 ----
    fh_xlim: list[float] = None     # FH 图横轴范围 (与 P 有关)
    fh_ylim: list[float] = None     # FH 图纵轴范围

    # ---- 参数图参数 (para 共用) ----
    para_xlim: list[float] = None   # 参数图横轴范围 (z 范围)
    # 每个参数图的纵轴范围, 如 {'c0': [0, 1], 'c1': [-0.5, 0.5], ...}
    # chi2 不用设, 纵轴直接写死
    param_ylim: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class OutputParams:
    """路径管理 — 输出到 06 目录下"""
    base_dir: str = "1_result"
    conf_short: str = ""
    P: int = 0

    @property
    def result_dir(self):
        return os.path.join(os.getcwd(), self.base_dir,
                            self.conf_short, f"P{self.P}")

    def get_sub_dir(self, name):
        """返回子目录路径并创建"""
        d = os.path.join(self.result_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def sub_dir_path(self, name):
        """仅返回子目录路径, 不创建"""
        return os.path.join(self.result_dir, name)


# ===== 从命令行参数读取 =====
parser = argparse.ArgumentParser(
    description="Gluon unpolarized PDF: FH transform, fit and plot ratio data from 08_zch_ratio")
parser.add_argument("-c", type=str, default="L24x72",
                    help="conf_short, e.g. L24x72 (default: L24x72)")
parser.add_argument("-s", type=int, default=1,
                    choices=[1, 2, 3],
                    help="start part: 1=FH, 2=fit, 3=plot (default: 1)")
parser.add_argument("-e", type=int, default=3,
                    choices=[1, 2, 3],
                    help="end part: 1=FH, 2=fit, 3=plot (default: 3)")
args = parser.parse_args()
conf_short = args.c
part_start = args.s
part_end = args.e
# =========================================


# ===== 配置参数 (P 直接在下面修改) =====
if conf_short == "L24x72":
    readpa = ReadParams(
        conf_short="L24x72",
        P=4,
        nexmax=2,   # FH 时循环 nex=0..nexmax (设为 0 则只算 nex=0)
    )

    # 拟合窗口: 右端点与 P 有关
    dt_end = 10  # 固定右端点
    dt_start_list = np.arange(3, 4)  # 尝试不同的左端点

    # 为每个拟合窗口创建一个 FitParams
    fitpa_list = [
        FitParams(
            nex=1,      # fit/plot 时使用的 nex 值
            p0={
                'c0': 0.6,
                # 'c1': 0.0,
                'c2': 0.0,
                'dE': 0.3
            },
            prior={
                'c0': gv.gvar('0.7(0.1)'),
                # 'c1': gv.gvar('0.0(1)),
                'c2': gv.gvar('-1.5(0.5)'),
                'dE': gv.gvar('0.7(0.5)'),
            },
            dt_start=_dt,
            dt_end=dt_end,
            svdcut=1e-6,
        )
        for _dt in dt_start_list
    ]

    # 检查: 每个拟合窗口的 dt_start 必须 > 2*nex, 否则 FH 在该点无意义
    for _fitpa in fitpa_list:
        if _fitpa.dt_start <= 2 * _fitpa.nex:
            print(
                f"    Error: dt_start ({_fitpa.dt_start}) must be > 2*nex ({2 * _fitpa.nex})")
            print(f"    FH data at t <= 2*nex is meaningless (all zeros).")
            sys.exit(1)

    # FH 画图参数 (xlim 与 P 有关)
    _fh_xlim = {
        2: [2.5, 10.5],
        3: [2.5, 16.5],
        4: [2.5, 11.5],
        5: [2.5, 12.5],
        6: [2.5, 10.5],
    }
    # bestfit 参数 (与 P 有关, 手动调整)
    _bestfit_params = {
        2: {"dt_start": 5, "dt_end": 9, "nex": 1},
        3: {"dt_start": 5, "dt_end": 9, "nex": 1},
        4: {"dt_start": 7, "dt_end": 10, "nex": 2},
        5: {"dt_start": 7, "dt_end": 10, "nex": 2},
        6: {"dt_start": 5, "dt_end": 9, "nex": 1},
    }
    plotpa = PlotParams(
        z_list=list(range(8)),  # 前 8 个 z
        fh_xlim=_fh_xlim.get(readpa.P),
        fh_ylim=[-0.1, 1.1],
        para_xlim=[-0.5, 24.5],
        param_ylim={
            'c0': [-0.1, 1.0],
            'c1': [-0.5, 0.5],
            'c2': [-2, 0],
            'dE': [0.0, 2.0],
        },
    )
else:
    print(f"conf {conf_short} not exist.")
    sys.exit()


# debug 时清理 0_debug 目录
if debug:
    print("debug mode")
    _debug_dir = os.path.join(os.getcwd(), "0_debug")
    if os.path.exists(_debug_dir):
        print(f"  cleaning debug dir: {_debug_dir}")
        shutil.rmtree(_debug_dir)

# ===== OutputParams =====
_base_dir = "0_debug" if debug else "1_result"
outpa = OutputParams(
    base_dir=_base_dir,
    conf_short=readpa.conf_short,
    P=readpa.P,
)

outpa.get_sub_dir(os.path.join("fh"))
outpa.get_sub_dir(os.path.join("bestfit"))
_nex_fit = fitpa_list[0].nex
outpa.get_sub_dir(os.path.join(f"fit_nex{_nex_fit}"))
# =========================


# 模型: FH(t) ≈ c0 + (c1 + c2 * t) * exp(-dE * t)
def fh_model(t, p):

    return (p['c0'] * np.ones_like(t, dtype=float)
            # + p['c1'] * np.exp(-p['dE'] * t)
            + p['c2'] * t * np.exp(-p['dE'] * t)
            )


# ============================================================
# 拟合函数 (从 analysis_tools.py 复制, 每 100 次输出时间)
# ============================================================

def _calc_cov(arr: np.ndarray, jackknife: bool = False):
    """计算协方差矩阵与条件数"""
    diff = arr - arr.mean(0)
    n = arr.shape[0]
    cov = diff.T @ diff
    if jackknife:
        cov *= (n - 1) / n
    else:
        cov /= n
    eig = np.linalg.eigvalsh(cov)
    cond = eig[-1] / eig[0]
    return cov, cond


def _calc_chi2(y_data, y_fit, cov, svdcut=None):
    """计算 chi2 = diff^T C^{-1} diff"""
    diff = y_data - y_fit
    return diff @ np.linalg.solve(cov, diff)


def _calc_chi2_dof(y_data, y_fit, cov, n_params, svdcut=None):
    """计算 chi2/dof"""
    chi2 = _calc_chi2(y_data, y_fit, cov, svdcut)
    dof = len(y_data) - n_params
    return chi2 / dof, chi2, dof


def fit_with_progress(
    y_coor: np.ndarray,
    x_coor,
    model,
    fitpa,
    jackknife: bool = False,
    debug: bool = False,
    debugNfit=20,
):
    """
    对每个 sample 做 lsqfit 非线性拟合, 每 100 次输出时间.
    返回格式与 analysis_tools 的 fit() 一致.
    """
    Nsample, _ = y_coor.shape
    param_names = list(fitpa.p0.keys())
    n_params = len(param_names)

    if debug:
        Nfit = min(debugNfit, Nsample)
        print(f"debug mode, fit number: {Nfit}")
    else:
        Nfit = Nsample

    fit_result = {name: np.full(Nsample, np.nan) for name in param_names}
    fit_result["chi2"] = np.full(Nsample, np.nan)

    cov, cond = _calc_cov(y_coor, jackknife)

    use_prior = fitpa.prior is not None and len(fitpa.prior) > 0
    if use_prior:
        print('use prior to fit')
    else:
        print('use p0 to fit')

    last_fit_info = None
    _t_start = time.perf_counter()
    for _id in range(Nfit):
        y_gvar = gv.gvar(y_coor[_id], cov)

        if use_prior:
            _fit = lsqfit.nonlinear_fit(
                data=(x_coor, y_gvar),
                prior=fitpa.prior,
                fcn=model,
                svdcut=fitpa.svdcut,
            )
        else:
            _fit = lsqfit.nonlinear_fit(
                data=(x_coor, y_gvar),
                p0=fitpa.p0,
                fcn=model,
                svdcut=fitpa.svdcut,
            )
        last_fit_info = _fit

        for name in param_names:
            fit_result[name][_id] = _fit.pmean[name]

        chi2_dof_val, _, _ = _calc_chi2_dof(
            y_coor[_id],
            model(x_coor, _fit.pmean),
            cov,
            n_params,
            fitpa.svdcut,
        )
        fit_result["chi2"][_id] = chi2_dof_val

        # 每 100 次输出时间
        if (_id + 1) % 100 == 0:
            _t_now = time.perf_counter()
            print(
                f"    fit progress: {_id+1}/{Nfit}, time: {(_t_now - _t_start):.1f}s")

    _t_end = time.perf_counter()
    print(f"    fit total: {Nfit} samples, time: {(_t_end - _t_start):.1f}s")

    return fit_result, cov, cond, last_fit_info


########################################################################################
def load_ratio(readpa: ReadParams) -> np.ndarray:
    """
    读取 zch_ratio.py 生成的 ratio 数据 (无方向, 直接读取).

    返回: ratio_array, 参数为 (Nsample, dt, dtau, z)
    """
    # 08_zch_ratio 的结果目录
    _project_dir = Path(__file__).resolve().parent.parent
    _load_path = os.path.join(
        str(_project_dir), "08_zch_ratio", "1_result",
        readpa.conf_short, f"P{readpa.P}",
        "ratio.npy"
    )

    print(f"  loading ratio from: {_load_path}")
    if not os.path.exists(_load_path):
        print(f"    Error: file not found: {_load_path}")
        sys.exit(1)

    ratio = np.load(_load_path)
    # ratio shape: (Nsample, dt, dtau, z)
    print(f"    loaded shape: (Nsample, dt, dtau, z) = {ratio.shape}")

    return ratio


########################################################################################
def compute_fh(ratio: np.ndarray, save_path: str, nex: int = 0):
    """
    计算 FH_n(t) = Σ_{τ=nex}^{t+1-nex} R(t+1, τ) - Σ_{τ=nex}^{t-nex} R(t, τ)

    对每个 dt, 在 τ 方向去掉两端各 nex 个点后求和, 再做差分.

    参数:
        ratio: (Nsample, dt, dtau, z)
        save_path: 保存路径
        nex: 两端各去掉的点数 (默认 0, 即原始定义)

    返回: FH_array, 参数为 (Nsample, dt, z)
    """
    print(f"  computing FH (nex={nex}) ")

    Nsample, dtmax, _, Nz = ratio.shape

    # temp(t) = Σ_{τ=nex}^{t-nex} R(t, τ): sample, dt, z
    temp = np.zeros((Nsample, dtmax, Nz))
    for dt in range(2 * nex, dtmax):  # 确保求和区间非空
        temp[:, dt] = ratio[:, dt, nex:dt-nex+1, :].sum(axis=1)

    # FH(t) = temp(t) - temp(t-1)
    fh = temp - np.roll(temp, 1, axis=1)

    # fh(2nex-1) \neq 0, 但该值无意义, 边界点要注意
    # if nex > 0:
    #     fh[:, 2 * nex - 1] = 0
    print(f"    FH shape: (Nsample, dt, z) = {fh.shape}")

    # 保存 FH 结果
    np.save(save_path, fh)
    print(f"    FH saved to: {save_path}")

    return fh


########################################################################################
def plot_fh(
    all_fh: dict,
    save_dir: str,
    readpa: ReadParams,
    plotpa: PlotParams,
    c0_data: dict = None,
    band_t_range: tuple = None,
):
    """
    画 FH 图.

    参数:
        all_fh: {nex: fh_array}, 每个 fh_array shape 为 (Nsample, dt, z)
        save_dir: 保存目录
        readpa: ReadParams
        plotpa: PlotParams
        c0_data: {z: (c0_mean, c0_err)}, 若提供则画 c0 色带
        band_t_range: (t_start, t_end), 色带覆盖的 t 范围, 默认全范围
    """
    print(f"  plotting FH to: {save_dir}")

    # 取第一个 nex 的 shape 作为参考
    _first_nex = next(iter(all_fh.keys()))
    _, dt_max, Nz = all_fh[_first_nex].shape

    t_vals = np.arange(dt_max)

    for _iz in plotpa.z_list:
        if _iz >= Nz:
            print(f"    Warning: z={_iz} exceeds Nz={Nz}, skipping")
            continue

        # 收集所有 nex 的数据
        _data = {}
        for _nex, _fh in sorted(all_fh.items()):
            _mean = _fh.mean(0)[:, _iz]
            _err = sem(_fh, jackknife=False)[:, _iz]
            _data[f"nex={_nex}"] = (_mean, _err)

        # 若提供 c0_data, 添加色带
        _show_band = False
        _band_x = None
        _band_y_down = None
        _band_y_up = None
        _band_label = None
        if c0_data is not None and _iz in c0_data:
            c0_mean, c0_err = c0_data[_iz]
            _show_band = True
            # 色带只覆盖拟合用的 t 范围
            if band_t_range is not None:
                _band_x = np.arange(
                    band_t_range[0], band_t_range[1] + 1, dtype=float)
            else:
                _band_x = t_vals
            _band_y_down = np.full_like(_band_x, c0_mean - c0_err, dtype=float)
            _band_y_up = np.full_like(_band_x, c0_mean + c0_err, dtype=float)
            _band_label = f"c0 = {c0_mean:.3f} ± {c0_err:.3f}"

        save_path = os.path.join(save_dir, f"z{_iz}.png")
        plot_errbar(
            t_vals, _data, save_path,
            xlabel="t", ylabel="FH",
            xlim=plotpa.fh_xlim,
            ylim=plotpa.fh_ylim,
            x_offset=plotpa.xoffset,
            title=f"{readpa.conf_short}, P={readpa.P}, z={_iz}",
            show_band=_show_band,
            band_x=_band_x,
            band_y_down=_band_y_down,
            band_y_up=_band_y_up,
            band_label=_band_label,
        )


########################################################################################
def do_fit_and_report(fh: np.ndarray, save_dir: str, fitpa: FitParams):
    """
    对一个方向的 FH 变换结果, 用给定的 fitpa 做 fit, 输出报告并保存结果.

    拟合模型: fh_model
    FH shape: (Nsample, dt, z)
    对每个 z 独立拟合, 结果保存到 save_dir 下.
    """
    Nsample, _, Nz = fh.shape
    dt_start = fitpa.dt_start
    dt_end = fitpa.dt_end
    param_names = list(fitpa.p0.keys())
    window_tag = f"dt{dt_start}_{dt_end}"

    t_vals = np.arange(dt_start, dt_end+1, dtype=int)

    # debug 模式下只保留前 debugsample 个样本
    if debug:
        Nfit = min(debugsample, Nsample)
    else:
        Nfit = Nsample

    print(
        f"\n    fitting window: t = [{dt_start}, {dt_end}], Nfit = {Nfit}/{Nsample}")

    # 收集所有 z 的拟合结果
    # fit() 返回 Nsample 大小, 但这里只取前 Nfit 个 (debug 模式下后面的全是 NaN)
    all_fit_result = {name: np.zeros((Nfit, Nz))
                      for name in param_names + ["chi2"]}
    all_cond = np.zeros(Nz)

    # 报告行
    lines = []
    sep_line = "=" * 72
    lines.append(sep_line)
    lines.append(f"  Fit Report, {window_tag}, nex={fitpa.nex}")
    lines.append(sep_line)
    lines.append(f"  model : FH(t) = c0")
    lines.append(f"  fitpa : {fitpa}")
    lines.append(sep_line)
    lines.append("")

    # 对每个 z 做拟合
    for _iz in range(Nz):
        _tz = time.perf_counter()
        y_data = fh[:, t_vals, _iz]  # (Nsample, Nt_fit)

        _fit_result, _cov, _cond, _last_fit = fit_with_progress(
            y_coor=y_data,
            x_coor=t_vals,
            model=fh_model,
            fitpa=fitpa,
            jackknife=False,
            debug=debug,
            debugNfit=debugsample,
        )
        _tz_end = time.perf_counter()
        print(f"z = {_iz}, time = {(_tz_end-_tz):.2f}s")

        # fit() 返回 Nsample 大小, 只取前 Nfit 个有效数据
        for name in param_names + ["chi2"]:
            all_fit_result[name][:, _iz] = _fit_result[name][:Nfit]
        all_cond[_iz] = _cond

        lines.append(f"  z = {_iz}: condition number = {_cond:.3g}")
        lines.append(_last_fit.format(maxline=True))
        lines.append("")

    # 汇总表格
    lines.append(sep_line)
    lines.append(f"  Summary Table, {window_tag}, nex={fitpa.nex}")
    lines.append(sep_line)

    summary_tbl = PrettyTable()
    summary_tbl.field_names = ["z"] + param_names + ["chi2/dof"]
    for name in summary_tbl.field_names:
        summary_tbl.align[name] = "c"

    for _iz in range(Nz):
        row = [f"{_iz}"]
        for name in param_names:
            mean = all_fit_result[name][:, _iz].mean()
            err = sem(all_fit_result[name][:, _iz], False)
            row.append(f"{mean:.3f}({err * 1e3:.0f})")
        row.append(f"{all_fit_result['chi2'][:, _iz].mean():.2g}")
        summary_tbl.add_row(row)

    lines.append(str(summary_tbl))
    lines.append("")

    # 保存报告
    report_path = os.path.join(save_dir, f"report_{window_tag}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"    report saved to: {report_path}")

    # 保存拟合结果 (所有 z 在一起)
    result_path = os.path.join(save_dir, f"fit_{window_tag}.npz")
    np.savez(result_path, **all_fit_result)
    print(f"    fit result saved to: {result_path}")


########################################################################################
def plot_para(
    all_fit_result: dict,
    save_dir: str,
    fitpa: FitParams,
    readpa: ReadParams,
    plotpa: PlotParams,
):
    """
    对一个拟合窗口, 画 4 个参数 + chi2 随 z 变化的图 (单窗口).

    参数用 plot_errbar (带误差棒), chi2 用 plot_scatter (散点图).

    参数:
        all_fit_result: {name: (Nsample, Nz)} 拟合结果 (从 npz 加载)
        save_dir: 该窗口的子目录
        fitpa: FitParams (含 dt_start, dt_end, p0)
        readpa: ReadParams
        plotpa: PlotParams (含 z_list)
    """
    param_names = list(fitpa.p0.keys())
    Nz = all_fit_result[param_names[0]].shape[1]
    z_vals = np.arange(Nz)

    # ---- 4 个参数: errorbar 图 ----
    for _name in param_names:
        _arr = all_fit_result[_name]  # (Nsample, Nz)
        _mean = _arr.mean(0)
        _err = sem(_arr, jackknife=False)
        _data = {_name: (_mean, _err)}

        save_path = os.path.join(save_dir, f"{_name}.png")
        plot_errbar(
            z_vals, _data, save_path,
            xlabel="z", ylabel=_name,
            xlim=plotpa.para_xlim,
            ylim=plotpa.param_ylim.get(_name),
            title=f"{readpa.conf_short}, P={readpa.P}, fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}]",
            x_offset=0.3,
            figsize=(10, 6), dpi=150,
        )

    # ---- chi2: 散点图 ----
    _chi2_arr = all_fit_result["chi2"]  # (Nsample, Nz)
    _chi2_data = {"chi2/dof": _chi2_arr.mean(0)}

    _chi2_save_path = os.path.join(save_dir, "chi2.png")
    plot_scatter(
        z_vals, _chi2_data, _chi2_save_path,
        xlim=plotpa.para_xlim,
        ylim=[0, 2],
        xlabel="z", ylabel="chi2/dof",
        title=f"{readpa.conf_short}, P={readpa.P}, fit: tsep=[{fitpa.dt_start},{fitpa.dt_end}]",
        x_offset=0.3,
        figsize=(10, 6), dpi=150,
        show_hline=True,
        hline_y=1.0,
        hline_label="chi2/dof=1",
    )


########################################################################################
def plot_para_cmp(
    fit_data: dict,
    save_dir: str,
    readpa: ReadParams,
    fitpa_list: list[FitParams],
    plotpa: PlotParams,
):
    """
    画 4 个参数 + chi2 随 z 变化的对比图, 不同拟合窗口叠加在同一张图上.

    参数用 plot_errbar (带误差棒), chi2 用 plot_scatter (散点图).
    按 plotpa.z_step 间隔取 z 点, 避免点过密.

    参数:
        fit_data: {window_tag: npz_object} 所有窗口的拟合结果
        save_dir: fit_nex{N}/ 目录
        readpa: ReadParams
        fitpa_list: 拟合参数列表 (用于获取窗口标签和参数名)
        plotpa: PlotParams (含 z_step)
    """
    param_names = list(fitpa_list[0].p0.keys())

    # 从第一个窗口数据获取 Nz
    _first_tag = next(iter(fit_data.keys()))
    _first_arr = fit_data[_first_tag][param_names[0]]  # (Nsample, Nz)
    Nz = _first_arr.shape[1]
    z_vals = np.arange(Nz)[::plotpa.z_step]  # 按步长取 z

    # ---- 4 个参数: errorbar 图 (不同窗口 = 多组数据) ----
    for _name in param_names:
        _data = {}
        for _fitpa in fitpa_list:
            _window_tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
            _label = f"dt: {_fitpa.dt_start}~{_fitpa.dt_end}"
            _arr = fit_data[_window_tag][_name]  # (Nsample, Nz)
            _mean = _arr.mean(0)[::plotpa.z_step]
            _err = sem(_arr, jackknife=False)[::plotpa.z_step]
            _data[_label] = (_mean, _err)

        save_path = os.path.join(save_dir, f"{_name}.png")
        plot_errbar(
            z_vals, _data, save_path,
            xlabel="z", ylabel=_name,
            xlim=plotpa.para_xlim,
            ylim=plotpa.param_ylim.get(_name),
            title=f"{readpa.conf_short}, P={readpa.P}, {_name}",
            x_offset=0.3,
            figsize=(10, 6), dpi=150,
        )

    # ---- chi2: 散点图 (用 plot_scatter) ----
    _chi2_data = {}
    for _fitpa in fitpa_list:
        _window_tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
        _label = f"dt: {_fitpa.dt_start}~{_fitpa.dt_end}"
        _chi2_arr = fit_data[_window_tag]["chi2"]  # (Nsample, Nz)
        _chi2_data[_label] = _chi2_arr.mean(
            0)[::plotpa.z_step]  # (Nz_sampled,)

    _chi2_save_path = os.path.join(save_dir, "chi2.png")
    plot_scatter(
        z_vals, _chi2_data, _chi2_save_path,
        xlabel="z", ylabel="chi2/dof",
        xlim=plotpa.para_xlim,
        ylim=[0, 2],
        title=f"{readpa.conf_short}, P={readpa.P}, chi2/dof",
        x_offset=0.3,
        figsize=(10, 6), dpi=150,
        show_hline=True,
        hline_y=1.0,
        hline_label="chi2/dof=1",
    )


if __name__ == "__main__":
    print("conf_short:", readpa.conf_short)
    print("P:", readpa.P)

    # ---- Part 1: FH (读取 ratio → FH 变换 → 画 FH 图) ----
    if part_start <= 1:
        _t0 = time.perf_counter()

        # 直接读取 zch_ratio.py 生成的 ratio (无方向平均)
        print(f"\n{'='*60}")
        ratio = load_ratio(readpa)

        # 对 ratio 做 FH
        _fh_dir = outpa.get_sub_dir(os.path.join("fh"))
        all_fh = {}
        for _nex in range(readpa.nexmax + 1):
            _fh_path = os.path.join(_fh_dir, f"FH_nex{_nex}.npy")
            fh = compute_fh(ratio, _fh_path, nex=_nex)
            all_fh[_nex] = fh

        # 画对比图: 所有 nex 画在同一张图上, legend 标注 nex=?
        plot_fh(all_fh, _fh_dir, readpa, plotpa)

        _t1 = time.perf_counter()
        print(f"    FH time: {(_t1 - _t0):.2f}s\n")

        if part_end == 1:
            print("job finish")
            sys.exit(0)
    else:
        print("===== skip FH =====")

    # ---- Part 2: fit (读取 FH 结果 → 做 fit) ----
    if part_start <= 2 <= part_end:
        print(f"\n{'='*60}")
        print(f"  processing [fit]")

        # 读取 FH 结果 (始终从 1_result 读取)
        _nex_fit = fitpa_list[0].nex
        _fh_dir = os.path.join(
            os.getcwd(), "1_result",
            readpa.conf_short, f"P{readpa.P}",
            "fh"
        )
        _fh_path = os.path.join(_fh_dir, f"FH_nex{_nex_fit}.npy")
        print(f"  loading FH (nex={_nex_fit}) from: {_fh_path}")
        if not os.path.exists(_fh_path):
            print(f"    Error: FH file not found: {_fh_path}")
            sys.exit(1)
        fh = np.load(_fh_path)

        print(f"    loaded FH shape: {fh.shape}")

        # fit_nex{N}/ 目录 (已在上方创建)
        _fit_dir = outpa.sub_dir_path(os.path.join(f"fit_nex{_nex_fit}"))

        # 对每个拟合窗口做拟合
        for _fitpa in fitpa_list:
            _t_fit = time.perf_counter()
            # 每个拟合窗口一个子目录
            _window_tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
            _window_dir = outpa.get_sub_dir(
                os.path.join(f"fit_nex{_nex_fit}", _window_tag))
            do_fit_and_report(fh, _window_dir, _fitpa)
            _t_fit_end = time.perf_counter()
            print(f"      fit time: {(_t_fit_end - _t_fit):.2f}s")

        if part_end == 2:
            print("job finish")
            sys.exit(0)
    else:
        print("===== skip fit =====")

    # ---- Part 3: plot (读取 fit 结果 → 画图) ----
    if part_start <= 3 <= part_end:
        _t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  processing [plot]")

        _nex_fit = fitpa_list[0].nex
        _base_dir = "0_debug" if debug else "1_result"

        # 读取 fit 结果
        _fit_dir = os.path.join(
            os.getcwd(), _base_dir,
            readpa.conf_short, f"P{readpa.P}",
            f"fit_nex{_nex_fit}"
        )
        print(f"  loading fit results from: {_fit_dir}")

        fit_data = {}
        for _fitpa in fitpa_list:
            _window_tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
            _load_path = os.path.join(
                _fit_dir, _window_tag, f"fit_{_window_tag}.npz")
            print(f"    loading: {_load_path}")
            if not os.path.exists(_load_path):
                print(f"    Error: fit file not found: {_load_path}")
                sys.exit(1)
            fit_data[_window_tag] = np.load(_load_path)

        # ---- 画图 1: 每个窗口的参数 vs z 图 ----
        print(f"\n  --- plotting per-window parameter vs z ---")
        for _fitpa in fitpa_list:
            _window_tag = f"dt{_fitpa.dt_start}_{_fitpa.dt_end}"
            _window_dir = os.path.join(_fit_dir, _window_tag)
            print(f"\n  window: {_window_tag}")
            plot_para(
                fit_data[_window_tag], _window_dir,
                _fitpa, readpa, plotpa)

        # ---- 画图 2: 多窗口参数对比图 ----
        print(f"\n  --- plotting multi-window parameter comparison ---")
        plot_para_cmp(
            fit_data, _fit_dir, readpa, fitpa_list, plotpa)

        # ---- 画图 3: best fit 的 FH + c0 色带图 ----
        print(f"\n  --- plotting best fit FH + c0 band ---")
        _bf_params = _bestfit_params.get(readpa.P)
        _bf_nex = _bf_params["nex"]
        _bf_dt_start = _bf_params["dt_start"]
        _bf_dt_end = _bf_params["dt_end"]
        _bf_window_tag = f"dt{_bf_dt_start}_{_bf_dt_end}"

        # 读取 bestfit 的 FH 数据 (始终从 1_result)
        _bf_fh_path = os.path.join(
            os.getcwd(), "1_result",
            readpa.conf_short, f"P{readpa.P}",
            "fh", f"FH_nex{_bf_nex}.npy")
        _bf_fh = np.load(_bf_fh_path)

        # 读取 bestfit 的 c0 数据
        _bf_fit_path = os.path.join(
            os.getcwd(), "1_result",
            readpa.conf_short, f"P{readpa.P}",
            f"fit_nex{_bf_nex}", _bf_window_tag,
            f"fit_{_bf_window_tag}.npz")
        _bf_fit = np.load(_bf_fit_path)
        _bf_c0 = _bf_fit["c0"]  # (Nsample, Nz)

        # 构建 c0_data: {z: (mean, err)}
        _bf_c0_data = {}
        for _iz in plotpa.z_list:
            if _iz < _bf_c0.shape[1]:
                _mean = _bf_c0[:, _iz].mean()
                _err = sem(_bf_c0[:, _iz], jackknife=False)
                _bf_c0_data[_iz] = (_mean, _err)

        # 用 plot_fh 画图, 传入 c0_data 和色带 t 范围
        _bf_fh_dict = {_bf_nex: _bf_fh}
        _bestfit_dir = outpa.get_sub_dir(os.path.join("bestfit"))
        plot_fh(_bf_fh_dict, _bestfit_dir, readpa, plotpa,
                c0_data=_bf_c0_data,
                band_t_range=(_bf_dt_start, _bf_dt_end))

        _t1 = time.perf_counter()
        print(f"\n  total plot time: {(_t1 - _t0):.2f}s\n")

    print("job finish")
