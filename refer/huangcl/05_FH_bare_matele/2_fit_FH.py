#!/public/home/huangcl/.venv/bin/python
"""
fit_FH.py

读取 FH 结果, 对每个拟合窗口做 fit, 保存拟合结果.

用法:
    # 使用 prior 拟合 (生产模式), 只拟合 z=3, 使用全部样本
    python fit_FH.py -c L24x72 -p 4 -z 3 -u

    # 使用 p0 拟合 (调试模式), 拟合所有 z, 每个 z 只取前 100 个样本
    python fit_FH.py -c L24x72 -p 4

输出:
    01_result/{conf_short}/P{P}/fit/para{model}_n{nex}_tsep{start}_{end}/
        fit.npz
        report.txt
    或
    00debug/{conf_short}/P{P}/fit/para{model}_n{nex}_tsep{start}_{end}/
        fit.npz
        report.txt
"""

import gvar as gv
import numpy as np
import argparse
import os
import sys
import time
from pathlib import Path
from prettytable import PrettyTable
from dataclasses import dataclass, field

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import sem, fit, FitParams  # noqa: E402


# ===== 定义 dataclass =====

@dataclass
class FitConfig:
    """拟合配置"""
    conf_short: str
    P: int
    model: int                 # 模型参数个数: 1, 3, 4
    nex: int                   # FH 时使用的 nex 值 (决定加载哪个 FH_nex{N}.npy)
    dt_end: int                # 拟合窗口右端点
    dt_start_list: list[int]   # 拟合窗口左端点列表
    p0: dict                   # 拟合参数初值 (从 prior 取 mean)
    prior: dict                # prior (gvar 格式)
    svdcut: float              # SVD cut (None 表示不使用)


@dataclass
class PathConfig:
    """路径配置"""
    fh_path: str               # FH 数据文件路径 (读取)
    fit_base_dir: str          # 拟合结果保存目录 (输出)


# ===== 初始化配置 (返回 cfg 和路径) =====

def init_config(conf_short: str, P: int, debug_mode: bool):
    """
    根据 conf_short 和 P 初始化配置, 创建输出目录.

    返回:
        cfg: FitConfig
        pcfg: PathConfig
    """
    # ===================================================
    # ===== 全局 prior (与 conf_short 无关, 用户填写) =====
    # ===================================================
    _prior = {
        'c0': gv.gvar(0.6, 0.5),
        'c2': gv.gvar(0.0, 0.5),
        'c1': gv.gvar(-1, 0.5),
        'dE': gv.gvar(0.7, 0.5),
    }
    # ===================================================

    # ===================================================
    # ===== P 相关的拟合参数 (手动调整) =====
    # ===================================================
    if conf_short == "L24x72":
        _fit_params_by_P = {
            0: {"model": 1, "nex": 2, "dt_start_list": list(range(6, 10)), "dt_end": 15},
            2: {"model": 1, "nex": 2, "dt_start_list": list(range(6, 9)), "dt_end": 11},
            3: {"model": 4, "nex": 1, "dt_start_list": list(range(4, 7)), "dt_end": 10},
            4: {"model": 3, "nex": 1, "dt_start_list": list(range(5, 8)), "dt_end": 10},
            5: {"model": 1, "nex": 1, "dt_start_list": list(range(6, 9)), "dt_end": 9},
            6: {"model": 1, "nex": 2, "dt_start_list": list(range(4, 5)), "dt_end": 7},
        }
    else:
        print(f"conf {conf_short} not exist.")
        sys.exit()

    if P not in _fit_params_by_P:
        print(f"Error: P={P} not configured in _fit_params_by_P")
        sys.exit(1)

    _p = _fit_params_by_P[P]
    model = _p["model"]
    # ===================================================

    # ===== 根据 model 从 _prior 选取, 统一装填 p0 和 prior =====
    # 模型: FH(t) = c0 + c1 * t * exp(-dE * t) + c2 * exp(-dE * t)
    if model == 1:
        prior = {'c0': _prior['c0']}
    elif model == 3:
        prior = {'c0': _prior['c0'], 'c1': _prior['c1'], 'dE': _prior['dE']}
    elif model == 4:
        prior = {
            'c0': _prior['c0'], 'c1': _prior['c1'],
            'c2': _prior['c1'], 'dE': _prior['dE'],
        }
    else:
        print(f"Error: model={model} not supported (must be 1, 3, or 4)")
        sys.exit(1)
    p0 = {k: v.mean for k, v in prior.items()}

    cfg = FitConfig(
        conf_short=conf_short,
        P=P,
        model=model,
        nex=_p["nex"],
        dt_end=_p["dt_end"],
        dt_start_list=_p["dt_start_list"],
        p0=p0,
        prior=prior,
        svdcut=1e-12,
    )

    # 检查: 每个拟合窗口的 dt_start 必须 > 2*nex, 否则 FH 在该点无意义
    for _dt in cfg.dt_start_list:
        if _dt < 2 * cfg.nex:
            print(
                f"    Error: dt_start ({_dt}) must be > 2*nex ({2 * cfg.nex})")
            print(f"    FH data at t <= 2*nex is meaningless (all zeros).")
            sys.exit(1)

    # 路径配置 (统一在 init_config 中管理)
    _base_dir = "00debug" if debug_mode else "01_result"
    pcfg = PathConfig(
        fh_path=os.path.join(
            os.getcwd(), "01_result", conf_short, f"P{P}",
            "fh", f"FH_nex{cfg.nex}.npy"),
        fit_base_dir=os.path.join(
            os.getcwd(), _base_dir, conf_short, f"P{P}", "fit"),
    )
    os.makedirs(pcfg.fit_base_dir, exist_ok=True)

    return cfg, pcfg


# ===== Model =====

def fh_model(t, p):
    """
    FH(t) 模型, 根据 p 中的参数键自动选择模型.

    model=1: FH(t) = c0
    model=3: FH(t) = c0 + c1 * t * exp(-dE * t)
    model=4: FH(t) = c0 + c1 * t * exp(-dE * t) + c2 * exp(-dE * t)

    """
    # model=1: 常数模型
    result = p['c0'] * np.ones_like(t, dtype=float)

    # model=3 或 4: 添加 c1 * t * exp(-dE * t)
    if 'c1' in p and 'dE' in p:
        result += p['c1'] * t * np.exp(-p['dE'] * t)

    # model=4: 添加 c2 * exp(-dE * t)
    if 'c2' in p:
        result += p['c2'] * np.exp(-p['dE'] * t)

    return result


# ===== Fit Logic =====

def do_fit_and_report(
    fh: np.ndarray,
    save_dir: str,
    fitpa: FitParams,
    model: int,
    z_fit: int = None,
    debug_mode: bool = False,
):
    """
    对 FH 变换结果, 用给定的 fitpa 做 fit, 输出报告并保存结果.

    参数:
        fh: (Nsample, dt, z)
        save_dir: 保存目录
        fitpa: FitParams
        model: 模型参数个数 (1, 3, 4)
        z_fit: 若指定, 只拟合该 z; 若为 None, 拟合所有 z
        debug_mode: 若为 True, 每个 z 只取前 100 个样本
    """
    Nsample, _, Nz = fh.shape
    dt_start = fitpa.dt_start
    dt_end = fitpa.dt_end
    param_names = list(fitpa.p0.keys())
    window_tag = f"tsep{dt_start}_{dt_end}"

    t_vals = np.arange(dt_start, dt_end + 1, dtype=int)

    # 确定样本数
    if debug_mode:
        Nfit = min(100, Nsample)
        print(f"    debug mode: Nfit = {Nfit}/{Nsample}")
    else:
        Nfit = Nsample

    # 确定要拟合的 z 列表
    if z_fit is not None:
        z_list = [z_fit]
    else:
        z_list = list(range(Nz))

    print(
        f"\n    fitting window: t = [{dt_start}, {dt_end}], "
        f"Nfit = {Nfit}/{Nsample}, z_list = {z_list}")

    # 收集拟合结果
    all_fit_result = {name: np.zeros((Nfit, len(z_list)))
                      for name in param_names + ["chi2"]}
    all_cond = np.zeros(len(z_list))

    # 报告行
    lines = []
    sep_line = "=" * 72
    lines.append(sep_line)
    lines.append(f"  Fit Report, {window_tag}, nex={fitpa.nex}")
    lines.append(sep_line)
    # 模型描述
    if model == 1:
        _model_str = "FH(t) = c0"
    elif model == 3:
        _model_str = "FH(t) = c0 + c1*t*exp(-dE*t)"
    elif model == 4:
        _model_str = "FH(t) = c0 + c1*t*exp(-dE*t) + c2*exp(-dE*t)"
    else:
        _model_str = "unknown"
    lines.append(f"  model : {_model_str}")
    lines.append(f"  fitpa : {fitpa}")
    if debug_mode:
        lines.append(f"  mode  : DEBUG (Nfit={Nfit})")
    else:
        lines.append(f"  mode  : PRODUCTION (z={z_fit})")
    lines.append(sep_line)
    lines.append("")

    # 对每个 z 做拟合
    for _idx, _iz in enumerate(z_list):
        _tz = time.perf_counter()
        y_data = fh[:, t_vals, _iz]  # (Nsample, Nt_fit)

        _fit_result, _cov, _cond, _last_fit = fit(
            y_coor=y_data,
            x_coor=t_vals,
            model=fh_model,
            fitpa=fitpa,
            jackknife=False,
            debug=debug_mode,
            debugNfit=100,
        )
        _tz_end = time.perf_counter()
        print(f"      z = {_iz}, time = {(_tz_end - _tz):.2f}s")

        # fit() 返回 Nsample 大小, 只取前 Nfit 个有效数据
        for name in param_names + ["chi2"]:
            all_fit_result[name][:, _idx] = _fit_result[name][:Nfit]
        all_cond[_idx] = _cond

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

    for _idx, _iz in enumerate(z_list):
        row = [f"{_iz}"]
        for name in param_names:
            mean = all_fit_result[name][:, _idx].mean()
            err = sem(all_fit_result[name][:, _idx], False)
            row.append(f"{mean:.3f}({err * 1e3:.0f})")
        row.append(f"{all_fit_result['chi2'][:, _idx].mean():.2g}")
        summary_tbl.add_row(row)

    lines.append(str(summary_tbl))
    lines.append("")

    # 保存报告 (生产模式加 z 编号, 避免 submit.sh 循环覆盖)
    if z_fit is not None:
        report_name = f"report_z{z_fit}.txt"
        result_name = f"fit_z{z_fit}.npz"
    else:
        report_name = "report.txt"
        result_name = "fit.npz"

    report_path = os.path.join(save_dir, report_name)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"    report saved to: {report_path}")

    # 保存拟合结果 (生产模式单 z 时 squeeze 掉多余的维度)
    _save_dict = {
        name: np.squeeze(arr) if arr.shape[1] == 1 else arr
        for name, arr in all_fit_result.items()
    }
    result_path = os.path.join(save_dir, result_name)
    np.savez(result_path, **_save_dict)
    print(f"    fit result saved to: {result_path}")


# ===== 主函数 =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit FH data. Use -z for single-z mode (all samples); "
                    "without -z for debug mode (all z, 100 samples each). "
                    "Use -u to use prior (default: use p0).")
    parser.add_argument("-c", type=str, default="L24x72",
                        help="conf_short (default: L24x72)")
    parser.add_argument("-p", type=int, default=2,
                        help="momentum P (default: 2)")
    parser.add_argument("-z", type=int, default=None,
                        help="z value to fit (default: None = debug mode, fit all z)")
    parser.add_argument("-u", "--use-prior", action="store_true",
                        help="use prior for fitting (default: use p0)")
    args = parser.parse_args()

    conf_short = args.c
    P = args.p
    z_fit = args.z
    debug_mode = (z_fit is None)
    use_prior = args.use_prior

    # 初始化配置和路径
    cfg, pcfg = init_config(conf_short, P, debug_mode)

    print(f"conf_short: {conf_short}, P: {P}, model: {cfg.model}")
    if use_prior:
        print(f"fit mode: using PRIOR")
    else:
        print(f"fit mode: using p0")
    if z_fit is not None:
        print(f"z mode: PRODUCTION, z = {z_fit}")
    else:
        print(f"z mode: DEBUG (no -z), fitting all z with 100 samples each")

    # ---- 加载 FH 数据 ----
    print(f"\nloading FH (nex={cfg.nex}) from: {pcfg.fh_path}")
    if not os.path.exists(pcfg.fh_path):
        print(f"  Error: FH file not found: {pcfg.fh_path}")
        sys.exit(1)
    fh = np.load(pcfg.fh_path)
    print(f"  loaded FH shape: {fh.shape}")

    # ---- 构建 FitParams 列表 ----
    # use_prior=True 时用 prior, 否则用 p0 (prior=None)
    fitpa_list = [
        FitParams(
            nex=cfg.nex,
            p0=cfg.p0,
            prior=cfg.prior if use_prior else None,
            dt_start=_dt,
            dt_end=cfg.dt_end,
            svdcut=cfg.svdcut,
        )
        for _dt in cfg.dt_start_list
    ]

    # ---- 对每个拟合窗口做拟合 ----
    for _fitpa in fitpa_list:
        _t_fit = time.perf_counter()
        _window_tag = f"tsep{_fitpa.dt_start}_{_fitpa.dt_end}"
        _sub_dir_name = f"para{cfg.model}_n{cfg.nex}_{_window_tag}"
        _window_dir = os.path.join(pcfg.fit_base_dir, _sub_dir_name)
        os.makedirs(_window_dir, exist_ok=True)

        do_fit_and_report(
            fh, _window_dir, _fitpa, cfg.model,
            z_fit=z_fit, debug_mode=debug_mode)

        _t_fit_end = time.perf_counter()
        print(f"    window fit time: {(_t_fit_end - _t_fit):.2f}s")

    print("\nfit_FH done.")
