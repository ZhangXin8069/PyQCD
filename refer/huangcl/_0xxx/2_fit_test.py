#!/public/home/huangcl/.venv/bin/python
"""
fit_FH.py

读取 FH 结果, 使用 prior 做测试拟合, 输出拟合过程和拟合结果报告.

输入:
    -c: conf_short.
    -p: momentum P.
    init_config() 中手动设置 para, nz 和 nsample.

输出:
    02_fit_test/{conf_short}/p{P}/
        nex{nex}_para{para}_nsam{nsample}.txt
        nex{nex}_para{para}_nsam{nsample}/tsep{tstart}_{tend}.txt

过程报告记录每个 (tstart, z) 的均值额外拟合结果.
结果报告由结果数组生成, 每一行结果数组对应一个 (tstart, z).
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gvar as gv
import numpy as np
from prettytable import PrettyTable

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import fit, sem  # noqa: E402


@dataclass
class FitConfig:
    """拟合配置."""
    conf_short: str
    P: int
    para: int
    use_prior: bool
    nz: int
    nsample: int
    nex: int
    dt_end: int
    dt_start_list: list[int]
    p0: dict
    prior: dict
    svdcut: float

@dataclass
class PathConfig:
    """路径配置."""
    fh_path: str
    fit_base_dir: str
    fit_report_dir: str


def init_config(conf_short: str, P: int):
    """
    初始化测试参数, 拟合参数和输入输出路径.

    输入:
        conf_short: 组态配置名称.
        P: 动量参数.

    输出:
        cfg: FitConfig.
        pcfg: PathConfig.
    """
    # ===== 测试参数, 用户手动修改 =====
    use_prior = True
    nz = 3
    nsample = 1000

    # ===== 全局 prior, 用户手动修改 =====
    _prior = {
        "c0": gv.gvar(0.6, 0.5),
        "c2": gv.gvar(0.0, 0.5),
        "c1": gv.gvar(-1, 0.5),
        "dE": gv.gvar(0.7, 0.5),
    }

    # ===== P 相关的拟合参数, 用户手动修改 =====
    if conf_short == "c24p29":
        _fit_params_by_P = {
            0: {"para": 1, "nex": 2, "dt_start_list": list(range(6, 10)), "dt_end": 15},
            2: {"para": 1, "nex": 2, "dt_start_list": list(range(6, 9)), "dt_end": 11},
            3: {"para": 1, "nex": 2, "dt_start_list": list(range(4, 7)), "dt_end": 10},
            4: {"para": 1, "nex": 2, "dt_start_list": list(range(4, 9)), "dt_end": 10},
            5: {"para": 1, "nex": 2, "dt_start_list": list(range(7, 8)), "dt_end": 10},
            6: {"para": 1, "nex": 2, "dt_start_list": list(range(7, 8)), "dt_end": 10},
        }
    else:
        print(f"conf {conf_short} not exist.")
        sys.exit(1)

    if P not in _fit_params_by_P:
        print(f"Error: P={P} not configured in _fit_params_by_P")
        sys.exit(1)

    _p = _fit_params_by_P[P]
    para = _p["para"]

    # ===== 根据 para 选择 prior, p0 保留用于 FitParams 接口 =====
    if para == 1:
        prior = {"c0": _prior["c0"]}
    elif para == 3:
        prior = {
            "c0": _prior["c0"],
            "c1": _prior["c1"],
            "dE": _prior["dE"],
        }
    elif para == 4:
        prior = {
            "c0": _prior["c0"],
            "c1": _prior["c1"],
            "c2": _prior["c1"],
            "dE": _prior["dE"],
        }
    else:
        print(f"Error: para={para} not supported (must be 1, 3, or 4)")
        sys.exit(1)
    p0 = {name: value.mean for name, value in prior.items()}

    cfg = FitConfig(
        conf_short=conf_short,
        P=P,
        para=para,
        use_prior=use_prior,
        nz=nz,
        nsample=nsample,
        nex=_p["nex"],
        dt_end=_p["dt_end"],
        dt_start_list=_p["dt_start_list"],
        p0=p0,
        prior=prior,
        svdcut=1e-12,
    )

    for dt_start in cfg.dt_start_list:
        if dt_start < 2 * cfg.nex:
            print(
                f"Error: dt_start ({dt_start}) must be => 2*nex "
                f"({2 * cfg.nex})")
            sys.exit(1)

    pcfg = PathConfig(
        fh_path=os.path.join(
            os.getcwd(),
            "01_result",
            conf_short,
            f"p{P}",
            "1_fh",
            f"FH_nex{cfg.nex}.npy",
        ),
        fit_base_dir=os.path.join(
            os.getcwd(), "02_fit_test", conf_short, f"p{P}"
        ),
        fit_report_dir=os.path.join(
            os.getcwd(),
            "02_fit_test",
            conf_short,
            f"p{P}",
            f"nex{cfg.nex}_para{cfg.para}_nsam{cfg.nsample}",
        ),
    )
    os.makedirs(pcfg.fit_base_dir, exist_ok=True)
    os.makedirs(pcfg.fit_report_dir, exist_ok=True)
    return cfg, pcfg


def fh_model(t, p):
    """
    FH(t) 模型.

    输入:
        t: 时间坐标.
        p: 拟合参数.

    输出:
        FH 模型值.
    """
    result = p["c0"] * np.ones_like(t, dtype=float)
    if "c1" in p and "dE" in p:
        result += p["c1"] * t * np.exp(-p["dE"] * t)
    if "c2" in p:
        result += p["c2"] * np.exp(-p["dE"] * t)
    return result


def model_description(para: int) -> str:
    """返回 FH 模型描述."""
    if para == 1:
        return "FH(t) = c0"
    if para == 3:
        return "FH(t) = c0 + c1*t*exp(-dE*t)"
    if para == 4:
        return "FH(t) = c0 + c1*t*exp(-dE*t) + c2*exp(-dE*t)"
    return "unknown"


def do_fit(
    fh: np.ndarray,
    dt_start: int,
    dt_end: int,
    nex: int,
    cfg: FitConfig,
    report_dir: str,
    prior=None,
    p0=None,
    svdcut=None,
) -> np.ndarray:
    """
    对一个 tstart 窗口和前 nz 个 z 做拟合.

    输入:
        fh: FH 数据, 第一轴为样本, 第二轴为时间, 第三轴为 z.
        dt_start: 拟合窗口起点.
        dt_end: 拟合窗口终点.
        nex: FH 拟合使用的激发态数.
        cfg: 测试配置.
        report_dir: 单窗口拟合报告目录.
        prior: 拟合先验参数.
        p0: 拟合初值.
        svdcut: SVD cut.

    输出:
        result_array: 当前窗口的结果数组.
    """
    n_sample = min(cfg.nsample, fh.shape[0])
    n_z = min(cfg.nz, fh.shape[2])
    t_vals = np.arange(dt_start, dt_end + 1, dtype=int)
    param_names = list(p0.keys() if p0 is not None else prior.keys())

    report_lines = [
        "=" * 72,
        f"tsep={dt_start}-{dt_end}, nex: {nex}, parameters: {cfg.para}",
        f"z = {n_z}, fit sample number: {n_sample}/{fh.shape[0]}",
        f"fit mode: {'prior' if cfg.use_prior else 'p0'}",
        "=" * 72,
        "",
    ]

    # 每行依次记录 tstart, z, 参数 mean, 参数 sem, chi2/dof.
    row_width = 2 + 2 * len(param_names) + 1
    result_array = np.full((n_z, row_width), np.nan, dtype=float)
    result_array[:, 0] = dt_start
    result_array[:, 1] = np.arange(n_z)

    for iz in range(n_z):
        time_start = time.perf_counter()

        # 传入全部样本构造协方差, 只拟合前 nsample 个样本.
        y_data = fh[:, t_vals, iz]
        fit_result, _cov, cond, mean_fit = fit(
            y_coor=y_data,
            x_coor=t_vals,
            model=fh_model,
            prior=prior,
            p0=p0,
            svdcut=svdcut,
            jackknife=False,
            debug=True,
            debugNfit=cfg.nsample,
        )

        time_end = time.perf_counter()
        print(
            f"      tsep={dt_start}, z={iz}, "
            f"time={(time_end - time_start):.2f}s")

        report_lines.extend([
            f"z = {iz}, condition number = {cond:.6g}",
            mean_fit.format(maxline=True),
            "",
        ])

        for iparam, name in enumerate(param_names):
            values = fit_result[name][:n_sample]
            result_array[iz, 2 + iparam] = np.mean(values)
            result_array[iz, 2 + len(param_names) + iparam] = sem(
                values, False
            )
        result_array[iz, -1] = np.mean(fit_result["chi2"][:n_sample])

    report_path = os.path.join(
        report_dir,
        f"tsep{dt_start}_{dt_end}.txt",
    )
    with open(report_path, "w") as report_file:
        report_file.write("\n".join(report_lines))
        report_file.write("\n")
    print(f"fit report saved to: {report_path}")
    return result_array


def summary(
    result_array: np.ndarray,
    cfg: FitConfig,
    pcfg: PathConfig,
) -> None:
    """
    只根据结果数组生成拟合结果报告.

    输入:
        result_array: 每行对应一个 (tstart, z) 的结果数组.
        cfg: 测试配置.
        pcfg: 路径配置.

    输出:
        结果报告保存到 02_fit_test/{conf_short}/p{P}/.
    """
    param_names = list(cfg.p0.keys())
    name_prefix = (
        f"nex{cfg.nex}_para{cfg.para}_nsam{cfg.nsample}"
    )
    result_path = os.path.join(
        pcfg.fit_base_dir, f"{name_prefix}.txt"
    )

    result_lines = [
        "=" * 72,
        "Fit Result Report",
        "=" * 72,
        f"conf_short: {cfg.conf_short}",
        f"P: {cfg.P}",
        f"model: {model_description(cfg.para)}",
        f"para: {cfg.para}",
        f"nex: {cfg.nex}",
        f"tsep end: {cfg.dt_end}",
        f"fit sample number: {cfg.nsample}",
        "covariance sample number: all input samples",
        f"z number: {cfg.nz}",
        f"fit mode: {'prior' if cfg.use_prior else 'p0'}",
        "",
    ]

    mean_start = 2
    sem_start = 2 + len(param_names)
    for tstart in np.unique(result_array[:, 0]).astype(int):
        table = PrettyTable()
        table.field_names = ["z"] + param_names + ["chi2/dof"]
        for field_name in table.field_names:
            table.align[field_name] = "c"

        rows = result_array[result_array[:, 0] == tstart]
        for row_data in rows:
            row = [str(int(row_data[1]))]
            row.extend(
                f"{mean:.3f}({sem*1000:.0f})"
                for mean, sem in zip(
                    row_data[mean_start:sem_start],
                    row_data[sem_start:-1],
                )
            )
            row.append(f"{row_data[-1]:.3f}")
            table.add_row(row)

        result_lines.extend([
            "=" * 72,
            f"tsep start = {tstart}, tsep end = {cfg.dt_end}",
            str(table),
            "",
        ])

    with open(result_path, "w") as result_file:
        result_file.write("\n".join(result_lines))
    print(f"result report saved to: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit FH test data with prior and write text reports."
    )
    parser.add_argument(
        "-c",
        type=str,
        default="c24p29",
        help="conf_short (default: c24p29)",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=0,
        help="momentum P (default: 0ß)",
    )
    args = parser.parse_args()

    cfg, pcfg = init_config(args.c, args.p)
    print(
        f"conf_short: {cfg.conf_short}, P: {cfg.P}, "
        f"para: {cfg.para}, use_prior: {cfg.use_prior}, "
        f"nz: {cfg.nz}, nsample: {cfg.nsample}"
    )

    print(f"\nloading FH (nex={cfg.nex}) from: {pcfg.fh_path}")
    fh = np.load(pcfg.fh_path)
    print("loaded FH data")

    result_arrays = [
        do_fit(
            fh,
            dt_start,
            cfg.dt_end,
            cfg.nex,
            cfg,
            pcfg.fit_report_dir,
            prior=cfg.prior if cfg.use_prior else None,
            p0=cfg.p0,
            svdcut=cfg.svdcut,
        )
        for dt_start in cfg.dt_start_list
    ]
    result_array = np.vstack(result_arrays)
    summary(result_array, cfg, pcfg)
    print("\nfit_FH test done.")
