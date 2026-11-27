#!/public/home/huangcl/.venv/bin/python
"""对指定 nex 的 z=0 FH 数据做常数模型拟合.

输入:
    01_data/<conf_short>/P<P>/fh_type<ratio_type>.npy.

输出:
    01_data/<conf_short>/P<P>/fit_z0_type<ratio_type>_nex<nex>.npz.
    02_fig_rep/<conf_short>/P<P>/fit_z0_type<ratio_type>_nex<nex>.txt.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

here = Path(__file__).resolve()
sys.path.append("/public/home/huangcl/07_pion_gluon_pdf/98_tools")

from analysis_tools import sem  # noqa: E402  # type: ignore
from output_tools import format_mean_err, save_npz, save_txt  # noqa: E402  # type: ignore
from feyn_hell_tools import fit_window  # noqa: E402  # type: ignore


@dataclass
class FitConfig:
    """z=0 FH 拟合参数与路径."""
    conf_short: str
    p: int
    ratio_type: int
    nex: int
    dt_start: int
    dt_end: int
    prior: dict
    tol: dict
    fh_path: Path
    data_dir: Path
    report_dir: Path


def init() -> FitConfig:
    """
    初始化拟合参数.

    输出:
    -----
    Fit: z=0 FH 拟合参数与路径.
    """
    CONF_SHORT = "c24p29"
    MOM = 5
    RATIO_TYPE = 0
    NEX = 2
    DT_START = 5
    DT_END = 7
    PRIOR = {"c0": 0.5}
    TOL = {"c0": 0.5}

    data_dir = here.parent / "01_data" / CONF_SHORT / f"P{MOM}"
    report_dir = here.parent / "02_fig_rep" / CONF_SHORT / f"P{MOM}"
    return FitConfig(
        conf_short=CONF_SHORT,
        p=MOM,
        ratio_type=RATIO_TYPE,
        nex=NEX,
        dt_start=DT_START,
        dt_end=DT_END,
        prior=PRIOR,
        tol=TOL,
        fh_path=data_dir / f"fh_type{RATIO_TYPE}.npy",
        data_dir=data_dir,
        report_dir=report_dir,
    )


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    Fit = init()
    print(f"conf_short: {Fit.conf_short}, P: {Fit.p}, ratio_type: "
          f"{Fit.ratio_type}, nex: {Fit.nex}, "
          f"fit range: [{Fit.dt_start}, {Fit.dt_end}]")

    # para: sample, nex, dt, z
    fh = np.load(Fit.fh_path)
    print(f"loaded FH: {Fit.fh_path}, shape={fh.shape}")

    # para: sample, dt
    fh_z0 = fh[:, Fit.nex, :, 0]
    fit_result, fit_mean, elapsed, n_bound = fit_window(
        fh_z0, Fit.dt_start, Fit.dt_end, Fit.prior, Fit.tol,
        print_detail=True)

    output_stem = f"fit_z0_type{Fit.ratio_type}_nex{Fit.nex}"
    save_npz(
        Fit.data_dir / f"{output_stem}.npz",
        **fit_result,
    )

    c0 = fit_result["c0"]
    chi2dof = fit_result["chi2dof"]
    report = (
        f"FH z=0 constant fit\n"
        f"conf_short: {Fit.conf_short}\n"
        f"P: {Fit.p}\n"
        f"ratio_type: {Fit.ratio_type}\n"
        f"nex: {Fit.nex}\n"
        f"fit range: [{Fit.dt_start}, {Fit.dt_end}]\n"
        f"FH data: {Fit.fh_path}\n"
        f"fit time: {elapsed:.2f}s\n"
        f"n_bound: {n_bound}\n"
        f"c0: {format_mean_err(c0.mean(), sem(c0))}\n"
        f"chi2/dof: {chi2dof.mean():.2f}\n"
        f"fit mean c0: {fit_mean['c0']:.3g}\n"
        f"fit mean chi2/dof: {fit_mean['chi2dof']:.2f}\n"
    )
    save_txt(Fit.output_dir / f"{output_stem}.txt", report)

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s, "
          f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
