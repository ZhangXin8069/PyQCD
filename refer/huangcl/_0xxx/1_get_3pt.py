#!/public/home/huangcl/.venv/bin/python
"""计算方向平均并重采样后的 pion-gluon disconnected 3pt 与 2pt.

输入:
    各方向的 pion 2pt 和胶子圈数据.

输出:
    01_data/<conf_short>/P<P>/3pt_disc.npz, keys: 3pt_disc, 2pt.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

here = Path(__file__).resolve()
project_dir = here.parents[1]
sys.path.insert(0, str(project_dir / "98_tools"))
sys.path.insert(0, str(project_dir / "01_check_2pt"))
from analysis_tools import resample  # noqa: E402  # type: ignore
from output_tools import save_npz  # noqa: E402  # type: ignore
from pion_io_tools import (  # noqa: E402  # type: ignore
    load_gluon_unpol_shenxm, load_pion_2pt,
)
from profiling_tools import print_mem_peak  # noqa: E402  # type: ignore
from ratio_tools import calc_gluon_3pt  # noqa: E402  # type: ignore
from snsc_params import get_conf  # noqa: E402  # type: ignore


@dataclass
class CalcConfig:
    """组态与数据路径参数."""
    conf_short: str
    conf_name: str
    conf_num: int
    exclude_ids: list[int]
    z_max: int
    space_sep_max: int
    p: int
    dt_max: int
    dirs: list[str]
    data_dir: Path


def init() -> CalcConfig:
    """
    初始化运行参数.

    输出:
    -----
    Calc: 组态与数据路径参数.
    """
    CONF_SHORT = "c24p29"
    MOM = 5
    Z_MAX = 18
    DT_MAX = 20

    CONF_PARAMS = {
        "c24p29": {"conf_num": 879, "exclude_ids": [12300],
                   "space_sep_max": 17, "dirs": ["x", "y", "z"]},
        "c32p29": {"conf_num": 984, "exclude_ids": [18350, 23050, 28400],
                   "space_sep_max": 17, "dirs": ["x", "y", "z"]},
    }
    params = CONF_PARAMS[CONF_SHORT]
    conf = get_conf(CONF_SHORT)
    data_dir = here.parent / "01_data" / CONF_SHORT / f"P{MOM}"

    return CalcConfig(
        conf_short=CONF_SHORT,
        conf_name=conf["conf_name"],
        conf_num=params["conf_num"],
        exclude_ids=params["exclude_ids"],
        z_max=Z_MAX,
        space_sep_max=params["space_sep_max"],
        p=MOM,
        dt_max=DT_MAX,
        dirs=params["dirs"],
        data_dir=data_dir,
    )


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    Calc = init()
    print("conf_short:", Calc.conf_short)
    print("conf_name:", Calc.conf_name)
    print("Nconf:", Calc.conf_num)
    print("data dir:", Calc.data_dir)

    corr3_sum = 0
    corr2_sum = 0
    loop_sum = 0
    for _dir in Calc.dirs:
        time0 = time.perf_counter()

        # para: conf, dt, tsrc
        correlators = load_pion_2pt(
            Calc.conf_short, Calc.conf_num, Calc.p, _dir, Calc.exclude_ids)
        # para: conf, tsrc, dt
        corr2 = correlators[:, :Calc.dt_max, :].transpose(0, 2, 1)

        # para: conf, tau, z
        gluon_loop = load_gluon_unpol_shenxm(
            Calc.conf_short, Calc.conf_num, _dir,
            Calc.space_sep_max, Calc.exclude_ids)
        gluon_loop = gluon_loop[:, :, :Calc.z_max]

        corr3_sum = corr3_sum + calc_gluon_3pt(corr2, gluon_loop).mean(axis=1)
        corr2_sum = corr2_sum + corr2.mean(axis=1)
        loop_sum = loop_sum + gluon_loop.mean(axis=1)

        print_mem_peak("Peak Memory", use_gpu=False)
        print(f"dir {_dir} spend time: {(time.perf_counter() - time0):.0f}s\n")

    n_dir = len(Calc.dirs)
    corr3_avg = corr3_sum / n_dir
    corr2_avg = corr2_sum / n_dir
    loop_avg = loop_sum / n_dir

    # para: sample, dt; sample, dt, dtau, z; sample, z
    corr2_re = resample(corr2_avg)
    corr3_re = resample(corr3_avg)
    loop_re = resample(loop_avg)
    corr3_disc = corr3_re - \
        corr2_re[:, :, np.newaxis, np.newaxis] * \
        loop_re[:, np.newaxis, np.newaxis, :]

    save_npz(Calc.data_dir / "corr.npz", name="data",
             **{"3pt_disc": corr3_disc.real, "2pt": corr2_re.real})

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s, "
          f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
