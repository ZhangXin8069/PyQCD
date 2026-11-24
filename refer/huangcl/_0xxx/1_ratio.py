#!/public/home/huangcl/.venv/bin/python
"""氘核胶子非极化 PDF ratio 计算主流程.

功能:
    按方向循环, 逐个方向读取零动量氘核 2pt 和胶子圈, 计算 3pt,
    在外部完成 tsrc 平均, tau 平均与方向平均, 再做 bootstrap, 然后
    调用 ratio_tools 计算 ratio, 保存 ratio.npy, 最后自动调用
    画图文件 2_plot_ratio.py 画图.

输出:
    01_data/<conf_short>/P<P>/ratio.npy
    02_fig/<conf_short>/P<P>/ratio_z<z>.png (由画图文件生成)
"""

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

here = Path(__file__).resolve()
project_dir = here.parents[1]
sys.path.insert(0, str(project_dir / "98_tools"))
sys.path.insert(0, str(project_dir / "07_get_channel_2pt"))
from analysis_tools import resample  # noqa: E402  # type: ignore
from get_channel_2pt import (  # noqa: E402  # type: ignore
    load_NPcorr, load_gluon_unpol, proj_deuteron_chan,
)
from output_tools import save_npy  # noqa: E402  # type: ignore
from profiling_tools import print_mem_peak  # noqa: E402  # type: ignore
from ratio_tools import (  # noqa: E402  # type: ignore
    calc_gluon_3pt, calc_gluon_ratio,
)
from snsc_params import get_conf  # noqa: E402  # type: ignore


@dataclass
class CalcConfig:
    """组态, 样本与输出路径参数."""
    conf_short: str
    conf_name: str
    conf_num: int
    exclude_ids: list[int]
    Nt: int
    Nx: int
    z_max: int
    delta_z: int
    p: int
    dt_max: int
    dirs: list[str]
    data_dir: Path


def init():
    """
    初始化运行参数.

    输出:
    -----
    Calc: CalcConfig 实例.
    """

    CONF_SHORT = "c24p29"
    MOM = 0
    Z_MAX = 20
    DT_MAX = 20

    # 各组态的读取参数, 键为 conf_short
    CONF_PARAMS = {
        "c24p29": {"conf_num": 359, "exclude_ids": [12300],
                   "delta_z": 30, "dirs": ["x"]},
        "c32p29": {"conf_num": 149, "exclude_ids": [18350],
                   "delta_z": 30, "dirs": ["x"]},
    }

    conf_params = CONF_PARAMS[CONF_SHORT]
    conf = get_conf(CONF_SHORT)

    # 数据保存目录
    data_dir = here.parent / "01_data" / CONF_SHORT / f"P{MOM}"

    Calc = CalcConfig(
        conf_short=CONF_SHORT,
        conf_name=conf["conf_name"],
        conf_num=conf_params["conf_num"],
        exclude_ids=conf_params["exclude_ids"],
        Nt=conf["Nt"],
        Nx=conf["Nx"],
        z_max=Z_MAX,
        delta_z=conf_params["delta_z"],
        p=MOM,
        dt_max=DT_MAX,
        dirs=conf_params["dirs"],
        data_dir=data_dir,
    )
    return Calc


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    Calc = init()

    print("conf_short:", Calc.conf_short)
    print("conf_name:", Calc.conf_name)
    print("Nconf:", Calc.conf_num)
    print("data dir:", Calc.data_dir)

    # ---- 逐方向读取 2pt 和胶子圈, 计算 3pt, 并做 tsrc 平均, tau 平均 ----
    corr2_sum = 0
    corr3_sum = 0
    loop_sum = 0
    for _dir in Calc.dirs:
        time0 = time.perf_counter()

        # 1. 读取该方向的零动量氘核 2pt
        correlators = load_NPcorr(Calc.conf_short, Calc.conf_num,
                                  Calc.p, _dir, Calc.exclude_ids)
        corr2 = proj_deuteron_chan(correlators).mean(axis=-1)
        # 换轴: conf, dt, t_src -> conf, t_src, dt
        corr2 = corr2[:, :Calc.dt_max, :].transpose(0, 2, 1)
        del correlators

        # 2. 读取该方向非极化胶子 PDF 的算符数据, para: conf, tau, z
        _gluon_loop = load_gluon_unpol(Calc.conf_short, Calc.conf_num, _dir,
                                       Calc.delta_z, Calc.exclude_ids)
        _gluon_loop = _gluon_loop[:, :, :Calc.z_max]

        # 3. 计算 gluon 的 3pt, para: conf, tsrc, dt, dtau, z
        _corr3 = calc_gluon_3pt(corr2, _gluon_loop)

        # 4. 对 tsrc 求平均, para: conf, dt, dtau, z
        corr3_sum = corr3_sum + _corr3.mean(axis=1)
        del _corr3

        # 5. 2pt 对 tsrc 求平均, para: conf, dt
        corr2_sum = corr2_sum + corr2.mean(axis=1)
        del corr2

        # 6. 胶子圈真空期望与时间无关
        # para: conf, z
        loop_sum = loop_sum + _gluon_loop.mean(axis=1)
        del _gluon_loop

        print_mem_peak("Peak Memory", use_gpu=False)
        print(f"dir {_dir} spend time: {(time.perf_counter() - time0):.2f}s\n")

    # 方向平均, 2pt: para: conf, dt; 3pt: para: conf, dt, dtau, z;
    # 胶子圈: para: conf, z
    n_dir = len(Calc.dirs)
    corr2_avg = corr2_sum / n_dir
    corr3_avg = corr3_sum / n_dir
    loop_avg = loop_sum / n_dir
    del corr2_sum, corr3_sum, loop_sum

    # ---- bootstrap 采样, 3pt: sample, dt, dtau, z; 2pt: sample, dt; 圈: sample, z ----
    corr2_re = resample(corr2_avg)          # para: sample, dt
    corr3_re = resample(corr3_avg)          # para: sample, dt, dtau, z
    loop_re = resample(loop_avg)            # para: sample, z
    del corr2_avg, corr3_avg, loop_avg

    # ---- 真空圈乘积减除并计算 ratio ----
    ratio = calc_gluon_ratio(corr3_re, corr2_re, loop_re)
    del corr3_re, corr2_re, loop_re

    # ---- 保存 ratio ----
    save_npy(Calc.data_dir / "ratio.npy", ratio, name="ratio")

    print_mem_peak("Peak Memory", use_gpu=False)

    # ---- 自动调用画图文件 (使用本次计算的 conf_short 和 P) ----
    plot_module = importlib.import_module("2_plot_ratio")
    plot_module.plot_saved_ratio(Calc.conf_short, Calc.p)

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s, "
          f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
