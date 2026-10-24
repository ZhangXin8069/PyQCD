#!/public/home/huangcl/.venv/bin/python
"""用 98_tools 的 comparison_data_tools 比较 Px=3 质子 2pt 结果与参考数据.

输入:
- 新结果: autotask11 的 momsmear_pp_Px3Py0Pz0_new_distillation_tools_conf4100.npy, para: t_sink, t_source.
- 参考: momsmear2x/4100 的 twopt_slice_pp_Px3Py0Pz0_eginphase2_Cg5g4_nopol_ss_conf4100.npy, para: t_sink, t_source.
处理: 参考按每个 t_source 滚动到相对时间, 结果与参考都取 dt=2..36 窗口.
输出: 用 compare_data 打印 nan, inf, 0 和差异统计量, 以及 all_match 判定.
"""

import sys
import time
from pathlib import Path

import numpy as np

# ---- 从项目目录 98_tools 导入通用函数 ----
HERE = Path(__file__).resolve()
PROJECT_DIR = next(p for p in HERE.parents if (p / "98_tools").is_dir())
sys.path.insert(0, str(PROJECT_DIR / "98_tools"))
from comparison_data_tools import compare_data  # noqa: E402  # type: ignore


def init() -> tuple:
    """
    初始化比较参数和输入文件路径.

    输出:
    -----
    conf_id, nt, dt_min, dt_max: 组态编号, 时间方向格点数, 比较窗口最小和最大 dt.
    result_file, ref_file: 新计算结果和参考数据文件.
    """
    CONF_ID = "4100"
    NT = 72
    DT_MIN = 2
    DT_MAX = 36
    RESULT_FILE = (
        "/public/home/huangcl/06_deuteron_gluon_pdf/test06_mom_dinucl_corr/"
        "autotask11/01_result/4100/proton_2pt/"
        "momsmear_pp_Px3Py0Pz0_new_distillation_tools_conf4100.npy"
    )
    REF_FILE = (
        "/public/group/lqcd/donghx/2pt_Result/"
        "beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear2x/4100/"
        "twopt_slice_pp_Px3Py0Pz0_eginphase2_Cg5g4_nopol_ss_conf4100.npy"
    )
    return CONF_ID, NT, DT_MIN, DT_MAX, RESULT_FILE, REF_FILE


def to_relative_time(reference: np.ndarray, nt: int) -> np.ndarray:
    """
    逐列把参考数据从绝对 sink 时间转换为相对时间 dt.

    参数:
    -----
    reference: 参考关联函数, para: t_sink_absolute, t_source.
    nt: 时间方向格点数.

    输出:
    -----
    result: 相对时间关联函数, para: dt, t_source.
    """
    result = np.empty_like(reference)
    for t_source in range(nt):
        result[:, t_source] = np.roll(reference[:, t_source], -t_source)
    return result


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    conf_id, nt, dt_min, dt_max, result_file, ref_file = init()
    print(f"result_file = {result_file}")
    print(f"ref_file = {ref_file}")

    # para: t_sink, t_source
    result = np.load(result_file, allow_pickle=False)
    # para: t_sink, t_source
    ref_raw = np.load(ref_file, allow_pickle=False)
    # para: dt, t_source
    reference = to_relative_time(ref_raw, nt)

    result_window = result[dt_min:dt_max + 1]
    reference_window = reference[dt_min:dt_max + 1]
    print(f"compare window: dt={dt_min}..{dt_max}, shape={result_window.shape}\n")

    compare_data(reference_window, result_window,
                 tag=f"conf {conf_id} Px3 new-tools")

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
