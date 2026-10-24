#!/public/home/huangcl/.venv/bin/python
"""用 98_tools 的 comparison_data_tools 比较 cg5 和 mom0 的 NP 2pt 结果.

输入:
- cg5: NPcorr_Px0Py0Pz0_conf11000.npy, para: dt, t_src, spin*4.
- mom0: NPcorr_conf11000_P0.npy, para: p_snk, p_src, dt, t_src, spin*4.
处理: 取 mom0 的 [0, 0] 零动量切片, 并把 cg5 截断到相同的 dt 范围.
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
    初始化输入文件路径.

    输出:
    -----
    cg5_file, mom0_file: cg5 和 mom0 的 NP 关联函数文件.
    """
    CG5_FILE = ("/public/home/huangcl/00_data/00_NP_2pt/cg5/c32p29/11000/"
                "NPcorr_Px0Py0Pz0_conf11000.npy")
    MOM0_FILE = ("/public/home/huangcl/00_data/00_NP_2pt/mom0/c32p29/11000/"
                 "NPcorr_conf11000_P0.npy")
    return CG5_FILE, MOM0_FILE


def load_compare_data(cg5_file: str, mom0_file: str) -> tuple:
    """
    读取 cg5 和 mom0 结果, 取出 mom0 零动量切片并对齐 dt 范围.

    参数:
    -----
    cg5_file, mom0_file: 两个 NP 关联函数文件路径.

    输出:
    -----
    cg5, mom0: 形状相同的 cg5 结果和 mom0 零动量切片, para: dt, t_src, spin*4.
    """
    # para: dt, t_src, spin*4
    cg5 = np.load(cg5_file, allow_pickle=False)

    # para: p_snk, p_src, dt, t_src, spin*4
    mom0_all = np.load(mom0_file, allow_pickle=False)
    # para: dt, t_src, spin*4
    mom0 = mom0_all[0, 0]

    n_dt = mom0.shape[0]
    return cg5[:n_dt], mom0


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    cg5_file, mom0_file = init()
    print(f"cg5_file = {cg5_file}")
    print(f"mom0_file = {mom0_file}")

    cg5, mom0 = load_compare_data(cg5_file, mom0_file)
    print(f"cg5 shape = {cg5.shape}, mom0[0,0] shape = {mom0.shape}")

    compare_data(cg5, mom0, tag="cg5_vs_mom0_p00")

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
