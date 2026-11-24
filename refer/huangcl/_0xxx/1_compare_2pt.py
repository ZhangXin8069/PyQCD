#!/public/home/huangcl/.venv/bin/python
# 功能: 对比 std 与 pyqcd 两套收缩代码产出的 2pt_NP 结果是否一致.
# 输入: std/NPcorr_Px2Py0Pz0_conf4050.npy 和 pyqcd/pyqcd_NPcorr_Px2Py0Pz0_conf4050.npy,
#       para: dt, t_src, spin1, spin2, spin3, spin4.
# 输出: 调用 98_tools 的 compare_data 打印一致性判定.

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/public/home/huangcl/06_deuteron_gluon_pdf/98_tools")
from comparison_data_tools import compare_data  # noqa: E402  # type: ignore

if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    base_dir = Path(__file__).resolve().parent
    std_file = base_dir / "std" / "NPcorr_Px2Py0Pz0_conf4050.npy"
    pyqcd_file = base_dir / "pyqcd" / "pyqcd_NPcorr_Px2Py0Pz0_conf4050.npy"

    # para: dt, t_src, spin1, spin2, spin3, spin4
    std_data = np.load(std_file, allow_pickle=False)
    # para: dt, t_src, spin1, spin2, spin3, spin4
    pyqcd_data = np.load(pyqcd_file, allow_pickle=False)

    print(f"std   = {std_file} {std_data.shape}")
    print(f"pyqcd = {pyqcd_file} {pyqcd_data.shape}")

    all_match = compare_data([(std_data, pyqcd_data, "2pt_NP")])
    print(f"all_match = {all_match}")
    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
