#!/public/home/huangcl/.venv/bin/python
"""
功能: 用 comparison_data_tools 比较 Px=-3 质子 2pt 计算结果与参考数据.
输入: 结果和参考 npy, para: dt, t_source.
处理: 参考按每个 t_source 滚动到相对时间, 结果与参考都取 dt=2..32 窗口.
输出: 终端打印 nan, inf, 0 和差异统计量; 直方图存到 01_result/.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "98_tools"))
from comparison_data_tools import compare_data  # noqa: E402  # type: ignore

conf_id = "4100"
Nt = 72
dt_min, dt_max = 2, 32

result_file = (HERE.parent / "test06_mom_dinucl_corr" / "autotask7" / "01_result"
               / conf_id / "proton_2pt"
               / f"momsmear_pp_Px-3Py0Pz0_conf{conf_id}.npy")
ref_file = (Path("/public/group/lqcd/donghx/2pt_Result/"
                 "beta6.20_mu-0.2770_ms-0.2400_L24x72/momsmear-2x")
            / conf_id
            / f"twopt_slice_pp_Px-3Py0Pz0_eginphase-2_Cg5g4_nopol_ss_conf{conf_id}.npy")

# para: dt, t_source
result = np.load(result_file, allow_pickle=False)
# para: dt, t_source
ref_raw = np.load(ref_file, allow_pickle=False)

# 把参考的每个 t_source 滚动到相对时间
reference = np.empty_like(ref_raw)
for t_source in range(Nt):
    reference[:, t_source] = np.roll(ref_raw[:, t_source], -t_source)

result_window = result[dt_min:dt_max + 1]
reference_window = reference[dt_min:dt_max + 1]

print(f"compare window: dt={dt_min}..{dt_max}, shape={result_window.shape}\n")
compare_data(result_window, reference_window, f"conf {conf_id}",
             save_dir=HERE / "01_result")
