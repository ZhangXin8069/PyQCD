#!/public/home/huangcl/.venv3_10/bin/python
"""
功能: 把本项目的 NPcorr 结果与参考数据比较.
输入: 两个写死路径的 .npy 文件.
输出: 终端打印的比较结果.
"""

import sys

import numpy as np

sys.path.append(str("/public/home/huangcl/06_deuteron_gluon_pdf/98_tools"))
from comparison_data_tools import compare_data  # noqa: E402  # type: ignore

conf = 4050
# para: ig1, ig2, t, r, q, z, y
# ref_file = ("/public/home/gengyq/NP/result/spin/"
#             f"beta6.20_mu-0.2770_ms-0.2400_L24x72/NPcorr_spin_{conf}.npy")

ref_file = ("/public/home/huangcl/00_data/00_NP_2pt/cg5/"
            f"c24p29/{conf}/NPcorr_Px0Py0Pz0_conf{conf}.npy")

# para: dt, t, r, q, z, y
new_file = ("/public/home/huangcl/06_deuteron_gluon_pdf/test11_backend_cg5/test/"
            f"c24p29/{conf}/NPcorr_Px0Py0Pz0_conf{conf}.npy")

print(ref_file)
print(new_file)
print()
# 参考数据的 [0, 0] 分量; 本项目结果对 dt 求和.
# ref = np.load(ref_file)[0, 0, :24]
# new = np.load(new_file).sum(axis=1)[:24]
ref = np.load(ref_file)
new = np.load(new_file)


print(f"ref shape: {ref.shape}")
print(f"new shape: {new.shape}")
compare_data([(ref, new, "cupy+pyqcd")], show_details=True)
