#!/public/home/huangcl/.venv3_10/bin/python
"""
功能: 逐个组态比较本项目 cg5 零动量 NPcorr 结果与 gengyq 参考数据, 仅适用于 p0.
输入: CONF_SHORT, CONF_NUM, EXCLUDE_IDS 决定的组态编号列表.
输出: 终端打印所有组态的总判定结果, 以及未通过比较的组态编号.
"""

import contextlib
import io
import sys

import numpy as np

sys.path.append(str("/public/home/huangcl/06_deuteron_gluon_pdf/98_tools"))
from comparison_data_tools import compare_data  # noqa: E402  # type: ignore
from snsc_params import get_conf_ids  # noqa: E402  # type: ignore

CONF_SHORT = "c24p29"
CONF_NUM = 450
EXCLUDE_IDS = [12300, 26550]

if __name__ == "__main__":
    conf_ids = get_conf_ids(CONF_SHORT, CONF_NUM, ids_exclude=EXCLUDE_IDS)

    conf_equals = []
    for conf in conf_ids:
        # para: mom1, mom2, t, r, q, z, y
        ref_file = ("/public/home/gengyq/NP/result/spin/"
                    f"beta6.20_mu-0.2770_ms-0.2400_L24x72/NPcorr_spin_{conf}.npy")
        # para: dt, t_src, r, q, z, y
        new_file = ("/public/home/huangcl/00_data/00_NP_2pt/cg5/"
                    f"{CONF_SHORT}/{conf}/NPcorr_Px0Py0Pz0_conf{conf}.npy")

        # 参考数据的 [0, 0] 分量; 本项目结果对 dt 求和.
        ref = np.load(ref_file)[0, 0, :24]
        new = np.load(new_file).sum(axis=1)[:24]

        # 屏蔽 compare_data 的逐组态打印, 只保留其返回值.
        with contextlib.redirect_stdout(io.StringIO()):
            conf_equal = compare_data([(ref, new, f"{conf}")])
        conf_equals.append(conf_equal)

    failed_conf = [
        conf for conf, equal in zip(conf_ids, conf_equals) if not equal
    ]
    print(f"job done, conf num = {len(conf_ids)}, all conf equal = {all(conf_equals)}, "
          f"failed conf = {failed_conf}")
