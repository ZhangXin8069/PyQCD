#!/public/home/huangcl/.venv/bin/python
"""
对比师兄的 ops_dz*.npz 和你的 3 个 OPE 文件的各种组合。
遍历 5 个组态 × 3 个方向，共 15 组数据。
"""

import numpy as np
import pdb

conf_ids = [4050, 14050, 24050, 34050, 44050]
axes = ["x", "y", "z"]
dz = 24

# 每个方向对应的 tdir1, tdir2
# axis='x': tdir1=1(y), tdir2=2(z)
# axis='y': tdir1=2(z), tdir2=0(x)
# axis='z': tdir1=0(x), tdir2=1(y)
tdir_map = {
    "x": (1, 2),
    "y": (2, 0),
    "z": (0, 1),
}

your_base = "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times"
senior_base = "/public/group/imp/zengch/LQCD/gluon_operator/output"

print(f"{'conf':>6} {'dir':>3} {'O_ij':>12} {'O_ti':>12} {'O_tj':>12} {'-O_ti-O_tj+2*O_ij':>22} {'best_match':>12}")
print("-" * 80)

for conf_id in conf_ids:
    for axis in axes:
        tdir1, tdir2 = tdir_map[axis]

        # 师兄的 ops_dz*.npz
        senior_path = (f"{senior_base}/L24x72/{axis}dir/{conf_id}/"
                       f"ops_dz{dz}_conf{conf_id}.npz")
        senior_ops = np.load(senior_path)['ops']

        # 你的 3 个文件
        ij_path = (f"{your_base}/L24x72/{axis}dir/{conf_id}/"
                   f"ops_mu{tdir1}_nu{tdir2}_dz{dz}_conf{conf_id}.npz")
        ti_path = (f"{your_base}/L24x72/{axis}dir/{conf_id}/"
                   f"ops_mu3_nu{tdir1}_dz{dz}_conf{conf_id}.npz")
        tj_path = (f"{your_base}/L24x72/{axis}dir/{conf_id}/"
                   f"ops_mu3_nu{tdir2}_dz{dz}_conf{conf_id}.npz")

        ops_ij = np.load(ij_path)['ops']
        ops_ti = np.load(ti_path)['ops']
        ops_tj = np.load(tj_path)['ops']

        # 计算各种组合的误差
        d_ij = np.max(np.abs(ops_ij - senior_ops))
        d_ti = np.max(np.abs(ops_ti - senior_ops))
        d_tj = np.max(np.abs(ops_tj - senior_ops))
        d_all = np.max(np.abs((-ops_ti - ops_tj + 2 * ops_ij) - senior_ops))

        # 找出最佳匹配
        diffs = [d_ij, d_ti, d_tj, d_all]
        labels = ["O_ij", "O_ti", "O_tj", "combined"]
        best_idx = int(np.argmin(diffs))
        best_label = labels[best_idx]

        print(f"{conf_id:>6} {axis:>3}  {d_ij:>8.2e}  {d_ti:>8.2e}  {d_tj:>8.2e}  {d_all:>8.2e}          {best_label:>10}")

print("\nDone. (未进入 pdb，如需调试请取消注释 pdb.set_trace())")
# pdb.set_trace()
