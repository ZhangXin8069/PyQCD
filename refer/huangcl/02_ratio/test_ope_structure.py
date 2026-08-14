#!/public/home/huangcl/.venv/bin/python
"""
对比师兄的 ops_dz*.npz 和你的 3 个 OPE 文件的各种组合。
zdir: tdir1=0(x), tdir2=1(y)
"""

import numpy as np
import pdb

conf_id = 4050
axis = "z"
dz = 24

# ===== 师兄的 ops_dz*.npz (zdir) =====
senior_path = (f"/public/group/imp/zengch/LQCD/gluon_operator/output/"
               f"L24x72/{axis}dir/{conf_id}/"
               f"ops_dz{dz}_conf{conf_id}.npz")
print(f"师兄文件: {senior_path}")
senior_file = np.load(senior_path)
senior_ops = senior_file['ops']
print(f"  shape: {senior_ops.shape}, dtype: {senior_ops.dtype}")
print()

# ===== 你的 3 个 OPE 文件 (zdir: tdir1=0, tdir2=1) =====
# ops_mu0_nu1 = O_ij
# ops_mu3_nu0 = O_ti
# ops_mu3_nu1 = O_tj
your_base = "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times"

ij_path = f"{your_base}/L24x72/{axis}dir/{conf_id}/ops_mu0_nu1_dz{dz}_conf{conf_id}.npz"
ti_path = f"{your_base}/L24x72/{axis}dir/{conf_id}/ops_mu3_nu0_dz{dz}_conf{conf_id}.npz"
tj_path = f"{your_base}/L24x72/{axis}dir/{conf_id}/ops_mu3_nu1_dz{dz}_conf{conf_id}.npz"

ij_file = np.load(ij_path)
ti_file = np.load(ti_path)
tj_file = np.load(tj_path)

ops_ij = ij_file['ops']
ops_ti = ti_file['ops']
ops_tj = tj_file['ops']

print(f"ops_ij (mu0_nu1) shape: {ops_ij.shape}")
print(f"ops_ti (mu3_nu0) shape: {ops_ti.shape}")
print(f"ops_tj (mu3_nu1) shape: {ops_tj.shape}")
print()

# ===== 各种组合与师兄对比 =====
# 组合1: 只用 O_ij
diff1 = np.max(np.abs(ops_ij - senior_ops))
print(f"组合1 (O_ij):          max diff = {diff1:.2e}")

# 组合2: 只用 O_ti
diff2 = np.max(np.abs(ops_ti - senior_ops))
print(f"组合2 (O_ti):          max diff = {diff2:.2e}")

# 组合3: 只用 O_tj
diff3 = np.max(np.abs(ops_tj - senior_ops))
print(f"组合3 (O_tj):          max diff = {diff3:.2e}")

# 组合4: -O_ti - O_tj + 2*O_ij
combined = -ops_ti - ops_tj + 2 * ops_ij
diff4 = np.max(np.abs(combined - senior_ops))
print(f"组合4 (-O_ti-O_tj+2*O_ij): max diff = {diff4:.2e}")

# 组合5: O_ij - O_ti
diff5 = np.max(np.abs(ops_ij - ops_ti - senior_ops))
print(f"组合5 (O_ij-O_ti):     max diff = {diff5:.2e}")

# 组合6: O_ij - O_tj
diff6 = np.max(np.abs(ops_ij - ops_tj - senior_ops))
print(f"组合6 (O_ij-O_tj):     max diff = {diff6:.2e}")

print()
# 找出最小误差的组合
diffs = [diff1, diff2, diff3, diff4, diff5, diff6]
labels = ["O_ij", "O_ti", "O_tj",
          "-O_ti-O_tj+2*O_ij", "O_ij-O_ti", "O_ij-O_tj"]
best_idx = np.argmin(diffs)
print(f"最佳匹配: {labels[best_idx]}, diff = {diffs[best_idx]:.2e}")

# ===== 暂停 =====
print("\n" + "=" * 60)
print("进入 pdb")
print("可用变量: senior_ops, ops_ij, ops_ti, ops_tj, combined")
print("=" * 60)
pdb.set_trace()

senior_file.close()
ij_file.close()
ti_file.close()
tj_file.close()
print("Done.")
