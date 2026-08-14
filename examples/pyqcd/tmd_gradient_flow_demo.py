"""
梯度流重整化核子胶子 TMD-PDF —— 完整流程示例（pyqcd 包调用）
================================================================

本示例演示 pyqcd 的核心目标计算链：

    1. 随机 SU(3) 规范场（或读 .lime 组态）
    2. Wilson flow 演化到流时间 τ（梯度流重整化）
    3. 流规范场上的胶子 TMD staple 算符 O(z, b⊥)
    4. 自重整化比值（Z_R 方案）
    5. NLO 匹配 / 共线极限 / 不变振幅提取（示例级）

运行：
    python examples/pyqcd/tmd_gradient_flow_demo.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy.linalg import expm

from pyqcd.tools import set_backend
from pyqcd.renorm import (
    wilson_flow, flow_action_density, gluon_tmd_operator,
    self_renormalized_ratio,
)
from pyqcd.operator import gluon_ope_operator_z0

set_backend('numpy')   # 或 'cupy'（GPU）


def random_su3_gauge(L=8, seed=1234):
    """随机 SU(3) 规范场 (L,L,L,L,4,3,3)（演示用；真实计算读 .lime）。"""
    rng = np.random.default_rng(seed)
    g = np.zeros((L, L, L, L, 4, 3, 3), dtype=complex)
    for idx in np.ndindex(L, L, L, L, 4):
        H = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
        H = H - H.conj().T - np.trace(H - H.conj().T) / 3 * np.eye(3)
        g[idx] = expm(H)
    return g


def main():
    L = 8
    print("=" * 60)
    print("梯度流重整化胶子 TMD-PDF 示例（pyqcd）")
    print("=" * 60)

    U = random_su3_gauge(L)
    print(f"随机 SU(3) 规范场: {U.shape}")

    # ── 梯度流：τ = 3a²（NieMiera et al. 2025 方案）──
    tau = 0.1
    V = wilson_flow(U, tau=tau, eps=0.05)
    E_in, E_out = flow_action_density(U).mean(), flow_action_density(V).mean()
    print(f"Wilson flow: τ = {tau}, E(t): {E_in:.4f} → {E_out:.4f} (递减=正确)")

    # ── 胶子 TMD 算符 O(z, b⊥) ──
    z_list = [0, 2, 4]
    b_list = [0, 2, 4]
    O = gluon_tmd_operator(V, z=2, b_perp=2)
    print(f"TMD 算符 O(z=2, b⊥=2) 逐格点（前 4×4）:")
    print(np.round(np.real(O[:4, :4, 0, 0]), 3))

    # ── 自重整化比值（Z_R 方案）──
    from pyqcd.renorm import tmd_matrix_elements
    M = tmd_matrix_elements(V, z_list, b_list)
    ratio = self_renormalized_ratio(M, M, z_s=0)
    print("O(z, b⊥) 矩阵元 (z × b⊥):")
    print(np.round(M, 4))
    print("自重整化比值 R(z, b⊥) = O(z,b⊥)/O(z_s=0,b⊥):")
    print(np.round(ratio, 4))

    # ── 共线 OPE 对照（O_01 分量）──
    O_ope = gluon_ope_operator_z0(V, 0, 1, 2, 4, L, L)
    print(f"共线 OPE O_01(z)（对照）: {np.round(O_ope[:4, 0], 4)}")

    # ── TMD 提取链：准 TMD-PDF + CS 核（完整物理链）──
    from pyqcd.renorm import (
        quasi_tmd_pdf, cs_kernel_from_ratio, sftx_gluon_matching_coeff,
    )
    # 用流场矩阵元（z 依赖）构造准 TMD-PDF（演示：z 网格 → x 空间）
    z_grid = np.linspace(0.1, 1.0, 32)
    M_z = tmd_matrix_elements(V, [0, 1, 2], [0])   # (nz, nb=1)
    # 外推/插值到 z_grid（演示用线性插值）
    z_data = np.array([0, 1, 2]) * 0.1053
    hr = np.interp(z_grid, z_data, M_z[:, 0])[:, None]
    x, xg = quasi_tmd_pdf(hr, z_grid, [0.2], pz_gev=1.87)
    print(f"准 TMD-PDF x·g̃(x): x∈[{x[0]:.2f},{x[-1]:.2f}], "
          f"xg[0:3]={np.round(xg[:3], 4)}")
    # CS 核（两动量比值，演示）
    K = cs_kernel_from_ratio(hr * 1.0, hr * 1.1, 2.5, 2.0)
    print(f"CS 核 K(b⊥)（演示）: {np.round(K, 4)}")
    al, c = sftx_gluon_matching_coeff(0.1, 2.0)
    print(f"SFTX 匹配系数: α_s/4π={al:.4f}, c(t,μ)={c:.4f}")

    print("\n完成：梯度流 → TMD staple 算符 → 自重整化 → 准TMD-PDF/CS核/SFTX 全链跑通。")


if __name__ == '__main__':
    main()
