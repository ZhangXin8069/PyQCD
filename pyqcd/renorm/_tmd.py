"""
胶子 TMD-PDF 提取（梯度流重整化方案）——本库核心目标
====================================================

参照 /root/PyQCD/refer/papers/gluon_tmd_gradient_flow_continuum.tex 构造：

1. 伴随表示下的胶子准 TMD 算符（Eq.:M_adj_tmd）：
       M^{μλ;νρ}(z, b⊥) = F^{μλ}(z, b⊥)·U_{adj}(staple)·F^{νρ}(0)

2. 基础表示格点实现（Eq.:staple_wilson, Eq.:M_fund_tmd）：
       W_⊏(z, b⊥) = U_z†((z+L)n̂_z + b⊥, b⊥)·U_⊥((z+L)n̂_z+b⊥, (z+L)n̂_z)
                     ·U_z((z+L)n̂_z, z n̂_z)
       M = Σ_x Tr[ F(x + z n̂_z + b⊥)·W_⊏·F(x)·W_⊏† ]

3. 可乘性重整化组合（Eq.:O_mult_tmd）：
       O(z, b⊥) = M^{tx;tx} + M^{ty;ty} − 2·M^{xy;xy}
   该组合的一圈 UV 反常量纲使整体（至少一圈水平）乘法可重整化；
   共线极限 b⊥→0 约化为胶子准 PDF 算符。

4. 不变振幅（Eq.:Mpp_extract_tmd, Eq.:Mpp_PDF_tmd）：
       M^{ti;it}(z,b⊥) + M^{ji;ij}(z,b⊥) = 2 p₀² M_pp(ν, b⊥)
       −M_pp(ν, b⊥) = ½ ∫₋₁¹ dx e^{−ixν} x·g(x, b⊥)

5. 梯度流重整化：先用 Wilson flow（_gradient_flow.wilson_flow）把规范场
   演化到流时间 τ（物理单位固定，如 τ = 3a²），再计算上述算符——
   Monahan–Orginos 2017 / NieMiera et al. 2025 的自重整化方案。

6. b⊥ 依赖的软函数 / Collins–Soper 核：TMD 重整化需软函数 S(b⊥)，
   本模块提供从准 TMD 矩阵元比值提取 CS 核的框架（LPC 2020 方案）。
"""
from __future__ import annotations

import numpy as np

from ..tools._backend import get_backend


def _to_cpu(x):
    """后端无关的 GPU→CPU 转换（运行时取后端，兼容 numpy/cupy/torch）。"""
    asnumpy = getattr(get_backend(), 'asnumpy', None)
    if asnumpy is not None:
        return asnumpy(x)
    return np.asarray(x)
from ..operator._gluon_ope import plaquette_clover, compute_dual_field_strength
from ._gradient_flow import wilson_flow


# ═══════════════════════════════════════════════════════════════════
# 基础表示 staple Wilson 线
# ═══════════════════════════════════════════════════════════════════

def staple_wilson_line(U, z, b_perp, z_dir=2, b_dir=0, L=None):
    """构造 staple Wilson 线 W_⊏(z, b⊥)（Eq.:staple_wilson）。

    W_⊏ = U_z†((z+L)n̂_z + b⊥, b⊥) · U_⊥((z+L)n̂_z + b⊥, (z+L)n̂_z)
          · U_z((z+L)n̂_z, z n̂_z)

    Args:
        U: 规范场 (Nt,Nz,Ny,Nx,4,3,3)。
        z: 纵向分离（格点单位）。
        b_perp: 横向位移（格点单位）。
        z_dir: 纵向方向（2=z 轴，默认）。
        b_dir: 横向方向（0=x 轴，1=y 轴；默认 0）。
        L: staple 臂长（默认 = z）。
    Returns:
        W_⊏，形状 (Nt,Nz,Ny,Nx,3,3)，逐格点（x 为起点）。
    """
    cp = get_backend()
    if L is None:
        L = abs(z)
    if z_dir not in (0, 1, 2) or b_dir not in (0, 1, 2):
        raise ValueError("z_dir 和 b_dir 必须是空间方向 0=x, 1=y, 2=z")
    if z_dir == b_dir:
        raise ValueError("staple 的纵向与横向方向必须不同")
    if L < 0:
        raise ValueError("staple 臂长 L 必须非负")

    # Zhang et al. (2022) Eq. (12):
    #   eta1 = -s L zhat,
    #   eta2 = s b bhat - L zhat,
    #   eta3 = b bhat + [-L + s(L+z)] zhat.
    # 因而 W 从 x 连到 x + b*bhat + z*zhat，按顺序走三段
    # -L zhat -> b bhat -> (L+z) zhat。
    nc = U.shape[-1]
    W = cp.broadcast_to(cp.eye(nc, dtype=U.dtype), U.shape[:4] + (nc, nc))
    displacement = [0, 0, 0]
    for direction, signed_length in (
            (z_dir, -L), (b_dir, b_perp), (z_dir, L + z)):
        W = _path_product(U, W, direction, signed_length, displacement)

    return W


def _shift_spatial(field, displacement):
    """返回 ``field(x + displacement)``；方向序为 x,y,z。"""
    cp = get_backend()
    shifted = field
    for direction, offset in enumerate(displacement):
        if offset:
            shifted = cp.roll(shifted, -offset, axis=3 - direction)
    return shifted


def _path_product(U, W, direction, signed_length, displacement):
    """从当前位移沿一个有向空间段累乘链接，并原位更新位移。"""
    cp = get_backend()
    step = 1 if signed_length >= 0 else -1
    for _ in range(abs(signed_length)):
        if step > 0:
            link = _shift_spatial(U[..., direction, :, :], displacement)
            displacement[direction] += 1
        else:
            displacement[direction] -= 1
            link = _shift_spatial(U[..., direction, :, :], displacement)
            link = cp.swapaxes(link.conj(), -1, -2)
        W = cp.einsum("...ab,...bc->...ac", W, link)
    return W


# ═══════════════════════════════════════════════════════════════════
# 基础表示 TMD 矩阵元
# ═══════════════════════════════════════════════════════════════════

def M_mu_lambda_nu_rho(U, mu, lam, nu, rho, z, b_perp, z_dir=2, b_dir=0,
                       L=None, compute_dtype=None):
    """M^{μλ;νρ}(z, b⊥)（Eq.:M_fund_tmd）：逐格点色迹（未空间求和）。

    M = Σ_x Tr[ F^{μλ}(x + z n̂_z + b⊥) · W_⊏(z, b⊥) · F^{νρ}(x) · W_⊏†(z, b⊥) ]
    """
    cp = get_backend()
    if compute_dtype is None:
        compute_dtype = U.dtype

    F_mu = plaquette_clover(U, mu, lam)
    F_nu = plaquette_clover(U, nu, rho)

    W = staple_wilson_line(U, z, b_perp, z_dir, b_dir, L)

    endpoint = [0, 0, 0]
    endpoint[z_dir] += z
    endpoint[b_dir] += b_perp
    F_mu_shift = _shift_spatial(F_mu, endpoint)
    W_dagger = cp.swapaxes(W.conj(), -1, -2)

    # W 从 x 连到 x+z+b，故闭合颜色指标为
    # Tr[F_nu(x) W(x,x+z+b) F_mu(x+z+b) W†(x,x+z+b)]。
    t1 = cp.einsum("...ab,...bc->...ac", F_nu, W)
    t2 = cp.einsum("...ab,...bc->...ac", F_mu_shift, W_dagger)
    return cp.einsum("...ab,...ba->...", t1, t2)


def gluon_tmd_operator(U, z, b_perp, z_dir=2, b_dir=0, L=None):
    """可乘性重整化组合 O(z, b⊥) = M^{tx;tx} + M^{ty;ty} − 2M^{xy;xy}（Eq.:O_mult_tmd）。

    Returns:
        逐格点 O 值，形状 (Nt,Nz,Ny,Nx)。
    """
    # Lorentz 方向沿用组态链接顺序 0=x, 1=y, 2=z, 3=t。
    M_txtx = M_mu_lambda_nu_rho(U, 3, 0, 3, 0, z, b_perp, z_dir, b_dir, L)
    M_tyty = M_mu_lambda_nu_rho(U, 3, 1, 3, 1, z, b_perp, z_dir, b_dir, L)
    M_xyxy = M_mu_lambda_nu_rho(U, 0, 1, 0, 1, z, b_perp, z_dir, b_dir, L)
    return M_txtx + M_tyty - 2.0 * M_xyxy


def tmd_matrix_elements(U, z_list, b_list, z_dir=2, b_dir=0, L=None,
                        spatial_sum=True):
    """批量计算 O(z, b⊥)：返回 (nz, nb) 实数数组（逐 t 时间片均分后求和）。

    胶子 TMD 组合 O = M^{tx;tx} + M^{ty;ty} − 2M^{xy;xy} 为实数值
    （每项 M 的虚部在色迹 + 空间求和后归零）。
    """
    cp = get_backend()
    out = np.zeros((len(z_list), len(b_list)), dtype=np.float64)
    for i, z in enumerate(z_list):
        for j, b in enumerate(b_list):
            O = gluon_tmd_operator(U, z, b, z_dir, b_dir, L)
            if spatial_sum:
                val = _to_cpu(cp.sum(O, axis=(1, 2, 3)))
            else:
                val = _to_cpu(O)
            out[i, j] = np.real(np.mean(val))  # 时间片平均
    return out


def tmd_matrix_elements_time(U, z_list, b_list, z_dir=2, b_dir=0, L=None):
    """批量计算 O(z, b⊥) 逐时间片（空间求和保留 t 轴）：返回 (nz, nb, Nt)。

    与 ``tmd_matrix_elements`` 的区别：不做时间片平均，保留每个 t 的
    空间求和值——供 disconnected 3pt 因子化 C3 = C2(dt)·OPE(dtau, z, b)
    使用（OPE(dtau, z, b) 需要逐时间片的算符矩阵元）。

    Returns:
        out: (nz, nb, Nt) 实数数组（每项为 Σ_{x,y,z} O(z,b⊥)(t)）。
    """
    cp = get_backend()
    Nt = U.shape[0]
    out = np.zeros((len(z_list), len(b_list), Nt), dtype=np.float64)
    for i, z in enumerate(z_list):
        for j, b in enumerate(b_list):
            O = gluon_tmd_operator(U, z, b, z_dir, b_dir, L)
            val = _to_cpu(cp.sum(O, axis=(1, 2, 3)))
            out[i, j] = np.real(val)
    return out


# ═══════════════════════════════════════════════════════════════════
# 梯度流重整化流程
# ═══════════════════════════════════════════════════════════════════

def gradient_flow_renormalized_tmd(U, tau, z_list, b_list, z_dir=2, b_dir=0,
                                   L=None, eps=0.01):
    """梯度流重整化的 TMD 矩阵元（Monahan–Orginos 2017 方案）。

    Args:
        U: 初始规范场。
        tau: 流时间（格点单位；NieMiera 2025 用 τ = 3a²）。
        z_list/b_list: 纵向/横向位移列表。
    Returns:
        O(z, b⊥) 矩阵（形状 (nz, nb)）。
    """
    V = wilson_flow(U, tau, eps=eps)
    return tmd_matrix_elements(V, z_list, b_list, z_dir, b_dir, L)


def self_renormalized_ratio(O_z, O_z0, z_s=2):
    """自重整化比值：R(z, b⊥) = O(z, b⊥) / O(z_s, b⊥)（Z_R 方案）。

    Args:
        O_z: O(z, b⊥) 数组 (nz, nb)。
        O_z0: O(0 或 z_s, b⊥) 数组（同一 b⊥ 网格）。
        z_s: 参考点 z 索引（默认 2，即短距参考）。
    Returns:
        比值数组（与 O_z 同形）。
    """
    return O_z / O_z0[z_s][None, :]


# ═══════════════════════════════════════════════════════════════════
# 不变振幅与 CS 核
# ═══════════════════════════════════════════════════════════════════

def invariant_amplitude(M_pp, x_grid, b_perp):
    """从不变振幅 M_pp(ν, b⊥) 傅里叶变换到胶子 TMD-PDF x·g(x, b⊥)（Eq.:Mpp_PDF_tmd）。

    −M_pp(ν, b⊥) = ½ ∫₋₁¹ dx e^{−ixν} x·g(x, b⊥)
    → x·g(x, b⊥) = −(1/π)·∫₀^∞ dν cos(xν)·2·Re[M_pp(ν,b⊥)]  （实部，奇偶性）

    Args:
        M_pp: 不变振幅（随 Ioffe 时间 ν 变化的数组，或 (nν, nb) 矩阵）。
        x_grid: x 网格（(0,1)）。
        b_perp: b⊥ 网格（用于维度标记，不参与计算）。
    Returns:
        xg(x, b⊥) 数组（形状 (len(x_grid), nb) 或 (len(x_grid),)）。
    """
    M_pp = np.asarray(M_pp, dtype=complex)
    if M_pp.ndim == 1:
        nν = len(M_pp)
        ν_grid = np.linspace(0, 2 * np.pi * nν, nν)  # Ioffe 时间网格
    else:
        nν, nb = M_pp.shape
        ν_grid = np.linspace(0, 2 * np.pi * nν, nν)

    dν = ν_grid[1] - ν_grid[0]
    out = np.zeros((len(x_grid),) + (1,) if M_pp.ndim == 1 else (nb,))
    for i, x in enumerate(x_grid):
        cos_mat = np.cos(np.outer(x, ν_grid))  # (nx, nν)
        if M_pp.ndim == 1:
            out[i, 0] = -(1.0 / np.pi) * dν * np.sum(cos_mat * (-M_pp).real)
        else:
            for j in range(nb):
                out[i, j] = -(1.0 / np.pi) * dν * np.sum(
                    cos_mat * (-M_pp[:, j]).real)
    return out[..., 0] if M_pp.ndim == 1 else out


def collins_soper_kernel(R_b, b_list, z_list, pz_gev):
    """从准 TMD 矩阵元比值提取 Collins–Soper 核（LPC 2020 方案框架）。

    CS 核 γ_ζ(b⊥) 通过不同 P_z 下矩阵元比值的幂律行为提取：
        R(b⊥) = O(Pz₁, b⊥)/O(Pz₂, b⊥) ~ (Pz₁/Pz₂)^{γ_ζ(b⊥)·...}

    这里提供比值—斜率最小二乘框架：
        γ_ζ(b) = d ln R(b) / d ln(Pz)   （逐 b⊥）

    Args:
        R_b: 比值数组（(n_pz, nb)，不同 Pz 下的比值）。
        b_list: b⊥ 网格（fm）。
        z_list: 纵向位移（用于标记，不参与计算）。
        pz_gev: Pz 数组（GeV）。
    Returns:
        (γ_ζ(b), 逐 b 误差) —— 用 ln(Pz) 对 ln R 线性回归。
    """
    pz_gev = np.asarray(pz_gev, dtype=float)
    R_b = np.asarray(R_b, dtype=float)
    ln_pz = np.log(pz_gev)
    ln_R = np.log(R_b)  # (n_pz, nb)

    nb = R_b.shape[1]
    gamma = np.zeros(nb)
    gamma_err = np.zeros(nb)
    for j in range(nb):
        A = np.vstack([ln_pz, np.ones_like(ln_pz)]).T
        coef, res, *_ = np.linalg.lstsq(A, ln_R[:, j], rcond=None)
        gamma[j] = coef[0]
        gamma_err[j] = np.sqrt(np.sum(res ** 2) / max(len(ln_pz) - 2, 1))
    return gamma, gamma_err
