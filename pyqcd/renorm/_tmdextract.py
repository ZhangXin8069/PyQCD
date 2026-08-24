"""
准 TMD-PDF 提取链：傅里叶变换、混合方案匹配、软函数/CS 核、SFTX
=================================================================

依据 refer/papers/gluon_tmd_gradient_flow_continuum.tex 实现：

1. 准 TMD-PDF（Eq.:quasi_tmd）：
       x·g̃(x, b⊥, Pz) = (1/𝒩)·∫dz/(2πPz) e^{−ixzPz}·hR(z, b⊥, Pz)
       𝒩 = (Pz)²/(Pt)²

2. 混合方案匹配（Eq.:matching_tmd）：
       x·g̃(x, b⊥, Pz) = ∫(dy/|y|)·C^hybrid(x/y, λs, μ/(yPz))
                          ·[y·g(y,b⊥)/√S_I(b⊥,μ)]·exp[½ln(ζz/ζ)·K(b⊥,μ)]

3. CS 核（LPC 2020 方案）：K(b⊥) 从不同 Pz 的准 TMD 矩阵元比值提取
       K(b⊥) ≈ ln[hR(Pz₁)/hR(Pz₂)] / ln(Pz₁/Pz₂)

4. SFTX 匹配系数（Suzuki 2013 / Mereghetti 2022，1 圈）：
       O_MS(μ) = [1 + α_s/(4π)·c(t,μ)]·O_flow(t)
"""
from __future__ import annotations

import numpy as np

from ._const import CA, CF, gammaE, pi, b0
from ._ensembles import fm_to_GeV
from ._matching import _matching_kernels, A_s_run


def quasi_tmd_pdf(hR_z, z_grid, b_perp, pz_gev, p_t=None, x_grid=None,
                  n_pts=2048, z_max=None):
    """准 TMD-PDF（Eq.:quasi_tmd）。

    Args:
        hR_z: 重整化矩阵元 hR(z, b⊥, Pz)，形状 (nz, nb) 或 (nz,)。
        z_grid: z 网格（fm）。
        b_perp: b⊥ 网格（fm）。
        pz_gev: Pz（GeV）。
        p_t: 核子能量分量 Pt（GeV）；None 时取 √(M²+Pz²)，M=0.94。
        x_grid: x 网格（默认 −1.5..1.5，128 点）。
        z_max: 积分截断 z（默认 z_grid[-1] 外推点，用 λ 外推）。
    Returns:
        (x_grid, xg(x, b⊥))。
    """
    hR_z = np.asarray(hR_z, dtype=float)
    z = np.asarray(z_grid, dtype=float) / fm_to_GeV   # fm → GeV⁻¹
    if hR_z.ndim == 1:
        hR_z = hR_z[:, None]

    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.5, 256)
    if z_max is None:
        z_max = z[-1]

    if p_t is None:
        M_N = 0.94
        p_t = np.sqrt(M_N ** 2 + pz_gev ** 2)
    norm = (pz_gev ** 2) / (p_t ** 2)

    # 余弦傅里叶（hR 为实）：∫dz cos(x·z·Pz)·hR(z)
    mask = z <= z_max
    zz = z[mask]
    hr = hR_z[mask]
    dz = zz[1] - zz[0] if len(zz) > 1 else 1.0

    out = np.empty((len(x_grid), hr.shape[1]))
    for j in range(hr.shape[1]):
        integrand = np.cos(np.outer(x_grid, zz * pz_gev)) @ (hr[:, j] * dz)
        out[:, j] = integrand / (2.0 * pi * pz_gev) / norm
    if hR_z.shape[1] == 1:
        out = out[:, 0]
    return x_grid, out


def quasi_pdf_gluon(h_z, z_grid, pz_gev, x_grid=None):
    """collinear 胶子准 PDF（sin 变换通道，照抄 zhangxin
    gluon_pdf_full_workflow.fourier_transform_to_quasi_pdf）。

        g̃(x, Pz) = (2Pz/x)·∫₀^{z_max} dz h(z, Pz)·sin(x·Pz·z)

    反对称部分（非极化胶子 collinear 准 PDF）用 sin 变换；x≈0 保护置 0
    （原版 abs(x)<1e-15 分支）。与 ``quasi_tmd_pdf`` 的 cos 变换（TMD
    约定、b⊥ 依赖）互补：b⊥ 积分极限下的交叉校验通道。

    Args:
        h_z: 坐标空间矩阵元 h(z, Pz)，形状 (nz,)（实部参与积分）。
        z_grid: z 网格（fm，内部转 GeV⁻¹）。
        pz_gev: Pz（GeV）。
        x_grid: Bjorken x 网格（默认 −1.5..1.5，256 点）。
    Returns:
        (x_grid, g̃(x)) 形状 (nx,)。
    """
    h = np.asarray(h_z, dtype=float).real
    z = np.asarray(z_grid, dtype=float) / fm_to_GeV   # fm → GeV⁻¹
    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.5, 256)
    x_grid = np.asarray(x_grid, dtype=float)

    small = np.abs(x_grid) < 1e-15
    xv = np.where(small, 1.0, x_grid)                 # 防除零，末尾回填 0
    integrand = h[None, :] * np.sin(np.outer(xv, pz_gev * z))
    trapz = getattr(np, "trapz", None) or np.trapezoid   # numpy 2.x 兼容
    integral = trapz(integrand, z, axis=1)            # 梯形法则（原版 np.trapz）
    out = (2.0 * pz_gev / xv) * integral
    out[small] = 0.0
    return x_grid, out


def cs_kernel_from_ratio(hR_pz1, hR_pz2, pz1, pz2):
    """CS 核提取（LPC 2020 比值法）。

    K(b⊥) = ln[hR(z=z₀, b⊥, Pz₁)/hR(z=z₀, b⊥, Pz₂)] / ln(Pz₁/Pz₂)

    Args:
        hR_pz1/hR_pz2: 两动量下的重整化矩阵元（(nz, nb) 或 (nz,)）。
        pz1/pz2: 动量（GeV）。
    Returns:
        K(b⊥) 数组（沿 z=0 提取；若 z 维首点噪声大可用 z=z₀ 索引）。
    """
    h1 = np.asarray(hR_pz1, dtype=float)
    h2 = np.asarray(hR_pz2, dtype=float)
    if h1.ndim == 1:
        h1, h2 = h1[:, None], h2[:, None]
    ratio = h1[0] / h2[0]           # z=0 处（b⊥ 依赖保留）
    return np.log(np.maximum(ratio, 1e-30)) / np.log(pz1 / pz2)


def cs_kernel_two_momentum(c01, c02, pz1_gev, pz2_gev, z_ref=1,
                           k_clip=(-3.0, 3.0)):
    """两动量 c0 比值 CS 核提取——工程封装（整合 test9 示例内联实现）。

    与 cs_kernel_from_ratio 的差异：
        1) 直接吃裸 c0(z,b)（未经 z₀ 归一化，避免归一点噪声放大）；
        2) z_ref 可选参考 z 索引（默认 1：z=0 处 c0≡1 无信息，
           z=1 为最低含 Wilson 线信息的分离）；
        3) k_clip 数值保护：噪声使 |K| 越界时截断
           （CS 核物理量级 |K| ≲ O(1) GeV²·fm² 量级，|K|>3 视为噪声）。

    Args:
        c01/c02: 两动量裸矩阵元 (nz, nb) 或 (nb,)（同 z 网格）。
        pz1_gev/pz2_gev: 两动量（GeV）。
        z_ref: 参考 z 行索引（负数按 Python 惯例回绕）。
        k_clip: (K_min, K_max) 截断区间；None 则不截断。
    Returns:
        K(b⊥) 数组（沿所选 z 行）。
    """
    a = np.asarray(c01, dtype=float)
    b_ = np.asarray(c02, dtype=float)
    if a.ndim == 1:
        a, b_ = a[None, :], b_[None, :]
    if a.shape != b_.shape:
        raise ValueError(f"c01/c02 形状不一致: {a.shape} vs {b_.shape}")
    z_ref = int(z_ref if z_ref >= 0 else len(a) + z_ref)
    if not (0 <= z_ref < len(a)):
        raise ValueError(f"z_ref={z_ref} 超出 z 维 [0, {len(a)-1}]")
    ratio = a[z_ref] / b_[z_ref]
    k = np.log(np.maximum(ratio, 1e-30)) / np.log(pz1_gev / pz2_gev)
    return np.clip(k, *k_clip) if k_clip is not None else k


def soft_function_intrinsic(R_square, b_perp, mu=2.0):
    """内禀软函数 S_I(b⊥, μ)（从平方比值矩阵元提取的框架）。

    S_I(b⊥) = R²(b⊥)/R²(b⊥→0) 型归一（文献中常取短距归一化），
    此处实现通用归一化框架：输入逐 b⊥ 的比值矩阵元，输出归一化软函数。
    """
    R = np.asarray(R_square, dtype=float)
    return R / R[0]


def tmd_matching_hybrid(x_grid, y_grid=None, b_perp=None, mu=2.0, pz_gev=2.0,
                        cs_kernel=0.0, soft_factor=1.0, pz_scale=2.0,
                        x_tmd=None, lambda_s_fm=0.3):
    """混合方案 TMD 匹配（Eq.:matching_tmd 的离散实现，1 圈核）。

    x·g̃(x, b⊥) = ∫(dy/|y|)·C^hybrid(x/y, λs, μ/(yPz))
                  ·[y·g(y, b⊥)/√S_I(b⊥,μ)]·exp[½ln(ζz/ζ)·K(b⊥)]

    1 圈匹配核 C^hybrid(ξ) = δ(1−ξ) + (α_s C_A/2π)·g_xy(ξ)，其中
    g_xy 复用 _matching._matching_kernels 的胶子核 g_0..g_3（ξ<0/0<ξ<1/ξ>1
    分区 + g_0 的 Si 项，含 μ/(yPz) 标度与 λ_s 截断）。

    数值结构：Z_ij = δ_ij + (α_s C_A/2π)·dx·(x/|x|)·[g_ij/y_j
    − δ_ij·y_i·Σ_k(g_ik/y_k²)]，x·g̃ = Z⁻¹·[y·g/√S_I·e^{½ln(ζz/ζ)K}]
    ——与 _matching.hR_PDF 同构（矩阵形式自然处理 ξ=1 主值奇点，
    δ 项由 Z⁻¹ 的 LO 部分还原，匹配求和规则经 test_matching_sum_rule 验证）。

    Args:
        x_grid: 输出 x 网格（(0,1)）。
        y_grid: 积分 y 网格；None 时取 x_grid（匹配矩阵结构要求同维）。
        b_perp: b⊥（fm，标记用）。
        mu: 匹配标度（GeV）。
        pz_gev: Pz（GeV）。
        cs_kernel: K(b⊥)（标量或数组）。
        soft_factor: √S_I(b⊥)（标量或数组）。
        pz_scale: 参考快度标度 ζ 对应的动量（GeV）。
        x_tmd: 输入光锥 TMD y·g(y, b⊥)（数组，与 y_grid 同形）。
        lambda_s_fm: 大 λ 截断（fm，默认 0.3，同 _matching.hR_PDF）。
    Returns:
        (x_grid, x·g̃(x, b⊥))——1 圈匹配。
    """
    x = np.asarray(x_grid, dtype=float)
    y = x if y_grid is None else np.asarray(y_grid, dtype=float)
    if len(y) != len(x):
        raise ValueError("TMD 匹配矩阵结构要求 y 网格与 x 网格同维")
    yg = np.asarray(x_tmd, dtype=float)
    if yg.ndim == 1:
        yg = yg[:, None]
    nb = yg.shape[1]
    n = len(x)

    dx = x[1] - x[0]
    lambda_s = lambda_s_fm / fm_to_GeV * pz_gev
    # 与 _matching.hR_PDF 同构：A_s ≡ α_s/(4π)，×4π 还原真耦合
    # （zengch 端 matching*.py 均为 alpha_s = A_s(mu)*4π）
    alpha_s = A_s_run(mu) * 4.0 * pi

    # 匹配核矩阵：g_ij = g_xy(x_i / y_j)（1 圈硬核，含主值奇点）
    cxi = x[:, None] / y[None, :]
    g_ij = _matching_kernels(cxi, np.broadcast_to(y[None, :], cxi.shape),
                             pz_gev, mu, lambda_s)

    # Z_ij 结构（与 _matching.hR_PDF 同构）：
    #   c_alp_lo = diag(x/|x|)·dx·α_s C_A/(2π)
    #   m_ij     = g_ij/y_j − δ_ij·y_i·Σ_k(g_ik/y_k²)
    c_alp_lo = np.diag(x / np.abs(x)) * dx * alpha_s * CA / (2.0 * pi)
    m_ij = (g_ij / y[None, :]
            - np.eye(n) * (y[:, None]
                           * np.sum(g_ij / y[None, :] ** 2.0, axis=1)))
    z_ij = np.eye(n) + c_alp_lo @ m_ij
    z_inv = np.linalg.inv(z_ij)

    # 快度演化因子与软函数（逐 b⊥）
    K = np.asarray(cs_kernel, dtype=float).ravel()
    S = np.asarray(soft_factor, dtype=float).ravel()
    rap = np.exp(0.5 * np.log((2.0 * pz_gev) ** 2 / (2.0 * pz_scale) ** 2) * K)

    out = np.empty((n, nb))
    for j in range(nb):
        v = yg[:, j] * rap[j % rap.size] / np.sqrt(S[j % S.size])
        out[:, j] = z_inv @ v
    if nb == 1:
        out = out[:, 0]
    return x, out


def _alpha_s(mu, Lambda_QCD=0.23, nf=3.0):
    return 2.0 * np.pi / (b0(nf) * np.log(mu / Lambda_QCD))


# ═══════════════════════════════════════════════════════════════════
# SFTX：小流时展开匹配系数（梯度流 → MS-bar，1 圈）
# ═══════════════════════════════════════════════════════════════════

def sftx_gluon_matching_coeff(t, mu, Lambda_QCD=0.23, nf=3.0):
    """胶子算符的 SFTX 1 圈匹配系数（Suzuki 2013 / Mereghetti 2022）。

    O_MS(μ) = [1 + α_s(μ)/(4π)·c(t,μ)]·O_flow(t)

    c(t,μ) = 2·b₀·ln(2μ²t) + c₁
    其中 b₀ = 11 − 2Nf/3，c₁ = 2·b₀·γ_E 为常数项
    （胶子能量动量张量/双线性算符的 1 圈 SFTX 常数项，Suzuki 2013）。

    Args:
        t: 流时间（格点单位 t = τ/a²，物理单位需乘 a²）。
        mu: MS-bar 标度（GeV）。
    Returns:
        (alpha_s/(4π), c(t,μ))——匹配系数。
    """
    al4pi = _alpha_s(mu, Lambda_QCD, nf) / (4.0 * np.pi)
    b0_ = b0(nf)
    # 1 圈 SFTX：O_MS(μ) = O_flow(t)·[1 + α_s/(4π)·(2b₀·ln(2μ²t) + 2b₀·γ_E)]
    c = 2.0 * b0_ * (np.log(2.0 * mu ** 2 * t) + gammaE)
    return al4pi, c


def sftx_energy_density_t0(E_t, t, mu, Lambda_QCD=0.23, nf=3.0):
    """⟨E(t)⟩ 的 SFTX → MS-bar ⟨¼F²⟩(μ)（Suzuki 2013）。

    ⟨E(t)⟩ = 3(N²−1)/(16π²t²)·[1 + α_s/(4π)·(2b₀·ln(2μ²t) + c₂)]·(1 + O(t·⟨F²⟩/Nc))

    此处实现 ⟨¼F²⟩_MS = E(t)/[1 + α_s/(4π)·c(t)] 的 1 圈反解。
    """
    al4pi, c = sftx_gluon_matching_coeff(t, mu, Lambda_QCD, nf)
    return np.asarray(E_t) / (1.0 + al4pi * c)
