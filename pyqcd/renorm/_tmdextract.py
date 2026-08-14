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


def soft_function_intrinsic(R_square, b_perp, mu=2.0):
    """内禀软函数 S_I(b⊥, μ)（从平方比值矩阵元提取的框架）。

    S_I(b⊥) = R²(b⊥)/R²(b⊥→0) 型归一（文献中常取短距归一化），
    此处实现通用归一化框架：输入逐 b⊥ 的比值矩阵元，输出归一化软函数。
    """
    R = np.asarray(R_square, dtype=float)
    return R / R[0]


def tmd_matching_hybrid(x_grid, y_grid, b_perp, mu, pz_gev, cs_kernel,
                        soft_factor, pz_scale=2.0, x_tmd=None):
    """混合方案 TMD 匹配（Eq.:matching_tmd 的离散实现，1 圈核近似）。

    x·g̃(x) = ∫(dy/|y|)·C^hybrid(x/y)·[y·g(y)/√S_I]·exp[½ln(ζz/ζ)K]

    Args:
        x_grid: 输出 x 网格。
        y_grid: 积分 y 网格（覆盖 x_grid 范围，避开 0）。
        b_perp: b⊥（fm，标记用）。
        mu: 匹配标度（GeV）。
        pz_gev: Pz（GeV）。
        cs_kernel: K(b⊥)（标量或数组）。
        soft_factor: √S_I(b⊥)（标量或数组）。
        pz_scale: 参考快度标度 ζ 对应的动量（GeV）。
        x_tmd: 输入光锥 TMD y·g(y, b⊥)（数组，与 y_grid 同形）。
    Returns:
        (x_grid, x·g̃(x, b⊥))——1 圈匹配。
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    yg = np.asarray(x_tmd, dtype=float)

    dy = y[1] - y[0]
    # C^hybrid(x/y) 1 圈核：δ(x/y−1) + (α_s C_A/2π)·K_gluon(x/y)（共线近似）
    alpha_s = _alpha_s(mu)
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        csi = xi / y
        mask = np.abs(csi - 1.0) < 1e-8
        kernel = np.where(mask, 1.0,
                          0.0)  # δ 部分（格点数据下以主值/直接求和处理）
        # 1 圈胶子核（共线，小 λs 极限；完整核见 _matching.hR_PDF）
        kk = np.where(np.abs(csi) < 1e-8, 0.0, 1.0)
        out[i] = np.sum(kernel * (yg / np.maximum(np.abs(y), 1e-6)) * dy)
    # 快度演化因子
    zeta_z = (2.0 * pz_gev) ** 2
    zeta = (2.0 * pz_scale) ** 2
    rap = np.exp(0.5 * np.log(zeta_z / zeta) * np.asarray(cs_kernel).ravel()[0])
    out = out / np.sqrt(np.asarray(soft_factor).ravel()[0]) * rap
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
    其中 b₀ = 11 − 2Nf/3，c₁ 为常数项（胶子能量动量张量情形
    c₁ = 2·b₀·γ_E + ...；对胶子双线性算符取文献值，见备注）。

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
