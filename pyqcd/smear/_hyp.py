"""
HYP 涂抹（Hasenbusch 2001）——梯度流/线性发散抑制的备选方案
============================================================

理论文档（refer/papers/gluon_tmd_gradient_flow_continuum.tex）对比了 HYP5
涂抹与 Wilson flow：两者定性一致。本模块实现完整 HYP 三级迭代
（Hasenbusch 2001 精确公式，带 (μ,ν) 对结构）：

    级别 3：V_μν^{(3)} = proj[(1−α₃)U_μ + (α₃/6)Σ_{ρ≠μ,ν} S(U; μ,ρ)]
    级别 2：V_μν^{(2)} = proj[(1−α₂)V_μν^{(3)} + (α₂/6)Σ_{ρ≠μ,ν} S(V^{(3)}; μ,ρ)]
    级别 1：V_μ^{(1)}  = proj[(1−α₁)V_μμ̄^{(2)} + (α₁/4)Σ_{ν≠μ} S(V^{(2)}; μ,ν)]

S(·; μ,ρ) 为 (μ,ρ) staple：V_ρμ(x)·V_μρ(x+ρ̂)·V_ρμ†(x+μ̂) + 反向。
标准参数 α₁=0.75, α₂=0.6, α₃=0.3。
"""
from __future__ import annotations

import numpy as np

from ..tools._backend import get_backend


def proj_su3(A):
    """投影到 SU(3)：极分解 A = U·Σ·V† → P = U·V†，再 det 归一。

    用 SVD 实现（numpy 批量 eigh 的特征向量相位在不同数组形状下
    不一致，SVD 的 U·V† 构造更稳健）。
    """
    cp = get_backend()
    U, _S, Vh = cp.linalg.svd(A)
    P = cp.matmul(U, Vh)
    det = cp.linalg.det(P)
    return P * (det ** (-1.0 / 3.0))[..., None, None]


def _staple_munu(V, mu, nu, U_dir=None):
    """(μ,ν) 对级 staple：用链接场 V[μ,ν]（第二指标随方向变化的 fat link）。

    S(x;μ,ν) = V_νμ(x)·V_μν(x+ν̂)·V_νμ†(x+μ̂)
             + V_νμ†(x−ν̂)·V_μν(x−ν̂)·V_νμ(x−ν̂+μ̂)
    其中 V 为 (…, 4, 4, 3, 3) 数组（V[..., μ, ν] = 方向 μ、对指标 ν 的链接）。
    当 U_dir 提供时使用其 (μ,ν) 切片（级别 3 用原始 U 的 (μ,ρ) 形式）。
    """
    cp = get_backend()
    e = cp.einsum
    if U_dir is not None:
        def G(a, b):
            return U_dir[..., a, :, :] if a == b else U_dir[..., a, :, :]
        Vnm = U_dir[..., nu, :, :]          # U_ν（级别 3：原始链接）
        Vmu = U_dir[..., mu, :, :]          # U_μ
    else:
        Vnm = V[..., nu, mu, :, :]          # V_νμ
        Vmu = V[..., mu, nu, :, :]          # V_μν
    a_mu, a_nu = 3 - mu, 3 - nu

    # 正向：V_νμ(x)·V_μν(x+ν̂)·V_νμ†(x+μ̂)
    t1 = e("...ab,...bc->...ac", Vnm, cp.roll(Vmu, -1, axis=a_nu))
    t1 = e("...ab,...cb->...ac", t1, cp.roll(Vnm, -1, axis=a_mu).conj())
    # 反向：V_νμ†(x−ν̂)·V_μν(x−ν̂)·V_νμ(x−ν̂+μ̂)
    t2 = e("...ab,...cb->...ac", cp.roll(Vnm, 1, axis=a_nu).conj(),
           cp.roll(Vmu, 1, axis=a_nu))
    t2 = e("...ab,...bc->...ac", t2,
           cp.roll(cp.roll(Vnm, 1, axis=a_nu), -1, axis=a_mu))
    return t1 + t2


def _staple_rho(V, mu, nu, rho):
    """级别 3/2 的 ρ 求和项：S(V; μ, ρ)（ρ≠μ,ν）。

    三项链接：V_ρμ(x)·V_μρ(x+ρ̂)·V_ρμ†(x+μ̂) + 反向。
    级别 3 时 V = U（原始），V[a,b] 退化为 U_a。
    """
    cp = get_backend()
    e = cp.einsum
    a_mu, a_rho = 3 - mu, 3 - rho
    if V.ndim == 7:   # 原始链接 (…,4,3,3)
        Vrm = V[..., rho, :, :]
        Vmr = V[..., mu, :, :]
    else:             # 对级链接 (…,4,4,3,3)
        Vrm = V[..., rho, mu, :, :]
        Vmr = V[..., mu, rho, :, :]
    t1 = e("...ab,...bc->...ac", Vrm, cp.roll(Vmr, -1, axis=a_rho))
    t1 = e("...ab,...cb->...ac", t1, cp.roll(Vrm, -1, axis=a_mu).conj())
    t2 = e("...ab,...cb->...ac", cp.roll(Vrm, 1, axis=a_rho).conj(),
           cp.roll(Vmr, 1, axis=a_rho))
    t2 = e("...ab,...bc->...ac", t2,
           cp.roll(cp.roll(Vrm, 1, axis=a_rho), -1, axis=a_mu))
    return t1 + t2


def hyp_smear(U, alpha1=0.75, alpha2=0.6, alpha3=0.3):
    """标准 HYP 涂抹（完整 (μ,ν) 对结构，Hasenbusch 参数）。

    Args:
        U: 规范场 (Nt,Nz,Ny,Nx,4,3,3)。
    Returns:
        涂抹后的规范场 V1 (Nt,Nz,Ny,Nx,4,3,3)（SU(3) 保持）。
    """
    cp = get_backend()
    Nd = 4
    # 级别 3：V3[μ,ν] = proj[(1−α3)U_μ + (α3/6)Σ_{ρ≠μ,ν} S(U; μ,ρ)]
    V3 = cp.zeros(U.shape[:-2] + (Nd, 3, 3), dtype=U.dtype)
    for mu in range(Nd):
        for nu in range(Nd):
            if nu == mu:
                continue
            acc = None
            for rho in range(Nd):
                if rho == mu or rho == nu:
                    continue
                s = _staple_rho(U, mu, nu, rho)
                acc = s if acc is None else acc + s
            V3[..., mu, nu, :, :] = proj_su3(
                (1.0 - alpha3) * U[..., mu, :, :] + (alpha3 / 6.0) * acc)

    # 级别 2：V2[μ,ν] = proj[(1−α2)V3[μ,ν] + (α2/6)Σ_{ρ≠μ,ν} S(V3; μ,ρ)]
    V2 = cp.zeros_like(V3)
    for mu in range(Nd):
        for nu in range(Nd):
            if nu == mu:
                continue
            acc = None
            for rho in range(Nd):
                if rho == mu or rho == nu:
                    continue
                s = _staple_rho(V3, mu, nu, rho)
                acc = s if acc is None else acc + s
            V2[..., mu, nu, :, :] = proj_su3(
                (1.0 - alpha2) * V3[..., mu, nu, :, :] + (alpha2 / 6.0) * acc)

    # 级别 1：V1[μ] = proj[(1−α1)V2[μ,ν₀] + (α1/4)Σ_{ν≠μ} S(V2; μ,ν)]
    V1 = cp.zeros_like(U)
    for mu in range(Nd):
        nu0 = (mu + 1) % Nd
        acc = None
        for nu in range(Nd):
            if nu == mu:
                continue
            s = _staple_rho(V2, mu, nu, nu)   # 级别 1 staple 用 V2[μ,ν]
            acc = s if acc is None else acc + s
        V1[..., mu, :, :] = proj_su3(
            (1.0 - alpha1) * V2[..., mu, nu0, :, :] + (alpha1 / 4.0) * acc)
    return V1
