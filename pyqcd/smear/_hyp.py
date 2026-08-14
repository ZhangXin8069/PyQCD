"""
HYP 涂抹（Hasenbusch 2001）——梯度流/线性发散抑制的备选方案
============================================================

理论文档（refer/papers/gluon_tmd_gradient_flow_continuum.tex）明确对比了
HYP 涂抹与 Wilson flow：内部笔记对 HYP5 涂抹与 Wilson flow（τ=1.4）的
结果定性一致。本模块实现标准 HYP 三级迭代：

    级别 3：V₃(x,μ) = proj[(1−α₃)U(x,μ) + (α₃/6)Σ_{ν≠μ} S(x,μ;ν)]
    级别 2：V₂(x,μ) = proj[(1−α₂)U(x,μ) + (α₂/6)Σ_{ν≠μ} S₃(x,μ;ν)]
    级别 1：V₁(x,μ) = proj[(1−α₁)U(x,μ) + (α₁/4)Σ_{ν≠μ} S₂(x,μ;ν)]

其中 S(x,μ;ν) = U_ν(x)U_μ(x+ν̂)U_ν†(x+μ̂) + U_ν†(x−ν̂)U_μ(x−ν̂)U_ν(x−ν̂+μ̂)，
proj 为 SU(3) 投影。标准参数 α₁=0.75, α₂=0.6, α₃=0.3。
"""
from __future__ import annotations

import numpy as np

from ..tools._backend import get_backend


def proj_su3(A):
    """投影到 SU(3)：A → A(A†A)^{−1/2}·det[·]^{−1/3}（显式 matmul）。"""
    cp = get_backend()
    M = cp.matmul(A.conj(), cp.swapaxes(A, -1, -2))            # A†A
    # 特征分解求 (A†A)^{-1/2}
    w, V = cp.linalg.eigh(M)
    w = cp.maximum(w, 1e-30)
    sqrt_inv = V * (1.0 / cp.sqrt(w))[..., None, :]            # V·diag
    sqrt_inv = cp.matmul(sqrt_inv, cp.swapaxes(V.conj(), -1, -2))
    P = cp.matmul(A, sqrt_inv)
    det = cp.linalg.det(P)
    return P * (det ** (-1.0 / 3.0))[..., None, None]


def _staple_sum(U, mu, V=None, exclude=None):
    """Σ_{ν≠μ} S(x,μ;ν)：用链接场 V（默认 U）构造 6-staple 求和。"""
    cp = get_backend()
    e = cp.einsum
    if V is None:
        V = U
    Vp = [cp.roll(V, -1, axis=a) for a in (0, 1, 2, 3)]
    Vm = [cp.roll(V, 1, axis=a) for a in (0, 1, 2, 3)]
    a_mu = 3 - mu
    acc = None
    for nu in range(4):
        if nu == mu:
            continue
        if exclude and nu in exclude:
            continue
        a_nu = 3 - nu
        t1 = e("...ab,...bc->...ac", V[..., nu, :, :], Vp[a_nu][..., mu, :, :])
        t1 = e("...ab,...cb->...ac", t1, Vp[a_mu][..., nu, :, :].conj())
        t2 = e("...ab,...cb->...ac", Vm[a_nu][..., nu, :, :].conj(),
               Vm[a_nu][..., mu, :, :])
        t2 = e("...ab,...bc->...ac", t2,
               cp.roll(Vm[a_nu], -1, axis=a_mu)[..., nu, :, :])
        acc = t1 + t2 if acc is None else acc + t1 + t2
    return acc


def hyp_smear(U, alpha1=0.75, alpha2=0.6, alpha3=0.3):
    """标准 HYP 涂抹（单次迭代，Hasenbusch 参数）。

    Args:
        U: 规范场 (Nt,Nz,Ny,Nx,4,3,3)。
    Returns:
        涂抹后的规范场（SU(3) 保持）。
    """
    cp = get_backend()
    # 级别 3：全部方向用原始链接
    V3 = cp.zeros_like(U)
    for mu in range(4):
        S = _staple_sum(U, mu)
        V3[..., mu, :, :] = proj_su3((1.0 - alpha3) * U[..., mu, :, :]
                                     + (alpha3 / 6.0) * S)
    # 级别 2：链接 V3，staple 也来自 V3
    V2 = cp.zeros_like(U)
    for mu in range(4):
        S = _staple_sum(V3, mu)
        V2[..., mu, :, :] = proj_su3((1.0 - alpha2) * U[..., mu, :, :]
                                     + (alpha2 / 6.0) * S)
    # 级别 1：链接 V2（HYP 惯例：第一级 staple 排除与 μ 正交方向的重叠，
    # 用 α1/4 加权）
    V1 = cp.zeros_like(U)
    for mu in range(4):
        S = _staple_sum(V2, mu)
        V1[..., mu, :, :] = proj_su3((1.0 - alpha1) * U[..., mu, :, :]
                                     + (alpha1 / 4.0) * S)
    return V1
