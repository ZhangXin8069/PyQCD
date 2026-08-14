"""
梯度流（Gradient Flow）——Wilson flow 格点演化
================================================

实现 Luescher (2010, JHEP 1008:071) 的 Wilson flow：

    ∂_t B_μ = D_ν G_{νμ},    B_μ(0) = A_μ

格点形式：V̇(x,μ) = −g₀²{∂_{x,μ}S_W(V)}·V(x,μ) = Z(V)(x,μ)·V(x,μ)，
其中 {∂_{x,μ}S_W(V)} 用 6-staple 求和 Ω_μ(x) 构造（Wilson 作用量的流导数），
并投影到无迹反厄米部分以保持 SU(3)：

    Z(V)(x,μ) = P_{ah}[ Ω_μ(x)·V_μ†(x) ]，  P_{ah}[M] = (M−M†)/2 − Tr(M−M†)/(2N)·I

时间积分采用 Luescher 的三阶 Runge–Kutta 格式（Eq.(3.8)）：

    W₀ = V_t
    W₁ = exp( ε/4 · Z(W₀) ) · W₀
    W₂ = exp( 8ε/9·Z(W₁) − 17ε/36·Z(W₀) ) · W₁
    V_{t+ε} = exp( 3ε/4·Z(W₂) − 8ε/9·Z(W₁) + 17ε/36·Z(W₀) ) · W₂

本模块为梯度流重整化（Monahan–Orginos 2017；NieMiera et al. 2025 采用
τ = 3a² 的 Wilson flow 涂抹）提供数值引擎：输出流时间 t 处的流规范场 V_t，
供胶子算符（场强张量 + Wilson 线）与核子胶子 TMD-PDF 矩阵元计算使用。
"""
from __future__ import annotations

import numpy as np

from ..tools._backend import get_backend


# ═══════════════════════════════════════════════════════════════════
# SU(3) 辅助函数
# ═══════════════════════════════════════════════════════════════════

def proj_su3(A):
    """把矩阵投影到 SU(3)：保持行列式归一（用于 exp 后的重投影）。"""
    cp = get_backend()
    det = cp.linalg.det(A)
    scale = det ** (-1.0 / 3.0)
    return scale[..., None, None] * A


def su3_exp(A):
    """SU(3) 矩阵指数 exp(A)（A 为无迹反厄米，截断级数 + 投影）。"""
    cp = get_backend()
    # exp(A) ≈ I + A + A²/2 + A³/6 + A⁴/24（对 ε·Z 足够，ε 小）
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    res = (cp.eye(3, dtype=A.dtype)[None, None, None, None, ...]
           + A + 0.5 * A2 + (1.0 / 6.0) * A3 + (1.0 / 24.0) * A4)
    return proj_su3(res)


# ═══════════════════════════════════════════════════════════════════
# 6-staple（Wilson 作用量流导数）
# ═══════════════════════════════════════════════════════════════════

def staple_6(U):
    """计算 6-staple 求和 Ω_μ(x)（Wilson 作用量 ∂S 的链接求和）。

    Ω_μ(x) = Σ_{ν≠μ} [ U_μ(x,ν) + U_μ(x,−ν) ]，其中
        U_μ(x,ν)  = U_ν(x)·U_μ(x+ν̂)·U_ν†(x+μ̂)
        U_μ(x,−ν) = U_ν†(x−ν̂)·U_μ(x−ν̂)·U_ν(x−ν̂+μ̂)

    Args:
        U: 规范场 (Nt,Nz,Ny,Nx,4,3,3)（t,z,y,x 序）。
    Returns:
        Ω (Nt,Nz,Ny,Nx,4,3,3)。
    """
    cp = get_backend()
    e = cp.einsum
    res = cp.zeros_like(U)
    for mu in range(4):
        a_mu = 3 - mu
        acc = None
        for nu in range(4):
            if nu == mu:
                continue
            a_nu = 3 - nu
            # U_μ(x,ν) = U_ν(x) U_μ(x+ν̂) U_ν†(x+μ̂)
            t1 = e("...ab,...bc->...ac", U[..., nu, :, :],
                   cp.roll(U, -1, axis=a_nu)[..., mu, :, :])
            t1 = e("...ab,...cb->...ac", t1,
                   cp.roll(U, -1, axis=a_mu)[..., nu, :, :].conj())
            # U_μ(x,−ν) = U_ν†(x−ν̂) U_μ(x−ν̂) U_ν(x−ν̂+μ̂)
            t2 = e("...ab,...cb->...ac",
                   cp.roll(U, 1, axis=a_nu)[..., nu, :, :].conj(),
                   cp.roll(U, 1, axis=a_nu)[..., mu, :, :])
            t2 = e("...ab,...bc->...ac", t2,
                   cp.roll(cp.roll(U, 1, axis=a_nu), -1, axis=a_mu)[..., nu, :, :])
            acc = t1 + t2 if acc is None else acc + t1 + t2
        res[..., mu, :, :] = acc
    return res


# ═══════════════════════════════════════════════════════════════════
# 流方程右端
# ═══════════════════════════════════════════════════════════════════

def flow_derivative(U):
    """Z(V) = P_{ah}[Ω·V†]·V —— 流方程右端（含投影，保持 SU(3)）。

    P_{ah}[M] = (M − M†)/2 − Tr(M − M†)/(2N)·I
    """
    cp = get_backend()
    Omega = staple_6(U)
    X = cp.einsum("...ab,...cb->...ac", Omega, U.conj())
    X_ah = 0.5 * (X - X.conj().transpose(0, 1, 2, 3, 5, 4))
    tr = cp.einsum("...aa->...", X_ah)
    X_ah = X_ah - (tr / 3.0)[..., None, None] * cp.eye(3, dtype=X_ah.dtype)
    return cp.einsum("...ab,...bc->...ac", X_ah, U)


def wilson_flow_step(U, eps):
    """单步 RK3（Luescher 2010 Eq.(3.8)）。"""
    cp = get_backend()
    W0 = U
    Z0 = flow_derivative(W0)
    W1 = su3_exp(0.25 * eps * Z0) @ W0
    Z1 = flow_derivative(W1)
    W2 = su3_exp((8.0 / 9.0) * eps * Z1 - (17.0 / 36.0) * eps * Z0) @ W1
    Z2 = flow_derivative(W2)
    return su3_exp((3.0 / 4.0) * eps * Z2 - (8.0 / 9.0) * eps * Z1
                   + (17.0 / 36.0) * eps * Z0) @ W2


def wilson_flow(U, tau, eps=0.01, n_steps=None, verbose=False):
    """Wilson flow 演化到流时间 tau。

    Args:
        U: 初始规范场 (Nt,Nz,Ny,Nx,4,3,3)。
        tau: 目标流时间（格点单位：t = tau/a²）。
        eps: RK3 步长（默认 0.01，Luescher 建议 ε ≲ 0.05 保证 O(ε³) 精度）。
        n_steps: 步数（默认 tau/eps 取整）。
    Returns:
        流规范场 V_tau（与 U 同形状同 dtype）。
    """
    if n_steps is None:
        n_steps = max(int(round(tau / eps)), 1)
    V = U
    for k in range(n_steps):
        V = wilson_flow_step(V, eps)
        if verbose and (k % 10 == 0 or k == n_steps - 1):
            print(f"[wilson_flow] step {k + 1}/{n_steps}, t = {(k + 1) * eps:.4f}")
    return V


# ═══════════════════════════════════════════════════════════════════
# 流的观测量：E(t) = ¼ G²_{μν}（SFTX 与尺度设定用）
# ═══════════════════════════════════════════════════════════════════

def flow_action_density(U):
    """E(t,x) = ¼ G_{μν}^a G_{μν}^a（用 Clover 场强张量近似）。"""
    cp = get_backend()
    E = None
    from ..operator._gluon_ope import plaquette_clover
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F = plaquette_clover(U, mu, nu)
            term = cp.einsum("...ab,...ba->...", F, F.conj()).real
            E = term if E is None else E + term
    return 0.25 * E


def scale_setting_t0(U, target=0.3, tau_min=0.01, tau_max=1.0,
                     n_probe=20, eps=0.01):
    """尺度设定：求流时间 t₀ 使 t²⟨E(t)⟩ = target（Luescher 2010 惯例）。"""
    taus = np.linspace(tau_min, tau_max, n_probe)
    V = U
    t_prev, E2_prev = 0.0, 0.0
    for i, tau in enumerate(taus):
        V = wilson_flow(U, tau, eps=eps)
        E2 = tau ** 2 * flow_action_density(V).mean()
        if i > 0 and (E2 - target) * (E2_prev - target) < 0:
            from scipy.interpolate import interp1d
            f = interp1d([t_prev, tau], [E2_prev, E2])
            return float(f(target))
        t_prev, E2_prev = tau, E2
    raise ValueError(f"target t²⟨E⟩={target} 未在 τ∈[{tau_min},{tau_max}] 内达到")
