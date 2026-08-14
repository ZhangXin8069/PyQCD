"""
胶子算符：Clover 场强张量 + Wilson 线的非定域胶子 OPE 算符
=============================================================

实现（照抄 docker-v20260805 ``compute_ope.py`` 的 donghx 算法，不改逻辑）：

    O_{μν}(z) = Σ_{x⊥} Tr[ F_{μν}(x + z) · W†(z→0) · F̃_{μν}(x) · W(0→z) ]

其中 F̃_{μν} = ½ ε_{μνρσ} F_{ρσ} 为对偶场强张量，W 为沿 z 方向的 Wilson 线
（roll 链接乘积构造）。

TMD 扩展（本库新增，供梯度流重整化 TMD-PDF 使用）：
    ``staple_operator`` 构造 staple 型 Wilson 线（含 b_⊥ 位移与 staple 回线），
    对应核子胶子 TMD 的非定域算符组合 M^{tx;tx} + M^{ty;ty} − 2M^{xy;xy}。

张量约定：gauge 为 (Nt, Nz, Ny, Nx, 4, 3, 3)（t,z,y,x 序，与成功实例一致）。
"""
from __future__ import annotations

import numpy as np

from ..tools._backend import get_backend


# ═══════════════════════════════════════════════════════════════════
# Tensor4 = ½ ε_{μνρσ}（Levi-Civita，系数查表）
# ═══════════════════════════════════════════════════════════════════

def build_tensor4() -> np.ndarray:
    """Build Tensor4[μ,ν,ρ,σ] = ½·ε_{μνρσ}（donghx Operator.py 原版）。"""
    T = np.zeros((4, 4, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            a = 1.0 if i > j else 0.0
            for k in range(4):
                b = (1.0 if i > k else 0.0) + (1.0 if j > k else 0.0)
                for l in range(4):
                    c = ((1.0 if i > l else 0.0)
                         + (1.0 if j > l else 0.0)
                         + (1.0 if k > l else 0.0))
                    if len({i, j, k, l}) == 4:
                        T[i, j, k, l] = 1.0 if int(a + b + c) % 2 == 0 else -1.0
    return 0.5 * T


_TENSOR4 = build_tensor4()


def plaquette_clover(g, mu: int, nu: int):
    """F_{μν} = -i/8 Σ_k (P_k - P_k†)，四叶 Clover 平均（donghx）。

    Args:
        g: gauge，形状 (Nt,Nz,Ny,Nx,4,3,3)，CPU(ndarray) 或 GPU(cupy)。
        mu, nu: Lorentz 指标 (0=t, 1=z, 2=y, 3=x)，mu != nu。
    Returns:
        F_{μν}，形状 (Nt,Nz,Ny,Nx,3,3)。
    """
    cp = get_backend()
    e = cp.einsum
    a_mu = 3 - mu   # 空间轴
    a_nu = 3 - nu

    g_lu = cp.roll(g, 1, axis=a_mu)
    g_rd = cp.roll(g, 1, axis=a_nu)
    g_ld = cp.roll(g_lu, 1, axis=a_nu)

    # P1 = P_{μν}
    p1 = e("tzyxab,tzyxbc->tzyxac", g[..., mu, :, :],
           cp.roll(g, -1, axis=a_mu)[..., nu, :, :])
    p1 = e("tzyxab,tzyxcb->tzyxac", p1,
           cp.roll(g, -1, axis=a_nu)[..., mu, :, :].conj())
    p1 = e("tzyxab,tzyxcb->tzyxac", p1, g[..., nu, :, :].conj())

    # P2 = P_{ν,-μ}
    p2 = e("tzyxab,tzyxcb->tzyxac",
           cp.roll(g_lu, -1, axis=a_mu)[..., nu, :, :],
           cp.roll(g_lu, -1, axis=a_nu)[..., mu, :, :].conj())
    p2 = e("tzyxab,tzyxcb->tzyxac", p2, g_lu[..., nu, :, :].conj())
    p2 = e("tzyxab,tzyxbc->tzyxac", p2, g_lu[..., mu, :, :])

    # P3 = P_{-μ,-ν}
    p3 = e("tzyxba,tzyxcb->tzyxac",
           cp.roll(g_ld, -1, axis=a_nu)[..., mu, :, :].conj(),
           g_ld[..., nu, :, :].conj())
    p3 = e("tzyxab,tzyxbc->tzyxac", p3, g_ld[..., mu, :, :])
    p3 = e("tzyxab,tzyxbc->tzyxac", p3,
           cp.roll(g_ld, -1, axis=a_mu)[..., nu, :, :])

    # P4 = P_{-ν,μ}
    p4 = e("tzyxba,tzyxbc->tzyxac", g_rd[..., nu, :, :].conj(),
           g_rd[..., mu, :, :])
    p4 = e("tzyxab,tzyxbc->tzyxac", p4,
           cp.roll(g_rd, -1, axis=a_mu)[..., nu, :, :])
    p4 = e("tzyxab,tzyxcb->tzyxac", p4,
           cp.roll(g_rd, -1, axis=a_nu)[..., mu, :, :].conj())

    tr = (0, 1, 2, 3, 5, 4)  # 共轭转置（色指标 Hermitian 共轭）
    ans = (p1 - p1.conj().transpose(*tr)
           + p2 - p2.conj().transpose(*tr)
           + p3 - p3.conj().transpose(*tr)
           + p4 - p4.conj().transpose(*tr))
    return cp.array(-1j, dtype=ans.dtype) * ans / cp.array(8.0, dtype=ans.real.dtype)


def compute_dual_field_strength(F_dict: dict, mu: int, nu: int):
    """F̃_{μν} = ½ Σ_{ρσ} ε_{μνρσ} F_{ρσ}。"""
    cp = get_backend()
    result = None
    for rho in range(4):
        for sigma in range(4):
            coeff = _TENSOR4[mu, nu, rho, sigma]
            if abs(coeff) < 1e-10 or rho == sigma:
                continue
            F_rs = F_dict.get((rho, sigma))
            if F_rs is None:
                continue
            term = cp.array(coeff, dtype=F_rs.dtype) * F_rs
            result = term if result is None else result + term
    return result


def gluon_ope_operator_z0(gauge, mu: int, nu: int, z_dir: int, delta_z: int,
                          Nt: int, Nx: int, compute_dtype=None):
    """O_{μν}(z)（z = 0..delta_z-1，全部时间片）。

    返回 (delta_z, Nt) 复数数组（CPU）。照抄 compute_ope.py 的 donghx roll
    Wilson 线算法。
    """
    cp = get_backend()
    if compute_dtype is None:
        compute_dtype = gauge.dtype
    if mu == nu:
        return np.zeros((delta_z, Nt), dtype=compute_dtype)

    z_axis = 3 - z_dir   # Wilson 线方向的空间轴

    need_pairs = {(mu, nu)}
    for rho in range(4):
        for sigma in range(4):
            if abs(_TENSOR4[mu, nu, rho, sigma]) > 1e-10 and rho != sigma:
                need_pairs.add((rho, sigma))

    F_dict = {pair: plaquette_clover(gauge, pair[0], pair[1])
              for pair in need_pairs}
    F = F_dict[(mu, nu)]
    F_tilde = compute_dual_field_strength(F_dict, mu, nu)
    del F_dict

    U_z = gauge[..., z_dir, :, :]   # (Nt,Nz,Ny,Nx,3,3)

    spatial_axes = (1, 2, 3)
    ope = np.zeros((delta_z, Nt), dtype=np.float64)

    for zi in range(delta_z):
        if zi == 0:
            ope_t = cp.einsum("tzyxab,tzyxba->tzyx", F, F_tilde)
            ope[0] = cp.asnumpy(cp.sum(ope_t, axis=spatial_axes)).real
            continue

        ope_t = cp.roll(F, -zi, axis=z_axis)
        for step in range(zi):
            U_conj = cp.roll(U_z, -(zi - 1 - step), axis=z_axis).conj()
            ope_t = cp.einsum("...ab,...cb->...ac", ope_t, U_conj)
        ope_t = cp.einsum("...ab,...bc->...ac", ope_t, F_tilde)
        for step in range(zi):
            U_fwd = cp.roll(U_z, -step, axis=z_axis)
            ope_t = cp.einsum("...ab,...bc->...ac", ope_t, U_fwd)

        trace = cp.einsum("...aa->...", ope_t)
        ope[zi] = cp.asnumpy(cp.sum(trace, axis=spatial_axes)).real

    return ope.astype(compute_dtype)


# ═══════════════════════════════════════════════════════════════════
# TMD staple 型 Wilson 线算符（本库新增）
# ═══════════════════════════════════════════════════════════════════
# 核子胶子 TMD-PDF 的非定域算符（参照 refer/papers/gluon_tmd_gradient_flow_
# continuum.tex）：横向位移 b_⊥ 的 staple Wilson 线组合。
# ═══════════════════════════════════════════════════════════════════

def _wilson_line(gauge, start, length: int, direction: int):
    """沿 direction 方向从 start 出发的 Wilson 线乘积（roll 构造）。

    Args:
        gauge: (Nt,Nz,Ny,Nx,4,3,3)。
        start: (t, x1, x2, x3) 元组（t,z,y,x 序）。
        length: 链接数。
        direction: 空间轴（0=x, 1=y, 2=z）。
    Returns:
        (3,3) 色矩阵（与 gauge 同后端）。
    """
    cp = get_backend()
    W = cp.eye(3, dtype=gauge.dtype)
    t, x1, x2, x3 = start
    pos = [t, x1, x2, x3]
    for _ in range(length):
        W = W @ gauge[tuple(pos)][:, :]
        pos[direction + 1] += 1
    return W


def staple_operator(gauge, mu: int, nu: int, z: int, b_perp: int,
                    z_dir: int = 2, b_dir: int = 0):
    """staple 型胶子算符（TMD 用）：F 在 z 方向分离 z、横向位移 b_perp。

    O_{μν}(z, b_⊥) = Tr[ F̃_{μν}(x + z·ẑ + b_⊥·b̂) · W(staple) ]
    其中 W(staple) 为：沿 +z 走 z 步 → 沿 +b̂ 走 b_perp 步 → 沿 −z 走 z 步。

    返回 (3,3) 色迹标量（逐格点），供后续空间求和。
    """
    cp = get_backend()
    F = plaquette_clover(gauge, mu, nu)
    F_tilde = compute_dual_field_strength({(mu, nu): F}, mu, nu)

    Nt, Nz, Ny, Nx = gauge.shape[:4]
    axes = (t, x1, x2, x3) = (0, 1, 2, 3)
    # b_dir: 0=x→轴 1，1=y→轴 2，2=z→轴 3
    b_axis = 1 + b_dir
    z_axis = 1 + z_dir

    # 对每个格点计算 Tr[ F̃(x)·U(x→x+z·ẑ + b·b̂) ]
    # roll 构造：先沿 z 走 z 步、再横向 b_perp 步、再沿 −z 走 z 步（staple 回线）
    U_z = gauge[..., z_dir, :, :]
    U_b = gauge[..., b_dir, :, :]

    ope_t = cp.roll(F_tilde, -(z + b_perp) if z_dir == b_dir else -(z), axis=z_axis)
    ope_t = cp.roll(ope_t, -b_perp, axis=b_axis) if b_dir != z_dir else ope_t

    for step in range(z):
        U_conj = cp.roll(U_z, -(z - 1 - step + b_perp), axis=z_axis).conj()
        ope_t = cp.einsum("...ab,...cb->...ac", ope_t, U_conj)
    for step in range(b_perp):
        U_bc = cp.roll(U_b, -(b_perp - 1 - step), axis=b_axis).conj()
        ope_t = cp.einsum("...ab,...cb->...ac", ope_t, U_bc)
    for step in range(z):
        U_fwd = cp.roll(U_z, -(b_perp - step), axis=z_axis)
        ope_t = cp.einsum("...ab,...bc->...ac", ope_t, U_fwd)

    trace = cp.einsum("...aa->...", ope_t)
    return trace  # (Nt,Nz,Ny,Nx)
