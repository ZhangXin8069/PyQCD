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

import os

import numpy as np

from ..tools._backend import get_backend


def _to_cpu(x):
    """后端无关的 GPU→CPU 转换（cupy 禁止 np.asarray 隐式转换，须用 asnumpy/get）。"""
    b = get_backend()
    asnumpy = getattr(b, 'asnumpy', None)
    if asnumpy is not None:
        return asnumpy(x)
    getter = getattr(x, 'get', None)
    if getter is not None:
        return getter()
    return np.asarray(x)


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
    m = cp.matmul
    a_mu = 3 - mu   # 空间轴
    a_nu = 3 - nu

    g_lu = cp.roll(g, 1, axis=a_mu)
    g_rd = cp.roll(g, 1, axis=a_nu)
    g_ld = cp.roll(g_lu, 1, axis=a_nu)

    tr = (0, 1, 2, 3, 5, 4)  # 共轭转置（色指标 Hermitian 共轭）

    # P1 = P_{μν}（einsum→cuBLAS matmul 等价改写，逐位一致）
    p1 = m(g[..., mu, :, :],
           cp.roll(g, -1, axis=a_mu)[..., nu, :, :])
    p1 = m(p1, cp.roll(g, -1, axis=a_nu)[..., mu, :, :].conj()
           .transpose(*tr))
    p1 = m(p1, g[..., nu, :, :].conj().transpose(*tr))

    # P2 = P_{ν,-μ}
    p2 = m(cp.roll(g_lu, -1, axis=a_mu)[..., nu, :, :],
           cp.roll(g_lu, -1, axis=a_nu)[..., mu, :, :].conj()
           .transpose(*tr))
    p2 = m(p2, g_lu[..., nu, :, :].conj().transpose(*tr))
    p2 = m(p2, g_lu[..., mu, :, :])

    # P3 = P_{-μ,-ν}
    p3 = m(cp.roll(g_ld, -1, axis=a_nu)[..., mu, :, :].conj()
           .transpose(*tr),
           g_ld[..., nu, :, :].conj().transpose(*tr))
    p3 = m(p3, g_ld[..., mu, :, :])
    p3 = m(p3, cp.roll(g_ld, -1, axis=a_mu)[..., nu, :, :])

    # P4 = P_{-ν,μ}
    p4 = m(g_rd[..., nu, :, :].conj().transpose(*tr),
           g_rd[..., mu, :, :])
    p4 = m(p4, cp.roll(g_rd, -1, axis=a_mu)[..., nu, :, :])
    p4 = m(p4, cp.roll(g_rd, -1, axis=a_nu)[..., mu, :, :].conj()
           .transpose(*tr))

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
            ope[0] = _to_cpu(cp.sum(ope_t, axis=spatial_axes)).real
            continue

        ope_t = cp.roll(F, -zi, axis=z_axis)
        for step in range(zi):
            U_conj = cp.roll(U_z, -(zi - 1 - step), axis=z_axis).conj()
            ope_t = cp.matmul(ope_t, U_conj.transpose(0, 1, 2, 3, 5, 4))
        ope_t = cp.matmul(ope_t, F_tilde)
        for step in range(zi):
            U_fwd = cp.roll(U_z, -step, axis=z_axis)
            ope_t = cp.matmul(ope_t, U_fwd)

        trace = cp.einsum("...aa->...", ope_t)
        ope[zi] = _to_cpu(cp.sum(trace, axis=spatial_axes)).real

    return ope.astype(compute_dtype)


# ═══════════════════════════════════════════════════════════════════
# Gauge reader（ILDG .lime，照抄 compute_ope.py）
# ═══════════════════════════════════════════════════════════════════

def _first_link_unitarity(raw: np.ndarray, Nc: int = 3) -> float:
    """首 3×3 链接的幺正性偏差 |U·U† − I|。"""
    U = raw[:Nc * Nc * 2].reshape(Nc, Nc, 2)
    Uc = U[..., 0] + 1j * U[..., 1]
    return float(np.abs(Uc @ Uc.conj().T - np.eye(Nc)).max())


def read_gauge_lime(filepath: str, Nt: int, Nx: int, Nc: int = 3) -> np.ndarray:
    """读 .lime 规范组态 → complex128 (Nt,Nx,Nx,Nx,4,Nc,Nc)。

    ILDG .lime 为大端 float64 + XML 头 + 尾部记录；扫描 ±16KB 窗口
    按幺正性定位数据起点。
    """
    expected_elems = Nt * Nx * Nx * Nx * 4 * Nc * Nc * 2
    expected_bytes = expected_elems * 8
    file_size = os.path.getsize(filepath)
    approx_off = file_size - expected_bytes

    def _read_at(off):
        with open(filepath, 'rb') as f:
            f.seek(off)
            return np.fromfile(f, dtype='>f8', count=expected_elems)

    if 0 <= approx_off < file_size:
        raw = _read_at(approx_off)
        if raw.size == expected_elems and _first_link_unitarity(raw, Nc) < 1e-3:
            return _gauge_from_raw(raw, Nt, Nx, Nc)

    for delta in range(-16384, 16385, 8):
        off = approx_off + delta
        if off < 0 or off + expected_bytes > file_size:
            continue
        raw = _read_at(off)
        if raw.size == expected_elems and _first_link_unitarity(raw, Nc) < 1e-3:
            return _gauge_from_raw(raw, Nt, Nx, Nc)

    raise ValueError(f"No valid gauge data found in {filepath} "
                     f"(size={file_size} bytes)")


def _gauge_from_raw(raw: np.ndarray, Nt: int, Nx: int, Nc: int = 3) -> np.ndarray:
    raw = raw.reshape(Nt, Nx, Nx, Nx, 4, Nc, Nc, 2)
    tg = raw[..., 0] + 1j * raw[..., 1]
    return tg.astype(np.complex128, copy=False)


def _read_gauge_or_skip(filepath: str, Nt: int, Nx: int, Nc: int = 3):
    """尝试读组态；文件不存在返回 None（供管线占位）。"""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        return read_gauge_lime(filepath, Nt, Nx, Nc)
    except (ValueError, OSError):
        return None


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
