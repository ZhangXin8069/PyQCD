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
                          Nt: int, Nx: int, compute_dtype=None, *,
                          mu2: int | None = None, nu2: int | None = None,
                          direction: int = 1,
                          second_insert: str = "Ftilde"):
    """O_{μν;μ₂ν₂}(z)（z = 0..delta_z-1，全部时间片）。

    默认 F_{μν}(z)·W†·F̃_{μν}(0)·W（+z Wilson 线，照抄 compute_ope.py 的
    donghx roll 算法）。扩展参数（照抄 zhangxin workflow / Operator.py）：

    - ``mu2/nu2``：F̃ 插入的独立 Lorentz 对（gauge_fix_helicity 交叉混合，
      如 (3,0;2,1)）；默认与 (mu, nu) 相同。
    - ``direction=-1``：负 z 方向 Wilson 线变体
      （Operator.py ``operators_new_z0_mz_mu2``：F(−z)·W(−z→0)·F̃(0)·W†）。

    返回 (delta_z, Nt) 数组；同对 +z 保持原版实数行为，交叉对或 −z 为复数。
    """
    cp = get_backend()
    if compute_dtype is None:
        compute_dtype = gauge.dtype
    if mu == nu:
        return np.zeros((delta_z, Nt), dtype=compute_dtype)
    if mu2 is None:
        mu2 = mu
    if nu2 is None:
        nu2 = nu

    z_axis = 3 - z_dir   # Wilson 线方向的空间轴
    cross = (mu2 != mu) or (nu2 != nu)
    complex_out = bool(cross or direction < 0)

    need_pairs = {(mu, nu), (mu2, nu2)}
    for a, b in ((mu, nu), (mu2, nu2)):
        for rho in range(4):
            for sigma in range(4):
                if abs(_TENSOR4[a, b, rho, sigma]) > 1e-10 and rho != sigma:
                    need_pairs.add((rho, sigma))

    F_dict = {pair: plaquette_clover(gauge, pair[0], pair[1])
              for pair in need_pairs}
    F = F_dict[(mu, nu)]
    F_tilde = compute_dual_field_strength(F_dict, mu2, nu2)
    if second_insert == "F":
        # donghx Calc_ope_unpol 通道：第二插入为 F（非对偶）
        F_tilde = F_dict[(mu2, nu2)]
    del F_dict

    U_z = gauge[..., z_dir, :, :]   # (Nt,Nz,Ny,Nx,3,3)

    spatial_axes = (1, 2, 3)
    out_dtype = np.complex128 if complex_out else np.float64
    ope = np.zeros((delta_z, Nt), dtype=out_dtype)

    def _store(zi, arr_cpu):
        ope[zi] = arr_cpu if complex_out else arr_cpu.real

    for zi in range(delta_z):
        if zi == 0:
            ope_t = cp.einsum("tzyxab,tzyxba->tzyx", F, F_tilde)
            _store(0, _to_cpu(cp.sum(ope_t, axis=spatial_axes)))
            continue

        if direction > 0:
            # +z：F(+z)·U† 链 → F̃(0) → U 链（compute_ope.py 原算法）
            ope_t = cp.roll(F, -zi, axis=z_axis)
            for step in range(zi):
                U_conj = cp.roll(U_z, -(zi - 1 - step), axis=z_axis).conj()
                ope_t = cp.matmul(ope_t, U_conj.transpose(0, 1, 2, 3, 5, 4))
            ope_t = cp.matmul(ope_t, F_tilde)
            for step in range(zi):
                U_fwd = cp.roll(U_z, -step, axis=z_axis)
                ope_t = cp.matmul(ope_t, U_fwd)
        else:
            # −z（operators_new_z0_mz_mu2）：F(−z)·U 链 → F̃(0) → U† 链
            ope_t = cp.roll(F, zi, axis=z_axis)
            for step in range(zi):
                U_k = cp.roll(U_z, zi - step, axis=z_axis)
                ope_t = cp.matmul(ope_t, U_k)
            ope_t = cp.matmul(ope_t, F_tilde)
            for step in range(zi):
                Uc_k = cp.roll(U_z, step + 1, axis=z_axis).conj()
                ope_t = cp.matmul(ope_t, Uc_k.transpose(0, 1, 2, 3, 5, 4))

        trace = cp.einsum("...aa->...", ope_t)
        _store(zi, _to_cpu(cp.sum(trace, axis=spatial_axes)))

    return ope.astype(compute_dtype)


def gluon_ff_operator_z0(gauge, mu: int, nu: int, delta_z: int,
                         Nt: int, Nx: int, *, mu2: int | None = None,
                         nu2: int | None = None, direction: int = 1):
    """固定规范（无 Wilson 线）FF 关联（照抄 Operator.py operators_FF_z0/_mz）。

    F_{μν}(±z·ẑ) 与 F̃_{μ₂ν₂}(0) 逐点收缩后全空间求和；无 Wilson 线，
    仅适用于已固定规范（Coulomb/Landau）的系综——规范协变算符的
    系统误差对照通道。z 方向按参考实现硬编码。

    返回 (delta_z, Nt) 复数数组。
    """
    cp = get_backend()
    if mu2 is None:
        mu2 = mu
    if nu2 is None:
        nu2 = nu
    z_axis = 3 - 2   # 参考实现硬编码 z_dir=2

    need_pairs = {(mu, nu), (mu2, nu2)}
    for a, b in ((mu, nu), (mu2, nu2)):
        for rho in range(4):
            for sigma in range(4):
                if abs(_TENSOR4[a, b, rho, sigma]) > 1e-10 and rho != sigma:
                    need_pairs.add((rho, sigma))
    F_dict = {pair: plaquette_clover(gauge, pair[0], pair[1])
              for pair in need_pairs}
    F = F_dict[(mu, nu)]
    F_tilde = compute_dual_field_strength(F_dict, mu2, nu2)
    del F_dict

    out = np.zeros((delta_z, Nt), dtype=np.complex128)
    for zi in range(delta_z):
        ope_t = cp.roll(F, -direction * zi, axis=z_axis)
        ope_t = cp.matmul(ope_t, F_tilde)
        trace = cp.einsum("...aa->...", ope_t)
        out[zi] = _to_cpu(cp.sum(trace, axis=(1, 2, 3)))
    return out


def get_ope_lorentz_pairs(zdir: int, mode: str = "unpol"):
    """OPE 计算的 (mu, nu, mu2, nu2) Lorentz 指派表（照抄 zhangxin workflow
    get_ope_lorentz_pairs，源自 donghx Calc_ope_* 的 rank 分派）。

    Args:
        zdir: Wilson 线方向（0=x, 1=y, 2=z）。
        mode: "unpol" / "helicity"（同表，后者用对偶叠）、
              "gauge_fix_unpol"（3 对）、"gauge_fix_helicity"（4 对交叉混合）。
    """
    if mode in ("unpol", "helicity"):
        pairs = [
            (3, (zdir + 1) % 3, 3, (zdir + 1) % 3),
            (3, (zdir + 2) % 3, 3, (zdir + 2) % 3),
            ((zdir + 1) % 3, (zdir + 2) % 3,
             (zdir + 1) % 3, (zdir + 2) % 3),
        ]
    elif mode == "gauge_fix_unpol":
        pairs = [
            (3, 0, 3, 0),   # F_{t,x}
            (3, 1, 3, 1),   # F_{t,y}
            (0, 1, 0, 1),   # F_{x,y}
        ]
    elif mode == "gauge_fix_helicity":
        pairs = [
            (3, 0, 2, 1),
            (3, 1, 0, 2),
            (3, 2, 0, 1),
            (0, 1, 3, 2),
        ]
    else:
        raise ValueError(f"Unknown OPE mode: {mode}")
    return pairs


# ═══════════════════════════════════════════════════════════════════
# Gauge reader（ILDG .lime，照抄 compute_ope.py）
# ═══════════════════════════════════════════════════════════════════

def _first_link_unitarity(raw: np.ndarray, Nc: int = 3) -> float:
    """首 3×3 链接的幺正性偏差 |U·U† − I|。

    对乱字节偏移（数据起点未对齐时）可能产生 inf/nan，这里用 errstate
    抑制溢出告警并以 isfinite 兜底，避免刷屏、保证错偏移被正确拒绝。
    """
    U = raw[:Nc * Nc * 2].reshape(Nc, Nc, 2)
    Uc = U[..., 0] + 1j * U[..., 1]
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            dev = float(np.abs(Uc @ Uc.conj().T - np.eye(Nc)).max())
        except (FloatingPointError, ValueError):
            return float('inf')
    return dev if np.isfinite(dev) else float('inf')


def _resolve_ildg_binary_record(filepath):
    """Resolve an ILDG file or an extracted ``.lime.contents`` directory.

    ``lime_contents`` stores the payload as the canonical
    ``msg02.rec04.ildg-binary-data`` record.  Keep the resolution explicit so
    an unrelated file in the directory can never be selected silently.
    """
    filepath = os.fspath(filepath)
    if not os.path.isdir(filepath):
        return filepath
    record = os.path.join(filepath, "msg02.rec04.ildg-binary-data")
    if os.path.isfile(record):
        return record
    raise ValueError(
        f"ILDG contents directory has no msg02.rec04.ildg-binary-data: "
        f"{filepath}"
    )


def read_gauge_lime(filepath: str, Nt: int, Nx: int, Nc: int = 3) -> np.ndarray:
    """读 .lime 规范组态 → complex128 (Nt,Nx,Nx,Nx,4,Nc,Nc)。

    ILDG .lime 为大端 float64 + XML 头 + 尾部记录；扫描 ±16KB 窗口
    按幺正性定位数据起点。

    ``filepath`` 也可以是解包后的 ``.lime.contents`` 目录；此时读取其
    标准 ``msg02.rec04.ildg-binary-data`` 记录。

    优化：幺正性判定只需首链接（18 个 double = 144B）。先用微小探针定位
    合法偏移，命中后才全量读取一次，避免每个候选偏移都读整份（≈573MB）
    造成的 TB 级读盘与乱字节数据导致的 matmul 溢出告警。扫描顺序、阈值
    与回退逻辑与原实现完全一致。
    """
    filepath = _resolve_ildg_binary_record(filepath)
    expected_elems = Nt * Nx * Nx * Nx * 4 * Nc * Nc * 2
    expected_bytes = expected_elems * 8
    file_size = os.path.getsize(filepath)
    approx_off = file_size - expected_bytes

    def _read_at(off):
        with open(filepath, 'rb') as f:
            f.seek(off)
            return np.fromfile(f, dtype='>f8', count=expected_elems)

    def _probe_unitarity(off: int) -> bool:
        """仅读首链接（144B）判定偏移合法性。"""
        if off < 0 or off + expected_bytes > file_size:
            return False
        with open(filepath, 'rb') as f:
            f.seek(off)
            head = np.fromfile(f, dtype='>f8', count=Nc * Nc * 2)
        if head.size != Nc * Nc * 2:
            return False
        return _first_link_unitarity(head, Nc) < 1e-3

    if 0 <= approx_off < file_size and _probe_unitarity(approx_off):
        raw = _read_at(approx_off)
        if raw.size == expected_elems and _first_link_unitarity(raw, Nc) < 1e-3:
            return _gauge_from_raw(raw, Nt, Nx, Nc)

    for delta in range(-16384, 16385, 8):
        off = approx_off + delta
        if _probe_unitarity(off):
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
