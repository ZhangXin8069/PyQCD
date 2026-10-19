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

from ._const import CA, CF, gammaE, pi, b0, alpha_s as _alpha_s_const
from ._ensembles import fm_to_GeV
from ._matching import (
    _matching_matrix, _validate_uniform_grid, A_s_run,
)


def quasi_tmd_pdf(hR_z, z_grid, b_perp, pz_gev, p_t=None, x_grid=None,
                  n_pts=2048, z_max=None):
    """准 TMD-PDF（Eq.:quasi_tmd）。

    Args:
        hR_z: 正半轴重整化矩阵元 hR(z, b⊥, Pz)，形状 (nz, nb)
            或 (nz,)；负半轴按 hR(-z)=hR(z)* 作 Hermitian 延拓。
        z_grid: 非负 z 网格（fm）。
        b_perp: b⊥ 网格（fm）。
        pz_gev: Pz（GeV）。
        p_t: 核子能量分量 Pt（GeV）；None 时取 √(M²+Pz²)，M=0.94。
        x_grid: x 网格（默认 −1.5..1.5，256 点）。
        n_pts: 在输入 z 支撑上作均匀细化后的积分点数（至少 2）。
        z_max: 积分截断 z（fm；默认 z_grid 的最大值）。
    Returns:
        (x_grid, xg(x, b⊥))。
    """
    hR_z = np.asarray(hR_z)
    input_is_real = not np.iscomplexobj(hR_z)
    z_fm = np.asarray(z_grid, dtype=float)
    if hR_z.ndim == 1:
        hR_z = hR_z[:, None]
    if z_fm.ndim != 1 or len(z_fm) < 2:
        raise ValueError("z_grid 必须是一维且至少含两个点")
    if hR_z.ndim != 2 or hR_z.shape[0] != len(z_fm):
        raise ValueError("hR_z 的 z 维必须与 z_grid 一致")
    if not (np.all(np.isfinite(z_fm)) and np.all(np.isfinite(hR_z))):
        raise ValueError("hR_z 与 z_grid 必须有限")
    if not (isinstance(n_pts, (int, np.integer)) and not isinstance(n_pts, bool)
            and n_pts >= 2):
        raise ValueError("n_pts 必须是至少为 2 的整数")

    order = np.argsort(z_fm)
    z_fm = z_fm[order]
    hR_z = hR_z[order]
    if np.any(np.diff(z_fm) <= 0.0):
        raise ValueError("z_grid 不能含重复点")
    if z_fm[0] < 0.0:
        raise ValueError(
            "quasi_tmd_pdf 接收非负 z 半轴；负半轴由 Hermitian 延拓生成")
    if z_fm[0] != 0.0:
        raise ValueError("quasi_tmd_pdf 的 z_grid 必须从 z=0 开始")

    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.5, 256)
    x_grid = np.asarray(x_grid, dtype=float)
    if z_max is None:
        z_max_fm = z_fm[-1]
    else:
        z_max_fm = float(z_max)
        if not np.isfinite(z_max_fm):
            raise ValueError("z_max 必须是有限的 fm 长度")
    if z_max_fm < z_fm[0]:
        raise ValueError("z_max 不能小于 z_grid 的最小 fm 值")
    z_upper_fm = min(z_max_fm, z_fm[-1])

    if p_t is None:
        M_N = 0.94
        p_t = np.sqrt(M_N ** 2 + pz_gev ** 2)
    norm = (pz_gev ** 2) / (p_t ** 2)

    # 接口截断始终在 fm 完成；仅在积分网格确定后统一换算 GeV^-1。
    z_int_fm = np.linspace(z_fm[0], z_upper_fm, n_pts)
    hr_dtype = np.result_type(hR_z.dtype, np.float64)
    hr = np.empty((n_pts, hR_z.shape[1]), dtype=hr_dtype)
    for b_index in range(hR_z.shape[1]):
        if input_is_real:
            hr[:, b_index] = np.interp(
                z_int_fm, z_fm, hR_z[:, b_index])
        else:
            hr[:, b_index] = (
                np.interp(z_int_fm, z_fm, hR_z[:, b_index].real)
                + 1j * np.interp(
                    z_int_fm, z_fm, hR_z[:, b_index].imag))
    zz = z_int_fm / fm_to_GeV

    # 正半轴数据按 h(-z)=h(z)* 延拓：
    # ∫_{-L}^{L} dz exp(-ikz)h(z)
    #   = 2∫_0^L dz [cos(kz) Re h(z) + sin(kz) Im h(z)]。
    phase = np.multiply.outer(x_grid, zz * pz_gev)
    sine = np.sin(phase) if not input_is_real else None
    np.cos(phase, out=phase)
    dz = np.diff(zz)
    weights = np.empty_like(zz)
    weights[0] = 0.5 * dz[0]
    weights[-1] = 0.5 * dz[-1]
    weights[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    hr *= weights[:, None]
    integral = phase @ hr.real
    if sine is not None:
        integral += sine @ hr.imag
    integral *= 2.0
    out = integral / (2.0 * pi * pz_gev) / norm
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
    h = _finite_real_array(h_z, "h_z")
    z_fm = np.asarray(z_grid, dtype=float)
    if z_fm.ndim != 1 or z_fm.size < 2:
        raise ValueError("z_grid 必须是一维且至少含两个点")
    if h.shape != z_fm.shape:
        raise ValueError(
            f"h_z 与 z_grid 形状必须一致: {h.shape} vs {z_fm.shape}")
    if not np.isfinite(z_fm).all() or np.any(np.diff(z_fm) <= 0.0):
        raise ValueError("z_grid 必须全部有限且严格递增")
    if z_fm[0] != 0.0:
        raise ValueError("quasi_pdf_gluon 的 z_grid 必须从 z=0 开始")
    pz_gev = _positive_finite_scalar(pz_gev, "pz_gev")
    z = z_fm / fm_to_GeV                              # fm → GeV⁻¹
    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.5, 256)
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size == 0 or not np.isfinite(x_grid).all():
        raise ValueError("x_grid 必须是非空有限一维数组")

    small = np.abs(x_grid) < 1e-15
    xv = np.where(small, 1.0, x_grid)                 # 防除零，末尾回填 0
    integrand = h[None, :] * np.sin(np.outer(xv, pz_gev * z))
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:                              # NumPy < 1.20
        trapezoid = np.trapz
    integral = trapezoid(integrand, z, axis=1)         # 与原版梯形法等价
    out = (2.0 * pz_gev / xv) * integral
    out[small] = 0.0
    return x_grid, out


def _positive_finite_scalar(value, name):
    """Validate a strictly positive finite scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限正数，不得为布尔值")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限正数") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return value


def _finite_real_array(values, name):
    """Validate finite real data without silently discarding an imaginary part."""
    raw = np.asarray(values)
    if raw.dtype.kind == 'b':
        raise ValueError(f"{name} 必须是实数数组，不得为布尔数组")
    if np.iscomplexobj(raw):
        if np.any(np.imag(raw) != 0.0):
            raise ValueError(f"{name} 必须是实数值，不能静默丢弃虚部")
        raw = np.real(raw)
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 必须全部有限")
    return result


def cs_kernel_from_ratio(hR_pz1, hR_pz2, pz1, pz2):
    """CS 核提取（LPC 2020 比值法）。

    K(b⊥) = ln[hR(z=z₀, b⊥, Pz₁)/hR(z=z₀, b⊥, Pz₂)] / ln(Pz₁/Pz₂)

    Args:
        hR_pz1/hR_pz2: 两动量下的重整化矩阵元（(nz, nb) 或 (nz,)）。
        pz1/pz2: 动量（GeV）。
    Returns:
        K(b⊥) 数组（沿 z=0 提取；若 z 维首点噪声大可用 z=z₀ 索引）。
    """
    h1 = _finite_real_array(hR_pz1, "hR_pz1")
    h2 = _finite_real_array(hR_pz2, "hR_pz2")
    if h1.shape != h2.shape:
        raise ValueError(f"hR_pz1/hR_pz2 形状不一致: {h1.shape} vs {h2.shape}")
    if h1.ndim == 1:
        h1, h2 = h1[:, None], h2[:, None]
    elif h1.ndim != 2:
        raise ValueError("hR_pz1/hR_pz2 必须是一维或二维数组")
    if h1.shape[0] == 0:
        raise ValueError("hR_pz1/hR_pz2 的 z 轴不得为空")
    pz1 = _positive_finite_scalar(pz1, "pz1")
    pz2 = _positive_finite_scalar(pz2, "pz2")
    if pz1 == pz2:
        raise ValueError("pz1 与 pz2 必须不同")
    ratio = h1[0] / h2[0]           # z=0 处（b⊥ 依赖保留）
    if not np.isfinite(ratio).all() or np.any(ratio <= 0.0):
        raise ValueError(
            "CS kernel requires a finite strictly positive hR ratio")
    return np.log(ratio) / np.log(pz1 / pz2)


def cs_kernel_two_momentum(c01, c02, pz1_gev, pz2_gev, z_ref=1,
                           k_clip=None):
    """两动量 c0 比值 CS 核提取——工程封装（整合 test9 示例内联实现）。

    与 cs_kernel_from_ratio 的差异：
        1) 直接吃裸 c0(z,b)（未经 z₀ 归一化，避免归一点噪声放大）；
        2) z_ref 可选参考 z 索引（默认 1：z=0 处 c0≡1 无信息，
           z=1 为最低含 Wilson 线信息的分离）；
        3) ``k_clip`` 仅在调用方显式要求时截断；默认保留原始核并让异常值
           暴露出来，不能用隐式 clip 把噪声伪装成有限物理结果。

    Args:
        c01/c02: 两动量裸矩阵元 (nz, nb) 或 (nb,)（同 z 网格）。
        pz1_gev/pz2_gev: 两动量（GeV）。
        z_ref: 参考 z 行索引（负数按 Python 惯例回绕）。
        k_clip: 可选 (K_min, K_max) 显式截断区间；默认不截断。
    Returns:
        K(b⊥) 数组（沿所选 z 行）。
    """
    a = _finite_real_array(c01, "c01")
    b_ = _finite_real_array(c02, "c02")
    if a.ndim == 1:
        a, b_ = a[None, :], b_[None, :]
    elif a.ndim != 2 or b_.ndim != 2:
        raise ValueError("c01/c02 必须是一维或二维数组")
    if a.shape != b_.shape:
        raise ValueError(f"c01/c02 形状不一致: {a.shape} vs {b_.shape}")
    if a.shape[0] == 0:
        raise ValueError("c01/c02 的 z 轴不得为空")
    pz1_gev = _positive_finite_scalar(pz1_gev, "pz1_gev")
    pz2_gev = _positive_finite_scalar(pz2_gev, "pz2_gev")
    if pz1_gev == pz2_gev:
        raise ValueError("pz1_gev 与 pz2_gev 必须不同")
    z_ref = int(z_ref if z_ref >= 0 else len(a) + z_ref)
    if not (0 <= z_ref < len(a)):
        raise ValueError(f"z_ref={z_ref} 超出 z 维 [0, {len(a)-1}]")
    ratio = a[z_ref] / b_[z_ref]
    if not np.isfinite(ratio).all() or np.any(ratio <= 0.0):
        raise ValueError(
            "CS kernel requires a finite strictly positive c0 ratio")
    if k_clip is not None:
        if len(k_clip) != 2:
            raise ValueError("k_clip 必须是 (K_min, K_max) 二元组或 None")
        lo, hi = (float(k_clip[0]), float(k_clip[1]))
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError("k_clip 边界必须是有限数且 K_min < K_max")
        k_clip = (lo, hi)
    k = np.log(ratio) / np.log(pz1_gev / pz2_gev)
    return np.clip(k, *k_clip) if k_clip is not None else k


def soft_function_intrinsic(R_square, b_perp, mu=2.0):
    """内禀软函数 S_I(b⊥, μ)（从平方比值矩阵元提取的框架）。

    S_I(b⊥) = R²(b⊥)/R²(b⊥→0) 型归一（文献中常取短距归一化），
    此处实现通用归一化框架：输入逐 b⊥ 的比值矩阵元，输出归一化软函数。
    """
    R = np.asarray(R_square, dtype=float)
    return R / R[0]


def tmd_matching_hybrid(x_grid, y_grid=None, b_perp=None, mu=2.0, pz_gev=2.0,
                        cs_kernel=None, soft_factor=None, pz_scale=2.0,
                        x_tmd=None, lambda_s_fm=0.3,
                        return_metadata=False):
    """混合方案 TMD 匹配（Eq.:matching_tmd 的离散实现，1 圈核）。

    x·g̃(x, b⊥) = ∫(dy/|y|)·C^hybrid(x/y, λs, μ/(yPz))
                  ·[y·g(y, b⊥)/√S_I(b⊥,μ)]·exp[½ln(ζz/ζ)·K(b⊥)]

    1 圈匹配核 C^hybrid(ξ) = δ(1−ξ) + (α_s C_A/2π)·g_xy(ξ)，其中
    g_xy 复用 _matching._matching_kernels 的胶子核 g_0..g_3（ξ<0/0<ξ<1/ξ>1
    分区 + g_0 的 Si 项，含 μ/(yPz) 标度与 λ_s 截断）。

    数值结构：离散卷积 ``K`` 包含 ξ 三分区核和固定的 ``y=x`` counterterm，
    严格按一圈作 ``q - α_s C_A/(2π) K q``；不做矩阵逆，也不把已修正量
    再次送入非截断乘积。

    Args:
        x_grid: 输出 x 网格（(0,1)）。
        y_grid: 积分 y 网格；None 时取 x_grid（匹配矩阵结构要求同维）。
        b_perp: b⊥（fm，标记用）。
        mu: 匹配标度（GeV）。
        pz_gev: Pz（GeV）。
        cs_kernel: 必填 K(b⊥)（标量或数组）。未测量时必须显式提供 0
            并把结果标为 proxy，不能依赖函数默认值。
        soft_factor: 必填 √S_I(b⊥)（标量或数组）。未测量时必须显式提供 1
            并把结果标为 proxy。
        pz_scale: 参考快度标度 ζ 对应的动量（GeV）。
        x_tmd: 输入光锥 TMD y·g(y, b⊥)（数组，与 y_grid 同形）。
        lambda_s_fm: 大 λ 截断（fm，默认 0.3，同 _matching.hR_PDF）。
        return_metadata: 为 True 时额外返回匹配状态字典。
    Returns:
        ``(x_grid, x·g̃(x,b⊥))``；``return_metadata=True`` 时再附状态字典。
    """
    x = _validate_uniform_grid(x_grid, "x_grid")
    y = (x if y_grid is None
         else _validate_uniform_grid(y_grid, "y_grid"))
    if len(y) != len(x):
        raise ValueError("TMD 匹配矩阵结构要求 y 网格与 x 网格同维")
    if x_tmd is None:
        raise ValueError("x_tmd 不得为 None")
    yg = np.asarray(x_tmd)
    if yg.ndim not in (1, 2):
        raise ValueError("x_tmd 必须是一维或二维数组")
    if yg.ndim == 1:
        yg = yg[:, None]
    if yg.shape[0] != len(x) or yg.shape[1] == 0:
        raise ValueError("x_tmd 的 y 轴必须与 x/y 网格一致且不得为空")
    if not np.isfinite(yg).all():
        raise ValueError("x_tmd 必须全部有限")
    nb = yg.shape[1]
    n = len(x)
    mu = _positive_finite_scalar(mu, "mu")
    pz_gev = _positive_finite_scalar(pz_gev, "pz_gev")
    pz_scale = _positive_finite_scalar(pz_scale, "pz_scale")
    lambda_s_fm = _positive_finite_scalar(lambda_s_fm, "lambda_s_fm")
    if cs_kernel is None:
        raise ValueError("cs_kernel 必须显式提供；proxy 使用 0 也要写明")
    if soft_factor is None:
        raise ValueError("soft_factor 必须显式提供；proxy 使用 1 也要写明")

    lambda_s = lambda_s_fm / fm_to_GeV * pz_gev
    # 与 _matching.hR_PDF 同构：A_s ≡ α_s/(4π)，×4π 还原真耦合
    # （zengch 端 matching*.py 均为 alpha_s = A_s(mu)*4π）
    alpha_s = A_s_run(mu) * 4.0 * pi

    matrix = _matching_matrix(y, x, pz_gev, mu, lambda_s)
    coupling = alpha_s * CA / (2.0 * pi)

    # 快度演化因子与软函数（逐 b⊥）
    K = _finite_real_array(cs_kernel, "cs_kernel").ravel()
    S = _finite_real_array(soft_factor, "soft_factor").ravel()
    if K.size == 0 or S.size == 0:
        raise ValueError("cs_kernel 与 soft_factor 不得为空")
    if K.size not in (1, nb):
        raise ValueError(
            f"cs_kernel length must be 1 or b-axis length {nb}; got {K.size}")
    if S.size not in (1, nb):
        raise ValueError(
            f"soft_factor length must be 1 or b-axis length {nb}; got {S.size}")
    K = np.broadcast_to(K, (nb,))
    S = np.broadcast_to(S, (nb,))
    if np.any(S <= 0.0):
        raise ValueError("soft_factor 必须严格为正")
    rap = np.exp(0.5 * np.log((2.0 * pz_gev) ** 2 / (2.0 * pz_scale) ** 2) * K)
    if not np.isfinite(rap).all():
        raise ValueError("快度演化因子含非有限值")

    out = np.empty((n, nb), dtype=np.result_type(yg.dtype, matrix.dtype,
                                                  rap.dtype, S.dtype))
    for j in range(nb):
        v = yg[:, j] * rap[j] / np.sqrt(S[j])
        out[:, j] = v - coupling * (matrix @ v)
    if nb == 1:
        out = out[:, 0]
    if return_metadata:
        proxy_terms = []
        if np.all(K == 0.0):
            proxy_terms.append("cs_kernel_zero")
        if np.all(S == 1.0):
            proxy_terms.append("soft_factor_one")
        metadata = {
            "status": "one_loop_hybrid_matching_diagnostic",
            "cs_kernel": "explicit_input",
            "soft_factor": "explicit_input",
            "proxy_terms": proxy_terms,
            "physical_acceptance": False,
            "light_cone_matching": "applied_in_fixed_one_loop_form",
        }
        return x, out, metadata
    return x, out


def _alpha_s(mu, Lambda_QCD=0.23, nf=3.0):
    return _alpha_s_const(mu, Lambda_QCD, nf)


# ═══════════════════════════════════════════════════════════════════
# SFTX：小流时展开匹配系数（梯度流 → MS-bar，1 圈）
# ═══════════════════════════════════════════════════════════════════

_HBARC_GEV_FM = 0.1973269804


def flow_time_gev_m2(tau, a_fm):
    """把 ``tau=t/a^2`` 与 ``a[fm]`` 换成物理流时间 ``t[GeV^-2]``。"""
    tau = float(tau)
    a_fm = float(a_fm)
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError('tau 必须是有限正数')
    if not (np.isfinite(a_fm) and a_fm > 0.0):
        raise ValueError('a_fm 必须是有限正数')
    return tau * (a_fm / _HBARC_GEV_FM) ** 2


def _resolve_sftx_flow_time(t_gev_m2, tau, a_fm):
    explicit = t_gev_m2 is not None
    lattice = tau is not None or a_fm is not None
    if explicit == lattice:
        raise ValueError(
            '须且只能提供 t_gev_m2，或同时提供 tau 与 a_fm')
    if explicit:
        t_gev_m2 = float(t_gev_m2)
        if not (np.isfinite(t_gev_m2) and t_gev_m2 > 0.0):
            raise ValueError('t_gev_m2 必须是有限正数')
        return t_gev_m2
    if tau is None or a_fm is None:
        raise ValueError('tau 与 a_fm 必须同时提供')
    return flow_time_gev_m2(tau, a_fm)


def sftx_gluon_matching_coeff(*, mu, t_gev_m2=None, tau=None, a_fm=None,
                               Lambda_QCD=0.23, nf=3.0):
    """胶子算符的 SFTX 1 圈匹配系数（Suzuki 2013 / Mereghetti 2022）。

    O_MS(μ) = [1 + α_s(μ)/(4π)·c(t,μ)]·O_flow(t)

    c(t,μ) = 2·b₀·ln(2μ²t) + c₁
    其中 b₀ = 11 − 2Nf/3，c₁ = 2·b₀·γ_E 为常数项
    （胶子能量动量张量/双线性算符的 1 圈 SFTX 常数项，Suzuki 2013）。

    Args:
        mu: MS-bar 标度（GeV）。
        t_gev_m2: 物理流时间（GeV⁻²）。
        tau/a_fm: 或提供无量纲 ``tau=t/a²`` 与格距（fm），内部严格换算。
    Returns:
        (alpha_s/(4π), c(t,μ))——匹配系数。
    """
    t = _resolve_sftx_flow_time(t_gev_m2, tau, a_fm)
    al4pi = _alpha_s(mu, Lambda_QCD, nf) / (4.0 * np.pi)
    b0_ = b0(nf)
    # 1 圈 SFTX：O_MS(μ) = O_flow(t)·[1 + α_s/(4π)·(2b₀·ln(2μ²t) + 2b₀·γ_E)]
    c = 2.0 * b0_ * (np.log(2.0 * mu ** 2 * t) + gammaE)
    return al4pi, c


def sftx_energy_density_t0(E_t, *, mu, t_gev_m2=None, tau=None, a_fm=None,
                           Lambda_QCD=0.23, nf=3.0):
    """⟨E(t)⟩ 的 SFTX → MS-bar ⟨¼F²⟩(μ)（Suzuki 2013）。

    ⟨E(t)⟩ = 3(N²−1)/(16π²t²)·[1 + α_s/(4π)·(2b₀·ln(2μ²t) + c₂)]·(1 + O(t·⟨F²⟩/Nc))

    此处实现 ⟨¼F²⟩_MS = E(t)/[1 + α_s/(4π)·c(t)] 的 1 圈反解。
    """
    al4pi, c = sftx_gluon_matching_coeff(
        mu=mu, t_gev_m2=t_gev_m2, tau=tau, a_fm=a_fm,
        Lambda_QCD=Lambda_QCD, nf=nf)
    return np.asarray(E_t) / (1.0 + al4pi * c)
