"""
功能: 提供胶子非极化圈算符计算常用的 gauge links 读取, clover 场强和 Wilson line 工具.

主要功能:
- 按大端 float64 原始二进制读取 gauge links.
- 构造 clover 场强 F_{mu,nu} 和完整的 Nmu x Nmu 反对称场强数组.
- 计算从 lD 指向 0 的直线 Wilson line, 其中 D 是单位有向线段.
- 计算非极化圈算符 Tr[F_{mu,nu}(z) W[z, 0] F_{mu,nu}(0) W[0, z]].

约定: gauge links 的 para: t, z, y, x, mu, color, color, 其中 mu = 0, 1, 2, 3
分别表示 x, y, z, t 方向; 每个 link 和场强都是 Nc x Nc 复矩阵.
"""

import numpy as np
from opt_einsum import contract


# ==================== 方向和平面约定 ====================

# NC: 颜色基础表示的数量; NDIR: 方向 mu 的总数.
NC = 3
NDIR = 4

# 方向 mu = 0, 1, 2, 3 分别表示 x, y, z, t, 而 gauge links 的
# para: t, z, y, x, mu, color, color, 因此 mu 方向到 gauge 坐标轴的映射为
# mu -> (3, 2, 1, 0)[mu].
GAUGE_AXIS = (3, 2, 1, 0)

# 独立平面指标对, 完整场强由 F_{nu,mu} = -F_{mu,nu} 补全.
INDEPENDENT_MU_NU = tuple(
    (mu, nu) for mu in range(NDIR) for nu in range(mu + 1, NDIR)
)


# ==================== gauge links 读取 ====================

def load_gauge_links(conf_file, Nt, Nx):
    """
    读取大端 float64 格式的 SU(3) gauge links.

    参数:
    -----
    conf_file: gauge links 文件完整路径, 实部和虚部按欧氏时间, z, y, x,
    link direction, color row, color column 顺序交错排列.
    Nt: 时间方向格点数.
    Nx: 空间方向格点数.

    输出:
    -----
    gauge: para: t, z, y, x, mu, color, color, mu = 0, 1, 2, 3 分别表示 x, y, z, t 方向.
    """
    raw_gauge = np.fromfile(conf_file, dtype=">f8")
    raw_gauge = raw_gauge.reshape(Nt, Nx, Nx, Nx, NDIR, NC, NC, 2)

    # para: t, z, y, x, mu, color, color
    gauge = raw_gauge[..., 0] + raw_gauge[..., 1] * 1j

    print(f"loaded gauge links: {conf_file}")
    return gauge


# ==================== 场强 ====================

def calc_clover_field_strength(gauge, mu, nu):
    r"""
    计算一个 mu, nu 平面的四叶 clover 场强 F_{mu,nu}.

    参数:
    -----
    gauge: gauge links, para: t, z, y, x, mu, color, color.
    mu, nu: 场强平面指标, 要求 mu != nu.

    输出:
    -----
    field_strength: 该平面的 clover 场强, para: t, z, y, x, color, color.
    """
    if mu == nu:
        raise ValueError("clover field strength requires mu != nu")

    coordinate_mu = GAUGE_AXIS[mu]
    coordinate_nu = GAUGE_AXIS[nu]

    gauge_rightup = gauge  # U(x)
    gauge_leftup = np.roll(gauge, 1, axis=coordinate_mu)  # U(x - mu)
    gauge_rightdown = np.roll(gauge, 1, axis=coordinate_nu)  # U(x - nu)
    gauge_leftdown = np.roll(
        gauge_leftup, 1, axis=coordinate_nu)  # U(x - mu - nu)

    # P_{mu,nu}(x) = U_mu(x) U_nu(x + mu) U_mu^dagger(x + nu) U_nu^dagger(x)
    # para: t, z, y, x, color, color, 下同.
    p_mu_nu = contract(
        "tzyxab,tzyxbc->tzyxac",
        gauge_rightup[..., mu, :, :],
        np.roll(gauge_rightup, -1, axis=coordinate_mu)[..., nu, :, :],
    )
    p_mu_nu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_mu_nu,
        np.roll(gauge_rightup, -1, axis=coordinate_nu)[..., mu, :, :].conj(),
    )
    p_mu_nu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_mu_nu,
        gauge_rightup[..., nu, :, :].conj(),
    )

    # P_{nu,-mu}(x) = U_nu(x) U_mu^dagger(x + nu - mu)
    #                  U_nu^dagger(x - mu) U_mu(x - mu)
    p_nu_minus_mu = contract(
        "tzyxab,tzyxcb->tzyxac",
        np.roll(gauge_leftup, -1, axis=coordinate_mu)[..., nu, :, :],
        np.roll(gauge_leftup, -1, axis=coordinate_nu)[..., mu, :, :].conj(),
    )
    p_nu_minus_mu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_nu_minus_mu,
        gauge_leftup[..., nu, :, :].conj(),
    )
    p_nu_minus_mu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_nu_minus_mu,
        gauge_leftup[..., mu, :, :],
    )

    # P_{-mu,-nu}(x) = U_mu^dagger(x - mu) U_nu^dagger(x - mu - nu)
    #                   U_mu(x - mu - nu) U_nu(x - nu)
    p_minus_mu_minus_nu = contract(
        "tzyxba,tzyxcb->tzyxac",
        np.roll(gauge_leftdown, -1, axis=coordinate_nu)[..., mu, :, :].conj(),
        gauge_leftdown[..., nu, :, :].conj(),
    )
    p_minus_mu_minus_nu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_minus_mu_minus_nu,
        gauge_leftdown[..., mu, :, :],
    )
    p_minus_mu_minus_nu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_minus_mu_minus_nu,
        np.roll(gauge_leftdown, -1, axis=coordinate_mu)[..., nu, :, :],
    )

    # P_{-nu,mu}(x) = U_nu^dagger(x - nu) U_mu(x - nu)
    #                  U_nu(x + mu - nu) U_mu^dagger(x)
    p_minus_nu_mu = contract(
        "tzyxba,tzyxbc->tzyxac",
        gauge_rightdown[..., nu, :, :].conj(),
        gauge_rightdown[..., mu, :, :],
    )
    p_minus_nu_mu = contract(
        "tzyxab,tzyxbc->tzyxac",
        p_minus_nu_mu,
        np.roll(gauge_rightdown, -1, axis=coordinate_mu)[..., nu, :, :],
    )
    p_minus_nu_mu = contract(
        "tzyxab,tzyxcb->tzyxac",
        p_minus_nu_mu,
        np.roll(gauge_rightdown, -1, axis=coordinate_nu)[..., mu, :, :].conj(),
    )

    q_mu_nu = p_mu_nu + p_nu_minus_mu + p_minus_mu_minus_nu + p_minus_nu_mu
    field_strength = (
        -1j * (q_mu_nu - np.swapaxes(q_mu_nu.conj(), -1, -2)) / 8.0
    )
    return field_strength


def calc_clover_field_strength_all(gauge):
    r"""
    构造完整的 Nmu x Nmu clover 场强数组.

    参数:
    -----
    gauge: gauge links, para: t, z, y, x, mu, color, color.

    输出:
    -----
    field_strength_all: clover 场强, para: mu, nu, t, z, y, x, color, color.
    """
    # para: mu, nu, t, z, y, x, color, color.
    field_strength_all = np.zeros(
        (NDIR, NDIR) + gauge.shape[:4] + (NC, NC),
        dtype=complex,
    )
    for mu, nu in INDEPENDENT_MU_NU:
        field_strength = calc_clover_field_strength(gauge, mu, nu)
        field_strength_all[mu, nu] = field_strength
        field_strength_all[nu, mu] = -field_strength
    return field_strength_all


# ==================== Wilson line ====================

def calc_neg_dir_wilson_line(gauge, direction, l):
    r"""
    记 D 是方向 d 的单位长度有向线段
    计算从起点 lD 指向终点 0 的直线 Wilson line.
        W(0; lD) = U_d(0) U_d(D) \cdots U_d((l-1)D),
    l = 0 时 W(0; 0) 为单位矩阵.

    参数:
    -----
    gauge: gauge links, para: t, z, y, x, mu, color, color.
    direction: Wilson line 的 link 方向, 0, 1, 2, 3 分别表示 x, y, z, t.
    l: Wilson line 的长度, 要求 l >= 0.

    输出:
    -----
    wilson_line: W(0; lD), para: t, z, y, x, color, color.
    """
    if l < 0:
        raise ValueError("l must be non-negative")

    coordinate_axis = GAUGE_AXIS[direction]
    # para: t, z, y, x, color, color.
    link = gauge[..., direction, :, :]
    # l = 0 时 Wilson line 退化为单位矩阵. 广播按最后一维对齐,
    # np.eye(Nc) 的 (Nc, Nc) 自动对应 link 的最后两维 color, 前面 t, z, y, x 维自动扩展.
    wilson_line = np.broadcast_to(
        np.eye(NC, dtype=gauge.dtype), link.shape
    ).copy()
    for step in range(l):
        wilson_line = contract(
            "tzyxab,tzyxbc->tzyxac",
            wilson_line,
            np.roll(link, -step, axis=coordinate_axis),
        )
    return wilson_line

# ==================== 非极化圈算符 ====================


def calc_ope_unpol(gauge, field_strength, l, direction, mu, nu):
    r"""
    计算一个平面 (mu, nu) 上的非极化圈算符.

    O(t) = sum_x Tr[F_{mu,nu}(z) W[z, 0] F_{mu,nu}(0) W[0, z]],
    其中 z = lD, D 是方向 d 的单位有向线段; W[0, z] 是 calc_neg_dir_wilson_line
    的结果, W[z, 0] 是它的共轭转置.

    参数:
    -----
    gauge: gauge links, para: t, z, y, x, mu, color, color.
    field_strength: clover 场强, para: mu, nu, t, z, y, x, color, color.
    l: 两端场强沿 d 方向的距离, 等于 Wilson line 包含的 link 数.
    direction: Wilson line 的 link 方向, 0, 1, 2, 3 分别表示 x, y, z, t.
    mu, nu: 两端场强所在的平面指标, 要求 mu != nu.

    输出:
    -----
    ope_unpol: 逐时间片的算符, para: t.
    """
    coordinate_axis = GAUGE_AXIS[direction]
    # neg_wl = W[0, z] = U_d(x) U_d(x + d) ... U_d(x + (l - 1) d),
    # 从 z 指向 0 的负方向 Wilson line, 用普通 link 连乘得到.
    neg_wl = calc_neg_dir_wilson_line(gauge, direction, l)
    # pos_wl = W[z, 0] = W^dagger(x) 由共轭转置得到, link 的乘积顺序会自动反过来.
    pos_wl = np.swapaxes(neg_wl.conj(), -1, -2)

    # F_{mu,nu}(z) W[z, 0]
    ope = contract(
        "tzyxab,tzyxbc->tzyxac",
        np.roll(field_strength[mu, nu], -l, axis=coordinate_axis),
        pos_wl,
    )
    # F_{mu,nu}(z) W[z, 0] F_{mu,nu}(0)
    ope = contract(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        field_strength[mu, nu],
    )
    # F_{mu,nu}(z) W[z, 0] F_{mu,nu}(0) W[0, z]
    ope = contract(
        "tzyxab,tzyxbc->tzyxac",
        ope,
        neg_wl,
    )

    ope_unpol = np.sum(np.trace(ope, axis1=-2, axis2=-1), axis=(1, 2, 3))
    return ope_unpol
