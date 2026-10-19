"""
功能: 读取蒸馏本征矢, vvv 和 perambulator, 并计算带动量投影的 vvv.
输出: NumPy 或 CuPy 复数组, 默认输出 CuPy . perambulator 输出可为 DR 或 DP 基.
后端: 各函数的 xp 参数决定计算后端. 若输出 NumPy, 不需要使用 GPU. 
eig_idx 表示本征矢索引, 沿用输入文件中对应本征值从小到大的排列.
"""
import os
import time

import cupy as cp
import numpy as np
from opt_einsum import contract

import gamma_matrix_DP
from profiling_tools import print_mem_peak, print_var_size


N_COLOR = 3
N_SPIN = 4

# ==================== 通用函数 ====================


def real_imag_to_complex(raw_data):
    """
    把 real, imag 交错排列的实数组转换为复数组.

    参数:
    -----
    raw_data: real, imag 交错排列的实数组, 最后一维长度为 2, 即 [..., real, imag].

    输出:
    -----
    complex_data: 复数组, 形状为 raw_data.shape[:-1], 与输入共享内存 (零拷贝).

    说明: 用 view 把每对相邻 real, imag 重解释为一个 complex128, 要求最后一维连续.
    """
    return raw_data.view(np.complex128).reshape(raw_data.shape[:-1])


# ==================== 本征矢与空间相位 ====================
def calc_fourier_phase(momentum, Nx, sign=-1, xp=cp):
    r"""
    计算格点动量的空间相位 $\exp(sign\,i\vec{p}\cdot\vec{x})$.

    参数:
    -----
    momentum: 傅里叶动量, 分量顺序为 px, py, pz.
    Nx: 立方空间格点边长.

    可选项:
    sign: 相位指数符号, -1 给出 $e^{-i\vec{p}\cdot\vec{x}}$, +1 给出 $e^{+i\vec{p}\cdot\vec{x}}$.
    xp: 计算后端, 默认 cupy; 传 numpy 时全程在 CPU 计算.

    输出:
    -----
    phase: 空间傅里叶相位, 轴顺序为 z, y, x, 可用 phase[z, y, x] 读取指定格点的相位.
    """
    px, py, pz = (float(component) for component in momentum)
    Nx = int(Nx)

    z = xp.arange(Nx)[:, None, None]
    y = xp.arange(Nx)[None, :, None]
    x = xp.arange(Nx)[None, None, :]
    angle = 2.0 * xp.pi * (px * x + py * y + pz * z) / Nx

    phase = xp.exp(sign * 1j * angle)

    # para: z, y, x
    return phase


def apply_momentum_smearing(eigvec, smear_momentum, Nx, xp=cp):
    r"""
    按 $\widetilde{V}(x)=\exp(+i\vec{q}\cdot\vec{x})V(x)$ 给本征矢注入单相位.

    说明: 文献约定的涂抹相位为 $e^{+i\vec{q}\cdot\vec{x}}$ (即 +iqx), 本函数按此约定提供相位;
    读取的 peram 用相反约定 $e^{-i\vec{q}\cdot\vec{x}}$ (即 -iqx) 涂抹, 两者相差一个符号.
    因此读取文件名为 mx-3 的 peram 数据时, 这里应使用 px=3 的涂抹动量.

    参数:
    -----
    eigvec: 本征矢, 轴顺序为 eig_idx, site, color, 其中 site = z * Nx**2 + y * Nx + x.
    smear_momentum: 动量涂抹的格点动量 $\vec{q}$, 分量顺序为 qx, qy, qz, 使用 $e^{+i\vec{q}\cdot\vec{x}}$.
    Nx: 立方空间格点边长.

    可选项:
    xp: 计算后端, 默认 cupy; 传 numpy 时全程在 CPU 计算.

    输出:
    -----
    eigvec: 加入动量涂抹相位后的本征矢, 轴顺序仍为 eig_idx, site, color.
    """
    phase_zyx = calc_fourier_phase(smear_momentum, Nx, sign=1, xp=xp)

    phase_site = phase_zyx.reshape(-1)

    # para: eig_idx, site, color; site = z * Nx**2 + y * Nx + x
    return contract("vsc,s->vsc", eigvec, phase_site)


def read_eigvec_timeslice(
    eigvec_file,
    num_eigvec,
    Nx,
    xp=cp,
):
    r"""
    读取一个时间片的蒸馏本征矢.

    参数:
    -----
    eigvec_file: 要读取的本征矢文件完整路径.
    num_eigvec: 实际计算中保留的本征矢数目 Nev1.
    Nx: 立方空间格点边长.

    可选项:
    xp: 计算和输出后端, 默认 cupy; 传 numpy 时全程在 CPU 计算.

    输出:
    -----
    eigvec: 截断后的本征矢, 轴顺序为 eig_idx, site, color, 其中 site = z * Nx**2 + y * Nx + x.
    """
    Nx = int(Nx)
    site_count = Nx**3
    raw_data = np.fromfile(eigvec_file, dtype="<f8")

    data_per_eigvec = site_count * N_COLOR * 2
    num_eigvec_file = raw_data.size // data_per_eigvec

    raw_data = raw_data.reshape(
        num_eigvec_file, site_count, N_COLOR, 2
    )
    eigvec = real_imag_to_complex(raw_data)
    eigvec = xp.asarray(eigvec[:num_eigvec])

    # para: eig_idx, site, color; site = z * Nx**2 + y * Nx + x
    return eigvec


# ==================== vvv 读取与计算 ====================
def calc_vvv_timeslice(
    eigvec,
    momentum,
    Nx,
    xp=cp,
):
    r"""
    计算一个时间片的重子结构元素 $vvv(\vec{p})$.

    具体计算

    .. math::

        vvv_{lmn}(\vec{p}, t) = \sum_{\vec{x}} \epsilon^{abc}
        e^{-i\vec{p}\cdot\vec{x}}
        V^{(l)}_a(\vec{x}, t) V^{(m)}_b(\vec{x}, t)
        V^{(n)}_c(\vec{x}, t).

    其中 l, m, n 是三个本征矢指标, a, b, c 是三个夸克的颜色指标.
    代码中的六项加减法实现 Levi-Civita 符号 $\epsilon^{abc}$ 的颜色反对称收缩.

    参数:
    -----
    eigvec: 本征矢, 轴顺序为 eig_idx, site, color, 其中 site = z * Nx**2 + y * Nx + x.
    momentum: vvv 的傅里叶动量 $\vec{p}$, 分量顺序为 px, py, pz.
    Nx: 立方空间格点边长.

    可选项:
    xp: 计算后端, 默认 cupy; 传 numpy 时全程在 CPU 计算.

    输出:
    -----
    vvv: 一个时间片的 vvv, 轴顺序为 eig1_idx, eig2_idx, eig3_idx.
    """
    Nx = int(Nx)

    fourier_phase_zyx = calc_fourier_phase(
        momentum, Nx, sign=-1, xp=xp
    )
    num_eigvec = eigvec.shape[0]
    vvv = xp.zeros(
        (num_eigvec, num_eigvec, num_eigvec),
        dtype=xp.complex128,
    )

    # 每次只收缩一个 z 切片, 避免产生过大的中间数组.
    sites_per_z = Nx * Nx
    for z_index in range(Nx):
        site_slice = slice(z_index * sites_per_z, (z_index + 1) * sites_per_z)

        # para: site(y, x)
        phase_z = fourier_phase_zyx[z_index].reshape(-1)

        # para: eig_idx, site(y, x), color
        eigvec_z = eigvec[:, site_slice]

        # para: eig_idx, site(y, x)
        color_0 = eigvec_z[:, :, 0]
        color_1 = eigvec_z[:, :, 1]
        color_2 = eigvec_z[:, :, 2]

        vvv += contract("x,ax,bx,cx->abc", phase_z, color_0, color_1, color_2)
        vvv += contract("x,ax,bx,cx->abc", phase_z, color_1, color_2, color_0)
        vvv += contract("x,ax,bx,cx->abc", phase_z, color_2, color_0, color_1)
        vvv -= contract("x,ax,bx,cx->abc", phase_z, color_2, color_1, color_0)
        vvv -= contract("x,ax,bx,cx->abc", phase_z, color_0, color_2, color_1)
        vvv -= contract("x,ax,bx,cx->abc", phase_z, color_1, color_0, color_2)

    # para: eig1_idx, eig2_idx, eig3_idx
    return vvv


def calc_vvv(
    eigvec_dir,
    conf_id,
    Nt,
    Nx,
    num_eigvec,
    momentum,
    smear_momentum=None,
    xp=cp,
):
    r"""
    从所有时间片的本征矢计算带傅里叶因子的 vvv.

    参数:
    -----
    eigvec_dir: 本征矢文件所在目录, 文件名为 eigvecs_t{time_index:03d}_{conf_id}.
    conf_id: 规范场构型编号.
    Nt: 时间方向格点数.
    Nx: 空间方向立方格点边长.
    num_eigvec: 每个时间片保留并用于收缩的本征矢数目 Nev1.
    momentum: vvv 的傅里叶动量 $\vec{p}$, 分量顺序为 px, py, pz.

    可选项:
    smear_momentum: 动量涂抹的格点动量 $\vec{q}$, 分量顺序为 qx, qy, qz; None 表示普通蒸馏.
    xp: 计算和输出后端, 默认 cupy; 传 numpy 时全程在 CPU 计算.

    输出:
    -----
    vvv: 所有时间片的 vvv, 轴顺序为 t, eig1_idx, eig2_idx, eig3_idx.
    """
    Nt = int(Nt)
    Nx = int(Nx)
    vvv = xp.zeros(
        (Nt, num_eigvec, num_eigvec, num_eigvec),
        dtype=xp.complex128,
    )
    print("\n========== start vvv calc ==========")
    t_start_all = time.time()
    for time_index in range(Nt):
        eigvec_file = os.path.join(
            eigvec_dir, f"eigvecs_t{time_index:03d}_{conf_id}"
        )
        eigvec = read_eigvec_timeslice(
            eigvec_file,
            num_eigvec,
            Nx,
            xp=xp,
        )

        if smear_momentum is not None:
            eigvec = apply_momentum_smearing(
                eigvec, smear_momentum, Nx, xp=xp
            )
        vvv[time_index] = calc_vvv_timeslice(
            eigvec,
            momentum,
            Nx,
            xp=xp,
        )
    print(eigvec_file)
    print(f"vvv calc done, total time used: {time.time() - t_start_all:.3f} s")
    print("========== end vvv calc ==========\n")
    print_mem_peak("after vvv calc", use_gpu=(xp is cp))
    print_var_size("after vvv calc", vvv=vvv)

    # para: t, eig1_idx, eig2_idx, eig3_idx
    return vvv


def _read_vvv_timeslice_host(vvv_file, num_eigvec):
    """
    在 CPU 上读取并截断一个时间片的 vvv.

    参数:
    -----
    vvv_file: 单个时间片 vvv 二进制文件路径.
    num_eigvec: 读取后保留的本征矢数目 Nev1.

    输出:
    -----
    vvv: 一个时间片的 vvv, 轴顺序为 eig1_idx, eig2_idx, eig3_idx.
    """
    # 文件按 real, imag, real, imag, ... 交错保存.
    raw_vvv = np.fromfile(vvv_file, dtype="<f8")
    raw_vvv = real_imag_to_complex(raw_vvv.reshape(-1, 2))

    # 一个时间片包含 Nev**3 个复数, 由此得到文件中的本征矢数目.
    num_eigvec_file = int(round(raw_vvv.size ** (1.0 / 3.0)))

    vvv = raw_vvv.reshape(
        num_eigvec_file, num_eigvec_file, num_eigvec_file
    )
    # para: eig1_idx, eig2_idx, eig3_idx
    return vvv[:num_eigvec, :num_eigvec, :num_eigvec]


def read_vvv_timeslice(
    vvv_dir,
    conf_id,
    time_index,
    momentum,
    num_eigvec,
    xp=cp,
):
    """
    读取单独保存的一个时间片 vvv 文件.

    参数:
    -----
    vvv_dir: vvv 文件所在目录, 文件名为 vvv.t{time_index:03d}.Px{px}Py{py}Pz{pz}.conf{conf_id}.
    conf_id: 规范场构型编号.
    time_index: 要读取的时间片编号 t.
    momentum: 文件名中的傅里叶动量, 分量顺序为 px, py, pz.
    num_eigvec: 读取后保留的本征矢数目 Nev1.

    可选项:
    xp: 输出后端, 默认 cupy; 读取和截断始终在 CPU 上完成.

    输出:
    -----
    vvv: 一个时间片的 vvv, 轴顺序为 eig1_idx, eig2_idx, eig3_idx.
    """
    px, py, pz = momentum
    vvv_file = os.path.join(
        vvv_dir,
        f"vvv.t{time_index:03d}.Px{px}Py{py}Pz{pz}.conf{conf_id}",
    )
    vvv = _read_vvv_timeslice_host(vvv_file, num_eigvec)
    print(vvv_file)

    # para: eig1_idx, eig2_idx, eig3_idx
    return xp.asarray(vvv)


def read_vvv(
    vvv_dir,
    conf_id,
    Nt,
    momentum,
    num_eigvec,
    file_layout="auto",
    xp=cp,
):
    """
    读取所有时间片的 vvv, 兼容合并文件和逐时间片文件.

    参数:
    -----
    vvv_dir: vvv 文件所在目录, 合并文件名为 vvv.Px{px}Py{py}Pz{pz}.conf{conf_id}.
    conf_id: 规范场构型编号.
    Nt: 时间方向格点数.
    momentum: 文件名中的傅里叶动量, 分量顺序为 px, py, pz.
    num_eigvec: 读取后保留的本征矢数目 Nev1.

    可选项:
    file_layout: vvv 文件布局, 可选 "auto", "combined" 或 "timeslices".
    xp: 输出后端, 默认 cupy; 读取和截断始终在 CPU 上完成.

    输出:
    -----
    vvv: 所有时间片的 vvv, 轴顺序为 t, eig1_idx, eig2_idx, eig3_idx.
    """
    px, py, pz = momentum
    combined_file = os.path.join(
        vvv_dir, f"vvv.Px{px}Py{py}Pz{pz}.conf{conf_id}"
    )
    if file_layout == "auto":
        file_layout = "combined" if os.path.isfile(
            combined_file) else "timeslices"

    if file_layout == "combined":
        # 文件按 real, imag, real, imag, ... 交错保存.
        raw_vvv = np.fromfile(combined_file, dtype="<f8")
        # 数据转成复数形式.
        raw_vvv = real_imag_to_complex(raw_vvv.reshape(-1, 2))

        # 每个时间片包含 Nev**3 个复数, 由此得到文件中的本征矢数目.
        num_eigvec_file = int(round((raw_vvv.size // Nt) ** (1.0 / 3.0)))
        vvv = raw_vvv.reshape(
            Nt,
            num_eigvec_file,
            num_eigvec_file,
            num_eigvec_file,
        )
        vvv = vvv[
            :, :num_eigvec, :num_eigvec, :num_eigvec
        ]
        file_to_print = combined_file
    elif file_layout == "timeslices":
        vvv = np.zeros(
            (Nt, num_eigvec, num_eigvec, num_eigvec),
            dtype=complex,
        )
        for time_index in range(Nt):
            timeslice_file = os.path.join(
                vvv_dir,
                f"vvv.t{time_index:03d}.Px{px}Py{py}Pz{pz}.conf{conf_id}",
            )
            vvv[time_index] = _read_vvv_timeslice_host(
                timeslice_file, num_eigvec
            )
        file_to_print = timeslice_file

    print(file_to_print)
    # para: t, eig1_idx, eig2_idx, eig3_idx
    return xp.asarray(vvv)


# ==================== Perambulator 读取与基变换 ====================
def read_peram_timeslice(
    perambulator_dir,
    conf_id,
    t_source,
    Nt,
    dt_length,
    num_eigvec,
    representation="DR",
    xp=cp,
):
    """
    读取固定 t_source 的 perambulator, 处理相对时间, 按需求返回 DR 或 DP 表象.

    说明: 路径输出已注释, 按时间片调用会重复输出路径, 如需路径输出请在外层循环外补上.

    参数:
    -----
    perambulator_dir: perambulator 文件所在目录, 文件名为 perams.{conf_id}.{spin_source}.{t_source}.
    conf_id: 规范场构型编号.
    t_source: 源时间片编号.
    Nt: 时间方向格点数.
    dt_length: dt 轴保留前 dt_length 个 dt, 即保留 dt=0, ..., dt_length-1.
    num_eigvec: eig_sink_idx 和 eig_source_idx 轴保留的长度.

    可选项:
    representation: 输出表象, 默认 "DR"; "DP" 表示在返回前做 DR -> DP 基变换.
    xp: 输出后端, 默认 cupy; 读取, 边界符号, roll 和截断始终在 CPU 上完成.

    输出:
    -----
    peram: perambulator, 轴顺序为 dt, spin_sink, spin_source, eig_sink_idx, eig_source_idx,
    其中 dt=(t_sink-t_source) % Nt.
    """
    # 四个文件分别固定一个 t_source 和一个 spin_source.
    raw_spin_data = []
    for spin_source in range(N_SPIN):
        peram_file = os.path.join(
            perambulator_dir, f"perams.{conf_id}.{spin_source}.{t_source}"
        )
        raw_spin_data.append(np.fromfile(peram_file, dtype="<f8"))
    # print(peram_file)  # 按时间片调用会重复输出路径, 已注释

    # 按 spin_source 拼接, 此时数据仍是 real, imag 交错的一维数组.
    raw_peram = np.concatenate(raw_spin_data)
    num_eigvec_file = int(
        np.sqrt(raw_peram.size / (N_SPIN * N_SPIN * Nt * 2))
    )

    # 文件原始轴: spin_source, t_sink, eig_source_idx, spin_sink, eig_sink_idx, real_imag.
    raw_peram = raw_peram.reshape(
        N_SPIN,
        Nt,
        num_eigvec_file,
        N_SPIN,
        num_eigvec_file,
        2,
    )

    peram = real_imag_to_complex(raw_peram)
    # 调整为: t_sink, spin_sink, spin_source, eig_sink_idx, eig_source_idx.
    peram = peram.transpose(1, 3, 0, 4, 2)

    # 原始 t_sink < t_source 的部分绕过时间边界, 反周期边界带来一个负号.
    boundary_sign = np.ones(Nt)
    boundary_sign[:t_source] = -1.0
    peram_DR = peram * boundary_sign[:, None, None, None, None]

    # 沿 t_sink 轴滚动 t_source, 使第 0 轴按 dt = t_sink - t_source 排列.
    peram_DR = np.roll(peram_DR, -t_source, axis=0)

    # 截断 dt、eig_sink_idx 和 eig_source_idx 轴.
    peram_DR = peram_DR[
        :dt_length, :, :, :num_eigvec, :num_eigvec
    ]

    peram = xp.asarray(peram_DR)

    # representation="DP" 时, 只变换 spin_sink 和 spin_source 两个轴.
    if representation == "DP":
        transform = xp.asarray(gamma_matrix_DP.s)
        transform_dagger = xp.conj(transform.T)
        peram = contract(
            "am,...mnij,nb->...abij",
            transform_dagger,
            peram,
            transform,
        )

    # para: dt = t_sink - t_source, spin_sink, spin_source, eig_sink_idx, eig_source_idx
    return peram
