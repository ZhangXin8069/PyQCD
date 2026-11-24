"""双核子 C gamma 通道投影工具.

主要功能:
    load_NPcorr: 按组态, 动量和方向读取 cg5/cg5gt 的 NP 2pt 数据.
    load_gluon_unpol: 读取指定方向非极化胶子 PDF 的算符组合数据.
    proj_deuteron_chan: 投影 I=0, S=1 的三条氘核自旋通道.
    proj_dinucleon_chan: 投影 I=1, S=0 的双核子通道.
    proj_deuteron_chan_std: 用标准方法投影三条氘核自旋通道.
    proj_dinucleon_chan_std: 用标准方法投影双核子通道.

参考:
    ../06_NPcorr_cg5/00_code/contract_str.py
        关联函数末尾四轴 r,q,z,y 分别表示:
        r: sink 端质子的旋量指标.
        q: source 端质子的旋量指标.
        z: sink 端中子的旋量指标.
        y: source 端中子的旋量指标.

    双核子算符推导.md
        双核子算符的构建, 式 (1) 给出归一化算符.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "98_tools"))
from distillation_tools import MOM_SMEAR_SIZE  # noqa: E402  # type: ignore
from gamma_matrix import gamma  # noqa: E402  # type: ignore
from snsc_params import get_conf, get_conf_ids  # noqa: E402  # type: ignore


NP_2PT_ROOT = Path("/public/home/huangcl/00_data/00_NP_2pt")


def _project_channel(
        correlators: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """
    用单个 C gamma 投影矩阵计算关联函数.

    输入:
    -----
    correlators: 关联函数数组, para: <others>, r, q, z, y.
    projector: 单个 C gamma 投影矩阵, para: r, z, 完整的 4 x 4 矩阵.

    输出:
    -----
    projected: 单个通道的投影结果, para: <others>.
    """
    g4 = gamma(4, "DP")
    proj_bar = g4 @ projector.conj().T @ g4.T
    # 旋量只保留上半分量, 收缩前截断到 2 x 2
    return np.einsum(
        "...rqzy,rz,yq->...",
        correlators,
        projector[:2, :2],
        proj_bar[:2, :2],
        optimize=True,
    )


def proj_deuteron_chan(correlators: np.ndarray) -> np.ndarray:
    """
    投影 I=0, S=1 的三条氘核自旋通道.
        S_z=+1: (C gamma_1 - i C gamma_2) / sqrt(2)
        S_z= 0: C gamma_3
        S_z=-1: -(C gamma_1 + i C gamma_2) / sqrt(2)

    输入:
    -----
    correlators: 关联函数数组, para: <others>, r, q, z, y.

    输出:
    -----
    projected: 投影结果, para: <others>, sz; sz 依次为 -1, 0, +1.
    """
    charge_conjugation = gamma(6, "DP")
    c_gamma_1 = charge_conjugation @ gamma(1, "DP")
    c_gamma_2 = charge_conjugation @ gamma(2, "DP")
    c_gamma_3 = charge_conjugation @ gamma(3, "DP")
    projectors = (
        -(c_gamma_1 + 1j * c_gamma_2) / np.sqrt(2.0),
        c_gamma_3,
        (c_gamma_1 - 1j * c_gamma_2) / np.sqrt(2.0),
    )
    channel_list = [
        _project_channel(correlators, projector)
        for projector in projectors
    ]
    # para: <others>, sz
    return np.stack(channel_list, axis=-1)


def proj_dinucleon_chan(correlators: np.ndarray) -> np.ndarray:
    """
    投影 I=1, S=0 的双核子通道.
        I_z=0: C gamma_5

    输入:
    -----
    correlators: 关联函数数组, para: <others>, r, q, z, y.

    输出:
    -----
    projected: 单个通道的投影结果, para: <others>.
    """
    projector = gamma(7, "DP")
    return _project_channel(correlators, projector)


def proj_deuteron_chan_std(correlators: np.ndarray) -> np.ndarray:
    """
    用标准方法投影 I=0, S=1 的三条氘核自旋通道.

    参数:
    -----
    correlators: 关联函数数组, para: <others>, r, q, z, y.

    输出:
    -----
    projected: 投影结果, para: <others>, sz; sz 依次为 -1, 0, +1.
    """
    # I = 0 投影: sqrt(2) O = p_s1 n_s2 - n_s1 p_s2 = p_s1 n_s2 + p_s2 n_s1
    # 已有算符是 pn 的形式, 无法直接构建出 np 形式. 但根据上式, 可通过交换自旋方向间接实现投影
    asym_isospin = (
        correlators
        + np.swapaxes(correlators, -3, -1)
        + np.swapaxes(correlators, -4, -2)
        + np.swapaxes(np.swapaxes(correlators, -4, -2), -3, -1)
    ) / 2.0

    # 固定 s1 和 s2(对称), 从而固定算符量子数 sz
    sz0 = (
        asym_isospin[..., 0, 0, 1, 1]
        + asym_isospin[..., 0, 1, 1, 0]
        + asym_isospin[..., 1, 1, 0, 0]
        + asym_isospin[..., 1, 0, 0, 1]
    ) / 2.0
    sz_plus1 = asym_isospin[..., 0, 0, 0, 0]
    sz_minus1 = asym_isospin[..., 1, 1, 1, 1]
    # para: <others>, sz
    return np.stack((sz_minus1, sz0, sz_plus1), axis=-1)


def proj_dinucleon_chan_std(correlators: np.ndarray) -> np.ndarray:
    """
    用标准方法投影 I=1, S=0 的双核子通道.

    参数:
    -----
    correlators: 关联函数数组, para: <others>, r, q, z, y.

    输出:
    -----
    projected: 单个通道的投影结果, para: <others>.
    """
    # I = 1 投影: sqrt(2) O = p_s1 n_s2 + n_s1 p_s2 = p_s1 n_s2 - p_s2 n_s1
    # 已有算符是 pn 的形式, 无法直接构建出 np 形式. 但根据上式, 可通过交换自旋方向间接实现投影
    sym_isospin = (
        correlators
        - np.swapaxes(correlators, -3, -1)
        - np.swapaxes(correlators, -4, -2)
        + np.swapaxes(np.swapaxes(correlators, -4, -2), -3, -1)
    ) / 2.0

    # 固定 s1 和 s2(反对称), 从而固定算符量子数 sz
    return (
        sym_isospin[..., 0, 0, 1, 1]
        - sym_isospin[..., 0, 1, 1, 0]
        + sym_isospin[..., 1, 1, 0, 0]
        - sym_isospin[..., 1, 0, 0, 1]
    ) / 2.0


def load_NPcorr(
        conf_short: str, conf_num: int, momentum: int,
        mom_dir: str, exclude_ids: list = ()) -> np.ndarray:
    """
    按组态, 动量和方向读取 cg5/cg5gt 的 NP 2pt 数据.

    参数:
    -----
    conf_short: 组态短标识, 如 c24p29.
    conf_num: 使用的组态数, 为剔除后实际使用的数目.
    momentum: 体系总动量大小, 取正值. 零动量读取 cg5,
              非零动量读取 cg5gt.
    mom_dir: 总动量方向, 取 "x" 或 "-x" 等, 符号表示强子动量方向.

    可选项:
    -----
    exclude_ids: 需要剔除的 conf_id 集合.

    输出:
    -----
    data: 关联函数数组, para: conf, dt, t_src, r, q, z, y.
    """
    print("\n==================== load NPcorr start ====================")
    print(f"momentum = {momentum}, mom_dir = {mom_dir}")

    conf_ids = get_conf_ids(conf_short, conf_num, ids_exclude=exclude_ids)

    if momentum == 0:
        data_dir = NP_2PT_ROOT / "cg5" / conf_short
        momentum_vec = [0, 0, 0]
    else:
        axis_names = ("x", "y", "z")
        axis = mom_dir[-1]
        dir_sign = mom_dir[:-1]
        mom_sign = -1 if dir_sign == "-" else 1
        momentum_vec = [0, 0, 0]
        momentum_vec[axis_names.index(axis)] = mom_sign * momentum
        smear_size = MOM_SMEAR_SIZE[conf_short]
        result_subdir = f"momsmear{dir_sign}{smear_size}{axis}"
        data_dir = NP_2PT_ROOT / "cg5gt" / result_subdir / conf_short

    data_list = []
    for conf_id in conf_ids:
        path = data_dir / str(conf_id) / (
            f"NPcorr_Px{momentum_vec[0]}Py{momentum_vec[1]}"
            f"Pz{momentum_vec[2]}_conf{conf_id}.npy"
        )
        # para: dt, t_src, r, q, z, y
        data_list.append(np.load(path))

    print("data load finish")
    print(f"example path: {path}")

    # para: conf, dt, t_src, r, q, z, y
    data = np.asarray(data_list, dtype=np.complex128)
    print(f"data shape: {data.shape}")

    print("==================== load NPcorr end ====================\n")
    return data


def load_gluon_unpol(
        conf_short: str, conf_num: int, dir: str,
        dl: int, exclude_ids: list = ()) -> np.ndarray:
    """
    读取指定方向非极化胶子 PDF 的算符组合数据.

    参数:
    -----
    conf_short: 组态短标识, 如 c32p29.
    conf_num: 使用的组态数.
    dir: 方向, 取 "x"/"-x"/"y"/"-y"/"z"/"-z".
        负方向与正方向读取同一份数据, 即 -x 等同 x.
    dl: nonlocal 算符拉开的距离.

    可选项:
    -----
    exclude_ids: 需要剔除的 conf_id 集合.

    输出:
    -----
    data: 算符数据, para: conf, tau, z.
    """
    print("\n==================== load gluon unpol start ====================")
    print(f"dir = {dir}")

    conf_name = get_conf(conf_short)["conf_name"]
    conf_ids = get_conf_ids(conf_short, conf_num, ids_exclude=exclude_ids)

    dir_by_axis = {
        "x": (
            "/public/home/chenc/gluon_unpo/generating_data/operator/"
            f"unfixed_operator_sum_xdir/{conf_name}/sum/"
        ),
        "y": (
            "/public/home/chenc/gluon_unpo/generating_data/operator/"
            f"unfixed_operator_sum_ydir/{conf_name}/sum/"
        ),
        "z": (
            "/public/home/chenc/gluon_unpo/generating_data/operator/"
            f"unfixed_operator_sum/{conf_name}/sum/"
        ),
    }
    # 负方向与正方向共用同一份算符数据
    axis = dir.lstrip("-")
    if axis not in dir_by_axis:
        print(f"unknown dir: {dir}")
        sys.exit(1)
    data_dir = dir_by_axis[axis]

    data_list = []
    for conf_id in conf_ids:
        path = Path(data_dir) / f"ops_sum_dz{dl}_conf{conf_id}.npz"
        # para: z, tau
        data_list.append(np.load(path)["ops"])

    print("data load finish")
    print(f"example path: {path}")

    # para: conf, z, tau
    data = np.asarray(data_list)
    # para: conf, tau, z
    data = data.transpose(0, 2, 1)
    print(f"data shape: {data.shape}")

    print("==================== load gluon unpol end ====================\n")
    return data
