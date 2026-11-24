"""pion 2pt 与胶子算符读取工具.

主要功能:
    load_pion_2pt: 按组态读取 shenxm 的 pion 2pt 数据, 并按周期边界重排为 (dt, t_src).
    load_gluon_unpol_shenxm: 按组态读取 shenxm 的胶子流算符, 对横向两轴求平均.
    load_gluon_unpol_chenc: 读取 chenc 的指定方向非极化胶子 PDF 算符数据.

参考:
    本目录的 get_channel_2pt.py
        本文件由该文件复制后改写, chenc 胶子算符读取的路径和轴约定保持不变.
"""

import sys
from pathlib import Path

import numpy as np

# ---- 从项目目录 98_tools 导入通用函数 ----
sys.path.append(str("/public/home/huangcl/07_pion_gluon_pdf/98_tools"))
from distillation_tools import MOM_SMEAR_SIZE  # noqa: E402  # type: ignore
from snsc_params import get_conf, get_conf_ids  # noqa: E402  # type: ignore


# pion 2pt 根目录
PION_ROOT = Path("/nexdata/project/imp/shenxm/LQCD/pion-2pt")

# 胶子流算符根目录, 各组态存放位置不一致, 按 conf_short 逐个指定
OG_DIR = {
    "c24p29": Path("/public/home/shenxm/share"),
    "c32p29": Path("/public/home/shenxm/share"),
}


def load_pion_2pt(
        conf_short: str, conf_num: int, momentum: int, direction: str,
        exclude_ids: list = ()) -> np.ndarray:
    """
    按组态读取 shenxm 的 pion 2pt 数据, 并按周期边界重排为相对时间表示.

    sink 与 source 动量相同, 且只有 direction 一个方向带非零动量.
    零动量读取未 smearing 的 G5, 带动量读取单次 smearing 的 G5Gt,
    smearing 幅度由 98_tools 的 MOM_SMEAR_SIZE 给出.

    参数:
    -----
    conf_short: 组态短标识, 如 c24p29.
    conf_num: 使用的组态数.
    momentum: 非零方向的动量大小, 取非负整数.
    direction: 带非零动量的方向, 取 "x"/"y"/"z";
              momentum = 0 时该参数不起作用, 可传任意值.

    可选项:
    -----
    exclude_ids: 需要剔除的 conf_id 集合.

    输出:
    -----
    data: 关联函数数组, para: conf, dt, t_src;
          dt = (t_sink - t_src) mod Nt.
    """
    print("\n==================== load pion 2pt start ====================")

    conf_name = get_conf(conf_short)["conf_name"]
    conf_ids = get_conf_ids(conf_short, conf_num, ids_exclude=exclude_ids)

    p_mom = [0, 0, 0]
    if momentum == 0:
        # 零动量: 未 smearing 的 G5
        operator = "G5"
    else:
        # 带动量: 单次 smearing 的 G5Gt, smearing 分量沿 direction
        axis_index = "xyz".index(direction)
        p_mom[axis_index] = momentum
        smear_mom = [0, 0, 0]
        smear_mom[axis_index] = -MOM_SMEAR_SIZE[conf_short]
        operator = (
            f"G5Gt_1smear{smear_mom[0]}.{smear_mom[1]}.{smear_mom[2]}"
        )
    mom_label = f"{p_mom[0]}.{p_mom[1]}.{p_mom[2]}"
    mom_dir = f"Psink{mom_label}.Psrc{mom_label}"

    data_list = []
    for conf_id in conf_ids:
        path = PION_ROOT / conf_name / mom_dir / (
            f"Pi_2pt_{operator}_{mom_dir}.conf{conf_id}.npy"
        )
        # para: t_sink, t_src
        data_list.append(np.load(path, allow_pickle=False))

    print("data load finish")
    print(f"example path: {path}")

    # para: conf, t_sink, t_src
    data_matrix = np.asarray(data_list, dtype=np.complex128)
    nt = data_matrix.shape[1]
    # para: t_src
    t_src_index = np.arange(nt)

    # 按周期边界把 t_sink 换成相对时间 dt, t_sink = (t_src + dt) % nt
    # para: conf, dt, t_src
    data = np.zeros_like(data_matrix)
    for dt in range(nt):
        # para: conf, t_src
        data[:, dt, :] = data_matrix[:, (t_src_index + dt) % nt, t_src_index]
    print(f"data shape: {data.shape}")

    print("==================== load pion 2pt end ====================\n")
    return data


def load_gluon_unpol_shenxm(
        conf_short: str, conf_num: int, dir: str, space_sep_max: int,
        exclude_ids: list = ()) -> np.ndarray:
    """
    按组态读取 shenxm 的胶子流算符, 并对横向两轴求平均.

    参数:
    -----
    conf_short: 组态短标识, 如 c32p29.
    conf_num: 使用的组态数.
    dir: 分离方向, 取 "x"/"y"/"z".
    space_sep_max: 最大分离长度, 文件名中记为 all{space_sep_max + 1}.

    可选项:
    -----
    exclude_ids: 需要剔除的 conf_id 集合.

    输出:
    -----
    data: 算符数据, para: conf, t_current, dz.
    """
    print("\n==================== load gluon unpol start ====================")

    conf_name = get_conf(conf_short)["conf_name"]
    conf_ids = get_conf_ids(conf_short, conf_num, ids_exclude=exclude_ids)

    data_list = []
    for conf_id in conf_ids:
        path = OG_DIR[conf_short] / conf_name / (
            f"Ogmp_d{dir}all{space_sep_max + 1}_conf{conf_id}.npy"
        )
        # para: dz, t_current, transverse_1, transverse_2
        og = np.load(path, allow_pickle=False)
        # 对横向两轴求平均, para: dz, t_current
        data_list.append(-og.sum(axis=(2, 3)))

    print("data load finish")
    print(f"example path: {path}")

    # para: conf, dz, t_current
    data = np.asarray(data_list, dtype=np.complex128)
    # para: conf, t_current, dz
    data = data.transpose(0, 2, 1)
    print(f"data shape: {data.shape}")

    print("==================== load gluon unpol end ====================\n")
    return data


def load_gluon_unpol_chenc(
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
