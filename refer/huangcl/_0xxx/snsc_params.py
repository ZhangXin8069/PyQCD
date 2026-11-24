#!/usr/bin/env python3
"""snsc 组态参数查询和组态编号生成模块."""

import sys

# 组态元数据来源为 ~/99_config/snsc.md.
# None 表示该参数未配置.
CONFIGS = {
    "c24p34": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 64,
        "Nx": 24,
        "lattice_spacing": 0.1053,
    },
    "c24p29": {
        "conf_name": "beta6.20_mu-0.2770_ms-0.2400_L24x72",
        "conf_id_first": 4050,
        "conf_id_step": 50,
        "conf_id_num": 880,
        "Nt": 72,
        "Nx": 24,
        "lattice_spacing": 0.1053,
    },
    "c32p29": {
        "conf_name": "beta6.20_mu-0.2770_ms-0.2400_L32x64",
        "conf_id_first": 11000,
        "conf_id_step": 50,
        "conf_id_num": 1000,
        "Nt": 64,
        "Nx": 32,
        "lattice_spacing": 0.1053,
    },
    "c24p23": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 64,
        "Nx": 24,
        "lattice_spacing": 0.1053,
    },
    "c32p23": {
        "conf_name": "beta6.20_mu-0.2790_ms-0.2400_L32x64",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 64,
        "Nx": 32,
        "lattice_spacing": 0.1053,
    },
    "c48p23": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 96,
        "Nx": 48,
        "lattice_spacing": 0.1053,
    },
    "c48p14": {
        "conf_name": "beta6.20_mu-0.2825_ms-0.2310_L48x96",
        "conf_id_first": 2000,
        "conf_id_step": 20,
        "conf_id_num": 300,
        "Nt": 96,
        "Nx": 48,
        "lattice_spacing": 0.1053,
    },
    "c64p14": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 128,
        "Nx": 64,
        "lattice_spacing": 0.1053,
    },
    "e32p29": {
        "conf_name": "beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 64,
        "Nx": 32,
        "lattice_spacing": 0.0897,
    },
    "e32p22": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 64,
        "Nx": 32,
        "lattice_spacing": 0.0897,
    },
    "f32p30": {
        "conf_name": "beta6.41_mu-0.2295_ms-0.2050_L32x96",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 96,
        "Nx": 32,
        "lattice_spacing": 0.07753,
    },
    "f48p30": {
        "conf_name": "beta6.41_mu-0.2295_ms-0.2050_L48x96",
        "conf_id_first": 2700,
        "conf_id_step": 20,
        "conf_id_num": 360,
        "Nt": 96,
        "Nx": 48,
        "lattice_spacing": 0.07753,
    },
    "f32p21": {
        "conf_name": "beta6.41_mu-0.2320_ms-0.2050_L32x64",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 64,
        "Nx": 32,
        "lattice_spacing": 0.07753,
    },
    "f48p21": {
        "conf_name": "beta6.41_mu-0.2320_ms-0.2050_L48x96",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 96,
        "Nx": 48,
        "lattice_spacing": 0.07753,
    },
    "f64p13": {
        "conf_name": "beta6.41_mu-0.2336_ms-0.2030_L64x128",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 128,
        "Nx": 64,
        "lattice_spacing": 0.07753,
    },
    "f64p14": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 128,
        "Nx": 64,
        "lattice_spacing": 0.07753,
    },
    "g36p29": {
        "conf_name": "beta6.498_mu-0.2150_ms-1926_L36x108",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 108,
        "Nx": 36,
        "lattice_spacing": 0.06887,
    },
    "g32p35": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 96,
        "Nx": 32,
        "lattice_spacing": 0.06887,
    },
    "h48p32": {
        "conf_name": "beta6.72_mu-0.1850_ms-0.1700_L48x144",
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 144,
        "Nx": 48,
        "lattice_spacing": 0.05199,
    },
    "i64p30": {
        "conf_name": None,
        "conf_id_first": None,
        "conf_id_step": None,
        "conf_id_num": None,
        "Nt": 128,
        "Nx": 64,
        "lattice_spacing": 0.03761,
    },
}


def get_conf(conf_short):
    """
    获取组态元数据.

    参数:
    -----
    conf_short: 组态短名.

    输出:
    -----
    conf: 包含 conf_name, conf_id 范围, Nt, Nx 和 lattice_spacing.
    """

    if conf_short not in CONFIGS:
        raise ValueError(f"unknown conf_short: {conf_short}")

    return dict(CONFIGS[conf_short])


def get_conf_ids(conf_short, id_num, ids_exclude=None, id_start=None):
    """
    生成使用的组态编号: 先按步长生成全部编号, 再剔除 ids_exclude, 最后从 id_start 起取 id_num 个.

    参数:
    -----
    conf_short: 组态短名.
    id_num: 使用的组态数.

    可选项:
    -----
    ids_exclude: 需要剔除的 conf_id 列表.
    id_start: 选取起点, 必须位于剔除后的编号列表中; 默认为第一个编号.

    输出:
    -----
    conf_ids: 使用的组态编号列表, 长度为 id_num.
    """

    conf = get_conf(conf_short)
    conf_id_first = conf["conf_id_first"]
    conf_id_step = conf["conf_id_step"]
    conf_id_num = conf["conf_id_num"]

    if conf_id_first is None or conf_id_step is None or conf_id_num is None:
        raise ValueError(f"incomplete parameters: {conf_short}")

    # 先生成全部组态编号, 再剔除指定编号.
    all_ids = [
        conf_id_first + index * conf_id_step for index in range(conf_id_num)
    ]
    ids_exclude = [] if ids_exclude is None else list(ids_exclude)
    conf_ids = [conf_id for conf_id in all_ids if conf_id not in ids_exclude]

    # 从 id_start 开始选取 id_num 个编号.
    if id_start is None:
        index_start = 0
    elif id_start in conf_ids:
        index_start = conf_ids.index(id_start)
    else:
        raise ValueError(f"invalid id_start: {id_start}")

    # 可用长度已扣除剔除的编号和 id_start 之前的编号, 不足 id_num 时报错.
    id_num_avail = len(conf_ids) - index_start
    if id_num > id_num_avail:
        raise ValueError(
            f"id_num = {id_num} exceeds available conf number "
            f"{id_num_avail} of {conf_short}: "
            f"id_start = {conf_ids[index_start]}, "
            f"exclude {len(ids_exclude)} ids"
        )

    return conf_ids[index_start:index_start + id_num]


def main():
    """命令行入口: 每个 conf_id 输出一行."""

    args = sys.argv[1:]
    if len(args) not in (2, 3):
        print(
            "usage: snsc_params.py CONF_SHORT ID_NUM [ID_START]",
            file=sys.stderr,
        )
        return 1

    try:
        id_start = int(args[2]) if len(args) == 3 else None
        conf_ids = get_conf_ids(args[0], int(args[1]), id_start=id_start)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1

    for conf_id in conf_ids:
        print(conf_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
