#!/usr/bin/env python3
"""snsc 组态参数查询模块."""

import sys

# 组态元数据来源为 ~/99_config/snsc.md.
# 无参数的位置 (None) 输出为 "-"
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
        "conf_id_num": 879,
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
        "conf_name": None,
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

CONF_FIELDS = (
    "conf_name",
    "conf_id_first",
    "conf_id_step",
    "conf_id_num",
    "Nt",
    "Nx",
    "lattice_spacing",
)


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
        raise KeyError(f"unknown conf_short: {conf_short}")

    return dict(CONFIGS[conf_short])


def format_conf_line(conf_short):
    """把组态元数据格式化为 Bash 使用的 8 列文本."""

    conf = get_conf(conf_short)
    values = []
    for key in CONF_FIELDS:
        value = conf[key]
        values.append("-" if value is None else str(value))

    return " ".join((conf_short, *values))


def main():
    """命令行入口: 输入 conf_short 并输出一行组态参数."""

    try:
        conf_short = sys.argv[1]
        print(format_conf_line(conf_short))
    except IndexError:
        print("usage: snsc_params.py CONF_SHORT")
        return 1
    except KeyError:
        print(f"unknown conf_short: {conf_short}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
