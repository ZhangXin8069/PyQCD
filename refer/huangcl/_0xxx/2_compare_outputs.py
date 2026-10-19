#!/public/home/huangcl/.venv/bin/python
"""对比本地计算的 NP 2pt 关联函数和 gengyq 的参考结果.

输入:
- 本地: {local_root}/{conf_id}/NPcorr_conf{conf_id}_P0.npy, para: p_snk, p_src, dt, t_src, spin*4
- 参考: {ref_root}/NPcorr_spin_mom4-8_conf{conf_id}.npy, para: p_snk, p_src, t, spin*4

处理: 本地先对 t_src 轴求和, 本地和参考的时间轴都只取前 N_t 个时间片.
输出: 每个组态和全部组态整体的绝对差, 相对差的最大值和 RMS, 以及 0 和 nan 的位置对应情况.
"""

import os

import numpy as np

local_root = "/public/home/huangcl/00_data/00_NP_2pt/mom0/c32p29"
ref_root = "/public/home/gengyq/NP/result/spin/beta6.20_mu-0.2770_ms-0.2400_L32x64"
# 从 11000 开始, 隔 50 取一个, 共 260 个, 跳过 18350 和 23050
conf_ids = np.array(
    [c for c in range(11000, 11000 + 50 * 260, 50) if c not in (18350, 23050)])
N_t = 24


def load_pair(conf_id):
    """
    读取一个组态的本地和参考关联函数, 时间轴都只取前 N_t 个时间片.

    参数:
    -----
    conf_id: 组态编号, 如 11000.

    输出:
    -----
    local, ref: 已截断的两组数据, para: p_snk, p_src, t, spin*4.
    local_file, ref_file: 实际读取的文件路径.
    """
    local_file = os.path.join(
        local_root, str(conf_id), f"NPcorr_conf{conf_id}_P0.npy")
    # para: p_snk, p_src, dt, t_src, spin*4
    local_all = np.load(local_file, allow_pickle=False)
    local = local_all.sum(axis=3)[:, :, :N_t]

    ref_file = os.path.join(ref_root, f"NPcorr_spin_mom4-8_conf{conf_id}.npy")
    # para: p_snk, p_src, t, spin*4
    ref = np.load(ref_file, allow_pickle=False)[:, :, :N_t]
    return local, ref, local_file, ref_file


def calc_stats(local, ref):
    """
    计算绝对差和相对差的最大值与 RMS.

    参数:
    -----
    local, ref: 形状相同的复数数组.

    输出:
    -----
    stats: max_abs_diff, rms_abs_diff, max_rel_diff, rms_rel_diff, 相对差只统计 ref 不为 0 的元素.
    """
    abs_diff = np.abs(local - ref).ravel()
    ref_abs = np.abs(ref).ravel()
    rel_diff = abs_diff[ref_abs > 0] / ref_abs[ref_abs > 0]
    return (
        np.max(abs_diff),
        np.sqrt(np.mean(abs_diff ** 2)),
        np.max(rel_diff),
        np.sqrt(np.mean(rel_diff ** 2)),
    )


def print_stats(tag, stats):
    """打印一行绝对差和相对差的最大值与 RMS."""
    print(f"{tag:>10} max_abs={stats[0]:.3g} rms_abs={stats[1]:.3g} "
          f"max_rel={stats[2]:.3g} rms_rel={stats[3]:.3g}")


def print_zero_nan(tag, local, ref):
    """打印 local 和 ref 中 0 和 nan 的个数, 以及位置是否逐个对应."""
    zero_local, zero_ref = local == 0, ref == 0
    nan_local, nan_ref = np.isnan(local), np.isnan(ref)
    print(f"{tag:>10} zero {np.count_nonzero(zero_local)}/{np.count_nonzero(zero_ref)} "
          f"mismatch={np.count_nonzero(zero_local != zero_ref)} "
          f"match={np.array_equal(zero_local, zero_ref)} "
          f"nan {np.count_nonzero(nan_local)}/{np.count_nonzero(nan_ref)} "
          f"mismatch={np.count_nonzero(nan_local != nan_ref)} "
          f"match={np.array_equal(nan_local, nan_ref)}")


if __name__ == "__main__":
    print(f"local_root = {local_root}")
    print(f"ref_root = {ref_root}, N_t = {N_t}")

    local_sum, ref_sum = [], []
    for conf_id in conf_ids:
        ref_file = os.path.join(
            ref_root, f"NPcorr_spin_mom4-8_conf{conf_id}.npy")
        if not os.path.exists(ref_file):
            print(f"conf {conf_id}: ref  data not exist, skipped")
            continue
        local, ref, local_file, ref_file = load_pair(conf_id)
        print_stats(f"conf {conf_id}", calc_stats(local, ref))
        print_zero_nan(f"conf {conf_id}", local, ref)
        local_sum.append(local)
        ref_sum.append(ref)

    print(f"last files: {local_file}, {ref_file}")
    print()
    local_all = np.concatenate(local_sum)
    ref_all = np.concatenate(ref_sum)
    print_stats("overall", calc_stats(local_all, ref_all))
    print_zero_nan("overall", local_all, ref_all)
