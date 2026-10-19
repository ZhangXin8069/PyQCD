#!/public/home/huangcl/.venv/bin/python
# 功能: 对比本地计算的胶子圈算符单平面结果和参考结果.
#
# 输入: 命令行参数 conf_short, conf_id, dir, 以及可选的 N_t, N_dz
# (只比较前 N_t 个时间片和前 N_dz 个位移, 不写表示不截断).
# - 本地: 01_result/{conf_short}/{conf_id}/{dir}dir/ops_mu*_nu*_dz*_conf*.npy, para: t, length.
# - 参考: ref_root/同名 .npz, 其中 key ops 的 para: length, t.
#
# 处理: 参考数组转置到 para: t, length, 再按 N_t 截断时间 dt, 按 N_dz 截断空间 dz, 最后比较.
#
# 输出: 每个文件和整体的形状, dtype, 有限值检查, np.allclose 结果,
# 绝对差和相对差 (只统计参考不为 0 的元素) 的最大值和 RMS,
# 以及 0 和 nan 的个数和位置对应情况.

import sys
from pathlib import Path

import numpy as np

# ==================== 配置适配开始 ====================
conf_short = sys.argv[1]
conf_id = int(sys.argv[2])
dir = sys.argv[3]
# 只取前 N_t 个时间片和前 N_dz 个位移, 不写表示不截断.
N_t = int(sys.argv[4]) if len(sys.argv) > 4 else None
N_dz = int(sys.argv[5]) if len(sys.argv) > 5 else None

# 本地结果目录, 相对本文件夹.
local_root = (
    Path(__file__).resolve().parents[0]
    / "02_result" / conf_short / str(conf_id) / f"{dir}dir"
)

# 参考路径: 别人给的数据, 直接复制完整路径到这里.
ref_root = Path(
    "/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/L24x72/zdir/6200"
)
# ==================== 配置适配结束 ====================


def find_local_files():
    """
    列出本地单平面结果文件.

    输出:
    -----
    local_files: 本地 ops_mu*_nu*_dz*_conf*.npy 路径, 按文件名排序.
    """
    return sorted(local_root.glob("ops_mu*_nu*_dz*_conf*.npy"))


def calc_stats(local, ref):
    """
    计算绝对差和相对差的最大值和 RMS.

    参数:
    -----
    local, ref: 形状相同的复数数组, para: t, length.

    输出:
    -----
    stats: max_abs, rms_abs, max_rel, rms_rel, 相对差只统计 ref 不为 0 的元素.
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


def print_info(tag, local, ref):
    """打印形状, dtype, 有限值检查和 np.allclose 结果."""
    print(f"{tag:>10} shape {local.shape}/{ref.shape} dtype {local.dtype}/{ref.dtype} "
          f"finite {np.isfinite(local).all()}/{np.isfinite(ref).all()} "
          f"allclose={np.allclose(local, ref)}")


def print_stats(tag, stats):
    """打印一行绝对差和相对差的最大值和 RMS."""
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
    print(f"conf_short = {conf_short}, conf_id = {conf_id}, dir = {dir}, "
          f"N_t = {N_t}, N_dz = {N_dz}")
    print(f"local_root = {local_root}")
    print(f"ref_root = {ref_root}")

    local_files = find_local_files()
    if not local_files:
        print(f"no local file: {local_root}")
        raise SystemExit

    local_sum, ref_sum = [], []
    for local_file in local_files:
        ref_file = ref_root / f"{local_file.stem}.npz"
        if not ref_file.exists():
            print(f"{local_file.stem}: ref data not exist, skipped")
            continue

        # para: t, length; 参考文件里 ops 的 para: length, t
        local = np.load(local_file, allow_pickle=False)
        ref = np.load(ref_file, allow_pickle=False)["ops"].T
        # 时间 dt 和空间 dz 截断
        local, ref = local[:N_t, :N_dz], ref[:N_t, :N_dz]
        if local.shape != ref.shape:
            print(
                f"{local_file.stem}: shape mismatch {local.shape}/{ref.shape}, skipped")
            continue

        print_info(local_file.stem, local, ref)
        print_stats(local_file.stem, calc_stats(local, ref))
        print_zero_nan(local_file.stem, local, ref)
        local_sum.append(local)
        ref_sum.append(ref)

    print(f"last files: {local_file}, {ref_file}")
    print()
    if local_sum:
        local_all = np.concatenate(local_sum)
        ref_all = np.concatenate(ref_sum)
        print_info("overall", local_all, ref_all)
        print_stats("overall", calc_stats(local_all, ref_all))
        print_zero_nan("overall", local_all, ref_all)
