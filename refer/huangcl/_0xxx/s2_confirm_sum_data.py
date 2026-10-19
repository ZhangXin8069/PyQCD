#!/public/home/huangcl/.venv/bin/python
# 功能: 对比本地合成的胶子非极化圈算符和参考的预组合数据.
#
# 输入: 下面两个完整文件路径 (含文件名), 直接在 py 里改.
# - 本地: ops_dz{length}_conf{conf_id}.npy, para: t, length.
# - 参考: ops_dz{length}_conf{conf_id}.npz, 其中 key ops 的 para: length, t.
#
# 处理: 参考数组转置到 para: t, length, 再按 N_t 截断时间 dt, 按 N_dz 截断空间 dz, 最后比较.
#
# 输出: 形状, dtype, 有限值检查, np.allclose 结果,
# 绝对差和相对差 (只统计参考不为 0 的元素) 的最大值和 RMS,
# 以及 0 和 nan 的个数和位置对应情况.

from pathlib import Path

import numpy as np

# ==================== 配置适配开始 ====================
# 本地合成文件 (含文件名), 直接复制路径到这里.
local_file = (
    "/public/home/huangcl/06_deuteron_gluon_pdf/03_gluon_unpol/02_result/c32p29/11000/zdir/ops_dz32_conf11000.npy"
)

# 参考合成文件 (含文件名), 直接复制路径到这里.
ref_file = (
    "/public/group/imp/zengch/LQCD/gluon_operator/output/L32x64_C32P29/zdir/11000/ops_dz30_conf11000.npz"
    # "/public/group/imp/zengch/LQCD/gluon_operator/output/L32x64_C32P29/zdir/11000/ops_dz24_conf11000.npz"
    # "/public/home/chenc/gluon_unpo/generating_data/operator/unfixed_operator_sum/beta6.20_mu-0.2770_ms-0.2400_L32x64/sum/ops_sum_dz30_conf11000.npz"
)

# 只取前 N_t 个时间片和前 N_dz 个位移, None 表示不截断.
N_t = None
N_dz = 30
# ==================== 配置适配结束 ====================


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
    print(f"local_file = {local_file}")
    print(f"ref_file = {ref_file}")
    print(f"N_t = {N_t}, N_dz = {N_dz}")

    if not Path(local_file).exists() or not Path(ref_file).exists():
        print(f"file not exist: {local_file} / {ref_file}")
        raise SystemExit

    # para: t, length
    local = np.load(local_file, allow_pickle=False)
    # 参考文件里 ops 的 para: length, t
    ref = np.load(ref_file, allow_pickle=False)["ops"].T
    # 时间 dt 和空间 dz 截断
    local, ref = local[:N_t, :N_dz], ref[:N_t, :N_dz]
    if local.shape != ref.shape:
        print(f"shape mismatch {local.shape}/{ref.shape}")
        raise SystemExit

    tag = Path(local_file).stem
    print_info(tag, local, ref)
    print_stats(tag, calc_stats(local, ref))
    print_zero_nan(tag, local, ref)
