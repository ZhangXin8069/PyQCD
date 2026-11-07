"""显存/内存预算守卫：L=24 空间格点下各组合是否安全。

只做算术与断言，**不分配**这些大张量。

判定规则（显式、可复核）：
    usable  = 可用显存 − 预留(默认 1 GiB：CUDA 上下文 + 常驻进程)
    ratio   = 峰值 / usable
    ratio ≤ 0.80            → SAFE
    0.80 < ratio ≤ 1.00     → MARGINAL（可行但不建议默认使用）
    ratio > 1.00            → UNSAFE（禁止）

峰值口径：E4 需要两个 prop + 一个同量级中间结果；E5 需要 prop + 共轭副本 + 输出。

用法：
    python -m pyqcd.testing.einsum.simple.check_memory
    python -m pyqcd.testing.einsum.simple.check_memory --bytes 8589934592
"""

from __future__ import annotations

import argparse

from . import _common

BYTES_PER_ELEMENT = 16  # complex128
GPU_BUDGET = 8 * 1024 ** 3
HOST_BUDGET = 12 * 1024 ** 3
DEFAULT_RESERVED = 1 * 1024 ** 3
SAFE_RATIO = 0.80


def prop_elements(spatial, n_t, nvec):
    return spatial ** 3 * 3 * n_t * 4 * 4 * nvec


def classify(peak, budget, reserved):
    usable = max(budget - reserved, 1)
    ratio = peak / usable
    if ratio <= SAFE_RATIO:
        return 'SAFE', ratio, usable
    if ratio <= 1.0:
        return 'MARGINAL', ratio, usable
    return 'UNSAFE', ratio, usable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bytes', type=int, default=GPU_BUDGET,
                        help='可用显存字节数（默认 8 GiB）')
    parser.add_argument('--host-bytes', type=int, default=HOST_BUDGET)
    parser.add_argument('--reserved-bytes', type=int, default=DEFAULT_RESERVED)
    args = parser.parse_args(argv)

    tally = _common.Tally('memory')
    results = {}
    for spatial, nvec, label in ((24, 50, 'L24'), (24, 24, 'L24n'),
                                 (24, 100, 'L24-N100')):
        prop_bytes = prop_elements(spatial, 2, nvec) * BYTES_PER_ELEMENT
        peak = 3 * prop_bytes
        verdict, ratio, usable = classify(peak, args.bytes, args.reserved_bytes)
        results[label] = (verdict, peak, ratio)
        print(f'{label}: spatial={spatial} nvec={nvec} '
              f'单算子={prop_bytes / 1024 ** 3:.2f} GiB '
              f'峰值={peak / 1024 ** 3:.2f} GiB '
              f'可用={usable / 1024 ** 3:.2f} GiB '
              f'占比={ratio:.3f} → {verdict}')

    tally.check(results['L24n'][0] == 'SAFE', 'budget/L24n-safe',
                f'实际 {results["L24n"][0]}')
    tally.check(results['L24'][0] == 'SAFE', 'budget/L24-safe',
                f'实际 {results["L24"][0]}，占比 {results["L24"][2]:.3f}')
    tally.check(results['L24-N100'][0] != 'SAFE', 'budget/L24-N100-not-safe',
                'L24 且 Nvec=100 不得判为 SAFE（应至少 MARGINAL）')
    tally.check(results['L24'][1] <= args.host_bytes, 'budget/L24-host',
                f'主机内存预算 {args.host_bytes / 1024 ** 3:.1f} GiB 不足')

    print('\n注意：以上为算术推导，未实际分配；真实峰值还包含 workspace、'
          'pack 缓冲与分配器碎片，故 MARGINAL 档不应作为默认生产配置。')
    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
