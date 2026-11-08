#!/usr/bin/env python3
"""`can_blas` 单函数对照：pyqcd.einsum vs opt_einsum。

`torch.einsum` 不做 BLAS 判定，故参考实现是 **opt_einsum**；
判定结果的实际数值正确性由 check_contract.py 的 torch 对照覆盖。
"""

from __future__ import annotations

import argparse
import itertools

try:
    from . import _common
except ImportError:  # 直接以脚本方式运行
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple import _common


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=30)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('blas')
    target = _common.target_module()
    oracle = _common.oracle_module()

    # 覆盖 GEMM / DOT / TDOT / OUTER / 不可用 五类典型二元收缩
    cases = [
        (['ij', 'jk'], 'ik', 'j'),
        (['ji', 'jk'], 'ik', 'j'),
        (['ij', 'kj'], 'ik', 'j'),
        (['ij', 'ij'], 'ij', ''),
        (['ij', 'ij'], '', 'ij'),
        (['i', 'i'], '', 'i'),
        (['i', 'j'], 'ij', ''),
        (['ijk', 'ijl'], 'kl', 'ij'),
        (['ab', 'bc', 'cd'], 'ad', ''),
    ]

    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        left_size = int(rng.integers(2, 5))
        right_size = int(rng.integers(2, 5))
        shared = int(rng.integers(1, min(left_size, right_size) + 1))
        left = 'ij'[:1] + 'k'[:shared]
        right = 'k'[:shared] + 'l'[:1]
        cases.append(([left, right], 'il', 'k'[:shared]))

    for index, (inputs, result, removed) in enumerate(cases):
        label = f'case{index}/{",".join(inputs)}->{result}'
        try:
            removed_set = set(removed)   # 契约类型是 set/frozenset，不接受 str
            target_value = target.blas.can_blas(list(inputs), result, removed_set)
        except Exception as exc:  # noqa: BLE001 - 记录而非中断
            tally.check(False, label, f'pyqcd 抛 {type(exc).__name__}: {exc}')
            continue
        oracle_value = oracle.blas.can_blas(list(inputs), result, removed_set)
        tally.check(str(target_value) == str(oracle_value), label,
                    f'{target_value!r} != {oracle_value!r}')

    # 三个及以上操作数不是二元收缩，两边都应判定为不可用
    triple = target.blas.can_blas(['ab', 'bc', 'cd'], 'ad', set())
    tally.check(not triple, 'n-ary-not-blas', f'三元组被误判为 {triple!r}')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
