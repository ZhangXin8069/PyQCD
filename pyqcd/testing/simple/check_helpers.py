#!/usr/bin/env python3
"""`helpers` 单函数对照：compute_size_by_dict / flop_count / find_contraction。

`torch.einsum` 不提供这些规划辅助函数，因此本文件的参考实现是 **opt_einsum**；
数值正确性由 `contract` 一侧的 torch 对照覆盖（见 check_contract.py）。
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from . import _common
except ImportError:  # 直接以脚本方式运行
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple import _common


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=20)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('helpers')
    target = _common.target_module()
    oracle = _common.oracle_module()

    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        labels = list(_common.SYMBOLS[:int(rng.integers(1, 6))])
        size_dict = {label: int(rng.integers(1, 5)) for label in labels}
        subset = list(rng.choice(labels, size=int(rng.integers(1, len(labels) + 1)),
                                replace=False))  # 子集大小不超过标签数

        target_size = target.helpers.compute_size_by_dict(subset, size_dict)
        oracle_size = oracle.helpers.compute_size_by_dict(subset, size_dict)
        tally.check(target_size == oracle_size, f'case{index}/compute_size_by_dict',
                    f'{target_size} != {oracle_size}')

        for inner in (True, False):
            terms = 2
            target_flops = target.helpers.flop_count(subset, inner, terms, size_dict)
            oracle_flops = oracle.helpers.flop_count(subset, inner, terms, size_dict)
            tally.check(target_flops == oracle_flops,
                        f'case{index}/flop_count-inner{int(inner)}',
                        f'{target_flops} != {oracle_flops}')

        # find_contraction：比较 (剩余集合, 新输入, 被消去集合)
        pair_size = min(2, len(labels))
        input_sets = [frozenset(rng.choice(labels, size=pair_size, replace=False))
                      for _ in range(2)]
        output_set = frozenset(rng.choice(labels, size=1))
        positions = (0, 1)
        target_result = target.helpers.find_contraction(
            positions, list(input_sets), output_set)
        oracle_result = oracle.helpers.find_contraction(
            positions, list(input_sets), output_set)
        target_norm = (sorted(target_result[0]),
                       [sorted(item) for item in target_result[1]],
                       sorted(target_result[2]))
        oracle_norm = (sorted(oracle_result[0]),
                       [sorted(item) for item in oracle_result[1]],
                       sorted(oracle_result[2]))
        tally.check(target_norm == oracle_norm, f'case{index}/find_contraction',
                    f'{target_norm} != {oracle_norm}')

    # 边界：空下标集合与零维
    tally.check(target.helpers.compute_size_by_dict([], {}) == 1,
                'edge/empty-size', '空下标应得 1')
    tally.check(target.helpers.flop_count([], False, 1, {}) == 1,
                'edge/empty-flop', '空下标乘法计数应得 1')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
