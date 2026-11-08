#!/usr/bin/env python3
"""`shared_intermediates` 单函数对照：pyqcd.einsum vs opt_einsum，参考 torch.einsum。

校验：缓存确实被填充、缓存计数与 opt_einsum 一致、重复计算复用结果且数值正确。
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
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('shared_intermediates')
    target = _common.target_module()
    oracle = _common.oracle_module()

    for index, rng in enumerate(_common.iter_seeds(args.rounds, args.seed)):
        left = (rng.standard_normal((3, 4))
                + 1j * rng.standard_normal((3, 4))).astype(np.complex128)
        middle = (rng.standard_normal((4, 5))
                  + 1j * rng.standard_normal((4, 5))).astype(np.complex128)
        right = (rng.standard_normal((5, 6))
                 + 1j * rng.standard_normal((5, 6))).astype(np.complex128)

        with target.shared_intermediates() as target_cache:
            target_first = target.contract('ab,bc->ac', left, middle, backend='numpy')
            target_second = target.contract('ab,bc->ac', left, middle, backend='numpy')
            target_third = target.contract('ac,cd->ad', target_first, right,
                                           backend='numpy')
        target_ops = target.count_cached_ops(target_cache)

        with oracle.shared_intermediates() as oracle_cache:
            oracle_first = oracle.contract('ab,bc->ac', left, middle, backend='numpy')
            oracle_second = oracle.contract('ab,bc->ac', left, middle, backend='numpy')
            oracle_third = oracle.contract('ac,cd->ad', oracle_first, right,
                                           backend='numpy')
        oracle_ops = oracle.sharing.count_cached_ops(oracle_cache)

        reference = _common.torch_reference('ab,bc,cd->ad',
                                            [left, middle, right])
        error = _common.relative_error(target_third, reference)
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'case{index}/numeric', f'rel={error:.2e}')
        tally.check(np.allclose(target_first, target_second),
                    f'case{index}/repeat-consistent', '重复调用结果不一致')
        tally.check(sum(target_ops.values()) > 0, f'case{index}/cache-used',
                    f'缓存为空: {dict(target_ops)}')
        tally.check(dict(target_ops) == dict(oracle_ops),
                    f'case{index}/cache-parity',
                    f'{dict(target_ops)} != {dict(oracle_ops)}')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
