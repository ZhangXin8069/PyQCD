#!/usr/bin/env python3
"""`contract` 单函数对照：pyqcd.einsum vs opt_einsum，参考实现 torch.einsum。

用法::

    python -m pyqcd.testing.simple.check_contract
    python pyqcd/testing/simple/check_contract.py --cases 40
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
    parser.add_argument('--cases', type=int, default=25)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('contract')
    target = _common.target_module()

    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        equation, operands = _common.random_case(rng)
        for optimize in ('greedy', 'dp'):
            _common.compare_three_way(equation, operands, tally,
                                      f'case{index}/{optimize}',
                                      optimize=optimize)

    # 隐式输出（无 "->"）：按 numpy 规则推导应当得到的输出
    rng = _common.seeded_rng(args.seed)
    equation, operands = _common.random_case(rng)
    terms = equation.split('->')[0]
    explicit = terms + '->' + _common.implicit_output(equation)
    reference = _common.torch_reference(explicit, operands)
    result = target.contract(terms, *operands, optimize='greedy')
    error = _common.relative_error(result, reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, 'implicit-output',
                f'rel={error:.2e}')

    # out= 语义：返回同一对象且写入正确
    rng = _common.seeded_rng(args.seed + 1)
    equation, operands = _common.random_case(rng, min_operands=2, max_operands=2)
    reference = _common.torch_reference(equation, operands)
    out = np.zeros(reference.shape, dtype=np.complex128)
    returned = target.contract(equation, *operands, out=out, optimize='greedy')
    tally.check(returned is out, 'out-identity', '返回对象不是传入的 out')
    error = _common.relative_error(out, reference)
    tally.check(error <= _common.RELATIVE_TOLERANCE, 'out-values',
                f'rel={error:.2e}')

    # 单操作数（转置 / trace）
    matrix = (np.arange(9).reshape(3, 3)
              + 1j * np.arange(9, 18).reshape(3, 3)).astype(np.complex128)
    transpose_error = _common.relative_error(
        target.contract('ij->ji', matrix, optimize=False),
        _common.torch_reference('ij->ji', [matrix]))
    tally.check(transpose_error <= _common.RELATIVE_TOLERANCE, 'unary/transpose',
                f'rel={transpose_error:.2e}')
    trace_error = _common.relative_error(
        target.contract('ii->', matrix, optimize=False),
        _common.torch_reference('ii->', [matrix]))
    tally.check(trace_error <= _common.RELATIVE_TOLERANCE, 'unary/trace',
                f'rel={trace_error:.2e}')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
