#!/usr/bin/env python3
"""`parser` 单函数对照：get_symbol / parse_einsum_input。

`torch.einsum` 没有对应的解析器接口，故参考实现是 **opt_einsum**；
解析结果的语义正确性（下标展开后能否算出同样结果）用 torch 校验。
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
    parser.add_argument('--cases', type=int, default=10)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('parser')
    target = _common.target_module()
    oracle = _common.oracle_module()

    # 1) get_symbol：0..51 必须与 opt_einsum 完全一致（a-zA-Z 序列）
    mismatch = []
    for index in range(52):
        target_symbol = target.get_symbol(index)
        oracle_symbol = oracle.get_symbol(index)
        if target_symbol != oracle_symbol:
            mismatch.append((index, target_symbol, oracle_symbol))
    tally.check(not mismatch, 'get_symbol/0-51', f'不一致: {mismatch[:5]}')

    # 2) parse_einsum_input：返回 (输入串, 输出串, 操作数列表)
    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        equation, operands = _common.random_case(rng)

        target_inputs, target_output, target_operands = target.parser.parse_einsum_input(
            (equation, *operands))
        oracle_inputs, oracle_output, oracle_operands = oracle.parser.parse_einsum_input(
            (equation, *operands))

        tally.check(target_inputs == oracle_inputs, f'case{index}/inputs',
                    f'{target_inputs!r} != {oracle_inputs!r}')
        tally.check(target_output == oracle_output, f'case{index}/output',
                    f'{target_output!r} != {oracle_output!r}')
        tally.check(len(target_operands) == len(oracle_operands),
                    f'case{index}/operand-count',
                    f'{len(target_operands)} != {len(oracle_operands)}')

        # 用解析后的显式表达式 + torch 校验语义
        explicit = f"{target_inputs}->{target_output}"
        reference = _common.torch_reference(explicit, operands)
        error = _common.relative_error(
            target.contract(equation, *operands, optimize='greedy'), reference)
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'case{index}/parsed-semantics', f'rel={error:.2e}')

    # 3) 隐式输出解析：输出串必须是"只出现一次的标签、按字母序"
    implicit_cases = ['ij,jk', 'ij,ji', 'ii', 'ijk,jil']
    for equation in implicit_cases:
        shapes = []
        for term in equation.split(','):
            shapes.append(tuple(2 for _ in term))
        operands = [np.ones(shape, dtype=np.complex128) for shape in shapes]
        target_inputs, target_output, _ = target.parser.parse_einsum_input(
            (equation, *operands))
        oracle_inputs, oracle_output, _ = oracle.parser.parse_einsum_input(
            (equation, *operands))
        tally.check((target_inputs, target_output) == (oracle_inputs, oracle_output),
                    f'implicit/{equation}',
                    f'{(target_inputs, target_output)} != {(oracle_inputs, oracle_output)}')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
