#!/usr/bin/env python3
"""`contract_path` 单函数对照：pyqcd.einsum vs opt_einsum，并用 torch.einsum 校验计划。

校验分三层：
1. **接口层**：`path`、`opt_cost`、`naive_cost`、`largest_intermediate` 与 opt_einsum 一致；
2. **数值层**：把 pyqcd 返回的 path 手工逐对执行，结果必须与 torch.einsum 一致；
3. **形状层**：`shapes=True` 时两者对同一组形状给出相同 path。
"""

from __future__ import annotations

import argparse
import decimal

import numpy as np

try:
    from . import _common
except ImportError:  # 直接以脚本方式运行
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple import _common


def _apply_path(equation: str, operands, path) -> np.ndarray:
    """按 path 手工逐对收缩（用 numpy.einsum），用于校验计划本身。"""
    terms = equation.split('->')[0].split(',')
    output = equation.split('->')[1]
    terms = list(terms)
    values = [np.asarray(operand) for operand in operands]
    for pair in path:
        left, right = int(pair[0]), int(pair[1])
        term_left, term_right = terms[left], terms[right]
        remaining = [term for index, term in enumerate(terms)
                     if index not in (left, right)]
        used = set(''.join(remaining)) | set(output)
        shared = set(term_left) & set(term_right)
        keep = ''.join(label for label in term_left + term_right
                       if label in shared or label in used)
        step_output = ''.join(dict.fromkeys(keep))
        if not remaining:  # 最后一步必须精确产出请求的输出顺序
            step_output = output
        value = np.einsum(f'{term_left},{term_right}->{step_output}',
                          values[left], values[right], optimize=False)
        for index in sorted((left, right), reverse=True):
            values.pop(index)
            terms.pop(index)
        values.append(value)
        terms.append(step_output)
    return values[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=15)
    parser.add_argument('--seed', type=int, default=_common.DEFAULT_SEED)
    args = parser.parse_args(argv)

    tally = _common.Tally('contract_path')
    target = _common.target_module()
    oracle = _common.oracle_module()

    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        equation, operands = _common.random_case(rng)
        shapes = [operand.shape for operand in operands]

        target_path, target_info = target.contract_path(
            equation, *operands, optimize='greedy')
        oracle_path, oracle_info = oracle.contract_path(
            equation, *operands, optimize='greedy')

        tally.check([tuple(step) for step in target_path]
                    == [tuple(step) for step in oracle_path],
                    f'case{index}/path-match',
                    f'{target_path} != {oracle_path}')
        tally.check(int(target_info.opt_cost) == int(oracle_info.opt_cost),
                    f'case{index}/opt-cost',
                    f'{target_info.opt_cost} != {oracle_info.opt_cost}')
        tally.check(int(target_info.naive_cost) == int(oracle_info.naive_cost),
                    f'case{index}/naive-cost',
                    f'{target_info.naive_cost} != {oracle_info.naive_cost}')
        tally.check(isinstance(target_info.opt_cost, decimal.Decimal),
                    f'case{index}/decimal-type',
                    type(target_info.opt_cost).__name__)

        reference = _common.torch_reference(equation, operands)
        error = _common.relative_error(
            _apply_path(equation, operands, target_path), reference)
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'case{index}/path-executes-correctly', f'rel={error:.2e}')

        for key in ('contraction_list', 'input_subscripts',
                    'output_subscript', 'indices', 'scale_list', 'size_list',
                    'size_dict', 'eq', 'path'):
            tally.check(hasattr(target_info, key), f'case{index}/field-{key}',
                        'PathInfo 缺字段')

        _, shape_target = target.contract_path(equation, *shapes,
                                               shapes=True, optimize='greedy')
        _, shape_oracle = oracle.contract_path(equation, *shapes,
                                               shapes=True, optimize='greedy')
        tally.check(int(shape_target.opt_cost) == int(shape_oracle.opt_cost),
                    f'case{index}/shapes-mode',
                    f'{shape_target.opt_cost} != {shape_oracle.opt_cost}')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
