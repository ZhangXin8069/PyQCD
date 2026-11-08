#!/usr/bin/env python3
"""`contract_expression` 单函数对照：pyqcd.einsum vs opt_einsum，参考 torch.einsum。

覆盖：首次求值、换一批输入复用、`out=` 语义、以及形状不符时的失败关闭。
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

    tally = _common.Tally('contract_expression')
    target = _common.target_module()
    oracle = _common.oracle_module()

    for index, rng in enumerate(_common.iter_seeds(args.cases, args.seed)):
        equation, operands = _common.random_case(rng)
        shapes = [operand.shape for operand in operands]

        target_expression = target.contract_expression(
            equation, *shapes, optimize='greedy')
        oracle_expression = oracle.contract_expression(
            equation, *shapes, optimize='greedy')

        reference = _common.torch_reference(equation, operands)
        target_value = target_expression(*operands)
        oracle_value = oracle_expression(*operands)

        error = _common.relative_error(target_value, reference)
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'case{index}/first-call', f'rel={error:.2e}')
        error = _common.relative_error(oracle_value, reference)
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'case{index}/oracle-first-call', f'rel={error:.2e}')

        # 换一批随机输入复用同一表达式
        fresh = [np.ascontiguousarray(
            (rng.standard_normal(operand.shape)
             + 1j * rng.standard_normal(operand.shape)).astype(np.complex128))
            for operand in operands]
        reference_fresh = _common.torch_reference(equation, fresh)
        error = _common.relative_error(target_expression(*fresh), reference_fresh)
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'case{index}/reuse-fresh-inputs', f'rel={error:.2e}')

        # out= 语义
        out = np.zeros(reference_fresh.shape, dtype=np.complex128)
        returned = target_expression(*fresh, out=out)
        tally.check(returned is out, f'case{index}/out-identity',
                    '返回对象不是传入的 out')
        error = _common.relative_error(out, reference_fresh)
        tally.check(error <= _common.RELATIVE_TOLERANCE,
                    f'case{index}/out-values', f'rel={error:.2e}')

        # 参数个数不符必须失败关闭（比改维度更确定：改尺寸可能仍然合法）
        try:
            target_expression(*operands[:-1])
        except Exception:  # noqa: BLE001 - 期望失败
            failed_closed = True
        else:
            failed_closed = False
        tally.check(failed_closed, f'case{index}/fail-closed',
                    '操作数个数不符时未报错')

    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
