"""API 兼容面检查：与 opt_einsum 3.4 的公开接口逐项对照。

用法：
    python -m pyqcd.testing.einsum.simple.check_compat
"""

from __future__ import annotations

import argparse
import decimal

import numpy as np

from . import _common

PATHINFO_FIELDS = (
    'contraction_list', 'input_subscripts', 'output_subscript', 'path',
    'indices', 'scale_list', 'naive_cost', 'opt_cost', 'speedup',
    'size_list', 'size_dict', 'shapes', 'eq', 'largest_intermediate',
)

CALLABLES = ('contract', 'contract_path', 'contract_expression',
             'shared_intermediates', 'get_symbol')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    del args

    oracle = _common.oracle_module()
    import pyqcd.einsum as target

    tally = _common.Tally('compat')
    for name in CALLABLES:
        tally.check(callable(getattr(target, name, None)), f'api/{name}',
                    '缺失或不可调用')

    for name in ('get_func', 'has_einsum', 'has_tensordot', 'infer_backend'):
        tally.check(callable(getattr(target.backends, name, None)),
                    f'api/backends.{name}', '缺失或不可调用')

    for name in ('register_path_fn', 'get_path_fn', 'PATH_OPTIONS'):
        tally.check(hasattr(target.paths, name), f'api/paths.{name}', '缺失')

    rng = np.random.default_rng(7)
    left = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    right = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))

    ours_path, ours_info = target.contract_path('ij,jk->ik', left, right,
                                                optimize='dp')
    oracle_path, oracle_info = oracle.contract_path('ij,jk->ik', left, right,
                                                    optimize='dp')
    tally.check(list(ours_path) == list(oracle_path), 'path/dp-match',
                f'{ours_path} != {oracle_path}')
    for field in PATHINFO_FIELDS:
        tally.check(hasattr(ours_info, field), f'pathinfo/{field}', '缺字段')
    for field in ('opt_cost', 'naive_cost', 'speedup', 'largest_intermediate'):
        value = getattr(ours_info, field, None)
        tally.check(isinstance(value, decimal.Decimal),
                    f'pathinfo/{field}-decimal', f'{type(value).__name__}')
    tally.check(int(ours_info.opt_cost) == int(oracle_info.opt_cost),
                'pathinfo/opt_cost-match',
                f'{ours_info.opt_cost} != {oracle_info.opt_cost}')

    values = target.contract('ij,jk->ik', left, right, optimize='dp')
    tally.check(_common.relative_error(values, left @ right)
                <= _common.RELATIVE_TOLERANCE, 'behavior/matmul', '')

    implicit = target.contract('ij,jk', left, right, optimize='dp')
    tally.check(_common.relative_error(implicit, left @ right)
                <= _common.RELATIVE_TOLERANCE, 'behavior/implicit-output', '')

    trace = target.contract('ii->', np.eye(4, dtype=np.complex128),
                            optimize=False)
    tally.check(abs(complex(trace) - 4.0) <= 1e-12, 'behavior/trace',
                f'得到 {trace}')

    hadamard = target.contract('ij,ij->ij', left, left, optimize=False)
    tally.check(_common.relative_error(hadamard, left * left)
                <= _common.RELATIVE_TOLERANCE, 'behavior/hadamard', '')

    ellipsis = target.contract('...ij,...jk->...ik',
                               np.ones((2, 3, 4), dtype=np.complex128),
                               np.ones((2, 4, 5), dtype=np.complex128),
                               optimize='dp')
    tally.check(tuple(ellipsis.shape) == (2, 3, 5), 'behavior/ellipsis',
                f'shape={ellipsis.shape}')

    expression = target.contract_expression('ij,jk->ik', (3, 4), (4, 5))
    product = expression(left, right)
    tally.check(_common.relative_error(product, left @ right)
                <= _common.RELATIVE_TOLERANCE, 'behavior/expression', '')

    with_target = target.shared_intermediates()
    tally.check(hasattr(with_target, '__enter__'), 'behavior/shared_intermediates',
                '不是上下文管理器')
    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
