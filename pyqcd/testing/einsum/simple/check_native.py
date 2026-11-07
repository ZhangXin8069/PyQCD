"""原生后端逐项检查：四 dtype、链式、批、一元、点积、compiled context 复用。

用法：
    python -m pyqcd.testing.einsum.simple.check_native --backend kupl
    PYQCD_EINSUM_NATIVE=cuda python -m pyqcd.testing.einsum.simple.check_native --backend cuda
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from . import _common


def _rand(shape, rng, dtype):
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return np.ascontiguousarray(data.astype(dtype))


def _rand_real(shape, rng, dtype):
    return np.ascontiguousarray(rng.standard_normal(shape).astype(dtype))


# 单精度量级：float32/complex64 的机器精度约 6e-8/1.2e-7，
# 与双精度参照比较只能用该量级的容差；双精度仍用 1e-12 契约值。
_SINGLE_TOLERANCE = 1e-5


def _tolerance(expected):
    dtype = np.asarray(expected).dtype
    if dtype in (np.float32, np.complex64):
        return _SINGLE_TOLERANCE
    return _common.RELATIVE_TOLERANCE


def _compare(tally, label, got, expected, backend):
    expected = np.asarray(expected)
    error = _common.relative_error(_common.to_host(got, backend), expected)
    tally.check(error <= _tolerance(expected), label,
                f'rel={error:.3e} tol={_tolerance(expected):.0e}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='kupl',
                        choices=('kupl', 'cuda', 'auto'))
    args = parser.parse_args(argv)
    backend = args.backend
    if backend == 'cuda' and not _common.cuda_available():
        print('SKIP: CUDA 不可用')
        return 0
    os.environ['PYQCD_EINSUM_NATIVE'] = backend

    import pyqcd.einsum as target
    from pyqcd.einsum import _native

    tally = _common.Tally(f'native[{backend}]')
    resolved = _native.backend_name()
    tally.check(resolved == backend, 'native/backend-selected',
                f'解析为 {resolved}，期望 {backend}')

    rng = np.random.default_rng(20261108)
    cases = (
        (np.complex128, 'complex128'),
        (np.complex64, 'complex64'),
        (np.float64, 'float64'),
        (np.float32, 'float32'),
    )
    for dtype, label in cases:
        make = _rand if np.issubdtype(dtype, np.complexfloating) else _rand_real
        left = make((8, 16), rng, dtype)
        right = make((16, 32), rng, dtype)
        host_left, host_right = (_common.to_backend([left, right], backend))
        got = target.contract('ij,jk->ik', host_left, host_right, optimize=False,
                              backend='cupy' if backend == 'cuda' else 'numpy')
        _compare(tally, f'matmul/{label}', got, left @ right, backend)

    left = _rand((6, 7), rng, np.complex128)
    mid = _rand((7, 8), rng, np.complex128)
    right = _rand((8, 9), rng, np.complex128)
    host = _common.to_backend([left, mid, right], backend)
    got = target.contract('ij,jk,kl->il', *host, optimize='greedy',
                          backend='cupy' if backend == 'cuda' else 'numpy')
    _compare(tally, 'chain', got, left @ mid @ right, backend)

    batch_left = _rand((4, 5, 6), rng, np.complex128)
    batch_right = _rand((4, 6, 7), rng, np.complex128)
    host = _common.to_backend([batch_left, batch_right], backend)
    got = target.contract('bij,bjk->bik', *host, optimize='greedy',
                          backend='cupy' if backend == 'cuda' else 'numpy')
    _compare(tally, 'batch', got,
             np.einsum('bij,bjk->bik', batch_left, batch_right), backend)

    cube = _rand((3, 4, 5), rng, np.complex128)
    host = _common.to_backend([cube], backend)
    got = target.contract('ijk->kji', *host, optimize=False,
                          backend='cupy' if backend == 'cuda' else 'numpy')
    _compare(tally, 'unary-permute', got, cube.transpose(2, 1, 0), backend)

    vector_a = _rand((11,), rng, np.complex128)
    vector_b = _rand((11,), rng, np.complex128)
    host = _common.to_backend([vector_a, vector_b], backend)
    got = target.contract('i,i->', *host, optimize=False,
                          backend='cupy' if backend == 'cuda' else 'numpy')
    _compare(tally, 'dot', got, np.einsum('i,i->', vector_a, vector_b),
             backend)

    expression = target.contract_expression(
        'ij,jk->ik', (8, 16), (16, 32), optimize='greedy')
    first = (_rand((8, 16), rng, np.complex128),
             _rand((16, 32), rng, np.complex128))
    second = (_rand((8, 16), rng, np.complex128),
              _rand((16, 32), rng, np.complex128))
    got_first = expression(*_common.to_backend(first, backend))
    got_second = expression(*_common.to_backend(second, backend))
    _compare(tally, 'expression/reuse-1', got_first, first[0] @ first[1],
             backend)
    _compare(tally, 'expression/reuse-2', got_second, second[0] @ second[1],
             backend)

    out_array = np.zeros((8, 32), dtype=np.complex128)
    host_out = (_common.to_backend([out_array], backend)[0])
    host_left = _common.to_backend([first[0]], backend)[0]
    host_right = _common.to_backend([first[1]], backend)[0]
    returned = target.contract('ij,jk->ik', host_left, host_right, out=host_out,
                               optimize='greedy',
                               backend='cupy' if backend == 'cuda' else 'numpy')
    same_object = returned is host_out
    tally.check(same_object, 'out/identity',
                '返回对象与传入 out 不是同一对象')
    _compare(tally, 'out/values', returned, first[0] @ first[1], backend)
    return tally.finish()


if __name__ == '__main__':
    raise SystemExit(main())
