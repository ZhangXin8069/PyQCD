"""Native KUPL/CUDA einsum correctness and cache-contract verification."""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Any

import numpy as np

from pyqcd.einsum import _native
from pyqcd.einsum import contract, contract_expression

RELATIVE_TOLERANCE = 1e-12
# 四类 dtype 一视同仁地覆盖；容差按元素精度取（单精度链上有固有散射）。
ALL_DTYPES = (np.complex128, np.complex64, np.float64, np.float32)
TOLERANCE = {'complex128': 1e-12, 'float64': 1e-12,
             'complex64': 1e-4, 'float32': 1e-4}


def _tolerance(dtype: Any) -> float:
    return TOLERANCE[np.dtype(dtype).name]


def _operand(shape, dtype, seed: int = 0) -> np.ndarray:
    """确定性操作数；复数才有虚部。"""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape)
    if np.dtype(dtype).kind == 'c':
        data = data + 1j * rng.normal(size=shape)
    return np.ascontiguousarray(data.astype(dtype))


class Tally:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def check(self, condition: bool, label: str, detail: str = '') -> None:
        if condition:
            self.passed += 1
            print(f'PASS {label}')
        else:
            self.failed += 1
            print(f'FAIL {label}: {detail}')

    def skip(self, label: str, reason: str) -> None:
        self.skipped += 1
        print(f'SKIP {label}: {reason}')


def _host(value: Any) -> np.ndarray:
    if type(value).__module__.split('.')[0] == 'cupy':
        import cupy

        cupy.cuda.Stream.null.synchronize()
        return cupy.asnumpy(value)
    return np.asarray(value)


def _relative_error(actual: Any, expected: np.ndarray) -> float:
    """按**输出最大幅值**归一的最大绝对偏差。

    曾经逐元素除以 ``|expected|``：sandwich 这类输出含近零元素（除零被 tiny 兜住）
    时，单个元素的微小绝对差会被放大成 1e-4 量级，把正确的 float32 结果判成失败。
    归一化口径必须与 ``_topology_kernel_contract`` / ``verify_dtype_parity`` 一致。
    """
    actual_host = _host(actual)
    if actual_host.shape != expected.shape:
        return float('inf')
    scale = float(np.max(np.abs(expected), initial=0.0))
    difference = float(np.max(np.abs(actual_host - expected), initial=0.0))
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float('inf')
    return difference / scale


def _to_backend(array: np.ndarray, backend: str) -> Any:
    if backend == 'cuda':
        import cupy

        return cupy.asarray(array)
    return array


def _backend_name(requested: str) -> str:
    if requested == 'auto':
        if _native._cuda_module() is not None:
            return 'cuda'
        return 'kupl'
    return requested


def _run_dtype_cases(tally: Tally, backend: str) -> None:
    for dtype in (np.complex128, np.complex64, np.float64, np.float32):
        left_host = np.arange(12, dtype=dtype).reshape(3, 4)
        right_host = np.arange(20, dtype=dtype).reshape(4, 5)
        expected = np.einsum('ij,jk->ik', left_host, right_host, optimize='greedy')
        actual = contract(
            'ij,jk->ik', _to_backend(left_host, backend),
            _to_backend(right_host, backend), optimize='greedy',
            backend='cupy' if backend == 'cuda' else 'numpy')
        error = _relative_error(actual, expected)
        tally.check(
            error <= RELATIVE_TOLERANCE and str(_host(actual).dtype) == np.dtype(dtype).name,
            f'{backend}/matmul/{np.dtype(dtype).name}',
            f'relative error={error:.3e}, dtype={_host(actual).dtype}')


def _run_equation_cases(tally: Tally, backend: str) -> None:
    """五类形态 × 四类 dtype：形态矩阵不能只在 complex128 上验。"""
    cases = [
        ('chain', 'ab,bc,cd->ad', [(2, 3), (3, 4), (4, 5)]),
        ('batch', 'bij,bjk->bik', [(2, 3, 4), (2, 4, 5)]),
        ('unary-permute', '...ij->...ji', [(2, 3, 4)]),
        ('dot', 'ij,ij->', [(4, 5), (4, 5)]),
        ('sandwich', 'ab,...bce,cd->...dae',
         [(4, 4), (1, 1, 1, 3, 2, 4, 4, 20), (4, 4)]),
    ]
    for dtype in ALL_DTYPES:
        name_of = np.dtype(dtype).name
        for name, equation, shapes in cases:
            hosts = [_operand(shape, dtype, seed=index + 1)
                     for index, shape in enumerate(shapes)]
            expected = np.einsum(equation, *hosts, optimize='greedy')
            operands = [_to_backend(array, backend) for array in hosts]
            actual = contract(
                equation, *operands, optimize='greedy',
                backend='cupy' if backend == 'cuda' else 'numpy')
            error = _relative_error(actual, expected)
            kept = str(_host(actual).dtype) == name_of
            tally.check(error <= _tolerance(dtype) and kept,
                        f'{backend}/{name}/{name_of}',
                        f'error={error:.3e}, dtype={_host(actual).dtype}')


def _run_mutation_and_out(tally: Tally, backend: str) -> None:
    for dtype in ALL_DTYPES:
        name_of = np.dtype(dtype).name
        for value in (1.0, 2.0, 3.0):
            left_host = np.full((2, 3), value, dtype=dtype)
            right_host = np.ones((3, 4), dtype=dtype)
            actual = contract(
                'ij,jk->ik', _to_backend(left_host, backend),
                _to_backend(right_host, backend), optimize='greedy',
                backend='cupy' if backend == 'cuda' else 'numpy')
            expected = np.full((2, 4), 3.0 * value, dtype=dtype)
            error = _relative_error(actual, expected)
            tally.check(error <= _tolerance(dtype),
                        f'{backend}/cache-value-{value:g}/{name_of}',
                        f'error={error:.3e}')

        left_host = np.arange(6, dtype=dtype).reshape(2, 3)
        right_host = np.arange(12, dtype=dtype).reshape(3, 4)
        if backend == 'cuda':
            import cupy

            output = cupy.empty((2, 4), dtype=dtype)
            backend_name = 'cupy'
        else:
            output = np.empty((2, 4), dtype=dtype)
            backend_name = 'numpy'
        returned = contract(
            'ij,jk->ik', _to_backend(left_host, backend),
            _to_backend(right_host, backend), out=output, optimize='greedy',
            backend=backend_name)
        expected = np.einsum('ij,jk->ik', left_host, right_host, optimize='greedy')
        tally.check(
            returned is output
            and _relative_error(output, expected) <= _tolerance(dtype)
            and str(_host(output).dtype) == name_of,
            f'{backend}/out-contract/{name_of}',
            f'dtype={_host(output).dtype}')


def _run_expression_and_fallback(tally: Tally, backend: str) -> None:
    for dtype in ALL_DTYPES:
        name_of = np.dtype(dtype).name
        left_host = np.arange(6, dtype=dtype).reshape(2, 3)
        right_host = np.arange(12, dtype=dtype).reshape(3, 4)
        expression = contract_expression(
            'ij,jk->ik', left_host.shape, right_host.shape, optimize='greedy')
        actual = expression(
            _to_backend(left_host, backend), _to_backend(right_host, backend),
            backend='cupy' if backend == 'cuda' else 'numpy')
        expected = left_host @ right_host
        tally.check(
            _relative_error(actual, expected) <= _tolerance(dtype)
            and str(_host(actual).dtype) == name_of,
            f'{backend}/contract-expression/{name_of}',
            f'dtype={_host(actual).dtype}')

        diagonal_host = np.arange(9, dtype=dtype).reshape(3, 3)
        fallback = contract(
            'ii->', _to_backend(diagonal_host, backend), optimize='greedy',
            backend='cupy' if backend == 'cuda' else 'numpy')
        tally.check(
            _relative_error(
                fallback,
                np.einsum('ii->', diagonal_host, optimize='greedy')) <= _tolerance(dtype),
            f'{backend}/repeated-axis-fallback/{name_of}')


def run_mixed_backends() -> tuple[bool, str]:
    """Load KUPL first and CUDA second in one process without symbol collision."""
    import os

    if _native._kupl_entry() is None:
        return True, 'KUPL unavailable'
    if _native._cuda_entry() is None:
        return True, 'CUDA unavailable'
    try:
        import cupy
    except ImportError:
        return True, 'CuPy unavailable'
    if not cupy.cuda.runtime.getDeviceCount():
        return True, 'no CUDA device'

    original = os.environ.get('PYQCD_EINSUM_NATIVE')
    try:
        worst = 0.0
        for dtype in ALL_DTYPES:
            left = np.arange(6, dtype=dtype).reshape(2, 3)
            right = np.arange(12, dtype=dtype).reshape(3, 4)
            os.environ['PYQCD_EINSUM_NATIVE'] = 'kupl'
            host = contract('ij,jk->ik', left, right, backend='numpy')
            os.environ['PYQCD_EINSUM_NATIVE'] = 'cuda'
            device = contract(
                'ij,jk->ik', cupy.asarray(left), cupy.asarray(right), backend='cupy'
            )
            cupy.cuda.Device().synchronize()
            error = _relative_error(device, left @ right)
            worst = max(worst, error / _tolerance(dtype))
        with open('/proc/self/maps', encoding='utf-8') as handle:
            maps = handle.read()
        loaded = (
            'libpyqcdeinsum_kupl.so' in maps
            and 'libpyqcdeinsum_cuda.so' in maps
        )
        return loaded and worst <= 1.0, (
            f'both_loaded={loaded}, worst_error/tolerance={worst:.3e}, '
            f'host_norm={np.linalg.norm(host):.3e}'
        )
    finally:
        if original is None:
            os.environ.pop('PYQCD_EINSUM_NATIVE', None)
        else:
            os.environ['PYQCD_EINSUM_NATIVE'] = original


def run(backend: str = 'auto', smoke: bool = False) -> tuple[Tally, bool]:
    selected = _backend_name(backend)
    tally = Tally(selected)
    if selected == 'cuda':
        try:
            import cupy
        except ImportError:
            tally.skip('cuda', 'CuPy unavailable')
            return tally, True
        if not cupy.cuda.runtime.getDeviceCount():
            tally.skip('cuda', 'no CUDA device')
            return tally, True
    entry = (
        _native._cuda_entry() if selected == 'cuda' else _native._kupl_entry()
    )
    if entry is None:
        if backend == 'auto':
            tally.skip(selected, 'native library unavailable')
            return tally, True
        tally.check(False, f'{selected}/native-library', 'requested entry is unavailable')
        return tally, False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _run_dtype_cases(tally, selected)
        if not smoke:
            _run_equation_cases(tally, selected)
            _run_mutation_and_out(tally, selected)
            _run_expression_and_fallback(tally, selected)
    unexpected_fallback = [
        str(item.message) for item in caught
        if 'using Python execution' in str(item.message)
        and "repeated axes in 'ii'" not in str(item.message)
    ]
    tally.check(
        not unexpected_fallback,
        f'{selected}/native-path-used',
        '; '.join(unexpected_fallback),
    )
    return tally, tally.failed == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Verify native qcd-einsum correctness')
    parser.add_argument('--backend', choices=('auto', 'kupl', 'cuda'), default='auto')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    tally, ok = run(backend=args.backend, smoke=args.smoke)
    print(
        f'{tally.name}: {tally.passed} passed, {tally.skipped} skipped, '
        f'{tally.failed} failed')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
