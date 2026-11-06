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
    actual_host = _host(actual)
    if actual_host.shape != expected.shape:
        return float('inf')
    denominator = np.maximum(np.abs(expected), np.finfo(float).tiny)
    return float(np.max(np.abs(actual_host - expected) / denominator, initial=0.0))


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
    rng = np.random.default_rng(20261106)

    def complex_operand(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex128)

    cases = [
        ('chain', 'ab,bc,cd->ad', [complex_operand((2, 3)), complex_operand((3, 4)), complex_operand((4, 5))]),
        ('batch', 'bij,bjk->bik', [complex_operand((2, 3, 4)), complex_operand((2, 4, 5))]),
        ('unary-permute', '...ij->...ji', [complex_operand((2, 3, 4))]),
        ('dot', 'ij,ij->', [complex_operand((4, 5)), complex_operand((4, 5))]),
        ('sandwich', 'ab,...bce,cd->...dae', [
            complex_operand((4, 4)), complex_operand((1, 1, 1, 3, 2, 4, 4, 20)),
            complex_operand((4, 4)),
        ]),
    ]
    for name, equation, hosts in cases:
        expected = np.einsum(equation, *hosts, optimize='greedy')
        operands = [_to_backend(array, backend) for array in hosts]
        actual = contract(
            equation, *operands, optimize='greedy',
            backend='cupy' if backend == 'cuda' else 'numpy')
        error = _relative_error(actual, expected)
        tally.check(error <= RELATIVE_TOLERANCE, f'{backend}/{name}', f'error={error:.3e}')


def _run_mutation_and_out(tally: Tally, backend: str) -> None:
    for value in (1.0, 2.0, 3.0):
        left_host = np.full((2, 3), value, dtype=np.complex128)
        right_host = np.ones((3, 4), dtype=np.complex128)
        actual = contract(
            'ij,jk->ik', _to_backend(left_host, backend),
            _to_backend(right_host, backend), optimize='greedy',
            backend='cupy' if backend == 'cuda' else 'numpy')
        expected = np.full((2, 4), 3.0 * value, dtype=np.complex128)
        error = _relative_error(actual, expected)
        tally.check(error <= RELATIVE_TOLERANCE, f'{backend}/cache-value-{value:g}', f'error={error:.3e}')

    left_host = np.arange(6, dtype=np.complex128).reshape(2, 3)
    right_host = np.arange(12, dtype=np.complex128).reshape(3, 4)
    if backend == 'cuda':
        import cupy

        output = cupy.empty((2, 4), dtype=cupy.complex128)
        backend_name = 'cupy'
    else:
        output = np.empty((2, 4), dtype=np.complex128)
        backend_name = 'numpy'
    returned = contract(
        'ij,jk->ik', _to_backend(left_host, backend), _to_backend(right_host, backend),
        out=output, optimize='greedy', backend=backend_name)
    expected = np.einsum('ij,jk->ik', left_host, right_host, optimize='greedy')
    tally.check(returned is output and _relative_error(output, expected) <= RELATIVE_TOLERANCE,
                f'{backend}/out-contract')


def _run_expression_and_fallback(tally: Tally, backend: str) -> None:
    left_host = np.arange(6, dtype=np.complex128).reshape(2, 3)
    right_host = np.arange(12, dtype=np.complex128).reshape(3, 4)
    expression = contract_expression(
        'ij,jk->ik', left_host.shape, right_host.shape, optimize='greedy')
    actual = expression(
        _to_backend(left_host, backend), _to_backend(right_host, backend),
        backend='cupy' if backend == 'cuda' else 'numpy')
    expected = left_host @ right_host
    tally.check(_relative_error(actual, expected) <= RELATIVE_TOLERANCE,
                f'{backend}/contract-expression')

    diagonal_host = np.arange(9, dtype=np.complex128).reshape(3, 3)
    fallback = contract(
        'ii->', _to_backend(diagonal_host, backend), optimize='greedy',
        backend='cupy' if backend == 'cuda' else 'numpy')
    tally.check(
        _relative_error(fallback, np.einsum('ii->', diagonal_host, optimize='greedy')) <= RELATIVE_TOLERANCE,
        f'{backend}/repeated-axis-fallback')


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
        left = np.arange(6, dtype=np.complex128).reshape(2, 3)
        right = np.arange(12, dtype=np.complex128).reshape(3, 4)
        os.environ['PYQCD_EINSUM_NATIVE'] = 'kupl'
        host = contract('ij,jk->ik', left, right, backend='numpy')
        os.environ['PYQCD_EINSUM_NATIVE'] = 'cuda'
        device = contract(
            'ij,jk->ik', cupy.asarray(left), cupy.asarray(right), backend='cupy'
        )
        cupy.cuda.Device().synchronize()
        error = _relative_error(device, left @ right)
        with open('/proc/self/maps', encoding='utf-8') as handle:
            maps = handle.read()
        loaded = (
            'libpyqcdeinsum_kupl.so' in maps
            and 'libpyqcdeinsum_cuda.so' in maps
        )
        return loaded and error <= RELATIVE_TOLERANCE, (
            f'both_loaded={loaded}, relative error={error:.3e}, '
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
