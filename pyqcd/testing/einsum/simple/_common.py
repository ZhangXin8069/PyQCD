"""simple 接口的共享工具：方程表、操作数、误差、计时与计数。"""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 复用既有基准的方程表与操作数构造，避免重复实现。
from pyqcd.testing.einsum.bench_distillation import (  # noqa: E402
    EQUATIONS,
    EQUATION_BY_NAME,
    PROFILES,
    build_operands,
)

DTYPE = 'complex128'
RELATIVE_TOLERANCE = 1e-12
DEFAULT_PROFILE = os.environ.get('PYQCD_SIMPLE_PROFILE', 'T0n')


class Tally:
    """PASS/FAIL/SKIP 计数与退出码。"""

    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def check(self, condition, label, detail=''):
        if condition:
            self.passed += 1
            print(f'PASS {label}')
        else:
            self.failed += 1
            self.failures.append((label, detail))
            print(f'FAIL {label}' + (f': {detail}' if detail else ''))

    def skip(self, label, reason):
        self.skipped += 1
        print(f'SKIP {label}: {reason}')

    def finish(self):
        print(f'\n{self.name}: {self.passed} passed, '
              f'{self.skipped} skipped, {self.failed} failed')
        for label, detail in self.failures:
            print(f'  FAILED {label}: {detail}')
        return 0 if self.failed == 0 else 1


def profile_config(name=None):
    return dict(PROFILES[name or DEFAULT_PROFILE])


def operands(eq_name, cfg, seed=20261108):
    """按方程名构造操作数（固定种子，可复现）。"""
    nvec = cfg['nvec']
    peram_shape = (72, 4, 4, nvec, nvec)
    rng = np.random.default_rng(seed)
    peram = (rng.standard_normal(peram_shape)
             + 1j * rng.standard_normal(peram_shape)).astype(np.complex128)
    return build_operands(eq_name, peram, cfg, seed=seed)


def oracle_module():
    import opt_einsum

    return opt_einsum


def relative_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float('inf')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return float('inf')
    denominator = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected)) / denominator)


def to_backend(arrays, backend):
    if backend == 'cuda':
        import cupy

        return [cupy.asarray(array) for array in arrays]
    return list(arrays)


def sync(backend):
    if backend == 'cuda':
        import cupy

        cupy.cuda.Stream.null.synchronize()


def to_host(value, backend):
    if backend == 'cuda':
        import cupy

        return cupy.asnumpy(value)
    return np.asarray(value)


def cuda_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - 探测失败按不可用处理
        return False


def median_seconds(function, iterations=10, warmup=2, backend='cpu'):
    """预热后取中位耗时。"""
    for _ in range(warmup):
        function()
    sync(backend)
    samples = []
    for _ in range(iterations):
        sync(backend)
        start = time.perf_counter()
        function()
        sync(backend)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def equation_names(default=None):
    return default or [item['name'] for item in EQUATIONS]
