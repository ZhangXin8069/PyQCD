"""Verify full-corpus Python caches and backend-specific native dispatch."""

from __future__ import annotations

import unittest

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import _api
from pyqcd.einsum import _kernel_kupl
from pyqcd.einsum import _specialize
from pyqcd.einsum import _wick_tasklist


def test_full_corpus_match_cache() -> None:
    equations = [task.equation for task in _wick_tasklist.TASKS]
    _specialize._match_of_normalized.cache_clear()
    _specialize._signature_of_normalized.cache_clear()
    _specialize._TOPOLOGY_ENTRY_CACHE.clear()

    assert all(_specialize.match(equation) is not None for equation in equations)
    first = _specialize._match_of_normalized.cache_info()
    assert first.currsize == len(equations)

    assert all(_specialize.match(equation) is not None for equation in equations)
    second = _specialize._match_of_normalized.cache_info()
    assert second.hits - first.hits == len(equations)
    assert len(_specialize._TOPOLOGY_ENTRY_CACHE) == len(equations)


def test_full_corpus_dedicated_plan_cache_capacity() -> None:
    cache = _kernel_kupl._DEDICATED_PLAN_CACHE
    with _kernel_kupl._DEDICATED_LOCK:
        saved = list(cache.items())
        saved_bytes = _kernel_kupl._DEDICATED_CACHE_BYTES
        cache.clear()
        _kernel_kupl._DEDICATED_CACHE_BYTES = 0

    original_planner = _kernel_kupl._cheapest_dedicated_plan
    original_flops = _kernel_kupl._plan_flops
    _kernel_kupl._cheapest_dedicated_plan = lambda *_args: (b"x", ())
    _kernel_kupl._plan_flops = lambda _plan: 1
    try:
        dtype = np.dtype(np.complex128)
        for index, task in enumerate(_wick_tasklist.TASKS):
            _kernel_kupl._dedicated_cached(
                f"eq{index}", ((1,),), dtype, "auto")
        assert len(cache) == len(_wick_tasklist.TASKS)
    finally:
        _kernel_kupl._cheapest_dedicated_plan = original_planner
        _kernel_kupl._plan_flops = original_flops
        with _kernel_kupl._DEDICATED_LOCK:
            cache.clear()
            cache.update(saved)
            _kernel_kupl._DEDICATED_CACHE_BYTES = saved_bytes


def test_backend_specific_dispatch() -> None:
    numpy_labels = [
        label for label, _module, _attempt
        in _api._specialized_kernel_attempts([np.empty((1,))])
    ]
    assert numpy_labels == ["kupl_dedicated", "kupl_topology"]

    class FakeCupyArray:
        pass

    FakeCupyArray.__module__ = "cupy.fake"
    cupy_labels = [
        label for label, _module, _attempt
        in _api._specialized_kernel_attempts([FakeCupyArray()])
    ]
    assert cupy_labels == ["cuda_dedicated", "cuda_topology"]


def test_contract_still_matches_numpy() -> None:
    task = _wick_tasklist.lookup(2584)
    terms = task.equation.split("->", 1)[0].split(",")
    rng = np.random.default_rng(20261112)
    operands = [
        rng.standard_normal((1,) * len(term))
        + 1j * rng.standard_normal((1,) * len(term))
        for term in terms
    ]
    expected = np.einsum(task.equation, *operands, optimize="greedy")
    got = pe.contract(task.equation, *operands)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-12)
    assert _kernel_kupl.LAST_INFO.get("path") == "kupl_dedicated"


TESTS = (
    test_full_corpus_match_cache,
    test_full_corpus_dedicated_plan_cache_capacity,
    test_backend_specific_dispatch,
    test_contract_still_matches_numpy,
)


def main() -> int:
    passed = 0
    failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            print(f"SKIP {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - command-line verifier
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
