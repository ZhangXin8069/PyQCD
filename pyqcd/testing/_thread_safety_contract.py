"""Concurrent use of the contraction engine.

The engine keeps module-level state (plan cache, kernel handles, the sharing
context), so every code path has to survive being called from several threads at
once. These tests are deliberately small — they check isolation and correctness,
not throughput.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import pyqcd.einsum as pe

NUCLEON_2PT = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"


def _nucleon_operands(seed: int, nvec: int = 4):
    rng = np.random.default_rng(seed)

    def draw(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    return [draw(4, 4, nvec, nvec), draw(4, 4, nvec, nvec), draw(4, 4, nvec, nvec),
            draw(4, 4), draw(4, 4), draw(4, nvec, nvec, nvec),
            draw(4, nvec, nvec, nvec), draw(4, 4)]


def test_threaded_matmul_is_correct() -> None:
    def work(seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((16, 24))
        b = rng.standard_normal((24, 12))
        for _ in range(5):
            assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(8)))


def test_threaded_specialised_path() -> None:
    def work(seed: int) -> None:
        operands = _nucleon_operands(seed)
        reference = np.einsum(NUCLEON_2PT, *operands, optimize="optimal")
        for _ in range(3):
            assert np.allclose(np.asarray(pe.contract(NUCLEON_2PT, *operands)),
                               reference)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(6)))


def test_shared_intermediates_is_thread_local() -> None:
    def work(seed: int) -> int:
        a = np.ones((4, 4))
        b = np.ones((4, 4)) * (seed + 1)
        with pe.shared_intermediates() as cache:
            pe.contract("ij,jk->ik", a, b)
            pe.contract("ij,jk->ik", a, b)
            assert isinstance(cache, dict)
            return len(cache)

    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes = list(pool.map(work, range(4)))
    assert len(set(sizes)) == 1, f"threads saw different cache sizes: {sizes}"


def test_backend_kwarg_is_per_call() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 8))
    b = rng.standard_normal((8, 8))

    def work(index: int) -> None:
        assert np.allclose(pe.contract("ij,jk->ik", a, b, backend="numpy"), a @ b)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(work, range(6)))
