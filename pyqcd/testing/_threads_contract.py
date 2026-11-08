"""Thread-configuration contract.

The engine picks OpenMP/BLAS thread defaults on import (see
``pyqcd/einsum/_threads.py``) because leaving both runtimes at "all logical
cores" is measurably slower. These tests pin the precedence rules so a refactor
cannot silently start overriding an explicit user setting.
"""

from __future__ import annotations

import contextlib
import os

from pyqcd.einsum import _threads


@contextlib.contextmanager
def _env(**values: str | None):
    """Temporarily set/clear environment variables and restore them afterwards."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_physical_cores_is_positive() -> None:
    assert _threads.physical_cores() >= 1


def test_auto_defaults_are_bounded() -> None:
    with _env(OMP_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None,
              PYQCD_EINSUM_OMP_THREADS=None, PYQCD_EINSUM_BLAS_THREADS=None,
              PYQCD_EINSUM_THREADS=None):
        for profile in ("kernel", "generic"):
            omp, blas, source = _threads.resolve(profile)
            assert 1 <= omp <= 8 and 1 <= blas <= 4
            assert profile in source


def test_profiles_differ_and_are_sane() -> None:
    table = _threads.profiles()
    assert set(table) == {"kernel", "generic"}
    (kernel_omp, kernel_blas), (generic_omp, generic_blas) = (
        table["kernel"], table["generic"])
    # 内核靠 OpenMP，通用路径靠 BLAS —— 两档必须真的不同，否则切换没有意义。
    assert kernel_omp >= generic_omp
    assert generic_blas >= kernel_blas


def test_profile_switching_is_observable() -> None:
    import io
    import contextlib

    import numpy as np
    import pyqcd.einsum as pe

    with _env(OMP_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None,
              PYQCD_EINSUM_OMP_THREADS=None, PYQCD_EINSUM_BLAS_THREADS=None):
        _threads.reset()
        expr = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"
        rng = np.random.default_rng(0)

        def draw(*shape):
            return (rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)).astype(np.complex128)

        n = 8
        nucleon = [draw(4, 4, n, n), draw(4, 4, n, n), draw(4, 4, n, n),
                   draw(4, 4), draw(4, 4), draw(4, n, n, n), draw(4, n, n, n),
                   draw(4, 4)]
        with contextlib.redirect_stdout(io.StringIO()):
            pe.contract(expr, *nucleon)
        assert _threads.current()["profile"] == "kernel"
        a, b, c = (rng.standard_normal((64, 64)) for _ in range(3))
        pe.contract("ij,jk,kl->il", a, b, c)
        assert _threads.current()["profile"] == "generic"
        with contextlib.redirect_stdout(io.StringIO()):
            pe.contract(expr, *nucleon)
        assert _threads.current()["profile"] == "kernel"


def test_disable_switch() -> None:
    with _env(PYQCD_EINSUM_THREADS="off"):
        assert _threads.resolve()[:2] == (None, None)


def test_standard_env_wins() -> None:
    with _env(OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="5"):
        omp, blas, source = _threads.resolve()
    assert (omp, blas) == (3, 5)
    assert source == "standard environment"


def test_project_env_is_second_priority() -> None:
    with _env(OMP_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None,
              PYQCD_EINSUM_OMP_THREADS="2", PYQCD_EINSUM_BLAS_THREADS="3"):
        omp, blas, source = _threads.resolve()
    assert (omp, blas) == (2, 3)
    assert source in ({"PYQCD_EINSUM_OMP_THREADS", "PYQCD_EINSUM_BLAS_THREADS"})


def test_mpi_ranks_divide_the_budget() -> None:
    cores = _threads.physical_cores()
    with _env(OMP_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None,
              PYQCD_EINSUM_OMP_THREADS=None, PYQCD_EINSUM_BLAS_THREADS=None,
              PYQCD_EINSUM_THREADS=None, OMPI_COMM_WORLD_SIZE="2"):
        omp, _blas, source = _threads.resolve()
    assert omp == min(max(cores // 2, 1), 8)
    assert "2 ranks" in source


def test_apply_is_idempotent() -> None:
    first = _threads.apply("kernel")
    second = _threads.apply("kernel")
    assert first["omp"] == second["omp"] and first["blas"] == second["blas"]


def test_thread_info_is_exposed() -> None:
    import pyqcd.einsum as pe

    info = pe.thread_info()
    assert set(("omp", "blas", "source", "profile")) <= set(info)
    assert set(pe.thread_profiles()) == {"kernel", "generic"}
