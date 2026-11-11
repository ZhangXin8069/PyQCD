"""契约测试：原生 KUPL 归档的 MPI 预加载重试与一次性诊断。

现场故障：``pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so`` 在盘上，但 ``dlopen`` 报
``undefined symbol: ompi_mpi_byte``（归档链了 OpenMPI，运行时却没进进程），于是
``pyqcd.einsum`` 永久走 per-call native / 纯 Python 路径。加载器现在的契约是：

* 只在 ``OSError`` 文本含 ``undefined symbol`` 时预加载 MPI 运行时并重试一次；
* 预加载自身容错（任何异常只记录原因），每个库路径只尝试一次；
* 重试成功写进 ``library_report()``（``load_retry`` / ``mpi_preload``），不产生任何
  ``pyqcd.einsum native`` 警告，收缩结果仍与 numpy 一致；
* 重试仍失败时保持一次性 ``RuntimeWarning``，且文案看得出"已尝试预加载 MPI"。

本模块不依赖 pytest，既可由统一测试入口调用，也可直接执行：

    python -m pyqcd.testing._native_loader_contract
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import _api
from pyqcd.einsum.kupl import einsum as kupl_einsum
from pyqcd.testing._native_install_contract import _fresh_native_state

_ARCHIVE = "libpyqcdeinsum_kupl.so"
_RUNTIME = "libmpi.so.40"
_EQUATION = "bij,bjk->bik"


@contextmanager
def _fresh_loader_state():
    """Forget the archive and any preload attempt, like a new process would."""
    with patch.object(kupl_einsum, "_LIBRARY", None), \
            patch.object(kupl_einsum, "_LOAD_FAILURE", None), \
            patch.object(kupl_einsum, "_LOAD_FAILURE_KIND", None), \
            patch.object(kupl_einsum, "_LOAD_RETRY", None), \
            patch.object(kupl_einsum, "_PRELOAD_ATTEMPTED", False), \
            patch.object(kupl_einsum, "_PRELOAD_RESULT", None), \
            patch.object(kupl_einsum, "_PRELOAD_FAILURE", None), \
            patch.object(kupl_einsum, "_PRELOAD_TRIED", {}), \
            patch.object(kupl_einsum, "_PROGRAM_CACHE", {}):
        yield


@contextmanager
def _isolated_loader():
    """A present archive plus a present MPI runtime, both inside a temp dir."""
    with tempfile.TemporaryDirectory() as tmpdir, \
            _fresh_native_state(), _fresh_loader_state():
        archive = Path(tmpdir) / _ARCHIVE
        archive.write_bytes(b"")
        runtime = Path(tmpdir) / _RUNTIME
        runtime.write_bytes(b"")
        with patch.object(kupl_einsum, "library_path", lambda: archive):
            yield archive, runtime


class _DlopenStub:
    """Stands in for ``ctypes.CDLL``: the archive needs MPI before it opens.

    ``archive_failures`` is how many archive loads fail before one succeeds;
    ``None`` keeps failing, which models "the preload did not help".
    """

    def __init__(self, runtime, archive_failures=1):
        self.runtime = str(runtime)
        self.archive_failures = archive_failures
        self.archive_attempts = 0
        self.calls: list[str] = []

    def __call__(self, path, mode=None):
        del mode
        resolved = str(path)
        self.calls.append(resolved)
        if resolved == self.runtime:
            return SimpleNamespace()
        self.archive_attempts += 1
        if self.archive_failures is None or self.archive_attempts <= self.archive_failures:
            raise OSError(f"{resolved}: undefined symbol: ompi_mpi_byte")
        return SimpleNamespace()


class _NumpyExecutor:
    """Native executor stub: the stub archive exports nothing, numpy runs the plan."""

    def build_info(self):
        return {"backend": "kupl", "array_backend": "numpy"}

    def compile(self, plan, backend="auto"):
        del plan, backend
        return lambda arrays, out=None: np.einsum(_EQUATION, *arrays, out=out)

    def execute(self, plan, arrays, out=None, backend="auto"):
        del plan, backend
        return np.einsum(_EQUATION, *arrays, out=out)


def _batched_operands():
    """``bij,bjk->bik`` keeps a batch index: the shape family taking the KUPL plan."""
    lhs = np.arange(24.0).reshape(2, 3, 4)
    rhs = (np.arange(24.0).reshape(2, 4, 3) - 7.0) / 8.0
    return lhs, rhs


def _native_warnings(run):
    """Return ``(result, messages)`` for the native fallback warnings of ``run``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    messages = [
        str(item.message) for item in caught
        if issubclass(item.category, RuntimeWarning)
        and "pyqcd.einsum native" in str(item.message)
    ]
    return result, messages


def test_undefined_symbol_preloads_mpi_and_retries_once():
    """首次 dlopen 报 undefined symbol → 预加载 MPI 后重试成功，且全程无警告。"""
    lhs, rhs = _batched_operands()
    with _isolated_loader() as (archive, runtime):
        dlopen = _DlopenStub(runtime, archive_failures=1)
        with patch.object(kupl_einsum, "_dlopen", dlopen), \
                patch.object(kupl_einsum, "_mpi_candidate_paths",
                             lambda: [str(runtime)]), \
                patch.object(_api, "_native_executor", _NumpyExecutor):
            result, messages = _native_warnings(
                lambda: (kupl_einsum._load_library(),
                         pe.contract(_EQUATION, lhs, rhs))[1])
            warned = _api._NATIVE_WARNING_EMITTED
            report = kupl_einsum.library_report()

    assert dlopen.calls == [str(archive), str(runtime), str(archive)], (
        f"expected one preload and exactly one retry, got {dlopen.calls}")
    assert report["load_retry"] == f"preloaded {_RUNTIME}", report
    assert report["mpi_preload"] == f"preloaded {_RUNTIME}", report
    assert report["load_failure"] is None, report
    assert report["status"] != "library failed to load", report
    assert not messages, f"a successful retry must stay silent: {messages}"
    assert not warned, "a successful retry must not arm the fallback warning"
    np.testing.assert_allclose(
        result, np.einsum(_EQUATION, lhs, rhs), rtol=1e-12, atol=1e-12)


def test_unresolved_symbol_after_preload_warns_once_with_the_attempt():
    """预加载成功但 dlopen 仍失败 → 只发一条警告，且文案点名 undefined symbol 与预加载。"""
    lhs, rhs = _batched_operands()
    with _isolated_loader() as (archive, runtime):
        dlopen = _DlopenStub(runtime, archive_failures=None)
        with patch.object(kupl_einsum, "_dlopen", dlopen), \
                patch.object(kupl_einsum, "_mpi_candidate_paths",
                             lambda: [str(runtime)]):
            first, messages = _native_warnings(
                lambda: pe.contract(_EQUATION, lhs, rhs))
            second = pe.contract(_EQUATION, lhs, rhs)
            report = kupl_einsum.library_report()

    assert len(messages) == 1, (
        f"expected exactly one native fallback warning, got {len(messages)}: {messages}")
    text = messages[0]
    assert "library failed to load" in text, (
        f"a present-but-unloadable archive must not be reported as missing: {text}")
    assert "undefined symbol: ompi_mpi_byte" in text, text
    assert str(archive) in text, text
    assert "MPI preload attempted" in text, (
        f"the warning must show the preload was attempted: {text}")
    assert f"preloaded {_RUNTIME}" in text, text
    assert "mpi_preload_attempted=" in text, text

    assert report["status"] == "library failed to load", report
    assert report["load_retry"] is None, report
    assert report["mpi_preload"] == f"preloaded {_RUNTIME}", report
    expected = np.einsum(_EQUATION, lhs, rhs)
    np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(second, expected, rtol=1e-12, atol=1e-12)


def test_preload_swallows_loader_errors_and_keeps_the_plain_failure():
    """预加载自身炸掉只记录原因：fail-open 语义不变，报告仍分类为 load 失败。"""
    lhs, rhs = _batched_operands()
    with _isolated_loader() as (archive, runtime):
        dlopen = _DlopenStub(runtime, archive_failures=None)
        with patch.object(kupl_einsum, "_dlopen", dlopen), \
                patch.object(kupl_einsum, "_mpi_candidate_paths",
                             lambda: [str(runtime)]), \
                patch.object(kupl_einsum, "_preload_mpi_runtime",
                             side_effect=_raising_preload):
            _, messages = _native_warnings(
                lambda: pe.contract(_EQUATION, lhs, rhs))
            report = kupl_einsum.library_report()

    assert dlopen.calls == [str(archive)], dlopen.calls
    assert report["status"] == "library failed to load", report
    assert report["mpi_preload"] == (
        "no usable MPI runtime (RuntimeError: mpi preload exploded)"), report
    assert len(messages) == 1, messages
    assert "undefined symbol" in messages[0], messages[0]
    assert "mpi preload exploded" in messages[0], messages[0]


def _raising_preload():
    """A preload hook that blows up instead of returning a runtime name."""
    raise RuntimeError("mpi preload exploded")


def test_mpi_candidates_are_ordered_and_deduplicated():
    """候选顺序：已加载的 mpi4py → find_library → mpicxx → LD_LIBRARY_PATH，且去重。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime = Path(tmpdir) / _RUNTIME
        runtime.write_bytes(b"")
        imported = ModuleType("mpi4py.MPI")
        imported.__file__ = str(runtime)
        with patch.dict(sys.modules, {"mpi4py.MPI": imported}), \
                patch.object(kupl_einsum, "_find_library_candidate",
                             lambda: [_RUNTIME]), \
                patch.object(kupl_einsum, "_mpicxx_candidates",
                             lambda: [str(runtime)]), \
                patch.dict(os.environ, {"LD_LIBRARY_PATH": tmpdir}):
            candidates = kupl_einsum._mpi_candidate_paths()

    assert candidates == [str(runtime), _RUNTIME], candidates


_TESTS = (
    test_undefined_symbol_preloads_mpi_and_retries_once,
    test_unresolved_symbol_after_preload_warns_once_with_the_attempt,
    test_preload_swallows_loader_errors_and_keeps_the_plain_failure,
    test_mpi_candidates_are_ordered_and_deduplicated,
)
TESTS = _TESTS


def main():
    passed = 0
    for test in _TESTS:
        test()
        print(f'PASS {test.__name__}')
        passed += 1
    print(f'{passed} passed, 0 failed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
