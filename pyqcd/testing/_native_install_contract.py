"""契约测试：原生 KUPL 库缺失时的诊断信息与 fail-open 行为。

``pip install pyqcd`` 在缺少 ``pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so`` 的机器上
只发一条埋在输出里的 ``RuntimeWarning``，调用方无法判断"结果是否可信""怎么修"。
本模块把这条诊断钉死成契约：

* 库路径解析必须落在安装目录的 ``pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so``；
* 警告文案必须同时给出缺失库的绝对路径、``build_info()`` 的 ``backend``、失败类别
  （缺库 vs 编译器上下文不可用）和一句可执行的修复建议；
* 同一个进程只发一次（沿用 ``_warn_native_fallback_once`` 的去重语义）；
* 原生库缺席只影响速度：收缩结果仍与 ``numpy`` 参考一致（fail-open，绝不抛异常）。

本模块不依赖 pytest，既可由统一测试入口调用，也可直接执行：

    python -m pyqcd.testing._native_install_contract
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import warnings

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import _api
from pyqcd.einsum.kupl import einsum as kupl_einsum
from pyqcd.einsum.kupl.einsum import NativeUnavailable

_ARCHIVE = "libpyqcdeinsum_kupl.so"
_EQUATION = "bij,bjk->bik"


@contextmanager
def _fresh_native_state():
    """Reset every process-wide native cache the fallback warning depends on."""
    with patch.object(_api, "_NATIVE_WARNING_EMITTED", False), \
            patch.object(_api, "_NATIVE_CONTEXT_UNAVAILABLE", set()), \
            patch.object(_api, "_NATIVE_DISABLED", set()), \
            patch.object(_api, "_NATIVE_CONTEXT_CACHE", OrderedDict()), \
            patch.object(_api, "_NATIVE_PLAN_CACHE", OrderedDict()), \
            patch.object(_api, "_CONTRACT_PROGRAM_CACHE", OrderedDict()), \
            patch.object(_api, "_CONTRACT_PLAN_CACHE", OrderedDict()), \
            patch.object(_api, "_LAST_CONTRACT_SIGNATURE", None), \
            patch.object(_api, "_LAST_CONTRACT_PROGRAM", None):
        yield


@contextmanager
def _missing_library(root: Path):
    """Point both archive candidates at absent paths, like a wheel without the build."""
    missing = root / "site-packages" / "pyqcd" / "einsum" / "kupl" / _ARCHIVE
    with _fresh_native_state(), \
            patch.object(kupl_einsum, "_PACKAGED_LIB", missing), \
            patch.object(
                kupl_einsum, "_REPO_LIB",
                root / "cpp" / "kupl" / "einsum" / _ARCHIVE), \
            patch.object(kupl_einsum, "_LIBRARY", None), \
            patch.object(kupl_einsum, "_LOAD_FAILURE", None), \
            patch.object(kupl_einsum, "_LOAD_FAILURE_KIND", None), \
            patch.object(kupl_einsum, "_PROGRAM_CACHE", {}), \
            patch.dict(os.environ, {"PYQCD_EINSUM_LIB": ""}):
        yield missing


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


class _FailingCompiler:
    """Native executor stub: the archive exists, the reusable context does not."""

    def build_info(self):
        return {"backend": "kupl", "array_backend": "numpy"}

    def compile(self, plan, backend="auto"):
        raise NativeUnavailable("native einsum compile failed with status 7")

    def execute(self, *args, **kwargs):
        raise NativeUnavailable("native einsum execution failed with status 7")


def test_library_path_resolves_the_packaged_kupl_archive():
    """安装后必须解析到 site-packages/pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so。"""
    expected = Path(kupl_einsum.__file__).resolve().with_name(_ARCHIVE)
    with patch.object(
            kupl_einsum, "_REPO_LIB",
            Path("/__pyqcd_contract_missing__") / _ARCHIVE), \
            patch.dict(os.environ, {"PYQCD_EINSUM_LIB": ""}):
        resolved = kupl_einsum.library_path()
        report = kupl_einsum.library_report()

    assert resolved.is_absolute(), f"library_path() is not absolute: {resolved}"
    assert resolved == expected, (
        f"library_path() resolved {resolved}, expected the packaged {expected}")
    assert resolved.parts[-4:] == ("pyqcd", "einsum", "kupl", _ARCHIVE), (
        f"library_path() is outside the package layout: {resolved}")
    assert report["path"] == str(expected), report
    assert report["present"] is expected.is_file(), report


def test_missing_library_warning_is_actionable_and_emitted_once():
    """缺库时必须点名绝对路径、build_info() 后端、失败类别与修复建议，且只发一次。"""
    lhs, rhs = _batched_operands()
    with tempfile.TemporaryDirectory() as tmpdir, \
            _missing_library(Path(tmpdir)) as missing:
        reported_backend = pe.build_info()["backend"]

        def run():
            return (pe.contract(_EQUATION, lhs, rhs),
                    pe.contract(_EQUATION, lhs, rhs))

        _, messages = _native_warnings(run)

    assert len(messages) == 1, (
        f"expected exactly one native fallback warning, got {len(messages)}: {messages}")
    text = messages[0]
    assert "library missing" in text, (
        f"the warning must name the missing-archive cause: {text}")
    assert os.path.abspath(str(missing)) in text, (
        f"the warning must carry the missing archive path: {text}")
    assert "present=False" in text, text
    assert f"backend={reported_backend}" in text, (
        f"the warning must report build_info()['backend']={reported_backend}: {text}")
    assert "reinstall pyqcd" in text, (
        f"the warning must suggest reinstalling the shipped archive: {text}")
    assert "python -m pyqcd" in text, (
        f"the warning must name the re-check command: {text}")


def test_missing_library_keeps_contraction_results_correct():
    """原生库缺席只降级速度：结果仍与 numpy 参考一致，绝不抛异常。"""
    lhs, rhs = _batched_operands()
    with tempfile.TemporaryDirectory() as tmpdir, _missing_library(Path(tmpdir)):
        result, messages = _native_warnings(
            lambda: pe.contract(_EQUATION, lhs, rhs))

    assert messages, "the degraded path must still be announced once"
    np.testing.assert_allclose(
        result, np.einsum(_EQUATION, lhs, rhs), rtol=1e-12, atol=1e-12)


def test_compiler_context_failure_is_distinguished_from_a_missing_library():
    """库在盘上但拿不到可复用编译上下文时，理由不能写成缺库。"""
    lhs, rhs = _batched_operands()
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / _ARCHIVE
        archive.write_bytes(b"")
        with _fresh_native_state(), \
                patch.object(kupl_einsum, "library_path", lambda: archive), \
                patch.object(kupl_einsum, "_LIBRARY", None), \
                patch.object(kupl_einsum, "_LOAD_FAILURE", None), \
                patch.object(kupl_einsum, "_LOAD_FAILURE_KIND", None), \
                patch.object(_api, "_native_executor", lambda: _FailingCompiler()), \
                patch.dict(os.environ, {"PYQCD_EINSUM_LIB": ""}):
            _, messages = _native_warnings(
                lambda: pe.contract(_EQUATION, lhs, rhs))

    assert len(messages) == 1, messages
    text = messages[0]
    assert "compiler context unavailable" in text, (
        f"a present archive must not be reported as missing: {text}")
    assert "library missing" not in text, text
    assert str(archive) in text, text
    assert "backend=kupl" in text, text


_TESTS = (
    test_library_path_resolves_the_packaged_kupl_archive,
    test_missing_library_warning_is_actionable_and_emitted_once,
    test_missing_library_keeps_contraction_results_correct,
    test_compiler_context_failure_is_distinguished_from_a_missing_library,
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
