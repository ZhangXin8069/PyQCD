"""契约：回落必须留痕 —— 四种"静默失败"在 2026-11-16 修复后的可观测性。

背景：这四处过去都会**静默**回落到通用路径，或只报"进程内第一次"故障：
`_shape_paths.try_execute` 吞异常、`_compile_native_expression` 吞异常、
`_warn_native_fallback_once` / `_warn_specialized_kernel_once` 用单个布尔去重。
后果不是"少说一句"，而是"把另一个故障说成第一个故障"——长进程里照着第一条去修，
永远修不好。本契约按**粒度**钉住修复：
  * 同一类失败只报一次（不刷屏）；
  * **不同类**失败各报一次（不被吞）；
  * 形状特化失败要写 `LAST_FAILURE`，让"在表里却回落"与"没注册"可区分。
"""

from __future__ import annotations

import contextlib
import io
import warnings
from unittest.mock import patch

from pyqcd.einsum import _api
from pyqcd.einsum import _shape_paths


def _reset() -> None:
    _api._NATIVE_WARNED_REASONS.clear()
    _api._SPECIALIZED_KERNEL_WARNED.clear()
    _shape_paths.LAST_FAILURE.clear()
    _shape_paths._execute_failed_announced.clear()


def test_native_fallback_warns_once_per_reason() -> None:
    """不同 reason 各报一次；同一 reason 重复调用不再报。"""
    _reset()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _api._warn_native_fallback_once("first failure", reason="lib-missing")
            _api._warn_native_fallback_once("first failure", reason="lib-missing")
            _api._warn_native_fallback_once("second failure", reason="dtype-unsupported")
            _api._warn_native_fallback_once("third failure", reason="lib-missing")
        messages = [str(item.message) for item in caught]
        assert messages == ["first failure", "second failure"], (
            f"期望每个不同 reason 各一次、同 reason 去重，实际 {messages}")
    finally:
        _reset()


def test_specialized_kernel_warns_once_per_distinct_failure() -> None:
    """不同异常类型各报一次；同类型同文本不再报。"""
    _reset()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _api._warn_specialized_kernel_once(ValueError("boom"))
            _api._warn_specialized_kernel_once(ValueError("boom"))
            _api._warn_specialized_kernel_once(TypeError("other"))
        assert len(caught) == 2, f"期望 2 条告警，实际 {len(caught)}"
        assert "ValueError" in str(caught[0].message)
        assert "TypeError" in str(caught[1].message)
    finally:
        _reset()


def test_shape_path_failure_is_recorded_and_announced_once() -> None:
    """形状特化失败要写 LAST_FAILURE，并按 (形态, 异常类型) 只提示一次。"""
    _reset()

    class _Boom(RuntimeError):
        pass

    fake = {"name": "fake-shape", "kernels": ("kupl_dedicated",), "note": "-",
            "run": lambda _operands: (_ for _ in ()).throw(_Boom("shape crashed"))}
    buffer = io.StringIO()
    try:
        with patch.object(_shape_paths, "lookup", return_value=fake), \
                patch.object(_shape_paths, "_notice_enabled", return_value=True), \
                contextlib.redirect_stdout(buffer):
            assert _shape_paths.try_execute("x", []) is None
            first = buffer.getvalue()
            assert _shape_paths.try_execute("x", []) is None
            second = buffer.getvalue()
        assert _shape_paths.LAST_FAILURE.get("name") == "fake-shape"
        assert "_Boom" in _shape_paths.LAST_FAILURE.get("reason", "")
        assert "shape crashed" in _shape_paths.LAST_FAILURE.get("reason", "")
        assert first.count("形状特化失败") == 1, "首次失败必须提示一行"
        assert second.count("形状特化失败") == 1, "同形态同类失败不得重复提示"
    finally:
        _reset()


def test_native_expression_compile_failure_leaves_a_warning() -> None:
    """`_compile_native_expression` 过去完全无声；现在必须留下可定位的告警。"""
    _reset()
    try:
        def _explode(*_args, **_kwargs):
            raise RuntimeError("encode exploded")

        with patch.object(_api, "_native_compatible", return_value=True), \
                patch.object(_api, "_native_backend", return_value="kupl"), \
                patch("pyqcd.einsum._native_plan.encode_execution_plan",
                      side_effect=_explode):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _api._compile_native_expression([], [], "kupl")
        assert result is None, "编译失败仍须回落（返回 None）"
        messages = [str(item.message) for item in caught]
        assert any("native expression compilation failed" in text
                   for text in messages), f"必须告警，实际 {messages}"
        assert any("encode exploded" in text for text in messages), (
            f"告警必须带上原因，实际 {messages}")
    finally:
        _reset()


TESTS = (
    test_native_fallback_warns_once_per_reason,
    test_specialized_kernel_warns_once_per_distinct_failure,
    test_shape_path_failure_is_recorded_and_announced_once,
    test_native_expression_compile_failure_leaves_a_warning,
)


def main() -> int:
    passed = failed = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
