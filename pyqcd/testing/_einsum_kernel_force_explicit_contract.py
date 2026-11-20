"""契约：``kernel_force`` 是**显式强制**属性——没有 ``auto``，缺省 ``generic``。

2026-11-18 决策：强制值不该带一个"未指定 → 自动分派"的隐藏档。自动选路交给 ``--tune``
的实测结论（tune 把胜出的一组写进配置表），于是输入侧只剩四个**具体**标签
（``generic`` / ``dedicated`` / ``topology`` / ``backend``）。``resolve()`` /
``forced_labels()`` / ``generic_only()`` 也随简化：没有 ``None`` 这种"不强制"的三态。

本模块只钉**取值解析**这一层（不调 ``contract()``，因此不依赖 ``_api`` 的接线）：

1. 显式 ``kernel_force="auto"`` 抛 ``ValueError``，正文点名 **tune** 并列出四个合法值；
2. ``None``（以及空串）解析为 ``("generic",)``；
3. CLI 不给 ``--kernel-force`` 时 argparse 是 ``None``（未指定），解析与报告口径仍是
   ``generic``——``None`` 不能被填成 ``generic``，否则 ``_api`` 会把每次默认调用都
   当成"显式 generic"，已 tune 的 ``kernel=backend`` 行永远套不上（2026-11-18）；
4. 环境变量 ``PYQCD_EINSUM_KERNEL_FORCE=auto`` 同样抛错（与显式参数一条消息）；
5. 四个合法标签都能**单独**解析；逗号链式（"依次尝试 A 再试 B"）已取消，一律抛
   ``ValueError`` 并说明"只接受单个值"。

可直接运行::

    python -m pyqcd.testing._einsum_kernel_force_explicit_contract
"""
from __future__ import annotations

import contextlib
import os
import unittest
from unittest.mock import patch

from ..einsum import _kernel_force
from ..einsum import test as cli

#: 合法取值（与 ``ALLOWED_LABELS`` 同源，避免测试里写第二份集合）。
LABELS = tuple(sorted(_kernel_force.ALLOWED_LABELS))

#: 被删除的自动分派取值：写成转义拼写，免得本文件成为第三处"它还合法"的线索。
REMOVED_AUTO = "\x61uto"


@contextlib.contextmanager
def _no_force_env():
    """临时清掉 ``PYQCD_EINSUM_KERNEL_FORCE``；退出时逐字恢复。"""
    saved = os.environ.pop(_kernel_force.KERNEL_FORCE_ENV, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[_kernel_force.KERNEL_FORCE_ENV] = saved


def _expect_value_error(call, what: str) -> str:
    """要求 ``call`` 抛 ``ValueError`` 且正文含 ``tune`` 与四个合法值；返回正文。"""
    try:
        call()
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError(f"{what} 必须抛 ValueError")
    assert "tune" in message, (what, message)
    for label in LABELS:
        assert label in message, (what, label, message)
    return message


def _expect_rejected_multiple(spec: str, what: str) -> str:
    """要求 ``parse(spec)`` 因"写了多个值"抛错，且正文说明规则并列出合法集合。"""
    try:
        _kernel_force.parse(spec)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError(f"{what} 必须抛 ValueError（逗号链式已取消）")
    assert "单个值" in message, (what, message)
    for label in LABELS:
        assert label in message, (what, label, message)
    return message


# ── 1. auto 是非法取值 ──────────────────────────────────────────────────────
def test_auto_is_rejected():
    """显式 ``kernel_force="auto"``：``ValueError`` + 点名 tune + 列出四个合法值。"""
    with _no_force_env():
        message = _expect_value_error(
            lambda: _kernel_force.resolve(REMOVED_AUTO), "kernel_force='auto'")
    # 大小写没有豁免：它不是"没写"，而是一个已删除的取值。
    with _no_force_env():
        _expect_value_error(lambda: _kernel_force.parse(REMOVED_AUTO.upper()),
                            "kernel_force='AUTO'")
    # 混在逗号链里先撞上"只接受单个值"那道门：这同样是拒绝，只是正文说的是
    # 单元值规则（并列出合法集合），不该要求它同时点名 tune。
    with _no_force_env():
        _expect_rejected_multiple(f"{REMOVED_AUTO},topology",
                                  "kernel_force='auto,topology'")
    assert REMOVED_AUTO not in _kernel_force.ALLOWED_LABELS, message


# ── 2. None → generic ───────────────────────────────────────────────────────
def test_default_is_generic():
    """``None`` 与空串都解析为 ``("generic",)``；不再有"未指定"的哨兵。"""
    generic = _kernel_force.GENERIC_LABEL
    assert _kernel_force.DEFAULT_LABEL == generic
    with _no_force_env():
        assert _kernel_force.parse(None) == (generic,)
        assert _kernel_force.resolve(None) == (generic,)
        assert _kernel_force.parse("") == (generic,)
        assert _kernel_force.parse("   ") == (generic,)
        # 语义简化：generic 不产生专用内核尝试 → forced_labels() 是空元组，
        # 而不是"没强制"的 None；generic_only() 直接为 True。
        assert _kernel_force.forced_labels(None) == ()
        assert _kernel_force.generic_only(None) is True
        assert _kernel_force.forced_labels() == ()
        assert _kernel_force.generic_only() is True


# ── 3. CLI 缺省 = 未指定（解析/报告口径仍是 generic）──────────────────────────
def test_cli_defaults_to_generic():
    """不给 ``--kernel-force`` 时 argparse 是 ``None``，解析结果与报告仍是 ``generic``。"""
    generic = cli.GENERIC_LABEL
    assert cli.DEFAULT_KERNEL == generic, cli.DEFAULT_KERNEL
    parser = cli.build_parser(language="en")
    args = parser.parse_args(["--equation", "ij,jk->ik"])
    # ``None`` = 未指定。**不能**填成 generic：`_api` 用 `kernel_force is not None`
    # 判断显式性，填上它会把每一次默认调用都判成"用户点名了 generic"。
    assert args.kernel_force is None, args.kernel_force
    with _no_force_env():
        # 报告/解析口径（最终回落值）仍是 generic。
        assert cli.kernel_labels(args.kernel_force) == (generic,)
        assert cli.kernel_requested(args.kernel_force) == generic
        # 完全省略该选项时，底层 `_kernel_force.resolve(None)` 同样落到 generic。
        assert _kernel_force.resolve(None) == (generic,)


# ── 4. 环境变量 auto 也报错 ─────────────────────────────────────────────────
def test_environment_auto_is_rejected():
    """``PYQCD_EINSUM_KERNEL_FORCE=auto``：与显式参数同一条错误消息。"""
    with patch.dict(os.environ, {_kernel_force.KERNEL_FORCE_ENV: REMOVED_AUTO}):
        _expect_value_error(_kernel_force.resolve, "环境变量 = auto")
        _expect_value_error(
            lambda: _kernel_force.resolve(None),
            "环境变量 = auto（resolve(None)）")


# ── 5. 五个合法标签（单值）──────────────────────────────────────────────────
def test_all_five_labels_and_order():
    """五个标签逐个可解析；逗号链式一律拒绝（一次只准写一个标签）。"""
    assert set(LABELS) == {"backend", "dedicated", "generic", "original",
                           "topology"}, LABELS
    with _no_force_env():
        for label in LABELS:
            assert _kernel_force.parse(label) == (label,), label
            assert _kernel_force.resolve(label) == (label,), label
            assert _kernel_force.parse(f"  {label.upper()}  ") == (label,), label
        # 逗号链式（"按书写次序依次尝试"）已取消：同一调用里两条互相竞争的路线，
        # 报告里看不出谁说了算。generic / backend 与其它标签更是必须早失败。
        for bad in ("topology,dedicated", "dedicated,topology",
                    "topology,topology", "generic,dedicated", "backend,topology",
                    "original,backend"):
            _expect_rejected_multiple(bad, repr(bad))
        # 具体标签原样交给 _api（generic 不产生 attempt 列表；backend / original
        # 都不进专用分派，但仍然是"强制"取值，所以在 forced_labels 里原样返回）。
        assert _kernel_force.forced_labels("topology") == ("topology",)
        assert _kernel_force.forced_labels("dedicated") == ("dedicated",)
        assert _kernel_force.forced_labels("generic") == ()
        assert _kernel_force.forced_labels("backend") == (
            _kernel_force.BACKEND_LABEL,)
        assert _kernel_force.forced_labels("original") == (
            _kernel_force.ORIGINAL_LABEL,)
        assert _kernel_force.generic_only("topology") is False
        assert _kernel_force.generic_only("backend") is False
        assert _kernel_force.generic_only("original") is False
        assert _kernel_force.generic_only("generic") is True
        # ``original`` 把本库的 optimize 叶子转成 opt_einsum 认的实参：连字符与
        # ``False`` 是**实测**结论（下划线与字符串 'none' 都会 KeyError）。
        assert _kernel_force.opt_einsum_path_name("branch_1") == "branch-1"
        assert _kernel_force.opt_einsum_path_name("branch_2") == "branch-2"
        assert _kernel_force.opt_einsum_path_name("branch_all") == "branch-all"
        assert _kernel_force.opt_einsum_path_name("none") is False
        assert _kernel_force.opt_einsum_path_name("greedy") == "greedy"
        assert _kernel_force.opt_einsum_path_name("optimal") == "optimal"
        assert _kernel_force.opt_einsum_path_name("dp") == "dp"
        assert _kernel_force.opt_einsum_path_name(None) == "greedy"
        assert _kernel_force.opt_einsum_path_name("bogus") is None


TESTS = (
    test_auto_is_rejected,
    test_default_is_generic,
    test_cli_defaults_to_generic,
    test_environment_auto_is_rejected,
    test_all_five_labels_and_order,
)


def test_einsum_kernel_force_explicit_contracts():
    """注册入口：逐条运行 ``TESTS``；全跳过时也 SkipTest（不许静默通过）。"""
    passed = skipped = 0
    failures = []
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条显式强制内核契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"explicit kernel-force contracts did not run ({skipped} skipped)")


def main() -> int:
    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", flush=True)
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
