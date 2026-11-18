"""契约：``optimize_topology_v2`` 只决定 **v2 回落**的计划，与 ``optimize`` 分家。

``topology`` 内部有两条路线（见 ``_kernel_kupl._TOPOLOGY_ROUTE_PATHS``）：

* 命中编译期步表 → **v3**，顺序固化在 ``topology_steps.h`` 里，任何 ``optimize``
  都不参与；
* 未命中 → 回落 **v2 通用执行器**，此时才需要一份 Python 生成的计划。

于是"用哪个优化器生成 v2 的计划"必须是一个**独立**参数：``optimize`` 已经另有
语义（其余内核的收缩序，对 ``topology`` 恒为"不规划"），拿它兼职会让同一份调用
在两条路线之间摇摆——v3 下看不出差别，v2 下换的是计划本身。本模块钉四件事：

1. **签名**：``try_topology_apply``（KUPL 与 CUDA 两侧）接受 ``optimize``，默认
   ``"greedy"``；``contract`` 接受 ``optimize_topology_v2``，默认也是 ``"greedy"``。
2. **传的是 ``optimize_topology_v2``，不是 ``optimize``**：包一层捕获实际实参，
   断言下发到内核的额外参数等于前者、且与用户给的 ``optimize`` 无关。
3. **非法值当场报错**，正文列出 7 个合法叶子（静默退回 ``greedy`` 会让"我要
   optimal"变成一句无声的空话）。
4. **v3 命中时两个参数都不影响数值**（与 ``np.einsum`` rel ≤ 1e-12）：顺序来自
   步表，任何取值都只是"没被读到"。

``none`` 的边界（有意为之，不是遗漏）
------------------------------------
``none`` 是 7 个合法叶子之一，**参数校验必须接受它**；但它表示"不规划"——一份
单步 N 元收缩，而原生执行器（v2 与 v3 共用）的单步模型是 arity ≤ 2。因此语料方程
（arity ≥ 6）上 ``none`` 的计划根本编码不出来，内核会 fail closed 并给出原因，
**绝不自作主张换成某个次序**。第 4 条因此断言：能编码的叶子数值必须一致；
``none`` 则必须"被接受但被明确拒绝"，两条都不能静默改数。

可直接运行::

    python -B -m pyqcd.testing._einsum_topology_v2_optimize_contract
"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import unittest
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _api, _kernel_cuda, _kernel_kupl, _kernel_force
from ..einsum import contract as pe_contract
from ..einsum._profile_cache import OPTIMIZE_LEAVES

TOLERANCE = 1e-12

#: 语料方程（arity 7，命中编译期 v3 步表）。形状之间必须自洽：同一标签只能有一个
#: 轴长，否则 numpy 会先报广播错误，测试根本走不到内核。
EQUATION = "ambn,codp,egfh,ce,mo,bdf,nph->ag"
SHAPES = ((2, 3, 4, 5), (6, 7, 8, 5), (6, 2, 8, 9), (6, 6), (3, 7), (4, 8, 8), (5, 5, 9))

#: 能编码成 arity ≤ 2 步表的叶子（``none`` 是单步 N 元收缩，见模块 docstring）。
PLANNABLE_LEAVES = tuple(leaf for leaf in OPTIMIZE_LEAVES if leaf != "none")

#: 第 4 条里与 ``optimize_topology_v2`` 正交扫描的 ``optimize`` 取值：``none`` 也要
#: 覆盖——它在 topology 上表示"顺序由内核决定"，正是最容易串味的那一格。
OPTIMIZE_SWEEP = ("auto", "greedy", "optimal", "none")


def _operands() -> list[np.ndarray]:
    rng = np.random.default_rng(20261118)
    return [np.ascontiguousarray(rng.standard_normal(shape)
                                 + 1j * rng.standard_normal(shape))
            for shape in SHAPES]


@contextlib.contextmanager
def _quiet():
    """静音 notice 与告警，并关掉画像（避免读到仓库根的 ``./.cache``）。"""
    saved = {name: os.environ.get(name)
             for name in ("PYQCD_EINSUM_NOTICE", "PYQCD_EINSUM_PROFILE")}
    os.environ["PYQCD_EINSUM_NOTICE"] = "0"
    os.environ["PYQCD_EINSUM_PROFILE"] = "0"
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _rel(actual: Any, expected: np.ndarray) -> float:
    scale = float(np.max(np.abs(expected))) or 1.0
    return float(np.max(np.abs(np.asarray(actual) - expected))) / scale


# ── 1. 签名与默认值 ─────────────────────────────────────────────────────────

def test_topology_apply_takes_optimize_defaulting_to_greedy() -> None:
    """两个后端的 ``try_topology_apply`` 都接受 ``optimize``，默认 ``"greedy"``。"""
    for module in (_kernel_kupl, _kernel_cuda):
        parameters = inspect.signature(module.try_topology_apply).parameters
        assert "optimize" in parameters, (
            f"{module.__name__}.try_topology_apply 缺 optimize 参数", list(parameters))
        default = parameters["optimize"].default
        assert default == "greedy", (
            f"{module.__name__} 的 optimize 默认值是 {default!r}，应为 'greedy'")
        assert parameters["optimize"].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{module.__name__} 的 optimize 必须是关键字参数（前面的 out/offsets "
            f"是位置参数，混在一起会改掉既有调用点的含义）")

    contract_parameters = inspect.signature(pe_contract).parameters
    assert "optimize_topology_v2" in contract_parameters, (
        "contract 缺 optimize_topology_v2 参数", list(contract_parameters))
    assert contract_parameters["optimize_topology_v2"].default == "greedy", (
        "contract 的 optimize_topology_v2 默认值应为 'greedy'，得到 "
        f"{contract_parameters['optimize_topology_v2'].default!r}")


# ── 2. 下发的是 optimize_topology_v2，不是 optimize ─────────────────────────

def test_the_value_handed_to_the_kernel_is_optimize_topology_v2() -> None:
    """捕获 ``try_topology_apply`` 的实际实参：只能是 ``optimize_topology_v2``。

    刻意让两个参数取**不同**的值（``optimize="dp"`` / ``optimize_topology_v2=
    "optimal"``）：若接线时误传了 ``optimize``，这里就会捕获到 ``"dp"``。
    """
    original = _kernel_kupl.try_topology_apply
    captured: list[dict[str, Any]] = []

    def spy(subscripts, operands, out=None, offsets=None, **kwargs):
        captured.append(dict(kwargs))
        return original(subscripts, operands, out=out, offsets=offsets, **kwargs)

    operands = _operands()
    expected = np.einsum(EQUATION, *operands)
    with _quiet(), patch.object(_kernel_kupl, "try_topology_apply", spy):
        result = pe_contract(EQUATION, *operands, kernel_force="topology",
                             optimize="dp", optimize_topology_v2="optimal")

    assert captured, "拓扑入口一次都没被调用：这条契约没有观察到任何实参"
    assert captured[-1].get("optimize") == "optimal", (
        "下发给 topology 的额外参数必须是 optimize_topology_v2='optimal'，"
        f"实际捕获 {captured[-1]!r}（若为 'dp' 说明接线用了 optimize）")
    assert _rel(result, expected) <= TOLERANCE, "换 optimize_topology_v2 改了数值"


# ── 3. 非法取值必须早失败，并列出合法叶子 ───────────────────────────────────

def test_invalid_optimize_topology_v2_is_rejected_with_the_legal_values() -> None:
    """非法拼写（含从规划器删掉的 ``auto``）抛 ``ValueError`` 并列出 7 个叶子。

    ``None`` 也在内：签名缺省已经是 ``greedy``，"没给"只有"不写"这一种写法；
    若把 ``None`` 归一成 ``str(None)``，它恰好命中叶子 ``"none"``，同一次 v2 调用
    就会悄悄换掉整份计划。
    """
    operands = _operands()
    for value in ("auto", "bogus", "", "GREEDY_2", None, 7):
        with _quiet():
            try:
                pe_contract(EQUATION, *operands, optimize_topology_v2=value)
            except ValueError as exc:
                message = str(exc)
            else:                                     # pragma: no cover - 失败路径
                raise AssertionError(
                    f"optimize_topology_v2={value!r} 必须抛 ValueError，"
                    "静默退回 greedy 会把'我要某个次序'变成一句无声的空话")
        for leaf in OPTIMIZE_LEAVES:
            assert leaf in message, (
                f"报错正文要列出合法值，缺 {leaf!r}：{message}")
        assert "topology" in message, (
            f"报错正文要说明这个参数只管 topology 的 v2 回落：{message}")


# ── 4. v3 命中时两个参数都不影响数值 ────────────────────────────────────────

def test_v3_numbers_ignore_both_optimize_and_optimize_topology_v2() -> None:
    """语料方程（v3 命中）上任意取值都必须给出同一个答案。

    这条不只是"数值一致"：每一步都断言 ``LAST_INFO['path'] == 'kupl_topology'``，
    否则内核根本没跑 v3，"参数不影响数值"就成了废话。``none`` 走单独一条断言：
    它**被接受**（不是校验错误），但单步 N 元计划编码不出来，必须 fail closed
    并说明原因——静默换一个次序才是真正要防的事故。
    """
    operands = _operands()
    expected = np.einsum(EQUATION, *operands)

    with _quiet():
        for optimize in OPTIMIZE_SWEEP:
            assert isinstance(optimize, str)
            for leaf in PLANNABLE_LEAVES:
                _kernel_kupl.LAST_INFO.clear()
                result = pe_contract(EQUATION, *operands, kernel_force="topology",
                                     optimize=optimize, optimize_topology_v2=leaf)
                path = _kernel_kupl.LAST_INFO.get("path")
                assert path == "kupl_topology", (
                    f"optimize={optimize!r} optimize_topology_v2={leaf!r} 没走 v3 "
                    f"步表（path={path!r}）：这条用例的前提不成立")
                assert _rel(result, expected) <= TOLERANCE, (
                    f"optimize={optimize!r} optimize_topology_v2={leaf!r} 改了数值")

        # ``none``：参数合法（走到这里说明没抛 ValueError），计划编码不出来。
        _kernel_kupl.LAST_INFO.clear()
        try:
            pe_contract(EQUATION, *operands, kernel_force="topology",
                        optimize_topology_v2="none")
        except _kernel_force.KernelUnavailableError as exc:
            message = str(exc)
        else:                                         # pragma: no cover - 失败路径
            raise AssertionError(
                "optimize_topology_v2='none' 在 arity>2 的语料方程上应 fail closed"
                "（单步 N 元收缩编码不出 ≤2 元的步表），而不是悄悄换一个次序")
        assert "arity" in message, (
            f"拒绝原因要说明计划模型（arity ≤ 2）：{message}")


TESTS = (
    test_topology_apply_takes_optimize_defaulting_to_greedy,
    test_the_value_handed_to_the_kernel_is_optimize_topology_v2,
    test_invalid_optimize_topology_v2_is_rejected_with_the_legal_values,
    test_v3_numbers_ignore_both_optimize_and_optimize_topology_v2,
)


def test_einsum_topology_v2_optimize_contracts() -> None:
    """套件入口：四条全过，否则汇总失败。"""
    failures: list[str] = []
    passed = skipped = 0
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
            f"{len(failures)} 条 topology-v2 计划参数契约失败："
            + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"topology-v2 optimize contracts did not run ({skipped} skipped)")


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
