"""契约：拓扑内核的计划收缩序 —— ``optimize_topology_v2`` **已删除**（0.5.9）。

历史背景
--------
``topology`` 曾经内部有两条路线：命中编译期步表 → **v3**；未命中 → 回落 **v2 通用
执行器**，那时才需要一份 Python 生成的计划，于是"用哪个优化器生成 v2 的计划"是一个
独立参数 ``optimize_topology_v2``。

**v2 通用执行器在 0.5.4 随 ``kupl_topology_v2`` 一并删除**，这个参数的唯一接收方
就此消失；但签名一直留着，于是它变成一个"传了也什么都不影响"的死参数——用户会以为
换它就能换拓扑的收缩序，而顺序其实固化在 ``topology_steps.h`` 里。0.5.9 把它删掉。

现在钉四件事
------------
1. **签名**：``contract`` **不**接受 ``optimize_topology_v2``（在 ``**kwargs`` 里
   传也要报 ``TypeError``，不能被后端关键字静默吃掉）；
2. **内部残留也已清掉**（0.5.12）：``_api._normalize_optimize_topology_v2`` **不存在**，
   配置表里的 ``optimize_topology_v2`` 那一列也**已删除**；
3. **内核侧仍收 ``optimize``**：``try_topology_apply`` 的 ``optimize`` 只影响 pack
   流量建模按哪份计划记账，不参与 v3 的收缩序，默认 ``"greedy"``；
4. **v3 命中时任何 ``optimize`` 都不改数值**（顺序来自步表，与 ``np.einsum``
   rel ≤ 1e-12）。

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

import numpy as np

from ..einsum import _api, _kernel_cuda, _kernel_kupl, _kernel_force
from ..einsum import contract as pe_contract

TOLERANCE = 1e-12

#: 语料方程（arity 7，命中编译期 v3 步表）。形状之间必须自洽：同一标签只能有一个
#: 轴长，否则 numpy 会先报广播错误，测试根本走不到内核。
EQUATION = "ambn,codp,egfh,ce,mo,bdf,nph->ag"
SHAPES = ((2, 3, 4, 5), (6, 7, 8, 5), (6, 2, 8, 9), (6, 6), (3, 7), (4, 8, 8), (5, 5, 9))

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

def test_the_internal_leftovers_are_gone() -> None:
    """0.5.12：内部残留也清掉——历史校验函数与配置表那一列都不再存在。

    为什么这条要单独钉：0.5.9 只删了签名，``_api._normalize_optimize_topology_v2``
    与配置表的 ``optimize_topology_v2`` 列还留着，于是 grep 仍能搜到这个名字，
    后来的人会以为"它还是个可用的东西、只是没接上"。彻底删除之后，任何 grep 命中都
    只可能出现在**历史注记**里，语义没有歧义。
    """
    assert not hasattr(_api, "_normalize_optimize_topology_v2"), (
        "历史校验函数应已删除（0.5.12）")
    from ..einsum import _profile_cache

    assert not hasattr(_profile_cache, "TOPOLOGY_V2_FIELD"), (
        "列常量应已删除")
    assert "optimize_topology_v2" not in _profile_cache.PROFILE_COLUMNS, (
        "配置表不该再有这一列", _profile_cache.PROFILE_COLUMNS)
    assert _profile_cache.PROFILE_SCHEMA_VERSION == 4, (
        "删列要升 schema 版本", _profile_cache.PROFILE_SCHEMA_VERSION)
    # 内部那份"计划次序"常量还在：``*_topology`` 入口照旧收一个 ``optimize``
    # （只影响 pack 流量建模），但它不再是任何调用方可以传的参数。
    assert _api._TOPOLOGY_PLAN_ORDER == "greedy", _api._TOPOLOGY_PLAN_ORDER


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
    assert "optimize_topology_v2" not in contract_parameters, (
        "contract 不该再有 optimize_topology_v2：v2 通用执行器在 0.5.4 删除后它没有"
        f"任何接收方，留着会让'换它就能换拓扑收缩序'看起来成立。当前参数 "
        f"{list(contract_parameters)}")


# ── 2. 旧参数必须显式报错，不能被 **kwargs 静默吃掉 ─────────────────────────

def test_the_removed_parameter_raises_instead_of_being_swallowed() -> None:
    """``optimize_topology_v2=`` 抛 ``TypeError``，正文说清为什么删、该用什么。

    ``contract`` 有 ``**kwargs``（转给后端），若不显式拦截，旧参数会落进那里被当成
    后端关键字——传了没反应比报错坏得多。
    """
    operands = _operands()
    for value in ("greedy", "optimal", None, 7):
        with _quiet():
            try:
                pe_contract(EQUATION, *operands, optimize_topology_v2=value)
            except TypeError as exc:
                message = str(exc)
            else:                                     # pragma: no cover - 失败路径
                raise AssertionError(
                    f"optimize_topology_v2={value!r} 必须抛 TypeError（参数已删除）")
        assert "optimize_topology_v2" in message and "0.5.9" in message, message
        assert "optimize=" in message, (
            f"报错正文要指出替代写法（改用 optimize=）：{message}")


# ── 3. 内核侧仍收 optimize，且只影响 pack 建模 ──────────────────────────────

def test_topology_apply_still_takes_a_keyword_only_optimize() -> None:
    """``try_topology_apply`` 的 ``optimize`` 仍在：它决定 pack 建模按哪份计划记账。

    它**不**参与 v3 的收缩序（顺序来自编译期步表），所以这里只钉签名形状——
    ``out`` / ``offsets`` 是位置参数，``optimize`` 必须是关键字参数，否则既有调用点
    的含义会变。
    """
    for module in (_kernel_kupl, _kernel_cuda):
        parameters = inspect.signature(module.try_topology_apply).parameters
        assert "optimize" in parameters, (
            f"{module.__name__}.try_topology_apply 缺 optimize 参数", list(parameters))
        assert parameters["optimize"].default == "greedy", (
            f"{module.__name__} 的 optimize 默认值是 {parameters['optimize'].default!r}")
        assert parameters["optimize"].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{module.__name__} 的 optimize 必须是关键字参数（前面的 out/offsets "
            f"是位置参数，混在一起会改掉既有调用点的含义）")


# ── 4. v3 命中时 optimize 不影响数值 ────────────────────────────────────────

def test_v3_numbers_ignore_optimize() -> None:
    """语料方程（v3 命中）上任意 ``optimize`` 都必须给出同一个答案。

    这条不只是"数值一致"：每一步都断言 ``LAST_INFO['path'] == 'kupl_topology'``，
    否则内核根本没跑 v3，"参数不影响数值"就成了废话。
    """
    operands = _operands()
    expected = np.einsum(EQUATION, *operands)

    with _quiet():
        for optimize in OPTIMIZE_SWEEP:
            _kernel_kupl.LAST_INFO.clear()
            result = pe_contract(EQUATION, *operands, kernel_force="topology",
                                 optimize=optimize)
            path = _kernel_kupl.LAST_INFO.get("path")
            assert path == "kupl_topology", (
                f"optimize={optimize!r} 没走 v3 步表（path={path!r}）："
                "这条用例的前提不成立")
            assert _rel(result, expected) <= TOLERANCE, (
                f"optimize={optimize!r} 改了数值")


TESTS = (
    test_the_internal_leftovers_are_gone,
    test_topology_apply_takes_optimize_defaulting_to_greedy,
    test_the_removed_parameter_raises_instead_of_being_swallowed,
    test_topology_apply_still_takes_a_keyword_only_optimize,
    test_v3_numbers_ignore_optimize,
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
