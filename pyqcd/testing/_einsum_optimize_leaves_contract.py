"""契约：``branch_1`` / ``branch_2`` / ``branch_all`` / ``none`` 都能接进 ``optimize``。

为什么值得单独立契约
--------------------
路由表（``paths._BUILTIN_ATTRS``）的键曾经全是**连字符**写法（``branch-1``），而
``_plan_py`` 的模块属性是**下划线**（``branch_1``）：``get_path_fn("branch_1")`` 于
是抛 ``KeyError``——名字看起来完全合理，报错却像"拼错了"。``none`` 更彻底，它根本
不在表里（"不规划"只由 ``optimize=False`` 表达），同一个概念于是有两种拼写，而其中
一种会炸。这种缺口数值测试永远抓不到：写错的名字只要不用，结果就一直是"对的"。

反过来，名字**太多**同样是安静的缺口：``optimize="eager"``/``"opportunistic"``/
``"random-greedy"`` 静默等同 ``greedy``，``branch`` 是 ``branch_all`` 的重复入口，
``auto``/``auto-hq`` 是缺省值而不是名字——它们"都能跑、数值也对"，于是"哪些拼写才
算数"变成一句没人能验证的宣言。所以这里还要钉住名字面的**上界**。

这里钉七条能各自独立失败的断言：

1. 七个名字都能被 ``paths.get_path_fn`` 解析，且 ``branch_1``/``branch_2``/
   ``branch_all`` 映射到 ``_plan_py`` 的**同名**属性；别名（``branch-1`` …
   ``dynamic-programming``）与主名解析到**同一个对象**（``is``，不是"另一份等价
   实现"）；
2. ``contract(..., optimize=<七者之一>)`` 全部成功，数值 ``rel <= 1e-12``；
3. ``optimize="none"`` 的语义随内核定型：对 ``generic``/``backend``/``dedicated``
   等同于 ``optimize=False``（解析成 ``False``、path 相同），对 ``topology`` 是
   **唯一合法**取值（顺序由内核决定）且不因 optimize 报错；
4. ``PATH_OPTIONS_MAIN`` 恰好是这七个名字（连字符别名不进主清单），且七者都在
   ``PATH_OPTIONS`` 里（别名合并进去不算违例）；
5. ``PATH_OPTIONS`` 去掉运行时注册的名字后**恰好**是七叶子 + :data:`ALIASES`：多一个
   旧拼写就红（:data:`REMOVED` 逐个断言不在集合里）；
6. :data:`REMOVED` 的七个拼写在 ``get_path_fn`` 上一律抛 ``KeyError``，消息列全七个
   合法值且不再点名它们——"能跑但没人敢写"的拼写必须真的消失，而不是留在表里；
   :data:`REJECTED`（六个）连 ``contract(optimize=...)`` 也必须拒；
7. ``auto`` 是 ``optimize`` 的**缺省值**，不是名字：它不在 ``PATH_OPTIONS``，但
   ``is_builtin_path_name("auto")`` 必须为真、``_plan_py.auto`` 必须还在，否则
   ``contract()``/``contract_path()`` 的缺省调用会当场 ``KeyError``（``_api`` 的
   ``_find_path`` 正是靠这个判定走规划器）。规划器的名字表因此只能比注册表多出
   ``auto``。

可直接运行::

    python -B -m pyqcd.testing._einsum_optimize_leaves_contract
"""
from __future__ import annotations

import re
import unittest
from typing import Any

import numpy as np

from ..einsum import _api, _kernel_force, _plan_py, contract, contract_path, paths

#: 七个必须在 ``optimize=`` 上可用的名字（下划线拼写，与 ``_plan_py`` 属性同名）。
LEAVES: tuple[str, ...] = (
    "greedy", "optimal", "dp", "branch_1", "branch_2", "branch_all", "none",
)

#: 兼容别名 → 主名：必须解析到**同一个**对象（旧调用方不破）。
#: 2026-11-18 用户裁定：**拼写统一下划线**，连字符别名与 ``dynamic-programming``
#: 已删除（它们让同一路径有两条拼写，README 与配置表会漂移）。此表因此为空——
#: 保留常量是为了让"不允许别名"这条规则可被断言，而不是再登记一批别名。
ALIASES: dict[str, str] = {}

#: 已删除的七个拼写：解析必须抛 ``KeyError``。
#:
#: ``auto``/``auto-hq`` 是缺省值（选择器归 ``--tune``，不是可写的名字）；``eager``/
#: ``opportunistic``/``random-greedy``/``random-greedy-128`` 静默等同 ``greedy``；
#: ``branch`` 是 ``branch_all`` 的重复入口。
REMOVED: tuple[str, ...] = (
    "auto", "auto-hq", "branch", "eager", "opportunistic",
    "random-greedy", "random-greedy-128",
)

#: 连 ``contract(optimize=...)`` 都必须拒的那几个。
#:
#: ``auto`` 是唯一例外：它是 ``optimize`` 的**签名缺省值**（``True``/``None`` 也归一
#: 到它），所以它仍能被 ``contract`` 接受，只从注册表消失——见第 7 条。
REJECTED: tuple[str, ...] = tuple(name for name in REMOVED if name != "auto")

#: ``_BUILTIN_ATTRS`` 的主键必须映射到同名属性（而不是"反正能跑"的替代实现）。
SAME_NAMED: tuple[str, ...] = ("branch_1", "branch_2", "branch_all")

#: 三个"optimize 有效"的内核：``none`` 对它们等价于 ``False``。
OPTIMIZE_KERNELS: tuple[str, ...] = (
    _kernel_force.GENERIC_LABEL, "backend", "dedicated",
)
#: ``optimize`` 无效的那个内核：顺序由内核决定。
TOPOLOGY = "topology"

EQUATION = "ij,jk,kl->il"
SHAPES = ((3, 4), (4, 5), (5, 6))
RTOL = 1e-12


# -- 张量 -----------------------------------------------------------------------
def _operands(dtype: Any = np.float64, seed: int = 20261118) -> list[np.ndarray]:
    """按方程左侧生成 C 连续操作数（轴长见 :data:`SHAPES`）。"""
    rng = np.random.default_rng(seed)
    return [np.ascontiguousarray(rng.standard_normal(shape).astype(dtype))
            for shape in SHAPES]


def _relative_error(actual: Any, expected: Any) -> float:
    """相对最大偏差（全零答案用 1 作尺度，与库内同口径）。"""
    expected_array = np.asarray(expected)
    scale = float(np.max(np.abs(expected_array))) or 1.0
    return float(np.max(np.abs(np.asarray(actual) - expected_array))) / scale


def _path_of(operands: list[np.ndarray], optimize: Any) -> list[tuple[int, ...]]:
    """``contract_path`` 给出的 path（规划缓存关掉：每次真的问一遍）。"""
    path, _info = contract_path(EQUATION, *operands, optimize=optimize, cache=False)
    return [tuple(int(index) for index in step) for step in path]


def _expect_keyerror(name: str) -> str:
    """``get_path_fn(name)`` 必须抛 ``KeyError``，返回它的消息。"""
    try:
        resolved = paths.get_path_fn(name)
    except KeyError as exc:
        return str(exc)
    raise AssertionError(("这个名字必须被拒绝，但它解析出来了", name, resolved))


def _options_listed(message: str) -> set[str]:
    """``KeyError`` 消息里 ``valid options are ...`` 之后列出的名字集合。

    按**整词**取，不用子串判定：``branch`` 是 ``branch-1`` 的子串，子串判定会让
    "已删除的 ``branch`` 又出现在合法选项里"这种错误漏过去。
    """
    tail = message.split("valid options are", 1)[-1]
    return set(re.findall(r"[0-9A-Za-z][0-9A-Za-z_-]*", tail))


# ── 1. 七个名字都解析得出来，别名与主名是同一个对象 ──────────────────────────
def test_seven_names_resolve_and_aliases_are_identical() -> None:
    """七个名字都能被 ``get_path_fn`` 解析；别名 ``is`` 主名；``none`` 不抛错。"""
    for name in LEAVES:
        resolved = paths.get_path_fn(name)
        assert callable(resolved), ("optimize 名字必须解析成可调用对象", name, resolved)

    # 主键映射到**同名**属性：下划线是模块属性拼写，这才是不抛 KeyError 的原因。
    for name in SAME_NAMED:
        assert paths.get_path_fn(name) is getattr(_plan_py, name), (
            "下划线主名必须直接落到 _plan_py 的同名属性", name,
            paths.get_path_fn(name), getattr(_plan_py, name))
        assert paths._BUILTIN_ATTRS[name][0] == name, (
            "_BUILTIN_ATTRS 的主键必须映射到同名属性", name,
            paths._BUILTIN_ATTRS[name])

    # 连字符别名不是"另一份实现"：同一个对象。
    for alias, target in ALIASES.items():
        assert paths.get_path_fn(alias) is paths.get_path_fn(target), (
            "连字符别名必须与下划线主名解析到同一个对象", alias, target,
            paths.get_path_fn(alias), paths.get_path_fn(target))

    # ``none`` 有哨兵：既不能 KeyError，也不能悄悄变成某个真的优化器。
    resolved_none = paths.get_path_fn("none")
    assert resolved_none is paths._NO_PATH, (
        "optimize='none' 必须解析成 _NO_PATH 哨兵", resolved_none)
    assert paths.get_path_fn("NONE") is resolved_none, "大小写也要归一"
    for name in ("greedy", "optimal", "dp", "branch_1", "branch_2", "branch_all"):
        assert paths.get_path_fn(name) is not resolved_none, (
            "none 不该与真的优化器共用同一个对象", name)


# ── 2. 七个名字都能跑通，数值一致 ──────────────────────────────────────────────
def test_contract_runs_all_seven_names() -> None:
    """``contract(..., optimize=<七者之一>)`` 全部成功，数值 ``rel <= 1e-12``。"""
    operands = _operands()
    expected = np.einsum(EQUATION, *operands)
    assert _relative_error(expected, expected) == 0.0, "参照本身不可信"

    for name in LEAVES:
        got = contract(EQUATION, *operands, optimize=name, specialize=False,
                       cache=False, profile=False)
        error = _relative_error(got, expected)
        assert error <= RTOL, ("contract(optimize=%r) 的数值超出容差" % name, error)
        assert np.asarray(got).shape == (SHAPES[0][0], SHAPES[-1][-1]), (
            "输出形状不对", name, np.asarray(got).shape)

    # 名字必须真的改变规划：全宽分支与 greedy 在该方程上给出不同 path，否则上面
    # 那几条断言区分不出"名字生效"还是"全被当成同一个东西"。
    assert _path_of(operands, "branch_all") != _path_of(operands, "greedy"), (
        "该方程上 branch_all 与 greedy 的 path 相同，这条断言区分不出名字是否生效")


# ── 3. none 的语义随内核定型 ─────────────────────────────────────────────────
def test_none_is_false_for_optimize_kernels() -> None:
    """``generic``/``backend``/``dedicated`` 上 ``none`` 等价于 ``optimize=False``。"""
    operands = _operands()
    off_path = _path_of(operands, False)
    assert off_path == [(0, 1, 2)], (
        "``optimize=False`` 应当是单步 n 元 einsum（整条方程一次收缩）", off_path)
    assert _path_of(operands, "none") == off_path, (
        "``optimize='none'`` 的 path 与 ``optimize=False`` 不同",
        _path_of(operands, "none"), off_path)

    expected = np.einsum(EQUATION, *operands)
    for kernel in OPTIMIZE_KERNELS:
        forced = _kernel_force.forced_labels(kernel)
        resolved = _api._resolve_optimize("none", forced)
        assert resolved is False, (
            "对 optimize 有效的内核，'none' 必须归一成 False", kernel, resolved)
        assert _api._resolve_optimize(" NONE ", forced) is False, (
            "大小写/空白也要归一", kernel)
        assert _api._resolve_optimize("greedy", forced) == "greedy", (
            "别的名字不许被这条归一吞掉", kernel)

        # 端到端：与 optimize=False 逐位相同（同一条 path，不是"碰巧接近"）。
        off = contract(EQUATION, *operands, optimize=False, kernel_force=kernel,
                       cache=False, profile=False)
        got = contract(EQUATION, *operands, optimize="none", kernel_force=kernel,
                       cache=False, profile=False)
        assert _relative_error(got, off) == 0.0, (
            "同一条 path 的两个写法必须给出同一结果", kernel,
            _relative_error(got, off))
        assert _relative_error(got, expected) <= RTOL, ("数值超出容差", kernel)


def test_none_is_accepted_for_topology() -> None:
    """``kernel_force='topology'`` 上 ``none`` 是合法输入：不因 optimize 报错。

    顺序由内核决定，所以这里**不**归一成 ``False``——原样保留，交给内核。
    内核本身仍可能因为 dtype/原生库缺席而跑不了；那不是 optimize 的错误，此时
    本用例跳过并写明原因（缺省是 complex128，本机可跑）。
    """
    forced = _kernel_force.forced_labels(TOPOLOGY)
    assert _api._resolve_optimize("none", forced) == "none", (
        "topology 上 'none' 必须原样保留（顺序由内核决定）",
        _api._resolve_optimize("none", forced))

    operands = _operands(np.complex128)
    expected = np.einsum(EQUATION, *operands)
    try:
        got = contract(EQUATION, *operands, optimize="none", kernel_force=TOPOLOGY,
                       cache=False, profile=False)
    except _kernel_force.KernelUnavailableError as exc:
        record = _api.last_kernel() or {}
        assert record.get("requested") == (TOPOLOGY,), (
            "已经走到内核分派：报错不能是 optimize 被拒", record)
        assert record.get("attempted"), (
            "强制即真试一次：必须留下 attempted 标签", record)
        raise unittest.SkipTest(
            f"本机跑不了 {TOPOLOGY} 内核（{type(exc).__name__}），"
            "optimize 一侧已验完，端到端不可达") from None
    assert _relative_error(got, expected) <= RTOL, ("数值超出容差", 
                                                   _relative_error(got, expected))


# ── 4. 主清单恰好这七个 ─────────────────────────────────────────────────────
def test_main_options_are_exactly_the_seven() -> None:
    """``PATH_OPTIONS_MAIN`` 恰好七个；七者都在 ``PATH_OPTIONS`` 里。"""
    main = set(paths.PATH_OPTIONS_MAIN)
    assert main == set(LEAVES), (
        "主清单必须恰好是这七个名字", "多出", sorted(main - set(LEAVES)),
        "缺少", sorted(set(LEAVES) - main))
    assert set(LEAVES) <= set(paths.PATH_OPTIONS), (
        "七个名字必须都在 PATH_OPTIONS 里",
        sorted(set(LEAVES) - set(paths.PATH_OPTIONS)))
    for alias in ALIASES:
        assert alias not in main, (
            "连字符别名不进主清单（它只是兼容拼写）", alias)


# ── 5. 名字面的上界：七叶子 + 别名，一个不多 ─────────────────────────────────
def test_options_are_exactly_leaves_plus_aliases() -> None:
    """``PATH_OPTIONS`` 去掉运行时注册的名字后**恰好**是七叶子 + 别名。"""
    registered = set(paths._REGISTERED)      # 调用方可以注册自己的名字
    declared = set(paths.PATH_OPTIONS) - registered
    allowed = set(LEAVES) | set(ALIASES)

    assert declared == allowed, (
        "PATH_OPTIONS 的名字面与约定不符",
        "多出（未登记，不许沉默）", sorted(declared - allowed),
        "缺少", sorted(allowed - declared))
    for name in REMOVED:
        assert name not in paths.PATH_OPTIONS, (
            "已删除的名字不许留在 PATH_OPTIONS", name)


# ── 6. 已删除的拼写必须真的被拒 ───────────────────────────────────────────────
def test_removed_options_raise_keyerror() -> None:
    """``REMOVED`` 的七个拼写一律 ``KeyError``，消息列全七个合法值。

    消息里**不许**再出现被删的名字：留给用户的合法选项必须与
    ``PATH_OPTIONS`` 一致，否则报错本身就在教人写一个会炸的名字。
    """
    for name in REMOVED:
        for spelling in (name, name.upper()):
            message = _expect_keyerror(spelling)
            assert spelling in message, (
                "KeyError 必须点名被拒的拼写", spelling, message)
            listed = _options_listed(message)
            assert set(LEAVES) <= listed, (
                "KeyError 必须列出七个合法值", spelling,
                sorted(set(LEAVES) - listed))
            assert not (set(REMOVED) & listed), (
                "KeyError 的合法选项里不该出现已删除的名字", spelling,
                sorted(set(REMOVED) & listed))
            assert listed == set(paths.PATH_OPTIONS), (
                "报错列出的选项集合必须就是 PATH_OPTIONS", spelling,
                sorted(listed ^ set(paths.PATH_OPTIONS)))

    # 端到端：删掉名字就要在 contract() 上失败，而不是换个名字继续跑。
    # ``auto`` 不在此列——它是签名缺省值，仍能被 ``contract`` 接受（见下一条用例）。
    operands = _operands()
    for name in REJECTED:
        try:
            contract(EQUATION, *operands, optimize=name, specialize=False,
                     cache=False, profile=False)
        except KeyError:
            pass
        else:
            raise AssertionError(("contract(optimize=%r) 必须失败" % name,))


# ── 7. auto 只是缺省值，不是名字 ─────────────────────────────────────────────
def test_auto_is_an_implicit_default_not_a_name() -> None:
    """``auto`` 不在 ``PATH_OPTIONS``，但规划侧必须仍认得它。

    ``_api.contract``/``contract_path`` 把 ``True``/``None`` 归一成 ``"auto"``，
    再经 ``paths.is_builtin_path_name`` 决定"走规划器还是走注册表"。删掉这一档等于
    让每一次缺省调用都 ``KeyError``——所以 ``auto`` 留在 ``_IMPLICIT_DEFAULT_NAMES``，
    只从**名字面**消失。
    """
    assert "auto" not in paths.PATH_OPTIONS, "auto 是缺省值，不是可写的名字"
    for spelling in ("auto", "AUTO"):
        assert paths.is_builtin_path_name(spelling), (
            "auto 必须仍被规划器承接，否则缺省 contract() 会 KeyError", spelling)
    assert callable(_plan_py.auto), "缺省选择器 _plan_py.auto 不能删"

    # 规划器的名字表只比注册表多出缺省值 auto：删掉的名字在规划器里也不该留着。
    extra = set(_plan_py._PATH_FNS) - set(paths.PATH_OPTIONS)
    assert extra == {"auto"}, (
        "规划器的名字表只能比注册表多出缺省值 auto", sorted(extra))

    # 端到端：不给 optimize 的调用必须仍能规划，且与 numpy 一致。
    operands = _operands()
    expected = np.einsum(EQUATION, *operands)
    for optimize in ("auto", None, True):
        got = contract(EQUATION, *operands, optimize=optimize, specialize=False,
                       cache=False, profile=False)
        error = _relative_error(got, expected)
        assert error <= RTOL, (
            "缺省 optimize（=auto）必须仍能收缩", optimize, error)


TESTS = (
    test_seven_names_resolve_and_aliases_are_identical,
    test_contract_runs_all_seven_names,
    test_none_is_false_for_optimize_kernels,
    test_none_is_accepted_for_topology,
    test_main_options_are_exactly_the_seven,
    test_options_are_exactly_leaves_plus_aliases,
    test_removed_options_raise_keyerror,
    test_auto_is_an_implicit_default_not_a_name,
)


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


def test_einsum_optimize_leaves_contracts() -> None:
    """套件入口：跑完 :data:`TESTS`，任何一条失败就抛出汇总。"""
    failures: list[str] = []
    passed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest:
            continue
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条 optimize 叶子名字契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("optimize 叶子名字契约全部跳过")


if __name__ == "__main__":
    raise SystemExit(main())
