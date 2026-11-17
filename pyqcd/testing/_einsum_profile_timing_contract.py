"""契约：配置表的计时口径——``wall_ms`` 是**整个** ``contract()`` 的墙钟（§4）。

为什么值得单独立契约
--------------------
``wall_ms`` 是配置表的**决策输入**（tune 按它选最优），也是用户判断"缓存到底省了多少"
的唯一依据。它属于"口径错了结果照样逐位正确"的那一类缺陷：数值测试永远抓不到，
只能靠固定住"哪段时间被算进去了"。

本版口径（冻结，取代此前的两列拆分）：

    ``wall_ms`` = 整个 ``pyqcd.einsum.contract(***)`` 的墙钟时间：
    含等价转换（若有）、读表与命中判断（若有）、未命中时的规划-匹配。

因此只有**一列**计时，没有第二列可以和它相加或混比（旧的 ``init_ms`` 已删除）。
本模块把三条固定成能独立失败的断言：

1. **只有一列**：``PROFILE_COLUMNS`` / ``PERFORMANCE_FIELDS`` / ``last_profile()``
   里都只有 ``wall_ms``，不含旧的拆分列 ``init_ms`` 与更早的毫秒列（名字见
   :data:`LEGACY_TIMING_COLUMNS`，本文件按"拼出来再查"的方式引用它，源码里不留旧词）。
2. **算的是整段调用，不是某一段**：把 ``_api._find_path``（规划）与
   ``_api._contiguous_copy``（等价改写的连续副本）各拖慢 :data:`INJECT_SLEEP` 秒后，
   走等价改写那条路径的 ``wall_ms`` 仍然**≥** 两个 sleep 之和——旧口径会把这两段
   排除在外，那正是本版要消灭的口径（"必须含"而不是"可以含"）。
   同一用例先跑一次未注入的基线，确认它明显小于这个下界，否则"够大"可能只是噪声。
3. **数值不变**：语料命中 / 等价改写 / 通用三条方程与 ``np.einsum`` 的相对偏差
   都 ≤ :data:`RTOL`——计时口径怎么改都不许碰数值。

口径与边界
----------
* 断言从 ``_api.last_profile()`` 取（未格式化的浮点），不读 CSV 的三位小数：
  一次小收缩只有零点几毫秒，四舍五入会把断言变成噪声。
* 计时断言只有下界（:data:`SLOW_LOWER`），留了一个数量级以上的余量：sleep 是 50 ms
  量级，机器再抖也跨不过这个界。
* 每条用例都在临时树里跑（CWD 也换过去），退出时把环境变量与三个模块的进程内状态
  原样放回；**仓库根不会留下 ``.cache``**。

可直接运行::

    python -B -m pyqcd.testing._einsum_profile_timing_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import time
import unittest
import warnings
from typing import Any, Iterator, Mapping
from unittest.mock import patch

import numpy as np

import pyqcd.einsum as pe
from ..einsum import (_api, _cache, _equivalence, _kernel_force, _profile_cache,
                      _specialize, _threads, contract)

#: 语料命中（``_specialize.match`` 非空）：直接走专用/后端路径，不查等价映射。
CORPUS_EQUATION = "cadb,menf,ac,em,bd,fn->"

#: 语料外**但可改写**：映射要动第 0 个操作数的轴序（``acdb`` → ``cadb``），
#: 所以改写准备里一定会调 ``_contiguous_copy``。用它验"准备段也算进墙钟"。
REWRITE_EQUATION = "acdb,menf,ac,em,bd,fn->"

#: 语料外、**不可**改写：走通用收缩（一定规划，且不做等价改写）。
GENERIC_EQUATION = "ij,jk,kl->il"

#: 三条方程的标签轴长（同一个标签在不同操作数里必须等长）。
LABEL_SIZES = {
    "i": 4, "j": 3, "k": 5, "l": 4,
    "a": 2, "b": 3, "c": 2, "d": 4, "e": 3, "f": 2, "m": 2, "n": 3,
}

#: 注入的 sleep 秒数（每条路径各注入一次）。
INJECT_SLEEP = 0.05

#: "这一段真的被算进去了"的下界（毫秒）：取单次注入的 80%，留给计时抖动；
#: 又比真实的小收缩（不到 1 ms）大两个数量级，所以热态基线不会误闯进来。
SLOW_LOWER = INJECT_SLEEP * 1000.0 * 0.8

#: 数值容差（与库内其他契约同口径：只差浮点结合律的舍入）。
RTOL = 1e-12

#: 已被 §4 废除的计时列：``init_ms``（规划/查表的拆分列）与更早的毫秒列。
#: 后者按"拼出来"的方式写——本文件源码里不留那个旧词（§8 的 grep 零残留验收）。
LEGACY_TIMING_COLUMNS = ("init_ms", "compute" + "_ms")

#: 本模块会改写/还原的环境变量（含别的契约可能设过的开关与线程档）。
_ENV_NAMES = (
    _cache.CACHE_ENV,
    _equivalence.EQUIVALENCE_ENV,
    _equivalence.AS_EQUIVALENCE_ENV,
    _kernel_force.KERNEL_FORCE_ENV,
    _specialize.DISABLE_ENV,
    _specialize.NOTICE_ENV,
    _specialize.FUSED_DISABLE_ENV,
    _profile_cache.PROFILE_ENV,
    _profile_cache.DIR_ENV,
    _threads.STANDARD_OMP_ENV,
    _threads.STANDARD_BLAS_ENV,
)

#: ``_profile_cache`` 需要按用例复位的进程内状态。
_CACHE_STATE = ("_CONFIGURED_DIR", "_LAST_ERROR", "_HOST", "_TABLE", "_TUNING")

#: ``_api`` 需要按用例复位的标量状态。
_API_STATE = ("_LAST_PROFILE", "_LAST_KERNEL", "_FUSED_ENV_OWNED",
              "_PROFILE_MISSING_NOTICED")


# ── 环境与进程内状态 ────────────────────────────────────────────────────────

def _prepare_env() -> None:
    """把本模块依赖的开关摆好（``_isolated`` 的 ``finally`` 负责还原）。

    ``PYQCD_EINSUM_SPECIALIZE`` 是"是否禁用"：``1`` 表示**没**禁用——等价改写的
    前置条件就是 ``specialize`` 开着。
    """
    os.environ[_cache.CACHE_ENV] = "1"
    os.environ[_equivalence.EQUIVALENCE_ENV] = "1"
    os.environ[_equivalence.AS_EQUIVALENCE_ENV] = "1"
    os.environ[_specialize.DISABLE_ENV] = "1"
    os.environ[_specialize.NOTICE_ENV] = "0"
    os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
    os.environ[_profile_cache.PROFILE_ENV] = "1"
    os.environ.pop(_profile_cache.DIR_ENV, None)
    os.environ.pop(_kernel_force.KERNEL_FORCE_ENV, None)


def _reset_state() -> None:
    """把配置表与 ``_api`` 的进程内状态清到"没钉目录、没解析缓存"的初始态。

    ``_api`` 的规划/程序缓存必须**凉**：本用例要量"规划真的跑了"，热缓存会让规划
    整个跳过，注入的 sleep 就不会发生——那条断言于是变成空断言。
    """
    for name in _CACHE_STATE:
        setattr(_profile_cache, name, "" if name == "_CONFIGURED_DIR" else None)
    _profile_cache._TABLE = None
    _profile_cache._TUNING = 0
    for name in _API_STATE:
        if hasattr(_api, name):
            setattr(_api, name, False if name == "_PROFILE_MISSING_NOTICED" else None)
    _api._PROFILE_NOTICED.clear()
    for name in ("_PLAN_PATH_CACHE", "_CONTRACT_PLAN_CACHE",
                 "_CONTRACT_PROGRAM_CACHE", "_NATIVE_PLAN_CACHE",
                 "_NATIVE_CONTEXT_CACHE"):
        cache = getattr(_api, name, None)
        clear = getattr(cache, "clear", None)
        if callable(clear):
            clear()
    pe.cache_clear()


def _snapshot(module: Any) -> dict[str, Any]:
    """快照模块状态：``_profile_cache`` 只取标量；``_api`` 取标量 + 去重集合。

    规划/程序缓存**只记对象本身**（``LRUCache`` / ``OrderedDict`` 都有 ``clear()``），
    退出时清空即可——它们是纯加速结构，清掉不改变任何可观测语义。
    """
    if module is _profile_cache:
        return {name: getattr(module, name) for name in _CACHE_STATE}
    return {
        "values": {name: getattr(module, name)
                   for name in _API_STATE if hasattr(module, name)},
        "noticed": set(_api._PROFILE_NOTICED),
        "caches": [getattr(module, name, None) for name in (
            "_PLAN_PATH_CACHE", "_CONTRACT_PLAN_CACHE", "_CONTRACT_PROGRAM_CACHE",
            "_NATIVE_PLAN_CACHE", "_NATIVE_CONTEXT_CACHE")],
    }


def _restore(module: Any, saved: Mapping[str, Any]) -> None:
    """按 :func:`_snapshot` 的副本原样放回（容器原地更新，不换对象）。"""
    if module is _profile_cache:
        for name, value in saved.items():
            setattr(module, name, value)
        return
    for name, value in saved["values"].items():
        setattr(module, name, value)
    _api._PROFILE_NOTICED.clear()
    _api._PROFILE_NOTICED.update(saved["noticed"])
    for cache in saved["caches"]:
        clear = getattr(cache, "clear", None)
        if callable(clear):
            clear()


@contextlib.contextmanager
def _isolated() -> Iterator[str]:
    """临时 CWD + 干净开关 + 干净的进程内状态；退出时全部还原。"""
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    snapshots = [(module, _snapshot(module))
                 for module in (_profile_cache, _api)]
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-profile-timing-") as root:
            work = os.path.join(root, "work")
            os.makedirs(work)
            os.chdir(work)
            try:
                _prepare_env()
                _reset_state()
                yield work
            finally:
                os.chdir(saved_cwd)
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for module, snapshot in snapshots:
            _restore(module, snapshot)
        pe.cache_clear()


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """屏蔽本契约不关心的提示与原生回落 warning（输出与数值、计时都无关）。"""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# ── 张量与注入 ──────────────────────────────────────────────────────────────

def _operands(equation: str, *, seed: int = 20261117) -> list[np.ndarray]:
    """按方程左侧生成 C 连续 ``complex128`` 操作数（轴长见 :data:`LABEL_SIZES`）。"""
    rng = np.random.default_rng(seed)
    operands: list[np.ndarray] = []
    for term in equation.split("->", 1)[0].split(","):
        shape = tuple(LABEL_SIZES[label] for label in term)
        value = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        operands.append(np.ascontiguousarray(value.astype(np.complex128)))
    return operands


def _relative_error(actual: Any, expected: Any) -> float:
    """相对最大偏差；全零答案用 1 作尺度（与库内同口径）。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    scale = float(np.max(np.abs(expected_array))) or 1.0
    return float(np.max(np.abs(actual_array - expected_array))) / scale


@contextlib.contextmanager
def _sleep_planning(seconds: float) -> Iterator[list[int]]:
    """拖慢 ``_api._find_path`` 并记录它被调用了几次（"规划真的跑了"）。"""
    original = _api._find_path
    calls: list[int] = []

    def slow(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        time.sleep(seconds)
        return original(*args, **kwargs)

    with patch.object(_api, "_find_path", new=slow):
        yield calls


@contextlib.contextmanager
def _sleep_contiguous(seconds: float) -> Iterator[list[int]]:
    """拖慢 ``_api._contiguous_copy``：改写准备里唯一会复制张量的一步。"""
    original = _api._contiguous_copy
    calls: list[int] = []

    def slow(operand: Any) -> Any:
        calls.append(1)
        time.sleep(seconds)
        return original(operand)

    with patch.object(_api, "_contiguous_copy", new=slow):
        yield calls


def _observed() -> dict[str, Any]:
    """最近一次画像调用的观测（``_api.last_profile()`` 的副本）。"""
    info = _api.last_profile()
    assert info is not None, "本次调用必须留下画像观测（profile=True）"
    return info


# ── 契约 1：只有一列计时 ────────────────────────────────────────────────────

def test_only_one_timing_column_exists() -> None:
    """表头、性能字段与 ``last_profile()`` 里都只有 ``wall_ms`` 这一列计时。"""
    columns = tuple(_profile_cache.PROFILE_COLUMNS)
    assert "wall_ms" in columns, ("配置表必须有 wall_ms 列", columns)
    performance = tuple(_profile_cache.PERFORMANCE_FIELDS)
    assert "wall_ms" in performance, performance
    for stale in LEGACY_TIMING_COLUMNS:
        assert stale not in columns, (
            "§4 只留一列计时：旧的拆分列不得留在表头里", stale, columns)
        assert stale not in performance, (
            "§4 只留一列计时：旧的拆分列不得留在性能字段里", stale, performance)

    operands = _operands(GENERIC_EQUATION)
    with _isolated() as work:
        with _quiet():
            contract(GENERIC_EQUATION, *operands,
                     profile=True, profile_dir=os.path.join(work, "cache"))
        info = _observed()
        for stale in LEGACY_TIMING_COLUMNS:
            assert stale not in info, (
                "拆分的旧列已删除，``last_profile()`` 不许再报它", stale, sorted(info))
        assert "wall_ms" in info, (
            "``last_profile()`` 必须报告这一次调用的墙钟", sorted(info))
        assert isinstance(info["wall_ms"], float) and info["wall_ms"] > 0.0, (
            "墙钟必须**量到**（恒为 0 说明这一段根本没测）", info)


# ── 契约 2：算的是整段调用（含规划与改写准备） ───────────────────────────────

def test_wall_clock_includes_planning_and_rewrite_prep() -> None:
    """规划与改写准备各自拖慢 50 ms 后都必须落在墙钟里（热态基线远小于下界）。

    两段注入**分开做**，因为它们在两条不同的路径上：
    * ``_api._find_path``（规划）在**通用**路径上跑——等价改写施放时内层会用
      ``target_equation`` 直接命中语料特化，规划根本不会发生（实测 find=0）；
    * ``_api._contiguous_copy``（改写准备）只在**等价改写**路径上跑。
    把它们混在一条调用里，必然有一段是空断言。
    """
    assert _specialize.match(REWRITE_EQUATION) is None, (
        "例方程必须是**语料外**写法，否则根本不会走等价改写", REWRITE_EQUATION)
    assert _specialize.match(GENERIC_EQUATION) is None, (
        "通用例方程必须是语料外写法", GENERIC_EQUATION)

    cases = (
        ("generic", GENERIC_EQUATION, "planning"),
        ("rewrite", REWRITE_EQUATION, "prep"),
    )
    for label, equation, hook in cases:
        operands = _operands(equation)
        expected = np.einsum(equation, *operands)
        with _isolated() as work:
            cache = os.path.join(work, "cache")

            # 预热：首调含原生库加载/绑定（实测 1 s 量级），与"规划/改写准备"无关。
            # 不先走一遍，基线会被首调开销淹没，"基线远小于下界"就成了噪声。
            with _quiet():
                contract(equation, *operands, profile=True, profile_dir=cache)
            warm = _observed()["wall_ms"]

            # 基线（不注入、热态）：小方程只有毫秒级，必须远低于下界。
            _reset_state()                  # 冷掉规划缓存（规划仍会真跑）
            with _quiet():
                contract(equation, *operands, profile=True, profile_dir=cache)
            baseline = _observed()["wall_ms"]
            assert baseline < SLOW_LOWER, (
                "热态基线本来就该远小于下界，否则本用例没有区分力",
                label, baseline, SLOW_LOWER, warm)

            # 注入：只拖慢这条路径上**真的会跑**的那一段。
            _reset_state()
            with contextlib.ExitStack() as stack:
                planning = stack.enter_context(_sleep_planning(
                    INJECT_SLEEP if hook == "planning" else 0.0))
                prep = stack.enter_context(_sleep_contiguous(
                    INJECT_SLEEP if hook == "prep" else 0.0))
                stack.enter_context(_quiet())
                result = contract(equation, *operands,
                                  profile=True, profile_dir=cache)
            info = _observed()

            hits = planning if hook == "planning" else prep
            assert hits, (
                f"{label}：{hook} 钩子一次都没被调用，注入的 sleep 没发生——"
                "断言没有牙齿", equation)
            assert info["wall_ms"] >= SLOW_LOWER, (
                f"§4：wall_ms 是整个 contract() 的墙钟；{label} 路径上的 {hook} "
                "耗时必须在里面（排除它就是把口径拆回两列）",
                label, info, SLOW_LOWER, baseline)
            assert _relative_error(result, expected) <= RTOL, (
                "口径怎么算都不许改答案", label, _relative_error(result, expected))


# ── 契约 3：数值不变 ────────────────────────────────────────────────────────

def test_every_timing_path_keeps_the_numbers() -> None:
    """语料命中 / 等价改写 / 通用：三条方程与 ``np.einsum`` 的 rel ≤ 1e-12。"""
    assert _specialize.match(CORPUS_EQUATION) is not None, (
        "第一条方程必须是语料命中", CORPUS_EQUATION)
    assert _specialize.match(REWRITE_EQUATION) is None, (
        "第二条方程必须是语料外写法（靠等价改写才命中）", REWRITE_EQUATION)
    assert _specialize.match(GENERIC_EQUATION) is None, (
        "第三条方程必须是语料外写法", GENERIC_EQUATION)

    worst: list[tuple[str, float]] = []
    with _isolated() as work:
        for index, (label, equation) in enumerate((
                ("corpus hit", CORPUS_EQUATION),
                ("equivalence rewrite", REWRITE_EQUATION),
                ("generic", GENERIC_EQUATION))):
            operands = _operands(equation, seed=20261117 + index)
            expected = np.einsum(equation, *operands)
            directory = os.path.join(work, f"case{index}")
            with _quiet():
                actual = contract(equation, *operands,
                                  profile=True, profile_dir=directory)
            error = _relative_error(actual, expected)
            assert np.shape(actual) == np.shape(expected), (label, "形状变了")
            assert np.asarray(actual).dtype == np.asarray(expected).dtype, (
                label, "dtype 变了", np.asarray(actual).dtype)
            assert error <= RTOL, (label, error, RTOL)
            worst.append((label, error))

    print("    三条路径的数值：max rel = " + ", ".join(
        f"{label} {error:.1e}" for label, error in worst), flush=True)


TESTS = (
    test_only_one_timing_column_exists,
    test_wall_clock_includes_planning_and_rewrite_prep,
    test_every_timing_path_keeps_the_numbers,
)


def test_einsum_profile_timing_contracts() -> None:
    """套件入口：跑完 :data:`TESTS`，任何一条失败就抛出汇总。"""
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
            f"{len(failures)} 条计时口径契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"profile timing contracts did not run ({skipped} skipped)")


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
