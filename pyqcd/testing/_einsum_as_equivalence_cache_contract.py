"""契约：等价映射自动施加与 einsum 缓存开关的集成行为。

本轮新增的 ``as_equivalence``、``cache`` 与 ``kernel_force`` 都可能在结果正确时
仍然选错路径：显式参数没有压过环境、缓存关闭后仍被读写、或强制内核错误地覆盖
自动分派。本模块只断言这些可观察语义，不做计时：

1. ``as_equivalence`` 的显式取值（``True`` / ``False`` / 字符串 ``"false"``）与 spy
   调用证据；
2. 操作数顺序、输出轴序及二者复合映射返回的仍是原方程答案；
3. ``contract(..., cache=False)`` 不改变任一缓存统计字段；
4. ``specialize.match`` / ``equivalence.normalize`` 的负结果缓存命中；
5. ``contract_path(..., cache=...)`` 的规划缓存与统一清空语义；
6. ``kernel_force='auto'`` 已删除：显式取值与环境变量两条路都必须抛 ``ValueError``；
7. ``kernel_force='generic'`` 与 ``out=`` / ``specialize=False`` 的非冲突组合。

可直接运行::

    python -B -m pyqcd.testing._einsum_as_equivalence_cache_contract
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
import warnings
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

import pyqcd.einsum as pe
from ..einsum import (_api, _cache, _equivalence, _kernel_force, _profile_cache,
                      _specialize, _threads)


# 同一语料族的三种查询：只换操作数位置、只换输出轴序、两者同时发生。
CORPUS_EQUATION = "cadb,menf,ac,em,Mbd,Nfn->MN"
OPERAND_ORDER_EQUATION = "menf,cadb,ac,em,Mbd,Nfn->MN"
OUTPUT_AXES_EQUATION = "cadb,menf,ac,em,Mbd,Nfn->NM"
COMPOSITE_EQUATION = "menf,cadb,ac,em,Mbd,Nfn->NM"

# 负结果测试分别使用两条语料外方程，避免共享一个缓存键。
SPECIALIZE_MISS_EQUATION = "ij,jk->i"
EQUIVALENCE_MISS_EQUATION = "ii->i"

CACHE_EQUATION = "ij,ij->ij"
PATH_EQUATION = "ij,jk->ik"
CACHE_NAMES = ("specialize.match", "equivalence.normalize", "api.find_path")
REQUIRED_CACHE_FIELDS = {
    "size", "maxsize", "hits", "misses", "stores", "evictions", "hit_rate",
}

#: ``_api`` 的集合型进程状态：都是"只报一次"的去重闸门，留着就等于让下一条用例
#: 静默通过（它以为告警已经打过了）。
_API_SET_FIELDS = (
    "_NATIVE_DISABLED", "_NATIVE_WARNED_REASONS", "_SPECIALIZED_KERNEL_WARNED",
    "_KERNEL_SKIP_WARNED", "_PROFILE_NOTICED",
)

#: ``_api`` 的标量/引用型状态：``_PROFILE_MISSING_NOTICED`` 是"缺表只提示一次"
#: 的闸门，``_LAST_KERNEL`` 是最近一次分派记录，``_LAST_PROFILE`` 是最近一次观测。
_API_VALUE_FIELDS = ("_PROFILE_MISSING_NOTICED", "_LAST_KERNEL", "_FUSED_ENV_OWNED",
                     "_LAST_PROFILE")

#: ``_profile_cache`` 的进程状态：本模块的用例会经 ``contract()`` 走到配置表层
#: （``profile`` 缺省为开），所以目录、解析缓存与遍历计数都会被改写。值即模块初值
#: （TV 重设计后表只保留这三类状态；索引/待写队列随写路径一并删除）。
_CACHE_PLAIN_FIELDS = {
    "_CONFIGURED_DIR": "", "_LAST_ERROR": None, "_HOST": None,
    "_TABLE": None, "_TUNING": 0,
}

LABEL_SIZES = {
    "i": 2, "j": 3, "k": 4,
    "c": 2, "a": 3, "d": 4, "b": 5,
    "m": 6, "e": 7, "n": 2, "f": 3,
    "M": 2, "N": 3,
}


@contextlib.contextmanager
def _isolated_state() -> Iterator[None]:
    """快照/复原本模块会触碰的**全部**进程级状态，并清空统一缓存。

    光清环境与 ``_LAST_KERNEL`` 不够——``contract()`` 缺省开着画像（``profile=None``），
    走一次就会改 ``_api._PROFILE_NOTICED`` / ``_LAST_PROFILE``、``_profile_cache`` 的
    目录与解析缓存，以及 ``_threads`` 的线程档位。这些都没人复原时，单跑永远通过、
    全套偶发失败：前一个模块把 ``_profile_cache._CONFIGURED_DIR`` 钉在它自己的目录上，
    本模块的第一次 ``contract()`` 就会按那张别人的表选路，于是预热调用走了表里的
    ``kernel=backend`` 建议、``specialize.match`` 一直是 0，
    ``test_contract_cache_false…`` 的逐字段比对就崩了。

    因此这里**进入前快照、退出后恢复**（不是清空了事）：任何字段被用例改坏，退出时
    都回到进入前的值，下一条用例（以及下一个模块）看到的是干净世界。

    画像目录也换到临时树：``contract()`` 缺省走画像（``profile=None``），不隔离目录时
    它会读环境变量指向的、或 CWD 下的 ``./.cache``——那里可能躺着别人写下的
    ``kernel=backend`` 建议，命中后预热调用根本不进 ``specialize.match``，
    ``test_contract_cache_false…`` 的逐字段比对就会看到全 0。同时这挡住了本模块
    往仓库根的 ``.cache/pyqcd.einsum.profile.csv`` 追加行（那是真实发生过的副作用）。
    """
    env_names = (
        _cache.CACHE_ENV,
        _equivalence.AS_EQUIVALENCE_ENV,
        _equivalence.EQUIVALENCE_ENV,
        _kernel_force.KERNEL_FORCE_ENV,
        _specialize.DISABLE_ENV,
        _specialize.NOTICE_ENV,
        _specialize.FUSED_DISABLE_ENV,
        _profile_cache.PROFILE_ENV,
        _profile_cache.DIR_ENV,
        _threads.STANDARD_OMP_ENV,
        _threads.STANDARD_BLAS_ENV,
    )
    saved_env = {name: os.environ.get(name) for name in env_names}
    saved_api = {name: set(getattr(_api, name)) for name in _API_SET_FIELDS}
    saved_api.update({name: getattr(_api, name) for name in _API_VALUE_FIELDS})
    saved_cache = {name: getattr(_profile_cache, name)
                   for name in _CACHE_PLAIN_FIELDS}
    saved_threads = dict(_threads._state)
    pe.cache_clear()
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-as-equivalence-") as root:
            os.environ[_profile_cache.DIR_ENV] = os.path.join(root, "profile")
            # 画像与线程档位都是"进程只读/只设一次"的口径，先复位才能验本模块自己。
            _reset_profile_state()
            _threads.reset()
            yield
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for name in _API_SET_FIELDS:
            getattr(_api, name).clear()
            getattr(_api, name).update(saved_api[name])
        for name in _API_VALUE_FIELDS:
            setattr(_api, name, saved_api[name])
        for name, value in saved_cache.items():
            setattr(_profile_cache, name, value)
        _threads._state.clear()
        _threads._state.update(saved_threads)
        pe.cache_clear()


def _reset_profile_state() -> None:
    """把画像缓存清到"没配置、没加载、没建目录"的初始态；不改环境变量。

    ``_TABLE`` 是解析缓存（按 ``(路径, mtime, size)`` 失效），必须置回 ``None``：
    留着上一条用例的表就等于把别人的行当成本用例的结果。
    """
    for name, initial in _CACHE_PLAIN_FIELDS.items():
        setattr(_profile_cache, name, initial)
    _api._PROFILE_MISSING_NOTICED = False


def _prepare_contract_environment() -> None:
    """打开本模块依赖的开关；各用例外层仍由 ``_isolated_state`` 负责恢复。"""
    os.environ[_cache.CACHE_ENV] = "1"
    os.environ[_equivalence.EQUIVALENCE_ENV] = "1"
    os.environ[_specialize.DISABLE_ENV] = "1"
    os.environ[_specialize.NOTICE_ENV] = "0"
    os.environ.pop(_kernel_force.KERNEL_FORCE_ENV, None)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """屏蔽本契约不关心的提示与原生回落 warning。"""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def _operands(equation: str, *, seed: int = 20261116) -> list[np.ndarray]:
    """按方程左侧生成标签长度互不相同的 complex128 操作数。"""
    rng = np.random.default_rng(seed)
    result = []
    for term in equation.split("->", 1)[0].split(","):
        shape = tuple(LABEL_SIZES[label] for label in term)
        value = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        result.append(np.ascontiguousarray(value.astype(np.complex128)))
    return result


def _relative_error(actual: Any, expected: Any) -> float:
    """相对最大偏差；全零答案用 1 作尺度。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    scale = float(np.max(np.abs(expected_array))) or 1.0
    return float(np.max(np.abs(actual_array - expected_array))) / scale


def _assert_original_answer(actual: Any, expected: Any, label: str) -> None:
    """分别断言原方程答案的形状、dtype 与 rel <= 1e-12。"""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, (
        label, actual_array.shape, expected_array.shape)
    assert actual_array.dtype == expected_array.dtype, (
        label, actual_array.dtype, expected_array.dtype)
    error = _relative_error(actual_array, expected_array)
    assert error <= 1e-12, (label, error)


def _cache_snapshot() -> dict[str, dict[str, Any]]:
    """复制全部缓存字段，避免后续调用原地改变快照。"""
    return {name: dict(info) for name, info in pe.cache_stats()["caches"].items()}


@contextlib.contextmanager
def _normalize_spy() -> Iterator[list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    """包住真实 ``normalize_equation``，记录它是否被自动施加路径调用。"""
    original = _equivalence.normalize_equation
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    with patch.object(_equivalence, "normalize_equation", new=spy):
        yield calls


def _profile_key(equation: str, operands: list[np.ndarray]) -> dict[str, Any]:
    """``_api._profile_begin`` 查表用的键（numpy / CPU / 单一 dtype 的固定写法）。"""
    return {
        "shape": _profile_cache.format_shape([o.shape for o in operands]),
        "dtype": _api._profile_dtype(operands),
        "device": "cpu",
        "equation": "".join(equation.split()),
        "backend": "numpy",
    }


@contextlib.contextmanager
def _profile_suggests_generic(equation: str, operands: list[np.ndarray]) -> Iterator[None]:
    """开画像并写一行 ``kernel=generic``，让缺省的 ``generic`` 走**画像**分支。

    为什么需要这一层：``kernel_force`` 的缺省值现在就是 ``generic``（``auto`` 已删除），
    所以普通调用在 ``_api.contract`` 的 ``generic_only`` 分支里**整段跳过**匹配与等价
    改写——本模块要观测的 ``as_equivalence`` / ``specialize.match`` 都不可达。画像给出的
    ``generic`` 是**计时结论**、不是"关掉等价改写"这种语义指令（见 ``_api.contract``
    里 ``generic_only`` 的注释），因此这条路径照旧匹配 + 改写，正是能观测开关的那条。

    表写在 ``_isolated_state`` 已指到临时树的 ``PYQCD_EINSUM_PROFILE_DIR`` 里：进出都
    复位表状态，仓库根不会多出 ``.cache``。
    """
    saved = os.environ.get(_profile_cache.PROFILE_ENV)
    os.environ[_profile_cache.PROFILE_ENV] = "1"
    try:
        _reset_profile_state()
        key = _profile_key(equation, operands)
        _profile_cache.write_winner(
            key, {"kernel": _kernel_force.GENERIC_LABEL, "optimize": "auto",
                  "wall_ms": 0.001})
        picked = _profile_cache.read_winner(key)
        assert picked is not None, ("画像行必须能被同一个键查回（键算错了）", key)
        yield
    finally:
        if saved is None:
            os.environ.pop(_profile_cache.PROFILE_ENV, None)
        else:
            os.environ[_profile_cache.PROFILE_ENV] = saved
        _reset_profile_state()


def test_as_equivalence_value_controls_the_normalization_path():
    """显式 ``True`` 施加改写、``False`` 不施加；字符串 ``"false"`` 不得被当成真。

    缺省 ``kernel_force`` 已是 ``generic``（``auto`` 已删除），普通调用整段跳过匹配与
    改写，所以这里借画像给出的 ``generic`` 建议把这条路径打开（见
    :func:`_profile_suggests_generic`）；断言仍然只看 ``normalize_equation`` 是否真被
    调用，以及结果是不是原方程的答案。
    """
    with _isolated_state():
        _prepare_contract_environment()
        operands = _operands(OPERAND_ORDER_EQUATION)
        expected = np.einsum(OPERAND_ORDER_EQUATION, *operands)
        cases = (
            ("explicit True applies", True, 1),
            ("explicit False skips", False, 0),
            # ``bool("false")`` 是 ``True``：把字符串原样交给布尔判定会**反转**开关。
            ("string 'false' skips", "false", 0),
        )

        for label, value, expected_calls in cases:
            pe.cache_clear()
            with _profile_suggests_generic(OPERAND_ORDER_EQUATION, operands), \
                    _normalize_spy() as calls, _quiet():
                actual = pe.contract(
                    OPERAND_ORDER_EQUATION,
                    *operands,
                    as_equivalence=value,
                    cache=True,
                    profile=True,
                )
            assert len(calls) == expected_calls, (label, len(calls), calls)
            if expected_calls:
                assert calls[0][0][0] == OPERAND_ORDER_EQUATION, label
            _assert_original_answer(actual, expected, label)


def test_equivalence_only_returns_the_answer_to_the_original_equation():
    """三条映射分别只含顺序、输出轴序及二者复合，均与原方程 numpy 答案一致。"""
    with _isolated_state():
        _prepare_contract_environment()
        os.environ[_equivalence.AS_EQUIVALENCE_ENV] = "1"
        cases = (
            (OPERAND_ORDER_EQUATION, "operand-order"),
            (OUTPUT_AXES_EQUATION, "output-axes"),
            (COMPOSITE_EQUATION, "operand-order+output-axes"),
        )

        for index, (equation, expected_kind) in enumerate(cases):
            mapping = _equivalence.normalize_equation(equation, use=False)
            assert mapping is not None, equation
            assert mapping.kind == expected_kind, (equation, mapping.kind)
            operands = _operands(equation, seed=20261116 + index)
            expected = np.einsum(equation, *operands)
            with _quiet():
                actual = pe.contract(
                    equation, *operands, as_equivalence=True, cache=True)
            _assert_original_answer(actual, expected, expected_kind)


def test_contract_cache_false_leaves_every_cache_field_unchanged():
    """预热后以 ``cache=False`` 重跑同一方程，所有缓存字段逐项完全相等。

    预热必须用**非逐元素**方程：``2026-11-17`` 起逐元素/纯置换方程在
    ``contract()`` 里提前委派给后端 ``einsum``（它们没有 BLAS 结构，也不可能有
    专用形态），因此不会再碰 ``specialize.match`` 与规划器。用 ``ij,ij->ij``
    预热会让"所有缓存都为 0"，下面的逐字段比对就退化成空断言。

    缺省 ``kernel_force`` 已是 ``generic``（``auto`` 已删除）时，``contract()`` 整段
    跳过匹配与改写，``specialize.match`` / ``equivalence.normalize`` 根本不会被写；
    这里借画像给出的 ``generic`` 建议把两条缓存都预热起来（见
    :func:`_profile_suggests_generic`），三个字段才是真的可观测。
    """
    warm_equation = "ij,jk->ik"
    with _isolated_state():
        _prepare_contract_environment()
        warm = _operands(warm_equation)
        pe.cache_clear()

        with _profile_suggests_generic(warm_equation, warm), _quiet():
            pe.contract(warm_equation, *warm, cache=True, profile=True)
        before = _cache_snapshot()
        # 两个匹配器缓存必须真的被暖起来，否则下面的逐字段比对是空断言。
        # ``api.find_path`` 可能被**别的**用例预先暖过（进程内的计划缓存不在
        # ``cache_clear`` 的范围内），所以对它只断言"不再写"，不断言"已经写过"。
        for name in ("specialize.match", "equivalence.normalize"):
            assert before[name]["size"] >= 1, (name, before)
            assert before[name]["stores"] >= 1, (name, before)

        for info in before.values():
            assert set(info) == REQUIRED_CACHE_FIELDS, info

        with _quiet():
            pe.contract(warm_equation, *warm, cache=False, profile=True)
        after = _cache_snapshot()
        assert after == before, (before, after)
        # 逐项再说一遍"关闭就是不写"：stores 只许原地不动，绝不许增长。
        for name in CACHE_NAMES:
            assert after[name]["stores"] <= before[name]["stores"], (
                name, before[name], after[name])
            assert after[name]["size"] <= before[name]["size"], (
                name, before[name], after[name])


def test_negative_results_are_cached_with_miss_then_hit():
    """两个匹配器的负结果都遵循 miss+store、再 hit，且 size 不增长。"""
    with _isolated_state():
        _prepare_contract_environment()
        cases = (
            ("specialize.match", _specialize.match, SPECIALIZE_MISS_EQUATION),
            ("equivalence.normalize", _equivalence.normalize_equation,
             EQUIVALENCE_MISS_EQUATION),
        )

        for cache_name, lookup, equation in cases:
            pe.cache_clear()
            first = lookup(equation, use=True)
            first_info = dict(pe.cache_stats()["caches"][cache_name])
            second = lookup(equation, use=True)
            second_info = dict(pe.cache_stats()["caches"][cache_name])

            assert first is None and second is None, (cache_name, first, second)
            assert first_info["misses"] == 1, (cache_name, first_info)
            assert first_info["stores"] == 1, (cache_name, first_info)
            assert first_info["hits"] == 0, (cache_name, first_info)
            assert second_info["misses"] == 1, (cache_name, second_info)
            assert second_info["stores"] == 1, (cache_name, second_info)
            assert second_info["hits"] == 1, (cache_name, second_info)
            assert second_info["size"] == first_info["size"] == 1, (
                cache_name, first_info, second_info)


def test_contract_path_cache_switch_and_global_clear_semantics():
    """``cache=True`` 第二次命中；``False`` 零变化；清空后三缓存归零。"""
    with _isolated_state():
        _prepare_contract_environment()
        shapes = ((2, 3), (3, 4))
        pe.cache_clear()

        first_path, _ = pe.contract_path(
            PATH_EQUATION, *shapes, shapes=True, optimize="greedy", cache=True)
        before_hit = dict(pe.cache_stats()["caches"]["api.find_path"])
        second_path, _ = pe.contract_path(
            PATH_EQUATION, *shapes, shapes=True, optimize="greedy", cache=True)
        after_hit = dict(pe.cache_stats()["caches"]["api.find_path"])
        assert second_path == first_path, (first_path, second_path)
        assert after_hit["hits"] == before_hit["hits"] + 1, (
            before_hit, after_hit)

        before_false = _cache_snapshot()
        pe.contract_path(
            PATH_EQUATION, *shapes, shapes=True, optimize="greedy", cache=False)
        after_false = _cache_snapshot()
        assert after_false == before_false, (before_false, after_false)

        _specialize.match(SPECIALIZE_MISS_EQUATION, use=True)
        _equivalence.normalize_equation(EQUIVALENCE_MISS_EQUATION, use=True)
        before_clear = pe.cache_stats()
        enabled_before = before_clear["enabled"]
        assert enabled_before is True, before_clear
        for name in CACHE_NAMES:
            assert before_clear["caches"][name]["size"] > 0, (name, before_clear)

        pe.cache_clear()
        cleared = pe.cache_stats()
        assert cleared["enabled"] == enabled_before, (enabled_before, cleared)
        for name in CACHE_NAMES:
            info = cleared["caches"][name]
            assert info["size"] == 0, (name, info)
            assert info["hits"] == 0, (name, info)
            assert info["misses"] == 0, (name, info)
            assert info["stores"] == 0, (name, info)
            assert info["evictions"] == 0, (name, info)
            assert info["hit_rate"] is None, (name, info)


def test_kernel_force_auto_is_rejected():
    """``auto`` 已删除：显式取值与环境变量两条路都抛 ``ValueError``。

    自动选路已交给 ``--tune`` 的实测结论（配置表）。这里钉两件事：正文必须点名
    "只接受强制值"并列出合法集合；对照调用（显式 ``generic``）仍然可用——报错是
    "这个取值不存在"，不是把整条路封死。
    """
    with _isolated_state():
        _prepare_contract_environment()
        os.environ[_kernel_force.KERNEL_FORCE_ENV] = "topology"
        operands = _operands(CORPUS_EQUATION)

        def _expect_rejected(**kwargs: Any) -> str:
            try:
                with _quiet():
                    pe.contract(CORPUS_EQUATION, *operands, **kwargs)
            except ValueError as exc:
                message = str(exc)
            else:
                raise AssertionError(f"{kwargs} 必须抛 ValueError（auto 已删除）")
            assert "只接受强制值" in message, message
            for label in sorted(_kernel_force.ALLOWED_LABELS):
                assert label in message, (label, message)
            return message

        # 显式 auto（大小写都算）：即使环境里写着 topology 也照样报错，不被环境救活。
        for value in ("auto", "AUTO"):
            _expect_rejected(kernel_force=value)
        # 环境变量里的 auto 同样报错（与显式参数一条消息）。
        os.environ[_kernel_force.KERNEL_FORCE_ENV] = "auto"
        _expect_rejected()

        # 对照：显式 generic 仍然生效，结果与 numpy 一致。
        os.environ[_kernel_force.KERNEL_FORCE_ENV] = "topology"
        expected = np.einsum(CORPUS_EQUATION, *operands)
        with _quiet():
            actual = pe.contract(
                CORPUS_EQUATION, *operands, kernel_force="generic")
        record = _api.last_kernel()
        assert record is not None and record["requested"] == (), record
        _assert_original_answer(actual, expected, "kernel_force=generic")


def test_generic_kernel_force_composes_with_out_and_specialize_false():
    """generic 与 ``out=`` / ``specialize=False`` 均不冲突，数值与 numpy 一致。"""
    with _isolated_state():
        _prepare_contract_environment()
        rng = np.random.default_rng(20261118)
        left = np.ascontiguousarray(
            rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)))
        right = np.ascontiguousarray(
            rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5)))
        expected = np.einsum("ij,jk->ik", left, right)

        out = np.empty_like(expected)
        with _quiet():
            returned = pe.contract(
                "ij,jk->ik", left, right, out=out, kernel_force="generic")
        assert returned is out, (type(returned), type(out))
        _assert_original_answer(returned, expected, "generic with out")

        with _quiet():
            no_specialize = pe.contract(
                "ij,jk->ik", left, right,
                kernel_force="generic", specialize=False)
        _assert_original_answer(no_specialize, expected, "generic without specialize")


TESTS = (
    test_as_equivalence_value_controls_the_normalization_path,
    test_equivalence_only_returns_the_answer_to_the_original_equation,
    test_contract_cache_false_leaves_every_cache_field_unchanged,
    test_negative_results_are_cached_with_miss_then_hit,
    test_contract_path_cache_switch_and_global_clear_semantics,
    test_kernel_force_auto_is_rejected,
    test_generic_kernel_force_composes_with_out_and_specialize_false,
)


def test_einsum_as_equivalence_cache_contracts() -> None:
    """注册入口：逐条执行本模块契约；全跳过时也显式 SkipTest。"""
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
            f"{len(failures)} 条等价映射/缓存契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"as_equivalence/cache contracts did not run ({skipped} skipped)")


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
