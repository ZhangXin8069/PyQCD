"""契约：``optimize="none"`` 必须与 ``optimize=False`` 完全等价。

为什么值得单独立契约
--------------------
``"none"`` 是**人写**的写法（配置表里也是这一列的可读拼写），而 ``False`` 是规划器
认的**机器**写法。两者只要有一处没接上，症状都很安静：``contract_path`` 会拿
``"none"`` 去查注册表（``KeyError`` 或悄悄退化成 ``greedy``），``contract`` 则会走
看起正常、其实换了一条收缩序的路径——数值仍然"对"，只是慢，或者路径与用户要求的
不同。数值测试抓不到"路径是否被尊重"，所以这里钉三条能独立失败的断言：

1. ``contract_path(eq, *ops, optimize="none")`` 的 path 与 ``optimize=False``
   **完全相等**（列表逐元素相同），且大小写/空白写法都归一；
2. ``contract(eq, *ops, optimize="none")`` 与 ``optimize=False`` 的数值
   ``rel <= 1e-15``；
3. **双向一致**：配置表的 ``optimize`` 列把 ``False`` 写成 ``none``（可读、可重放），
   而表里写 ``none`` 的那一行读回来就是 ``False``，施加后复现同一 path。

非法名字**不许**被这里吞掉：``"nonense"`` 之类仍按原有逻辑报错（见对照断言）。
这三条都在临时目录里跑，仓库根不会留下 ``.cache``。

可直接运行::

    python -B -m pyqcd.testing._einsum_optimize_none_contract
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
import unittest
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from unittest.mock import patch

import numpy as np

from ..einsum import _api, _profile_cache, contract, contract_path

EQUATION = "ij,jk,kl->il"
SHAPES = ((3, 4), (4, 5), (5, 6))
RTOL = 1e-15
#: 同一方程换成别的优化器时会得到**不同**的 path：用来确认它确实"被当成了一回事"。
OTHER_OPTIMIZE = "greedy"


# -- 张量 -----------------------------------------------------------------------
def _operands(dtype: Any = np.float64, seed: int = 20261117) -> list[np.ndarray]:
    """按方程左侧生成 C 连续操作数（轴长见 :data:`SHAPES`）。"""
    rng = np.random.default_rng(seed)
    return [np.ascontiguousarray(rng.standard_normal(shape).astype(dtype))
            for shape in SHAPES]


def _relative_error(actual: Any, expected: Any) -> float:
    """相对最大偏差（全零答案用 1 作尺度，与库内同口径）。"""
    expected_array = np.asarray(expected)
    scale = float(np.max(np.abs(expected_array))) or 1.0
    return float(np.max(np.abs(np.asarray(actual) - expected_array))) / scale


def _path_of(operands: Sequence[np.ndarray], optimize: Any) -> list[tuple[int, ...]]:
    """``contract_path`` 给出的 path（规划缓存关掉：每次真的问一遍规划器）。"""
    path, _info = contract_path(EQUATION, *operands, optimize=optimize, cache=False)
    return [tuple(int(index) for index in step) for step in path]


# -- 环境 -----------------------------------------------------------------------
def _env_names() -> list[str]:
    return [name for name in (_profile_cache.PROFILE_ENV, _profile_cache.DIR_ENV)]


def _reset_state() -> None:
    """把配置表的进程内状态清到初始态（别的契约可能已经钉过目录/打过提示）。"""
    _profile_cache._TUNING = 0
    _profile_cache._TABLE = None
    _profile_cache._CONFIGURED_DIR = ""
    _profile_cache._LAST_ERROR = None
    if hasattr(_api, "_PROFILE_MISSING_NOTICED"):
        _api._PROFILE_MISSING_NOTICED = False
    noticed = getattr(_api, "_PROFILE_NOTICED", None)
    if isinstance(noticed, set):
        noticed.clear()


@contextlib.contextmanager
def _isolated() -> Iterator[str]:
    """临时 CWD + 干净的开关与进程内状态；退出时全部还原。"""
    saved_cwd = os.getcwd()
    names = _env_names()
    saved_env = {name: os.environ.get(name) for name in names}
    saved_table = _profile_cache._TABLE
    saved_dir = _profile_cache._CONFIGURED_DIR
    saved_tuning = _profile_cache._TUNING
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-optimize-none-") as root:
            work = os.path.join(root, "work")
            os.makedirs(work)
            os.chdir(work)
            try:
                for name in names:
                    os.environ.pop(name, None)
                os.environ[_profile_cache.PROFILE_ENV] = "1"
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
        _profile_cache._TABLE = saved_table
        _profile_cache._CONFIGURED_DIR = saved_dir
        _profile_cache._TUNING = saved_tuning


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """屏蔽本契约不关心的提示与原生回落 warning。"""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# -- 配置表 ---------------------------------------------------------------------
def _blank_row() -> dict[str, str]:
    """一整行的空壳（列取自被测模块的 ``PROFILE_COLUMNS``）。"""
    return {str(name): "" for name in _profile_cache.PROFILE_COLUMNS}


def _row(key: Mapping[str, Any], *, optimize: str, kernel: str = "generic",
         timing: str = "0.900") -> dict[str, str]:
    """按键造一行性能配置（``optimize`` 列写给定的文本）。"""
    row = _blank_row()
    for name in _profile_cache.KEY_FIELDS:
        row[name] = str(key.get(name, ""))
    row["kernel"] = kernel
    row["optimize"] = optimize
    if "wall_ms" in row:
        row["wall_ms"] = timing
    return row


def _write_table(directory: str, rows: Sequence[Mapping[str, str]]) -> str:
    """把若干行写成一张 CSV（直接写文本：``write_winner`` 是 tune 独占的写口）。"""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, _profile_cache.PROFILE_FILENAME)
    columns = [str(name) for name in _profile_cache.PROFILE_COLUMNS]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: str(row.get(name, "")) for name in columns})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    return path


def _read_column(directory: str, column: str) -> list[str]:
    """按列名读出配置表里**每一数据行**的该列原文（写盘格式的唯一证词）。"""
    path = os.path.join(directory, _profile_cache.PROFILE_FILENAME)
    with open(path, encoding="utf-8", newline="") as handle:
        return [str(row.get(column, "")) for row in csv.DictReader(handle)]


def _capture_key(directory: str, operands: Sequence[np.ndarray]) -> dict[str, Any]:
    """让 ``_api`` 自己把键递过来（探针返回 ``None``，调用照常往下走）。"""
    seen: list[dict[str, Any]] = []

    def spy(key: Mapping[str, Any]) -> dict[str, str] | None:
        seen.append(dict(key))
        return None

    with _quiet(), patch.object(_profile_cache, "read_winner", spy):
        contract(EQUATION, *operands, profile=True, profile_dir=directory)
    assert seen, "普通调用没有经过 read_winner：读路径必须走配置表的精确匹配接口"
    return seen[-1]


# ── 1. contract_path：两种写法给出同一条 path ────────────────────────────────
def test_contract_path_treats_none_as_false() -> None:
    """``optimize="none"`` 的 path 必须与 ``optimize=False`` 完全相等。"""
    operands = _operands()
    off_path = _path_of(operands, False)
    assert off_path == [(0, 1, 2)], (
        "``optimize=False`` 应当是单步 n 元 einsum（整条链一次收缩）", off_path)
    for spelling in ("none", "NONE", "  None  ", "nOnE"):
        got = _path_of(operands, spelling)
        assert got == off_path, (
            f"optimize={spelling!r} 的 path 与 optimize=False 不同", got, off_path)
    # 对照：别的优化器给出另一条 path，否则上面那条断言区分不出"是否被归一"。
    assert _path_of(operands, OTHER_OPTIMIZE) != off_path, (
        "该方程上 greedy 与 False 的 path 相同，这条断言区分不出归一是否生效")
    # 非法名字不许被吞掉：仍按原有逻辑报错（这里用注册表之外的拼写）。
    try:
        _path_of(operands, "no-such-optimizer")
    except KeyError as exc:
        assert "no-such-optimizer" in str(exc), str(exc)
    else:                                            # pragma: no cover - 失败路径
        raise AssertionError("未知的 optimize 名字必须报错，不能被静默当成 none")


# ── 2. contract：两种写法数值等价 ───────────────────────────────────────────
def test_contract_numeric_matches_false() -> None:
    """``optimize="none"`` 与 ``optimize=False`` 的数值 ``rel <= 1e-15``。"""
    for dtype in (np.float64, np.complex128):
        operands = _operands(dtype, seed=41)
        with _quiet():
            expected = contract(EQUATION, *operands, optimize=False, specialize=False)
        for spelling in ("none", "NONE", " none "):
            with _quiet():
                got = contract(EQUATION, *operands, optimize=spelling, specialize=False)
            error = _relative_error(got, expected)
            assert error <= RTOL, (
                f"optimize={spelling!r} 与 optimize=False 的数值不一致",
                np.dtype(dtype).name, error)
            assert np.asarray(got).dtype == np.dtype(dtype), (
                "dtype 不许被归一化改掉", np.dtype(dtype).name, np.asarray(got).dtype)
        numpy_expected = np.einsum(EQUATION, *operands)
        assert _relative_error(expected, numpy_expected) <= 1e-12, "参照本身不可信"


# ── 3. 配置表双向一致：写 none ↔ 读 False ──────────────────────────────────
def test_profile_column_round_trips_none() -> None:
    """``False`` 写成 ``none``；表里写 ``none`` 读回 ``False`` 并复现同一 path。"""
    with _isolated() as work:
        cache_dir = os.path.join(work, "cache")
        operands = _operands(seed=7)
        off_path = _path_of(operands, False)

        # 写入方向：``False`` 必须落成可读的 ``none``（不是 ``"False"``）。
        assert _api._profile_optimize(False) == "none", _api._profile_optimize(False)
        assert _api._profile_optimize("none") == "none"
        assert _api._profile_optimize("NONE") == "none", (
            "任一 ``none`` 拼写都该落成同一个可读写法（大小写不敏感）",
            _api._profile_optimize("NONE"))
        assert _api._profile_optimize(" greedy ").strip() == "greedy", \
            "其余名字原样保留（写盘时按 _text 去空白）"
        assert _api._profile_optimize(True) == "True", "别的取值不许被改写"

        # 读取方向：表里的 ``none`` 读回 ``False``，施加后 path 与 False 一致。
        assert _api._profile_optimize_value("none") is False, \
            _api._profile_optimize_value("none")
        assert _api._profile_optimize_value(" none ") is False
        assert _api._profile_optimize_value("greedy") == "greedy"
        assert _api._profile_optimize_value("no-such-optimizer") is None, \
            "表里的未知名字只能退回调用方的值，不许当成 none"

        key = _capture_key(cache_dir, operands)

        # 双向一致：``False`` 的列写法由 ``_api._profile_optimize`` 定，落进真正的
        # 写入口后列里必须就是 ``none``；读回该列再转回来必须又是 ``False``。
        spelling = _api._profile_optimize(False)
        assert spelling == "none", ("False 的列写法必须是 none", spelling)
        _profile_cache.write_winner(key, {"kernel": "generic", "optimize": spelling,
                                          "wall_ms": 0.9})
        raw = _read_column(cache_dir, "optimize")
        assert raw == ["none"], (
            "配置表的 optimize 列应写成可读、可重放的 none", raw)
        written = _profile_cache.read_winner(key)
        assert written is not None, ("刚写下的行读不回来", key)
        assert written.get("optimize") == "none", written.get("optimize")
        assert _api._profile_optimize_value(written.get("optimize")) is False, (
            "表里的 optimize=none 必须读回 False", written.get("optimize"))

        # 手工写的那一行同样要能读回（表是数据，不假设它一定出自 write_winner）。
        _write_table(cache_dir, [_row(key, optimize="none")])
        row = _profile_cache.read_winner(key)
        assert row is not None, ("表里写下的 none 行读不回来", key)
        assert _api._profile_optimize_value(row.get("optimize")) is False, (
            "表里的 optimize=none 必须读回 False", row.get("optimize"))

        with _quiet():
            applied = contract(EQUATION, *operands, profile=True,
                               profile_dir=cache_dir, specialize=False)
        observed = _api.last_profile() or {}
        with _quiet():
            requested_off = contract(
                EQUATION, *operands, optimize=False, profile=False, specialize=False)
        assert _relative_error(applied, requested_off) <= RTOL, (
            "按表施加的 none 与 optimize=False 的数值不一致",
            _relative_error(applied, requested_off))
        assert observed.get("optimize") is False, (
            "画像必须把表里的 none 施加成 False（不是字符串）", observed.get("optimize"))

        # 复现同一 path：表驱动的路径等于 ``optimize=False`` 的路径。
        table_path = _path_of(operands, _api._profile_optimize_value(row["optimize"]))
        assert table_path == off_path, (
            "按表读回的取值复现出的 path 与 optimize=False 不同", table_path, off_path)

        # 冲突判定用同一把尺子：表里 none + 调用方显式 "none" 不是冲突。
        _api._profile_check_conflict(key, row, caller_force=None, optimize="none",
                                     as_equivalence=None, has_equivalence=False)
        try:
            _api._profile_check_conflict(key, row, caller_force=None,
                                         optimize=OTHER_OPTIMIZE,
                                         as_equivalence=None, has_equivalence=False)
        except _profile_cache.ProfileConflict as exc:
            assert exc.field == "optimize", exc.field
        else:                                        # pragma: no cover - 失败路径
            raise AssertionError("显式 greedy 撞上表里的 none 时必须报冲突，不能被静默忽略")


TESTS = (
    test_contract_path_treats_none_as_false,
    test_contract_numeric_matches_false,
    test_profile_column_round_trips_none,
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


def test_einsum_optimize_none_contracts() -> None:
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
            f"{len(failures)} 条 optimize=none 契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest("optimize=none 契约全部跳过")


if __name__ == "__main__":
    raise SystemExit(main())
