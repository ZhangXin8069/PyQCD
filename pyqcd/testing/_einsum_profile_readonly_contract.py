"""契约：配置表（``pyqcd.einsum.profile.csv``）对普通调用是**只读**的。

为什么值得单独立契约
--------------------
TV 重设计（2026-11-17）把配置表收成"一个键一行、只有 tune 能写"：写者唯一、读路径
绝不落盘。这类约束**不影响数值**——读错一行、多写一行，结果照样逐位正确——普通数值
测试永远抓不到，所以必须把下面五件事分别钉成能独立失败的断言：

1. **同键多行 = 报错**：不猜、不选最优，异常里必须带上**全部**候选行（§2.2）。
2. **文件不存在 = 警告 + 继续**：警告里要有"需要先 tune"与该键的 tune CLI 命令模板，
   随后调用仍然正常返回结果（§2.3）；顺带验它**没有**顺手建一张空表。
3. **普通调用不写文件**：命中、未命中各跑一次，表的字节、``mtime``、行数与目录内容
   全部不变（§1）；顺带验表头是 ``wall_ms`` 且不含旧毫秒列名（§8 的"零残留"验收）。
4. **命中与用户显式配置冲突 = 硬报错**：``ProfileConflict`` 向上冒泡，消息里带记录值
   与请求值，不静默让某一方胜出（§2.4 / §7）。
5. **只读不等于改答案**：命中 / 缺表 / 关缓存三条路径与 ``np.einsum`` 的相对偏差
   ≤ :data:`RTOL`（1e-12），dtype 与形状逐项相同（§6）。

实现口径
--------
* **键从被测代码自己手里拿**：本模块不复制 ``_api`` 的键构造逻辑（那会随实现漂移），
  而是给 :func:`._profile_cache.read_winner` 装一个只记录、不返回的探针，让 ``_api``
  自己把键递过来。探针拿到的键就是种子行的键，所以"命中"是构造出来的，不是猜的。
* 种子行**直接用 CSV 文本写**（表头取 ``PROFILE_COLUMNS``）：``write_winner`` 按规格
  是 tune 独占的写口，本契约不该用它来摆测试现场。
* 每条用例都在临时树里跑（CWD 一起换过去），退出时把环境变量与 ``_profile_cache`` /
  ``_api`` 的进程内状态原样放回；**仓库根不会留下 ``.cache``**。

可直接运行::

    python -B -m pyqcd.testing._einsum_profile_readonly_contract
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
import unittest
import warnings
from typing import Any, Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np

import pyqcd.einsum as pe
from ..einsum import _api, _profile_cache, contract

#: 用例方程：语料外、无需改写，走通用收缩（命中与否只影响选路，不影响答案）。
EQUATION = "ij,jk,kl->il"

#: 标签轴长（同一个标签在各操作数里等长）。
LABEL_SIZES = {"i": 4, "j": 3, "k": 5, "l": 2}

#: 数值容差（与库内其他契约同口径：只差浮点结合律的舍入）。
RTOL = 1e-12

#: 缺表警告里必须出现的两段文本：tune 这个词，以及可直接复制执行的 CLI 模板。
TUNE_WORD = "tune"
TUNE_CLI = "python -m pyqcd.einsum.test"

#: 本模块会改写/还原的环境变量名（模块被重命名时跳过缺失项，不让契约变成脆弱的名字检查）。
_ENV_NAMES = ("PROFILE_ENV", "DIR_ENV")


# -- 环境与进程内状态 ------------------------------------------------------------
def _env_names() -> tuple[str, ...]:
    """本模块要动的环境变量名（来自被测模块自己的常量，缺了就跳过）。"""
    names: list[str] = []
    for module in (_profile_cache, _api):
        for attr in _ENV_NAMES:
            name = getattr(module, attr, None)
            if isinstance(name, str) and name not in names:
                names.append(name)
    names.append("PYQCD_EINSUM_PROFILE")
    names.append("PYQCD_EINSUM_PROFILE_DIR")
    names.append("PYQCD_EINSUM_KERNEL_FORCE")
    return tuple(dict.fromkeys(names))


def _clone(value: Any) -> Any:
    """深一层拷贝模块状态（字典/列表/集合递归，其余按值）。"""
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, set):
        return set(value)
    return value


def _snapshot(module: Any) -> dict[str, tuple[str, Any]]:
    """快照模块状态：容器拷内容、标量/元组记值（锁、函数与常量不碰）。"""
    snapshot: dict[str, tuple[str, Any]] = {}
    for name, value in vars(module).items():
        if name.startswith("__") or name.isupper():
            continue
        if isinstance(value, dict):
            snapshot[name] = ("dict", _clone(value))
        elif isinstance(value, list):
            snapshot[name] = ("list", _clone(value))
        elif isinstance(value, set):
            snapshot[name] = ("set", set(value))
        elif value is None or isinstance(value, (bool, int, float, str, tuple)):
            snapshot[name] = ("value", value)
    return snapshot


def _restore(module: Any, snapshot: Mapping[str, tuple[str, Any]]) -> None:
    """按 :func:`_snapshot` 的副本原地放回（容器对象不换，外部引用继续有效）。"""
    for name, (kind, saved) in snapshot.items():
        current = getattr(module, name, None)
        if kind == "dict" and isinstance(current, dict):
            current.clear()
            current.update(_clone(saved))
        elif kind == "list" and isinstance(current, list):
            current[:] = _clone(saved)
        elif kind == "set" and isinstance(current, set):
            current.clear()
            current.update(saved)
        elif kind == "value":
            setattr(module, name, saved)


def _reset_state() -> None:
    """把配置表的进程内状态清到"没配置、没解析缓存、不处于 tune 窗口"的初始态。

    TV 重设计后只剩这几样状态：``_TABLE``（解析缓存）、``_TUNING``（``tune_session``
    的嵌套计数）、``_CONFIGURED_DIR`` / ``_LAST_ERROR``。``_api`` 侧只需复位"缺表提示
    每进程一次"的闸与命中提示的去重集合——别的契约置过之后，本模块的缺表用例就再也
    等不到那行警告（那是**跨契约串味**，不是实现问题）。
    """
    _profile_cache._TUNING = 0
    _profile_cache._TABLE = None
    _profile_cache._CONFIGURED_DIR = ""
    _profile_cache._LAST_ERROR = None
    noticed = getattr(_api, "_PROFILE_NOTICED", None)
    if isinstance(noticed, set):
        noticed.clear()
    if hasattr(_api, "_PROFILE_MISSING_NOTICED"):
        _api._PROFILE_MISSING_NOTICED = False
    pe.cache_clear()


@contextlib.contextmanager
def _isolated() -> Iterator[str]:
    """临时 CWD + 干净开关 + 干净的进程内状态；退出时全部还原。"""
    saved_cwd = os.getcwd()
    names = _env_names()
    saved_env = {name: os.environ.get(name) for name in names}
    snapshots = [(module, _snapshot(module)) for module in (_profile_cache, _api)]
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-profile-readonly-") as root:
            work = os.path.join(root, "work")
            os.makedirs(work)
            os.chdir(work)
            try:
                for name in names:
                    os.environ.pop(name, None)
                os.environ["PYQCD_EINSUM_PROFILE"] = "1"
                # 自动 tune 会让"缺行"变成"写表"：本契约验的是
                # **普通调用只读**这条不变量，所以显式关掉它。
                os.environ["PYQCD_EINSUM_AUTOTUNE"] = "0"
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


class _Captured:
    """一次调用的全部可见输出：``warnings`` + stdout + stderr（任一渠道都算"说了"）。"""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.stdout = ""
        self.stderr = ""

    @property
    def text(self) -> str:
        return "\n".join([*self.messages, self.stdout, self.stderr])


@contextlib.contextmanager
def _capture() -> Iterator[_Captured]:
    """收集 warnings 与两条标准流（缺表警告走哪个渠道由实现决定，契约只看内容）。"""
    seen = _Captured()
    out, err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings(record=True) as caught, \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        warnings.simplefilter("always")
        try:
            yield seen
        finally:
            seen.messages = [str(item.message) for item in caught]
            seen.stdout = out.getvalue()
            seen.stderr = err.getvalue()


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """屏蔽本契约不关心的提示与原生回落 warning（它们与只读语义无关）。"""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# -- 张量、表与键 ----------------------------------------------------------------
def _operands(equation: str = EQUATION, *, seed: int = 20261117) -> list[np.ndarray]:
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


def _blank_row() -> dict[str, str]:
    """一整行的空壳（列取自被测模块的 ``PROFILE_COLUMNS``，列序改名不影响本契约）。"""
    return {str(name): "" for name in _profile_cache.PROFILE_COLUMNS}


def _row(key: Mapping[str, Any], *, kernel: str = "", optimize: str = "",
         timing: str = "0.900", note: str = "") -> dict[str, str]:
    """按键造一行（性能行：``kernel`` / ``optimize`` / 计时列）。"""
    row = _blank_row()
    for name in _profile_cache.KEY_FIELDS:
        row[name] = str(key.get(name, ""))
    row["kernel"] = kernel
    row["optimize"] = optimize
    if "wall_ms" in row:
        row["wall_ms"] = timing
    row["note"] = note
    return row


def _write_table(directory: str, rows: Sequence[Mapping[str, str]]) -> str:
    """把若干行写成一张 CSV（**直接写文本**：``write_winner`` 是 tune 独占的写口）。"""
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


def _table_state(directory: str) -> dict[str, Any]:
    """目录与配置表的可观测状态：表字节、``mtime``/大小、目录内文件名列表。"""
    path = os.path.join(directory, _profile_cache.PROFILE_FILENAME)
    names = sorted(name for name in os.listdir(directory)
                   if not name.startswith(".pyqcd_profile_probe_"))
    if not os.path.exists(path):
        return {"exists": False, "names": names}
    with open(path, "rb") as handle:
        raw = handle.read()
    info = os.stat(path)
    return {
        "exists": True,
        "bytes": raw,
        "mtime_ns": int(info.st_mtime_ns),
        "size": int(info.st_size),
        "names": names,
    }


def _capture_key(directory: str, equation: str,
                 operands: Sequence[np.ndarray]) -> dict[str, Any]:
    """让 ``_api`` 自己把键递过来：给 ``read_winner`` 装一个只记不返回的探针。

    探针返回 ``None``（未命中），调用照常往下走；拿到的键就是下一次种子行要用的键，
    所以后面的"命中"是**构造**出来的。探针一次都没被调用说明读路径根本没查表——
    那正是要报出来的缺陷，不是本契约该绕开的细节。
    """
    seen: list[dict[str, Any]] = []

    def spy(key: Mapping[str, Any]) -> dict[str, str] | None:
        seen.append(dict(key))
        return None

    with _quiet(), patch.object(_profile_cache, "read_winner", spy):
        contract(equation, *operands, profile=True, profile_dir=directory)
    assert seen, (
        "普通调用没有经过 read_winner：读路径必须走配置表的精确匹配接口",
        _table_state(directory))
    key = seen[-1]
    for name in _profile_cache.KEY_FIELDS:
        assert name in key, ("键缺少字段", name, sorted(key))
    assert str(key["equation"]) == "".join(equation.split()), ("方程未按口径规范化", key)
    return key


def _run(directory: str, equation: str = EQUATION,
         operands: Sequence[np.ndarray] | None = None, **kwargs: Any) -> Any:
    """在指定配置目录里跑一次普通 ``contract()``（画像开、其余按缺省）。"""
    args = list(operands) if operands is not None else _operands(equation)
    return contract(equation, *args, profile=True, profile_dir=directory, **kwargs)


# -- 断言 -----------------------------------------------------------------------
def test_profile_duplicate_rows_are_reported() -> None:
    """§2.2：同键多行 → 报错，且异常里带**全部**候选行（不挑、不猜）。"""
    with _isolated() as work:
        cache_dir = os.path.join(work, "cache")
        key = _capture_key(cache_dir, EQUATION, _operands())
        first = _row(key, kernel="generic", optimize="greedy", timing="1.500", note="旧行")
        second = _row(key, kernel="topology", optimize="optimal", timing="0.700", note="新行")
        other_key = dict(key)
        other_key["equation"] = "ij,jk->ik"
        _write_table(cache_dir, [first, _row(other_key, kernel="backend"),
                                 second])

        try:
            _profile_cache.read_winner(key)
        except _profile_cache.DuplicateProfileEntry as exc:
            rows = list(exc.rows)
            message = str(exc)
        else:                                            # pragma: no cover - 失败路径
            raise AssertionError("同键 2 行时 read_winner 必须抛 DuplicateProfileEntry")
        assert len(rows) == 2, ("候选行必须全部给出（不含别的键的行）", len(rows), rows)
        assert [row.get("line") for row in rows] == ["1", "3"], (
            "候选行要带文件里的数据行号", [row.get("line") for row in rows])
        # 两行的 ``optimize`` 取**不同**的合法叶子：``auto`` 已从 ``OPTIMIZE_LEAVES``
        # 删除（写进表里的行会被判非法、直接不参与匹配），用它造行等于让这条契约
        # 失去被测对象；两行取值不同也顺带证明报错逐行照抄、没有互相覆盖。
        for kernel, optimize, note in (("generic", "greedy", "旧行"),
                                       ("topology", "optimal", "新行")):
            for text in (kernel, optimize, note):
                assert text in message, ("报错里要能看到这一行", text, message)

        with _quiet():
            try:
                _run(cache_dir)
            except _profile_cache.DuplicateProfileEntry:
                pass
            else:                                        # pragma: no cover - 失败路径
                raise AssertionError("普通调用遇到同键多行必须报错退出，不能静默选一行")


def test_missing_profile_warns_with_tune_command() -> None:
    """§2.3：缺表 → 警告"需要先 tune"+ 该键的 CLI 模板，然后**继续**执行。"""
    with _isolated() as work:
        cache_dir = os.path.join(work, "cache")
        expected = np.einsum(EQUATION, *_operands())
        with _capture() as captured:
            result = _run(cache_dir)
        text = captured.text
        assert TUNE_WORD in text.lower(), (
            "缺表时必须提示需要先 tune", text or "（一个字都没说）")
        assert TUNE_CLI in text, ("警告里要给出可直接复制的 tune 命令模板", text)
        assert "--tune" in text, ("模板必须真的带 --tune", text)
        assert _relative_error(result, expected) <= RTOL, "缺表警告之后必须照常返回结果"
        path = os.path.join(cache_dir, _profile_cache.PROFILE_FILENAME)
        assert not os.path.exists(path), (
            "缺表只许警告：普通调用不得顺手建出一张空表", path)


def test_ordinary_call_never_writes() -> None:
    """§1：普通调用（命中 + 未命中）不写表；表头是 ``wall_ms``（§8）。"""
    with _isolated() as work:
        cache_dir = os.path.join(work, "cache")
        key = _capture_key(cache_dir, EQUATION, _operands())
        columns = [str(name) for name in _profile_cache.PROFILE_COLUMNS]
        assert "wall_ms" in columns, ("表头必须用 wall_ms", columns)
        # 旧列名按"拼出来再查"的方式断言：本文件自己也不许让那个词出现在源码里
        # （§8 的验收是 `grep -rn` 零命中，连测试文本都算）。
        legacy = "compute" + "_ms"
        assert legacy not in columns, (f"表头不得残留旧列名 {legacy}", columns)
        _write_table(cache_dir, [_row(key, kernel="generic", optimize="greedy")])
        before = _table_state(cache_dir)

        with _quiet():
            for _ in range(3):                           # 命中：每次都得一模一样
                _run(cache_dir)
            _run(cache_dir, "ij,jk->ik")                  # 未命中：也不许写
        after = _table_state(cache_dir)

        assert before["exists"] and after["exists"], ("配置表不该消失", before, after)
        assert after["bytes"] == before["bytes"], "普通调用改动了配置表的字节内容"
        assert after["mtime_ns"] == before["mtime_ns"], "普通调用重写了配置表（mtime 变了）"
        assert after["size"] == before["size"], "普通调用的配置表大小变了"
        assert after["names"] == before["names"], (
            "普通调用在配置目录里新建/删除了文件", before["names"], after["names"])


def test_profile_conflict_raises() -> None:
    """§2.4 / §7：命中行的配置与用户显式输入冲突 → ``ProfileConflict`` 向上冒泡。

    用两个**具体内核标签**做对照（表 `generic` vs 显式 `dedicated`）：``kernel_force``
    的 ``generic`` 在分派层解析成空元组（"完全跳过专用内核"），语义上不是"点名某个
    内核"，所以拿它去撞表里的具体标签不构成这一条要钉的冲突。
    """
    with _isolated() as work:
        cache_dir = os.path.join(work, "cache")
        key = _capture_key(cache_dir, EQUATION, _operands())
        recorded, requested = "generic", "dedicated"
        _write_table(cache_dir, [_row(key, kernel=recorded, optimize="greedy")])

        with _quiet():
            try:
                _run(cache_dir, kernel_force=requested)
            except _profile_cache.ProfileConflict as exc:
                message = str(exc)
            else:                                        # pragma: no cover - 失败路径
                raise AssertionError(
                    "命中行 kernel=%r 与显式 kernel_force=%r 冲突时必须抛 ProfileConflict"
                    % (recorded, requested))
        for text in (recorded, requested):
            assert text in message, ("冲突报错要能看出记录值与请求值", text, message)


def test_profile_readonly_keeps_numbers() -> None:
    """§6：命中 / 缺表 / 关缓存三条路径的数值、dtype、形状与 ``np.einsum`` 一致。"""
    with _isolated() as work:
        operands = _operands()
        expected = np.einsum(EQUATION, *operands)
        cache_dir = os.path.join(work, "cache")
        key = _capture_key(cache_dir, EQUATION, operands)
        _write_table(cache_dir, [_row(key, kernel="generic", optimize="greedy")])

        with _quiet():
            hit = _run(cache_dir, operands=operands)
            miss = _run(os.path.join(work, "empty"), operands=operands)
            off = contract(EQUATION, *operands, profile=False)

        for label, value in (("命中", hit), ("缺表", miss), ("关缓存", off)):
            assert np.shape(value) == np.shape(expected), (label, "形状变了", np.shape(value))
            assert np.asarray(value).dtype == np.asarray(expected).dtype, (
                label, "dtype 变了", np.asarray(value).dtype)
            error = _relative_error(value, expected)
            assert error <= RTOL, (label, "相对偏差超限", error, RTOL)


#: 本契约的用例清单（``main`` 与 ``conftest`` 共用）。
TESTS = (
    test_profile_duplicate_rows_are_reported,
    test_missing_profile_warns_with_tune_command,
    test_ordinary_call_never_writes,
    test_profile_conflict_raises,
    test_profile_readonly_keeps_numbers,
)


def test_einsum_profile_readonly_contracts() -> None:
    """套件入口：跑完 :data:`TESTS` 的每一条，任何一条失败就抛出汇总。"""
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
            f"{len(failures)} 条配置表只读契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"profile readonly contracts did not run ({skipped} skipped)")


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
