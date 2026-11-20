"""契约：配置表（``pyqcd.einsum.profile.csv``）的读写口径——只读/单写者/同键一行。

为什么值得单独立契约
--------------------
TV 重设计（2026-11-17，§1/§2/§3/§7/§8）把这张表收成**标定产物**：只有 ``--tune``
能写，普通调用只读；同一个键**只允许一行**，出现多行就直接报错（不猜、不选最优）；
毫秒列的列名是 ``wall_ms``（整个 ``contract()`` 的墙钟）。这一层不参与数值计算，
所以普通数值测试永远碰不到它：多写一行、选错一行，结果照样逐位正确，只是脏、只是
慢、只是把别的运行环境带进来。本模块把下面十三件事分别钉成能独立失败的断言：

1. **冻结名字与表头**：``PROFILE_FILENAME`` / ``KEY_FIELDS`` / ``PROFILE_COLUMNS``
   （表头含 ``wall_ms``、不含旧毫秒列名——§8 的零残留验收在源码层也成立）。
2. **缺文件 → :class:`ProfileMissing`**：带该键的 tune 命令模板（§2.3 的提示就是它），
   并且**不建表、不建目录**。
3. **``read_winner`` 0 行**：文件在、键不在 → ``None``（不猜、不报错）。
4. **``read_winner`` 1 行**：命中返回副本（带数据行号 ``line``），改返回值改不动表。
5. **``read_winner`` 2 行**：抛 :class:`DuplicateProfileEntry`，``rows`` 是**全部**
   候选（含各自 ``line``），消息里能看到每一行；``best()`` 同样抛（不静默选一行）。
6. **读路径绝不写**：缺表不建目录；有表时命中与未命中都不改字节、``mtime``、大小。
7. **``write_winner`` 只留一行**：同键旧行（含更快的旧行）全部删除，别的键原样保留；
   ``wall_ms`` 按三位小数落盘。
8. **等价行**：``equivalence=`` 非空时该行只含 ``target_equation`` + 三个置换字段
   + ``note``，性能字段留空；:func:`equivalence_payload` 能把它解回可执行载荷。
9. **``tune_session`` 期间不读不写**：读当无命中、写静默不落盘（嵌套安全），
   退出后读写立刻恢复。
10. **关闭零副作用**：``PYQCD_EINSUM_PROFILE=0`` 时 ``enabled()`` 为假、读返回
    ``None``、写静默不写，且临时 CWD 里连目录都不出现。
11. **``format_shape``**：``x`` 连轴、``|`` 连操作数，已规范化字符串原样返回，
    无法解析给空串；列表写法与字符串写法命中同一把键。
12. **目录优先级**：``PYQCD_EINSUM_PROFILE_DIR`` 压过默认 ``./.cache``，``configure``
    的参数再压过环境变量；``ensure_configured`` 只补齐、不改写已钉住的目录。
13. **``summary``**：``rows`` / ``keys`` 如实计数（同键多行时 ``keys`` 仍是 1，用来
    一眼看出异常），缺表报 0 而不是抛。
14. **``wall_ms`` 是全程墙钟**：在 :func:`read_winner` 里注入一段睡眠——它不是收缩执行
    的时间，却必须出现在 ``wall_ms`` 里（§4 的"只有一列、且是整段 ``contract()``"）。
    观测口里不许残留旧字段名，也不许还有第二列拆分计时。
15. **``optimize`` 列的 7 叶子**：7 个叶子
    写入读回一致；连字符旧写法（``branch-1``）读入归一成下划线、**不**视为非法。
16. **``auto`` 是非法值**：读侧只警告**一次**，该行**不参与** ``read_winner`` 匹配
    （不静默当 ``greedy``）；``summary()`` 的 ``invalid`` 如实计数。
17. **带已删列（``optimize_topology_v2``）的老文件**照样能读：该列不再出现，缺省 ``greedy``。
18. **``schema_version``**：新写为 :data:`PROFILE_SCHEMA_VERSION`（``3``），老行（文件里
    没有这一列）读回 ``0`` 且仍可用；老行不降级、不删除。

口径与边界
----------
* 种子行**分两种写法**：要验 ``write_winner`` 就用它（那是被测的写口）；要摆
  "同键多行"这种**只能坏着存在**的现场，就直接写 CSV 文本（``write_winner`` 按规格
  不可能造出两行）。
* 每条用例都在临时树里跑（CWD 一起换过去），退出时把环境变量与 ``_profile_cache``
  的进程内状态原样放回；**仓库根不会留下 ``.cache``**。
* 不做任何计时断言：本模块只关心"写了什么、读了什么"，不关心它多快。

可直接运行::

    python -B -m pyqcd.testing._einsum_profile_cache_contract
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
import time
import unittest
import warnings
from typing import Any, Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np

from ..einsum import _api, _profile_cache, contract

#: 旧毫秒列名（按"拼出来再查"的方式断言：本文件也不许让那个词出现在源码里，
#: 否则 §8 的 ``grep`` 验收会被测试文本自己打回）。
LEGACY_MS_COLUMN = "compute" + "_ms"

#: 本模块会改写/还原的环境变量（目录与总开关）。
_ENV_NAMES = (_profile_cache.PROFILE_ENV, _profile_cache.DIR_ENV)

#: 第 14 条注入的"读表"睡眠（秒）：远大于噪声、又小到不拖慢本模块。
READ_SLEEP = 0.05

#: ``_profile_cache`` 的进程内状态（``_TABLE`` 是解析缓存，按内容拷贝）。锁与常量不碰。
#: 待发警告与"已警告过"的集合也要隔离：否则上一个用例发出的警告会把本用例的吞掉。
_STATE_NAMES = ("_TABLE", "_TUNING", "_CONFIGURED_DIR", "_LAST_ERROR", "_HOST",
                "_PENDING_WARNINGS", "_WARNED_CELLS")


# -- 环境与进程内状态 ------------------------------------------------------------
def _snapshot() -> dict[str, Any]:
    """状态快照：``_TABLE`` 是 ``(路径, mtime, size, 行列表)``，行也要拷一层。"""
    state: dict[str, Any] = {}
    for name in _STATE_NAMES:
        value = getattr(_profile_cache, name)
        if name == "_TABLE" and value is not None:
            value = (value[0], value[1], value[2], [dict(row) for row in value[3]])
        elif isinstance(value, list):
            value = list(value)
        elif isinstance(value, set):
            value = set(value)
        state[name] = value
    return state


def _restore(state: Mapping[str, Any]) -> None:
    """按 :func:`_snapshot` 的副本原样放回。"""
    for name, value in state.items():
        setattr(_profile_cache, name, value)


def _reset_state() -> None:
    """清到"没配置、没解析缓存、不处于 tune 窗口、没有待发/已发警告"的初始态。"""
    _profile_cache._TABLE = None
    _profile_cache._TUNING = 0
    _profile_cache._CONFIGURED_DIR = ""
    _profile_cache._LAST_ERROR = None
    _profile_cache._PENDING_WARNINGS = []
    _profile_cache._WARNED_CELLS = set()


def _forget() -> None:
    """丢掉解析缓存：本模块会直接改写 CSV 文本，别让 ``(路径, mtime, size)`` 侥幸命中。"""
    _profile_cache._TABLE = None


@contextlib.contextmanager
def _isolated() -> Iterator[str]:
    """临时 CWD + 干净开关 + 干净的进程内状态；退出时全部还原。"""
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    saved_state = _snapshot()
    try:
        with tempfile.TemporaryDirectory(prefix="pyqcd-profile-cache-") as root:
            work = os.path.join(root, "work")
            os.makedirs(work)
            os.chdir(work)
            try:
                for name in _ENV_NAMES:
                    os.environ.pop(name, None)
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
        _restore(saved_state)


# -- 键、行与表 ------------------------------------------------------------------
def _key(**overrides: Any) -> dict[str, Any]:
    """主表键的稀疏写法（``shape`` 用列表写法，顺带验归一化）。"""
    key = {
        "shape": [(2, 3), (3, 4)],
        "dtype": "complex128",
        "device": "cpu",
        "equation": "ij,jk->ik",
        "backend": "numpy",
    }
    key.update(overrides)
    return key


def _blank_row() -> dict[str, str]:
    """一整行的空壳（列取自被测模块的 ``PROFILE_COLUMNS``，列序改名不影响本契约）。"""
    return {str(name): "" for name in _profile_cache.PROFILE_COLUMNS}


def _row(key: Mapping[str, Any], *, kernel: str = "", optimize: str = "",
         wall_ms: str = "0.900", note: str = "") -> dict[str, str]:
    """按键造一行性能行（键列按实现口径规范化，免得写成"幽灵未命中"）。"""
    row = _blank_row()
    for name in _profile_cache.KEY_FIELDS:
        value = key.get(name, "")
        if name == "shape":
            value = _profile_cache.format_shape(value)
        row[name] = str(value)
    row["kernel"] = kernel
    row["optimize"] = optimize
    row["wall_ms"] = wall_ms
    row["note"] = note
    return row


def _table_path(directory: str) -> str:
    return os.path.join(directory, _profile_cache.PROFILE_FILENAME)


def _write_table(directory: str, rows: Sequence[Mapping[str, str]]) -> str:
    """直接写 CSV 文本（表头取 ``PROFILE_COLUMNS``）：造"只能坏着存在"的现场。"""
    os.makedirs(directory, exist_ok=True)
    path = _table_path(directory)
    columns = [str(name) for name in _profile_cache.PROFILE_COLUMNS]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: str(row.get(name, "")) for name in columns})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    _forget()
    return path


def _read_rows(path: str) -> list[list[str]]:
    """整份 CSV（含表头行）；用真实 ``open``。"""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _data_rows(path: str) -> list[dict[str, str]]:
    """数据行（表头按名字映射）。"""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _state_of(directory: str) -> dict[str, Any]:
    """目录与配置表的可观测状态：表字节、``mtime``/大小、目录内文件名列表。"""
    path = _table_path(directory)
    names = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
    if not os.path.exists(path):
        return {"exists": False, "names": names}
    with open(path, "rb") as handle:
        raw = handle.read()
    info = os.stat(path)
    return {"exists": True, "bytes": raw, "mtime_ns": int(info.st_mtime_ns),
            "size": int(info.st_size), "names": names}


# -- 1. 冻结名字与表头 -----------------------------------------------------------
def test_frozen_names_and_columns() -> None:
    """§7/§8：文件名、键字段与表头是冻结口径；毫秒列叫 ``wall_ms``，旧名零残留。"""
    assert _profile_cache.PROFILE_FILENAME == "pyqcd.einsum.profile.csv", (
        "配置表文件名是冻结口径", _profile_cache.PROFILE_FILENAME)
    assert tuple(_profile_cache.KEY_FIELDS) == (
        "shape", "dtype", "device", "equation", "backend"), _profile_cache.KEY_FIELDS
    columns = [str(name) for name in _profile_cache.PROFILE_COLUMNS]
    assert "wall_ms" in columns, ("表头必须用 wall_ms", columns)
    assert LEGACY_MS_COLUMN not in columns, (
        f"表头不得残留旧列名 {LEGACY_MS_COLUMN}", columns)
    for name in ("kernel", "optimize", "target_equation", "operand_order",
                 "operand_permutations", "output_permutation", "note"):
        assert name in columns, ("表头缺列", name, columns)
    # 三个异常类同属一族：CLI 才能"报错但不当崩溃"。
    for exc in (_profile_cache.ProfileMissing,
                _profile_cache.DuplicateProfileEntry,
                _profile_cache.ProfileConflict):
        assert issubclass(exc, _profile_cache.ProfileError), exc


# -- 2. 缺文件 → ProfileMissing --------------------------------------------------
def test_read_winner_missing_file_raises_profile_missing() -> None:
    """§2.3/§7：缺表 → ``ProfileMissing``，``cli_hint`` 是能直接复制的 tune 命令。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        try:
            _profile_cache.read_winner(_key())
        except _profile_cache.ProfileMissing as exc:
            hint, path, message = exc.cli_hint, exc.path, str(exc)
        else:                                            # pragma: no cover - 失败路径
            raise AssertionError("配置表不存在时 read_winner 必须抛 ProfileMissing")

        assert path == os.path.abspath(_table_path(directory)), path
        assert "python -m pyqcd.einsum.test" in hint, ("提示里要有 CLI 入口", hint)
        assert "--tune" in hint, ("提示里要有 --tune", hint)
        assert "--compare opt_einsum" in hint, ("提示里要有 --compare", hint)
        assert "--equation" in hint and "--shape" in hint, hint
        assert "需要先 tune" in message, message
        # 缺表只许"报"，不许"造"：目录与文件都不出现。
        assert not os.path.exists(directory), ("读路径不得建目录", os.listdir(work))


# -- 3. read_winner：0 行 --------------------------------------------------------
def test_read_winner_zero_rows_returns_none() -> None:
    """§7：文件在、键不在 → ``None``（不猜、不报错）；只有表头也一样。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        _write_table(directory, [])
        assert _profile_cache.read_winner(_key()) is None, "空表必须当未命中"

        path = _write_table(directory, [_row(_key(equation="ab,bc->ac"),
                                             kernel="backend")])
        assert _profile_cache.read_winner(_key()) is None, (
            "别的键的行不该被当成命中", _read_rows(path))


# -- 4. read_winner：1 行 --------------------------------------------------------
def test_read_winner_single_row_returns_copy() -> None:
    """§7：命中返回**副本**（带数据行号），改返回值改不动表。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        _profile_cache.write_winner(key, {"kernel": "kupl_dedicated",
                                         "optimize": "greedy", "wall_ms": 0.25})
        _forget()

        row = _profile_cache.read_winner(key)
        assert isinstance(row, dict), row
        assert row["kernel"] == "kupl_dedicated", row
        assert row["optimize"] == "greedy", row
        assert row["wall_ms"] == "0.250", ("三位小数落盘", row["wall_ms"])
        assert row["equation"] == "ij,jk->ik", row
        assert row["shape"] == "2x3|3x4", ("键列按归一化形状存", row["shape"])
        assert row["line"] == "1", ("命中要能钉到具体记录", row)

        row["kernel"] = "tampered"
        again = _profile_cache.read_winner(key)
        assert again["kernel"] == "kupl_dedicated", ("返回值必须是副本", again)


# -- 5. read_winner：2 行 → 报错并列候选 -----------------------------------------
def test_read_winner_duplicate_rows_raise_with_all_candidates() -> None:
    """§2.2/§7：同键多行 → 抛错并给出**全部**候选行（含行号），不挑、不猜。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        other = dict(key, equation="ab,bc->ac")
        _write_table(directory, [
            _row(key, kernel="kupl_dedicated", optimize="dp", note="旧行"),
            _row(other, kernel="backend"),
            _row(key, kernel="kupl_topology", optimize="greedy", note="新行"),
        ])
        try:
            _profile_cache.read_winner(key)
        except _profile_cache.DuplicateProfileEntry as exc:
            rows = [dict(item) for item in exc.rows]
            message = str(exc)
        else:                                            # pragma: no cover - 失败路径
            raise AssertionError("同键 2 行时 read_winner 必须抛 DuplicateProfileEntry")

        assert len(rows) == 2, ("候选行必须是全部、且不含别的键", len(rows), rows)
        assert [row.get("line") for row in rows] == ["1", "3"], (
            "候选行要带文件里的数据行号", [row.get("line") for row in rows])
        for text in ("kupl_dedicated", "kupl_topology", "旧行", "新行"):
            assert text in message, ("报错里要能看到这一行", text, message)
        assert "只允许 1 行" in message, message
        # 便利读法也不能偷偷选一行：数据错误照旧抛。
        try:
            _profile_cache.best(key)
        except _profile_cache.DuplicateProfileEntry:
            pass
        else:                                            # pragma: no cover - 失败路径
            raise AssertionError("best() 也不许多行时猜一行")
        # 同键多行只让**那个键**不可用：别的键仍然读得到。
        assert _profile_cache.read_winner(other) is not None, "异键必须仍可读"


# -- 6. 读路径绝不写 -------------------------------------------------------------
def test_read_winner_never_writes() -> None:
    """§1/§7：读路径零写入——缺表不建目录，有表时命中/未命中都不改文件。"""
    with _isolated() as work:
        missing = os.path.join(work, "missing")
        _profile_cache.configure(missing)
        try:
            _profile_cache.read_winner(_key())
        except _profile_cache.ProfileMissing:
            pass
        assert not os.path.exists(missing), ("读路径不得建目录", os.listdir(work))

        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        _profile_cache.write_winner(key, {"kernel": "generic", "optimize": "optimal",
                                         "wall_ms": 1.5})
        _forget()
        before = _state_of(directory)

        for _ in range(3):
            assert _profile_cache.read_winner(key) is not None
        assert _profile_cache.read_winner(_key(equation="ab,bc->ac")) is None
        assert _profile_cache.best(_key(equation="ab,bc->ac")) is None
        _profile_cache.summary()

        after = _state_of(directory)
        assert before["bytes"] == after["bytes"], "读路径改动了配置表的字节内容"
        assert before["mtime_ns"] == after["mtime_ns"], (
            "读路径重写了配置表（mtime 变了）")
        assert before["size"] == after["size"], "配置表大小变了"
        assert before["names"] == after["names"], (
            "读路径在配置目录里新建/删除了文件", before["names"], after["names"])


# -- 7. write_winner：同键只留一行 -----------------------------------------------
def test_write_winner_keeps_single_row_and_drops_same_key() -> None:
    """§3.3/§7：写前先删同键**所有**行（含更快的旧行），别的键原样保留。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        other = dict(key, equation="ab,bc->ac")
        _write_table(directory, [
            _row(key, kernel="stale_fast", optimize="dp", wall_ms="0.010"),
            _row(other, kernel="backend", optimize="branch_1", wall_ms="5.000"),
            _row(key, kernel="stale_slow", optimize="greedy", wall_ms="9.000"),
        ])
        path = _table_path(directory)
        _profile_cache.write_winner(key, {"kernel": "kupl_dedicated",
                                         "optimize": "optimal", "wall_ms": 2.0})
        _forget()

        table = _data_rows(path)
        same = [row for row in table if row["equation"] == "ij,jk->ik"]
        assert len(same) == 1, ("同键只许留 1 行", len(same), table)
        assert same[0]["kernel"] == "kupl_dedicated", same[0]
        assert same[0]["optimize"] == "optimal", same[0]
        assert same[0]["wall_ms"] == "2.000", same[0]
        assert "stale_fast" not in {row["kernel"] for row in table}, (
            "更快的旧行也必须被删（防异环境污染）", table)
        kept = [row for row in table if row["equation"] == "ab,bc->ac"]
        assert len(kept) == 1 and kept[0]["kernel"] == "backend", (
            "别的键的行不许被冲掉", table)

        # 整表重写（不是追加）：表头只出现一次。
        raw = _read_rows(path)
        assert raw[0] == [str(name) for name in _profile_cache.PROFILE_COLUMNS], raw[0]
        assert sum(1 for line in raw if line and line[0] == "shape") == 1, raw


# -- 8. 等价行只写载荷 -----------------------------------------------------------
def test_write_winner_equivalence_row_has_payload_only() -> None:
    """§3.3b/§7：等价行只写载荷 + ``note``，性能字段留空；能被解回可执行改写。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        payload = {
            "target_equation": "cadb,ac,bd->",
            "operand_order": [1, 0],
            "operand_permutations": [None, [1, 0]],
            "output_permutation": None,
        }
        _profile_cache.write_winner(key, {"kernel": "generic", "wall_ms": 0.1})
        _profile_cache.write_winner(key, equivalence=payload)
        _forget()

        table = _data_rows(_table_path(directory))
        assert len(table) == 1, ("等价行也该删掉同键的性能行", table)
        row = table[0]
        assert row["target_equation"] == "cadb,ac,bd->", row
        assert row["note"] == _profile_cache.EQUIVALENCE_NOTE, (
            "缺省备注就是冻结文案", row["note"])
        for name in _profile_cache.PERFORMANCE_FIELDS:
            assert row[name] == "", ("等价行不许带性能配置", name, row)

        decoded = _profile_cache.equivalence_payload(row)
        assert isinstance(decoded, dict), decoded
        assert decoded["target_equation"] == "cadb,ac,bd->", decoded
        assert tuple(decoded["operand_order"]) == (1, 0), decoded
        assert decoded["operand_permutations"][0] is None, decoded
        assert tuple(decoded["operand_permutations"][1]) == (1, 0), decoded
        assert decoded["output_permutation"] is None, decoded
        assert decoded["note"] == _profile_cache.EQUIVALENCE_NOTE, decoded

        # 性能行解不出载荷：不会被误当成可执行改写。
        _profile_cache.write_winner(key, {"kernel": "generic", "wall_ms": 0.1})
        _forget()
        assert _profile_cache.equivalence_payload(
            _profile_cache.read_winner(key)) is None, "性能行不该被解成载荷"


# -- 9. tune_session：期间不读不写 -----------------------------------------------
def test_tune_session_blocks_reads_and_writes() -> None:
    """§3.1/§7：``tune_session`` 期间读当无命中、写静默不落盘；退出即恢复。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        _profile_cache.write_winner(key, {"kernel": "generic", "optimize": "greedy",
                                         "wall_ms": 1.0})
        _forget()
        before = _state_of(directory)
        assert _profile_cache.read_winner(key) is not None, "会话之前读得到"

        with _profile_cache.tune_session():
            assert _profile_cache.read_winner(key) is None, (
                "遍历期间读表会把'照表施加'混进被测配置")
            _profile_cache.write_winner(key, {"kernel": "kupl_topology",
                                             "optimize": "greedy", "wall_ms": 0.001})
            _profile_cache.write_winner(key, equivalence={
                "target_equation": "cadb,ac,bd->", "operand_order": [1, 0]})
            with _profile_cache.tune_session():          # 嵌套：仍然不读不写
                assert _profile_cache.read_winner(key) is None
            assert _profile_cache.read_winner(key) is None, "内层退出不该提前恢复"
            during = _state_of(directory)
            assert during["bytes"] == before["bytes"], "遍历期间配置表被改写了"
            assert during["mtime_ns"] == before["mtime_ns"], (
                "遍历期间配置表被重写了")

        after_row = _profile_cache.read_winner(key)
        assert after_row is not None, "会话退出后读必须恢复"
        assert after_row["kernel"] == "generic", ("遍历期间的写不该落盘", after_row)
        assert _state_of(directory)["bytes"] == before["bytes"], (
            "会话期间一个字都不许落盘")
        _profile_cache.write_winner(key, {"kernel": "backend", "wall_ms": 9.0})
        _forget()
        assert _profile_cache.read_winner(key)["kernel"] == "backend", (
            "会话退出后写必须恢复")


# -- 10. 关闭零副作用 ------------------------------------------------------------
def test_disabled_profile_reads_nothing_and_writes_nothing() -> None:
    """§1/§6：``PYQCD_EINSUM_PROFILE=0`` → 读返回 ``None``、写静默、零目录。"""
    with _isolated() as work:
        os.environ[_profile_cache.PROFILE_ENV] = "0"
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        assert _profile_cache.enabled() is False
        assert _profile_cache.read_winner(_key()) is None, (
            "关闭时读当无命中（不抛 ProfileMissing）")
        assert _profile_cache.write_winner(_key(), {"kernel": "backend"}) is None
        assert _profile_cache.best(_key()) is None

        state = _profile_cache.summary()
        assert state["enabled"] is False, state
        assert state["rows"] == 0 and state["keys"] == 0, state
        assert os.listdir(work) == [], ("关闭时不得产生任何文件", os.listdir(work))
        assert not os.path.exists(directory), "关闭时不许建目录"


# -- 11. format_shape ------------------------------------------------------------
def test_format_shape_normalizes() -> None:
    """§7：``x`` 连轴、``|`` 连操作数；字符串原样、无法解析给空串。"""
    assert _profile_cache.format_shape([(2, 3), (3, 4)]) == "2x3|3x4"
    assert _profile_cache.format_shape([[2, 3], [3, 4]]) == "2x3|3x4"
    assert _profile_cache.format_shape([(2, 3)]) == "2x3"
    assert _profile_cache.format_shape(" 2x3|3x4 ") == "2x3|3x4"
    assert _profile_cache.format_shape(_profile_cache.format_shape(
        [(2, 3), (3, 4)])) == "2x3|3x4", "归一化必须幂等"
    assert _profile_cache.format_shape(None) == ""
    assert _profile_cache.format_shape(object()) == "", "不可解析必须给空串"
    assert _profile_cache.format_shape([2, 3]) == "2|3", (
        "裸轴长列表会被读成两个 0 维操作数（入参必须是操作数列表）")
    # 归一化之后是同一把键：列表写法与字符串写法命中同一行。
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        _profile_cache.write_winner(_key(shape=[(2, 3), (3, 4)]),
                                   {"kernel": "backend"})
        _forget()
        assert _profile_cache.read_winner(_key(shape="2x3|3x4")) is not None, (
            "两种形状写法必须落到同一把键上")


# -- 12. 目录优先级 --------------------------------------------------------------
def test_directory_priority_env_over_default_and_configure_over_env() -> None:
    """§7：参数 > ``PYQCD_EINSUM_PROFILE_DIR`` > 默认 ``./.cache``；ensure 只补齐。"""
    with _isolated() as work:
        default = os.path.join(os.path.abspath(work), _profile_cache.DEFAULT_DIR)
        assert _profile_cache.ensure_configured() == default, (
            "缺省目录是 CWD 下的 .cache", _profile_cache.ensure_configured())
        assert _profile_cache.profile_path() == _table_path(default)

        env_dir = os.path.join(work, "env_cache")
        os.environ[_profile_cache.DIR_ENV] = env_dir
        assert _profile_cache.configure(None) == os.path.abspath(env_dir), (
            "环境变量压过默认目录")
        assert _profile_cache.ensure_configured() == os.path.abspath(env_dir)

        pinned = os.path.join(work, "pinned")
        assert _profile_cache.configure(pinned) == os.path.abspath(pinned), (
            "参数压过环境变量")
        assert _profile_cache.ensure_configured() == os.path.abspath(pinned), (
            "ensure_configured 不得用环境变量改写已钉住的目录")
        assert not os.path.exists(pinned), "configure 只解析路径，读路径不建目录"
        assert not os.path.exists(env_dir), "只解析路径，不得按环境变量建目录"

        os.environ.pop(_profile_cache.DIR_ENV, None)
        assert _profile_cache.configure(None) == default, (
            "configure(None) 才按'env > 默认'重算")


# -- 13. summary 的行/键计数 ------------------------------------------------------
def test_summary_counts_rows_and_keys() -> None:
    """§7：``summary`` 如实报行数/键数（同键多行时 keys 仍是 1）；缺表报 0。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        empty = _profile_cache.summary()
        assert empty["rows"] == 0 and empty["keys"] == 0, empty
        assert empty["file"] == os.path.abspath(_table_path(directory)), empty
        assert empty["enabled"] is True, empty
        assert empty["last_error"] is None, empty

        key = _key()
        _profile_cache.write_winner(key, {"kernel": "backend", "wall_ms": 1.0})
        _profile_cache.write_winner(dict(key, equation="ab,bc->ac"),
                                   {"kernel": "generic", "wall_ms": 2.0})
        _forget()
        state = _profile_cache.summary()
        assert state["rows"] == 2 and state["keys"] == 2, state

        _write_table(directory, [_row(key, kernel="a"), _row(key, kernel="b")])
        broken = _profile_cache.summary()
        assert broken["rows"] == 2, broken
        assert broken["keys"] == 1, ("同键多行时 keys 仍应是 1", broken)


# -- 14. wall_ms = 整个 contract() 的墙钟 ---------------------------------------
def test_wall_ms_covers_the_whole_contract_call() -> None:
    """§4/§8：``wall_ms`` 是**整段** ``contract()`` 的墙钟（含读表这条腿）。

    在 :func:`read_winner` 里注入 :data:`READ_SLEEP` 秒：它不是"收缩执行"的时间，却
    必须出现在 ``wall_ms`` 里。旧的拆分计时会把它算进另一列（该列已随 §4 删除），
    所以这条断言正好钉住"只有一列、且是全程"。
    """
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        operands = [np.arange(12, dtype=np.complex128).reshape(3, 4),
                    np.arange(20, dtype=np.complex128).reshape(4, 5)]
        real = _profile_cache.read_winner
        calls: list[dict[str, Any]] = []

        def slow(key: Mapping[str, Any]) -> Any:
            calls.append(dict(key))
            time.sleep(READ_SLEEP)
            return real(key)

        with patch.object(_profile_cache, "read_winner", slow), \
                contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            result = contract("ij,jk->ik", *operands, profile=True,
                              profile_dir=directory)

        assert tuple(np.shape(result)) == (3, 5), np.shape(result)
        info = _api.last_profile()
        assert calls, "开启画像的普通调用必须经 read_winner"
        assert isinstance(info, dict), ("画像调用必须留下观测口", info)
        assert "wall_ms" in info, ("观测口必须用 wall_ms", sorted(info))
        assert LEGACY_MS_COLUMN not in info, ("观测口不得残留旧字段名", sorted(info))
        assert "init_ms" not in info, ("拆分计时已删除（§4）", sorted(info))
        assert float(info["wall_ms"]) >= READ_SLEEP * 1000.0 * 0.8, (
            "注入的读表耗时必须计入 wall_ms", info["wall_ms"])
        assert float(info["wall_ms"]) < 60_000.0, info["wall_ms"]
        # 表列名与观测口同名：写方按名字取值，改名会静默丢数字。
        assert "wall_ms" in _profile_cache.PERFORMANCE_FIELDS, (
            _profile_cache.PERFORMANCE_FIELDS)


# -- 15. 7 叶子：写入读回一致 + 连字符旧写法归一 -------------------------------
def _write_table_with_columns(directory: str, rows: Sequence[Mapping[str, str]],
                              columns: Sequence[str]) -> str:
    """按**指定列**写 CSV（造"老文件缺列"的现场：``PROFILE_COLUMNS`` 少几列）。"""
    os.makedirs(directory, exist_ok=True)
    path = _table_path(directory)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: str(row.get(name, "")) for name in columns})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    _forget()
    return path


def test_optimize_leaves_roundtrip_and_legacy_hyphen_normalizes() -> None:
    """§15：7 个叶子写入读回一致；连字符旧写法读入归一成下划线（不算非法）。"""
    assert tuple(_profile_cache.OPTIMIZE_LEAVES) == (
        "greedy", "optimal", "dp", "branch_1", "branch_2", "branch_all", "none"), (
        "7 叶子是冻结口径", _profile_cache.OPTIMIZE_LEAVES)
    assert _profile_cache.DEFAULT_OPTIMIZE == "greedy", _profile_cache.DEFAULT_OPTIMIZE
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        for leaf in _profile_cache.OPTIMIZE_LEAVES:
            _profile_cache.write_winner(key, {"kernel": "generic", "optimize": leaf,
                                             "wall_ms": 0.5})
            _forget()
            row = _profile_cache.read_winner(key)
            assert row is not None, ("写进去的叶子必须能读回", leaf)
            assert row["optimize"] == leaf, ("写入读回必须一致", leaf, row)
        # 0.5.12：``optimize_topology_v2`` 那一列已从表里彻底删除。
        assert "optimize_topology_v2" not in _profile_cache.PROFILE_COLUMNS, (
            "该列已删除，不该再出现在表头", _profile_cache.PROFILE_COLUMNS)
        assert not hasattr(_profile_cache, "TOPOLOGY_V2_FIELD"), (
            "常量也应一并删除，否则还会有人去读它")

        # 连字符旧写法（手工写进表）：读入归一为下划线，**不**视为非法。
        _write_table(directory, [_row(key, kernel="generic", optimize="branch-1")])
        row = _profile_cache.read_winner(key)
        assert row is not None, "连字符旧写法不是非法值（向后兼容）"
        assert row["optimize"] == "branch_1", ("连字符必须归一成下划线", row)

        # 写侧同样归一：大小写与连字符都折成叶子。
        _profile_cache.write_winner(key, {"kernel": "generic",
                                          "optimize": " Branch-All "})
        _forget()
        assert _profile_cache.read_winner(key)["optimize"] == "branch_all"


# -- 16. ``auto`` 旧值：警告一次 + 该行不参与匹配 --------------------------------
def test_auto_row_warns_once_and_is_not_matched() -> None:
    """§16：``auto`` 非法——读侧只警告一次，该行不匹配（绝不静默当 ``greedy``）。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        _write_table(directory, [_row(key, kernel="kupl_dedicated", optimize="auto")])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _profile_cache.read_winner(key) is None, (
                "auto 行**不参与**匹配（不能被当成 greedy 命中）")
            assert _profile_cache.read_winner(key) is None, "第二次读也不该命中"
            state = _profile_cache.summary()
        assert len(caught) == 1, (
            "同一个非法值只警告一次", len(caught), [str(item.message) for item in caught])
        message = str(caught[0].message)
        assert "auto" in message and "不参与匹配" in message, message
        assert issubclass(caught[0].category, RuntimeWarning), caught[0].category
        assert state["rows"] == 1, ("行还在表里（不删老行）", state)
        assert state["invalid"] == 1, ("无效行要如实计数", state)


# -- 17. 带已删列的老文件 ------------------------------------------------
def test_legacy_file_with_the_removed_column_still_reads() -> None:
    """§17：老表**多**一列（已删的 ``optimize_topology_v2``）也能读，且该列不再出现。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        legacy_columns = list(_profile_cache.PROFILE_COLUMNS)
        legacy_columns.insert(
            legacy_columns.index("optimize") + 1, "optimize_topology_v2")
        legacy_row = _row(key, kernel="backend", optimize="dp")
        legacy_row["optimize_topology_v2"] = "greedy"
        _write_table_with_columns(directory, [legacy_row], legacy_columns)

        row = _profile_cache.read_winner(key)
        assert row is not None, "带已删列的老文件必须照旧能读（不降级、不丢行）"
        assert row["kernel"] == "backend", row
        assert row["optimize"] == "dp", row
        assert "optimize_topology_v2" not in row, (
            "已删的列不该再出现在读回的行里", sorted(row))
        # 现存列一列不少地都在。
        for name in _profile_cache.PROFILE_COLUMNS:
            assert name in row, ("读回来的行必须补齐全部列", name, sorted(row))


# -- 18. ``schema_version``：新写 3，老行 0 且仍可用 ---------------------------
def test_schema_version_written_four_and_legacy_rows_read_zero() -> None:
    """§18：``schema_version`` 新行写 4；老文件（无该列）读回 0 且照旧命中。"""
    assert _profile_cache.PROFILE_SCHEMA_VERSION == 4, (
        _profile_cache.PROFILE_SCHEMA_VERSION)
    assert _profile_cache.LEGACY_SCHEMA_VERSION == 0, (
        _profile_cache.LEGACY_SCHEMA_VERSION)
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _profile_cache.configure(directory)
        key = _key()
        _profile_cache.write_winner(key, {"kernel": "generic", "optimize": "dp",
                                         "wall_ms": 1.0})
        _forget()
        written = _data_rows(_table_path(directory))[0]
        assert written["schema_version"] == "4", ("新行写 4", written)
        row = _profile_cache.read_winner(key)
        assert row["schema_version"] == "4", row
        # 等价行也带 schema 版本（它同样按这一版的列集合写）。
        _profile_cache.write_winner(key, equivalence={
            "target_equation": "cadb,ac,bd->", "operand_order": [1, 0]})
        _forget()
        assert _profile_cache.read_winner(key)["schema_version"] == "4", (
            "等价行也要记 schema 版本")

        # 老文件：连 ``schema_version`` 这一列都没有。
        legacy_columns = [name for name in _profile_cache.PROFILE_COLUMNS
                          if name != "schema_version"]
        _write_table_with_columns(
            directory, [_row(key, kernel="backend", optimize="greedy")],
            legacy_columns)
        old = _profile_cache.read_winner(key)
        assert old is not None, "老行必须仍可用（不降级、不删）"
        assert old["schema_version"] == "0", ("老行的版本号读作 0", old)
        assert old["kernel"] == "backend", old
        assert _read_rows(_table_path(directory))[0] == [
            "shape", "dtype", "device", "equation", "backend", "kernel", "optimize",
            "wall_ms", "target_equation", "operand_order", "operand_permutations",
            "output_permutation", "note", "host", "updated_at", "pyqcd_version"], (
            "老文件的表头不该被读路径改写")


#: 本契约的用例清单（``main`` 与 ``conftest`` 共用）。
TESTS = (
    test_frozen_names_and_columns,
    test_read_winner_missing_file_raises_profile_missing,
    test_read_winner_zero_rows_returns_none,
    test_read_winner_single_row_returns_copy,
    test_read_winner_duplicate_rows_raise_with_all_candidates,
    test_read_winner_never_writes,
    test_write_winner_keeps_single_row_and_drops_same_key,
    test_write_winner_equivalence_row_has_payload_only,
    test_tune_session_blocks_reads_and_writes,
    test_disabled_profile_reads_nothing_and_writes_nothing,
    test_format_shape_normalizes,
    test_directory_priority_env_over_default_and_configure_over_env,
    test_summary_counts_rows_and_keys,
    test_wall_ms_covers_the_whole_contract_call,
    test_optimize_leaves_roundtrip_and_legacy_hyphen_normalizes,
    test_auto_row_warns_once_and_is_not_matched,
    test_legacy_file_with_the_removed_column_still_reads,
    test_schema_version_written_four_and_legacy_rows_read_zero,
)


def test_einsum_profile_cache_contracts() -> None:
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
            f"{len(failures)} 条配置表读写契约失败：" + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"profile cache contracts did not run ({skipped} skipped)")


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
