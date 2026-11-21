"""契约：记录表（``pyqcd.einsum.record.csv``）**只作参考、绝不参与命中**。

为什么单独立契约
----------------
用户要的是"tune 的所有测试组的全量信息"：记录表把**每一组**测过的组合都留下来（不只是
胜出的那一格），与配置表同目录、列集合是配置表的超集。这张表一旦被当缓存用，语义就彻底
坏了——命中判定会开始从"没被选中的 29 组"里挑答案，"tune 的结论"变成"tune 的过程"。

这类错误**不会让任何数值测试变红**：多读一张表、多命中一行，答案照样逐位正确，只是
"为什么这次跑得这么慢/这么快"再也解释不清。所以这里把"只作参考"拆成能各自独立失败的
七条断言：

1. **同目录 + 冻结文件名**：记录表必须与 ``profile.csv`` 同目录（用户只记一个目录），
   文件名逐字 ``pyqcd.einsum.record.csv``（外部脚本/tune 报告已经按这个名字取）。
2. **列超集且公共列逐字相同**：``set(PROFILE_COLUMNS) <= set(RECORD_COLUMNS)``、公共列的
   **相对顺序**一致，所以"从记录表里挑一行贴进配置表"永远是合法操作；``peak_memory_mb``
   只属于记录表——配置表的列集合是冻结的（多条契约依赖它），往里加一列等于改配置表格式。
3. **整表重写而非追加**：两次 ``write_records`` 之后磁盘上只剩第二次那批行。追加会让同一
   把键在表里留下几十行历史噪音，用户拿它跟配置表对照时会看到一堆"没生效的组"。
4. **不参与命中**：``_api.py`` 里**没有一处**读记录表，``_profile_cache.read_winner``
   （唯一命中入口）的函数体里连常量名都不出现，整个 ``pyqcd/einsum/`` 包里也没有第二个
   模块读它。记录表回到纯粹的**报告**地位（2026-11-21 删掉"更省内存"提示之后，原先
   ``_api._cheaper_memory_hint`` 那一处例外也一并消失）。
5. **坏行跳过 / 缺列按空**：手写一张有坏行（少一列、多一列）的 CSV，``read_records()``
   不抛、好行照常返回、每行都补齐全部列键。记录表是"给报告添一列证据"，坏一行不该让
   报告拿不到结果。
6. **开关关闭时不写**：``PYQCD_EINSUM_PROFILE=0`` 下 ``write_records`` 返回 0 且**文件根本
   不存在**（连目录都不建）——"关掉配置表"必须是真的零 IO。
7. **旧位置只读回退**：生效目录是缺省（家目录）而家目录里没有表、旧位置
   （``<CWD>/.cache``）有表时，读路径把旧表原样当来源：**不复制、不移动、不删**，旧文件
   字节与 ``mtime`` 都不变，新位置里连目录都不出现。

隔离口径
--------
每个用例都在 ``data/tmp/`` 下的临时目录里跑（CWD 一起换过去），显式 ``configure(tmpdir)``
或设 ``PYQCD_EINSUM_PROFILE_DIR``；退出时把环境变量（含 ``HOME`` / ``XDG_CACHE_HOME``）与
``_profile_cache`` 的进程级状态（``_CONFIGURED_DIR`` / ``_TABLE`` / ``_LAST_ERROR`` /
``_TUNING`` / 旧位置说明 / 警告去重集合）原样放回。**本契约绝不碰
``~/.cache/pyqcd.einsum``**——那是用户真实的配置表与记录表。

可直接运行::

    python -B -m pyqcd.testing._einsum_record_table_contract
"""
from __future__ import annotations

import ast
import contextlib
import csv
import os
import shutil
import tempfile
import unittest
import warnings
from typing import Any, Iterator

from ..einsum import _api, _profile_cache

__all__ = ["test_einsum_record_table_contracts"]

#: 记录表的文件名（**冻结的对外契约**：tune 报告与外部脚本按这个字面量取文件）。
RECORD_FILE_LITERAL = "pyqcd.einsum.record.csv"

#: 记录表独有的那一列（性能段末尾的峰值内存）：只该在记录表里出现。
RECORD_MEMORY_LITERAL = "peak_memory_mb"

#: 记录表 schema 版本（列集合版本：配置表 v4 的全部列 + ``peak_memory_mb``）。列集合每变
#: 一次才 +1，所以"等于 5"本身就是"列集合没被悄悄改过"的证据。
#:
#: 0.5.12 起配置表删掉了 ``optimize_topology_v2`` 一列（3 → 4），记录表的版本随之 +1。
RECORD_SCHEMA_LITERAL = 5

#: 允许读记录表的**模块**（相对 ``pyqcd/einsum/``）：**空集**。
#:
#: 2026-11-21 起记录表是纯报告：``write_records`` 写、``read_records`` 给工具读，
#: 但没有任何运行时代码在 ``contract()`` 路径上读它。原先这里放行 ``_api.py``（那条
#: "更省内存"提示），该提示已按用户要求删除，白名单随之清空——把任何模块加回来都
#: 会让第 4 条断言红，这正是"别悄悄把 I/O 塞回热路径"的闸门。
ALLOWED_RECORD_READER_MODULES: frozenset[str] = frozenset()

#: 临时区：``data/tmp/``（相对仓库根），与仓库既有约定一致。
_TMP_ENV = "PYQCD_TEST_TMP"

#: 需要按用例隔离的进程级状态：前五项照抄 ``_einsum_profile_robustness_contract``，
#: 后四项是本契约会碰到的（旧位置说明 + 警告去重，见 :func:`_reset`）。缺哪个就跳过哪个
#: （实现删/改了某个内部名时，契约只该少隔离一项，不该整条炸掉）。
_STATE_NAMES = ("_CONFIGURED_DIR", "_LAST_ERROR", "_HOST", "_TABLE", "_TUNING",
                "_SOURCE_NOTE", "_SOURCE_PATH", "_WARNED_CELLS", "_PENDING_WARNINGS")

#: "模块上没有这个内部名"的哨兵（用于 :func:`_snapshot` 的可选状态）。
_MISSING = object()

#: 环境变量：配置表开关/目录（名字从模块常量取），以及"家目录"这一档的两个来源。
_EXTRA_ENV = ("HOME", "XDG_CACHE_HOME")


# ── 临时区 ──────────────────────────────────────────────────────────────────

def _repo_root() -> str:
    """仓库根目录（本文件在 ``<root>/pyqcd/testing/`` 下）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tmp_root() -> str:
    """临时根：``data/tmp/``；不可写时退回系统临时目录（报告里能看出来）。"""
    override = os.environ.get(_TMP_ENV)
    candidates = [override] if override else []
    candidates.append(os.path.join(_repo_root(), "data", "tmp"))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = tempfile.mkdtemp(prefix="record-contract-", dir=candidate)
            os.rmdir(probe)
            return candidate
        except Exception:                             # noqa: BLE001 - 换下一个
            continue
    return tempfile.gettempdir()


class _TempDir:
    """``with _TempDir() as path:`` —— 在 ``data/tmp/`` 下建一个用完即删的目录。"""

    def __init__(self, prefix: str = "record-contract-") -> None:
        self._prefix = prefix
        self.path = ""

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix=self._prefix, dir=_tmp_root())
        return self.path

    def __exit__(self, *exc_info: Any) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        return None


# ── 状态隔离（照抄 _einsum_profile_robustness_contract 的 _snapshot/_restore） ──

def _snapshot(module: Any) -> dict[str, Any]:
    """把模块的进程级状态抄一份（``_TABLE`` 的行列表、集合/列表容器都拷一层）。

    模块上不存在的内部名记成 :data:`_MISSING`，:func:`_restore` 据此"本来没有就别造出来"
    ——隔离层不该因为实现少了一个内部名而把整条契约拖崩。
    """
    saved: dict[str, Any] = {}
    for name in _STATE_NAMES:
        value = getattr(module, name, _MISSING)
        if value is _MISSING:
            saved[name] = _MISSING
        elif name == "_TABLE" and value is not None:
            saved[name] = (value[0], value[1], value[2], [dict(row) for row in value[3]])
        elif isinstance(value, (set, list)):
            saved[name] = type(value)(value)
        else:
            saved[name] = value
    return saved


def _restore(module: Any, saved: dict[str, Any]) -> None:
    """把 :func:`_snapshot` 抄下来的状态放回去（``_MISSING`` = 原来没有，跳过）。"""
    for name, value in saved.items():
        if value is _MISSING:
            continue
        setattr(module, name, value)


def _clear(module: Any, name: str, value: Any) -> None:
    """把 ``name`` 设成 ``value``；模块上本来没有这个内部名就跳过。"""
    if hasattr(module, name):
        setattr(module, name, value)


def _empty(module: Any, name: str) -> None:
    """清空一个容器型内部状态；模块上本来没有这个内部名就跳过。"""
    container = getattr(module, name, None)
    if isinstance(container, (set, list, dict)):
        container.clear()


def _reset(module: Any) -> None:
    """把模块清到"没配置、没解析缓存、不处于 tune 窗口"的初始态。

    警告去重集合（``_WARNED_CELLS`` / ``_PENDING_WARNINGS``）也一起清空：第 7 条要断言
    读旧位置时**真的**发了一条警告，留着上一个用例的去重记录会让它静默（同一签名只报一次）。
    """
    _clear(module, "_TABLE", None)
    _clear(module, "_TUNING", 0)
    _clear(module, "_CONFIGURED_DIR", "")
    _clear(module, "_LAST_ERROR", None)
    _clear(module, "_SOURCE_NOTE", None)
    _clear(module, "_SOURCE_PATH", None)
    _empty(module, "_WARNED_CELLS")
    _empty(module, "_PENDING_WARNINGS")


@contextlib.contextmanager
def _isolated(disabled: bool = False) -> Iterator[str]:
    """临时 CWD + 干净的环境变量与进程级状态；退出时全部还原。

    换 CWD 是必须的：第 7 条的旧位置（``<CWD>/.cache``）是**相对当前工作目录**的路径，
    如果不换 CWD，这条断言就会去读仓库根下的 ``.cache``（若真存在，还会读到别人的表）。
    配置目录一律显式钉到本临时树里；第 7 条还会把 ``HOME`` / ``XDG_CACHE_HOME`` 指向
    临时目录，其余用例根本不走缺省那一档——所以 ``~/.cache/pyqcd.einsum`` 在整个契约里
    既不会被读、也不会被写。
    """
    module = _profile_cache
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name)
                 for name in (module.PROFILE_ENV, module.DIR_ENV) + _EXTRA_ENV}
    saved = _snapshot(module)
    with _TempDir() as root:
        work = os.path.join(root, "work")
        os.makedirs(work)
        try:
            os.chdir(work)
            os.environ.pop(module.DIR_ENV, None)
            if disabled:
                os.environ[module.PROFILE_ENV] = "0"
            else:
                os.environ.pop(module.PROFILE_ENV, None)
            _reset(module)
            yield work
        finally:
            os.chdir(saved_cwd)
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _restore(module, saved)


# ── 小工具 ──────────────────────────────────────────────────────────────────

def _key(**overrides: Any) -> dict[str, Any]:
    """配置表/记录表键的稀疏写法（``shape`` 用列表写法，与实现同口径归一化）。"""
    key = {
        "backend": "numpy",
        "dtype": "complex128",
        "equation": "ij,jk->ik",
        "shape": [(2, 3), (3, 4)],
        "device": "cpu",
    }
    key.update(overrides)
    return key


def _record(**overrides: Any) -> dict[str, Any]:
    """记录表一行的稀疏写法（只给关心的列，其余由 ``write_records`` 补空）。"""
    row: dict[str, Any] = {
        "backend": "numpy",
        "dtype": "complex128",
        "equation": "ij,jk->ik",
        "shape": "2x3|3x4",
        "device": "cpu",
        "kernel": "backend",
        "optimize": "greedy",
        "wall_ms": 1.25,
        "peak_memory_mb": 12.5,
        "note": "recorded-by-contract",
    }
    row.update(overrides)
    return row


def _read_rows(path: str) -> list[list[str]]:
    """整份 CSV（含表头行）；用真实 ``open``，本契约不打补丁。"""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _write_csv(path: str, header: list[str], rows: list[list[str]]) -> None:
    """手写一张 CSV（表头 + 若干数据行）：本契约要用它造坏行与老表头。"""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _read_text(path: str) -> str:
    """读源码文本（UTF-8）。"""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _function_node(path: str, name: str) -> ast.AST:
    """源码里名为 ``name`` 的函数节点（必须**恰好**一个：多份定义等于契约失效）。"""
    nodes = [node for node in ast.walk(ast.parse(_read_text(path)))
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == name]
    if len(nodes) != 1:
        raise AssertionError(
            f"{os.path.basename(path)} 里名为 {name} 的函数定义了 {len(nodes)} 次"
            f"（{[(getattr(node, 'lineno', 0)) for node in nodes]}），必须恰好 1 次"
            "——记录表的读写边界只能有一个可检查的位置")
    return nodes[0]


def _needle_lines(lines: list[str], needle: str) -> list[int]:
    """``needle`` 出现过的行号（1-based），等价于 ``grep -n needle <file>``。"""
    return [index + 1 for index, line in enumerate(lines) if needle in line]


def _quote(lines: list[str], line_no: int) -> str:
    """失败信息里附上那一行的原文（超长截断，够定位就够）。"""
    text = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
    return f"{line_no}: {text[:160]}"


# ── 1. 同目录 + 冻结文件名 ──────────────────────────────────────────────────

def test_record_table_shares_the_profile_directory() -> None:
    """``record_path()`` 与 ``profile_path()`` 同目录，文件名是冻结的 ``.record.csv``。

    为什么：用户只需要记**一个**目录（家目录缓存），tune 报告里两张表也要能并排对照；
    文件名一旦漂移，``--profile-dir``、报告与外部脚本就各自找一张表。
    """
    module = _profile_cache
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        module.configure(directory)
        profile = module.profile_path()
        record = module.record_path()
        assert module.RECORD_FILENAME == RECORD_FILE_LITERAL, (
            f"RECORD_FILENAME 已漂移：{module.RECORD_FILENAME!r} != {RECORD_FILE_LITERAL!r}")
        assert os.path.basename(record) == RECORD_FILE_LITERAL, (
            f"record_path() 的文件名不是 {RECORD_FILE_LITERAL!r}：{record!r}")
        assert os.path.dirname(record) == os.path.dirname(profile), (
            "记录表必须与配置表同目录：\n"
            f"  profile={profile!r}\n  record ={record!r}")
        assert os.path.dirname(record) == os.path.abspath(directory), (
            f"目录没跟着 configure：{os.path.dirname(record)!r} != "
            f"{os.path.abspath(directory)!r}")
        assert record != profile, f"记录表与配置表不能是同一个文件：{record!r}"
        # 只算路径：两个函数都不许建目录、不许落文件。
        assert not os.path.exists(directory), (
            f"record_path()/profile_path() 只解析路径，不得建目录：{os.listdir(work)}")


# ── 2. 列超集、公共列同序、内存列只属于记录表 ───────────────────────────────

def test_record_columns_are_a_profile_superset() -> None:
    """``RECORD_COLUMNS`` 是 ``PROFILE_COLUMNS`` 的超集；公共列相对顺序一致。

    为什么：公共列逐字相同 + 同序，"把记录表某一行贴进 profile.csv"才是合法的选路操作；
    ``peak_memory_mb`` 则必须**只**在记录表里——配置表的列集合是冻结的，多条契约（表头
    逐字、``PERFORMANCE_FIELDS`` 只含 ``wall_ms``）都钉在它上面。
    """
    module = _profile_cache
    with _isolated() as work:
        module.configure(os.path.join(work, "cache"))
        profile = list(module.PROFILE_COLUMNS)
        record = list(module.RECORD_COLUMNS)
        assert len(profile) == len(set(profile)), f"配置表列重复：{profile}"
        assert len(record) == len(set(record)), f"记录表列重复：{record}"
        missing = [name for name in profile if name not in set(record)]
        assert not missing, (
            f"记录表缺了配置表的列 {missing}（记录表={record}）")
        common = [name for name in record if name in set(profile)]
        assert common == profile, (
            "公共列的相对顺序必须与配置表一致：\n"
            f"  profile={profile}\n  record 的公共列={common}\n"
            f"  第一个分歧位置={_first_divergence(profile, common)}")
        assert getattr(module, "RECORD_MEMORY_FIELD", None) == RECORD_MEMORY_LITERAL, (
            f"RECORD_MEMORY_FIELD 已漂移：{getattr(module, 'RECORD_MEMORY_FIELD', None)!r}"
            f" != {RECORD_MEMORY_LITERAL!r}")
        assert module.RECORD_MEMORY_FIELD not in profile, (
            f"{RECORD_MEMORY_LITERAL!r} 只属于记录表，配置表的列集合是冻结的：{profile}")
        extra = [name for name in record if name not in set(profile)]
        assert extra == [module.RECORD_MEMORY_FIELD], (
            f"记录表比配置表多出的列必须恰好是 {RECORD_MEMORY_LITERAL!r}，实际 {extra}")
        assert module.RECORD_SCHEMA_VERSION == RECORD_SCHEMA_LITERAL, (
            f"RECORD_SCHEMA_VERSION 应为 {RECORD_SCHEMA_LITERAL}（配置表 v"
            f"{module.PROFILE_SCHEMA_VERSION} 的列 + 一列），实际 "
            f"{module.RECORD_SCHEMA_VERSION}")


def _first_divergence(left: list[str], right: list[str]) -> str:
    """两个列名序列的第一个分歧（用于把"顺序不一致"说清到具体位置）。"""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"#{index}: profile={a!r} vs record={b!r}"
    if len(left) != len(right):
        return f"长度不同：profile={len(left)} vs 公共列={len(right)}"
    return "无分歧"


# ── 3. 整表重写而非追加 ─────────────────────────────────────────────────────

def test_write_records_rewrites_the_whole_table() -> None:
    """两次 ``write_records`` 之后磁盘上只剩第二次那批行（行数用 ``len`` 算）。

    为什么：记录表是"这一次 tune 的全量快照"。追加会让同一把键留下几十行历史噪音，
    用户拿它跟配置表对照时分不清"哪一行是这次跑出来的"。
    """
    module = _profile_cache
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        module.configure(directory)
        path = module.record_path()
        first = [
            _record(kernel="first-a", wall_ms=1.0, peak_memory_mb=10.0),
            _record(kernel="first-b", wall_ms=2.0, peak_memory_mb=20.0),
        ]
        second = [_record(kernel="second-only", wall_ms=3.0, peak_memory_mb=30.0)]

        written = module.write_records(first)
        assert written == len(first), (
            f"write_records 返回的行数不对：{written} != {len(first)}")
        lines = _read_rows(path)
        assert lines[0] == list(module.RECORD_COLUMNS), (
            "记录表表头必须逐字等于 RECORD_COLUMNS：\n"
            f"  磁盘={lines[0]}\n  常量={list(module.RECORD_COLUMNS)}")
        assert len(lines) - 1 == len(first), (
            f"第一次写入应有 {len(first)} 条数据行，磁盘上是 {len(lines) - 1} 条")

        written = module.write_records(second)
        assert written == len(second), (
            f"write_records 返回的行数不对：{written} != {len(second)}")
        lines = _read_rows(path)
        assert len(lines) - 1 == len(second), (
            f"整表重写而非追加：第二次写 {len(second)} 行后磁盘上应只剩 "
            f"{len(second)} 条数据行，实际 {len(lines) - 1} 条"
            f"（总共 {len(lines)} 行含表头，{len(first)} + {len(second)} 行 = 追加行为）")
        rows = module.read_records()
        assert len(rows) == len(second), (
            f"read_records() 只想看到第二次那 {len(second)} 行，实际 {len(rows)} 行")
        assert {row["kernel"] for row in rows} == {"second-only"}, (
            f"旧行没有被清掉：{sorted((row['kernel'], row['wall_ms']) for row in rows)}")
        assert rows[0]["wall_ms"] == "3.000", (
            f"留下的应是第二次写入的那一行，wall_ms={rows[0]['wall_ms']!r}")
        assert set(module.RECORD_COLUMNS) <= set(rows[0]), (
            f"读回来的行必须带全部列键，缺 "
            f"{sorted(set(module.RECORD_COLUMNS) - set(rows[0]))}")


# ── 4. 记录表不参与命中 ─────────────────────────────────────────────────────

def test_records_never_reach_the_hit_path() -> None:
    """``read_winner`` 里不出现记录表；**整个 einsum 包里也没有一处读它**。

    为什么：命中判定只走 :func:`._profile_cache.read_winner`。一旦那里（或其它任何选路
    代码）读了记录表，"没被选中的组"就会变成可选答案——记录表是"测过什么"，不是"该用
    什么"。

    口径演变（2026-11-21）：原先这里允许**一处**例外——``_api._cheaper_memory_hint``，
    它在每次命中时读一遍记录表，给出一行"存在更省内存的配置"的建议。该提示已按用户
    要求删除（"把'更省内存或显存的提示输出'功能删去"），于是例外也一起消失：现在
    除 ``_profile_cache`` 自己的定义与写入外，**任何模块出现 ``read_records`` 都算违约**。
    这同时把热路径上的一次纯 I/O（每次 ``contract()`` 读 29 行）彻底去掉。
    """
    module = _profile_cache
    with _isolated() as work:
        module.configure(os.path.join(work, "cache"))
        api_path = os.path.abspath(_api.__file__)
        cache_path = os.path.abspath(module.__file__)

        # (a) 那条提示已经不存在（删功能不是删断言：这里钉住"别再加回来"）。
        api_lines = _read_text(api_path).splitlines()
        assert not _needle_lines(api_lines, "def _cheaper_memory_hint"), (
            "_cheaper_memory_hint 已按用户要求删除，_api.py 里不该再有它的定义")
        assert not hasattr(_api, "_cheaper_memory_hint"), (
            "模块上也还能取到 _cheaper_memory_hint")
        api_hits = _needle_lines(api_lines, "read_records")
        assert not api_hits, (
            "_api.py 不该再读记录表（'更省内存'提示已删）："
            + "; ".join(_quote(api_lines, line) for line in api_hits))

        # (b) 唯一的命中入口（read_winner）里连常量名都不许出现。
        winner = _function_node(cache_path, "read_winner")
        cache_lines = _read_text(cache_path).splitlines()
        body = "\n".join(cache_lines[winner.lineno - 1:winner.end_lineno])
        for needle in ("read_records", "RECORD_FILENAME", "record_path"):
            count = body.count(needle)
            assert count == 0, (
                f"_profile_cache.read_winner（第 {winner.lineno}-{winner.end_lineno} 行）"
                f"出现了 {needle!r} {count} 次——命中路径只认 profile.csv：\n"
                + "\n".join(_quote(cache_lines, line)
                            for line in _needle_lines(cache_lines, needle)
                            if winner.lineno <= line <= winner.end_lineno))

        # (c) 整个 einsum 包里，除 read_records 自己的定义模块外，**没有**任何模块读它。
        #     白名单外的模块出现一次就算违约（"悄悄多读一处"必须红）。
        package = os.path.dirname(cache_path)
        offenders: list[str] = []
        for name in sorted(os.listdir(package)):
            path = os.path.join(package, name)
            if not name.endswith(".py") or os.path.abspath(path) == cache_path:
                continue
            lines = _read_text(path).splitlines()
            for line_no in _needle_lines(lines, "read_records"):
                offenders.append(f"{name}::{_quote(lines, line_no)}")
        assert not offenders, (
            "以下位置读了记录表（2026-11-21 起一处都不允许——记录表只是报告）："
            + "; ".join(offenders))


# ── 5. 坏行跳过 / 缺列按空 ──────────────────────────────────────────────────

def test_bad_rows_are_skipped_and_missing_columns_read_as_empty() -> None:
    """手写坏行（少一列、多一列）：不抛、坏行跳过、好行补齐全部列键（缺的是空串）。

    为什么：记录表是"给报告添一列证据"。它坏一行只该少一条证据，不该让报告拿不到结果
    ——降到半行数据（列数对不上却硬补）会把列错位成另一个字段，那比丢行更坏。
    第二段顺带钉住"缺列按空"的另一半：``4`` 之前的记录表头没有 ``peak_memory_mb``，
    读回来那一格必须是空串（不是缺键、不是 ``KeyError``）。
    """
    module = _profile_cache
    with _isolated() as work:
        module.configure(os.path.join(work, "cache"))
        path = module.record_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = list(module.RECORD_COLUMNS)

        def cells(**overrides: Any) -> list[str]:
            """按 ``header`` 摆一串单元格（``overrides`` 里的列换成指定值）。"""
            return [str(overrides.get(name, "")) for name in header]

        good_a = cells(equation="ij,jk->ik", kernel="good-a", wall_ms="1.000",
                       optimize="greedy", note="good-a")
        good_b = cells(equation="ab,bc->ac", kernel="good-b", wall_ms="2.000",
                       optimize="dp", note="good-b")
        short = good_a[:-1]                      # 少最后一列（schema_version）
        short[header.index("note")] = "short-row"
        long_row = good_a + ["extra"]            # 多一列
        long_row[header.index("note")] = "long-row"
        _write_csv(path, header, [good_a, short, good_b, long_row])

        rows = module.read_records()             # 不许抛
        assert len(rows) == 2, (
            f"两条坏行应被跳过、两条好行应留下，实际 {len(rows)} 行："
            f"{[row.get('note') for row in rows]}")
        assert [row["note"] for row in rows] == ["good-a", "good-b"], (
            f"留下的应是夹在坏行中间的好行，实际 {[row['note'] for row in rows]}")
        for row in rows:
            missing = sorted(set(module.RECORD_COLUMNS) - set(row))
            assert not missing, f"读回来的行缺列 {missing}：{row}"
            assert row[module.RECORD_MEMORY_FIELD] == "", (
                f"好行里那一格本来是空的，应读成空串："
                f"{row[module.RECORD_MEMORY_FIELD]!r}")

        # 第二段：``4`` 之前的记录表头（没有 peak_memory_mb）→ 那一格按空补齐。
        old_header = [name for name in header if name != module.RECORD_MEMORY_FIELD]
        old_row = [str({"equation": "ij,jk->ik", "kernel": "legacy-row",
                        "wall_ms": "9.000", "optimize": "greedy",
                        "note": "legacy-header"}.get(name, ""))
                   for name in old_header]
        _write_csv(path, old_header, [old_row])
        rows = module.read_records()
        assert len(rows) == 1, f"老表头那一行应被读出，实际 {len(rows)} 行"
        missing = sorted(set(module.RECORD_COLUMNS) - set(rows[0]))
        assert not missing, f"老表头的行必须补齐全部列键，缺 {missing}"
        assert rows[0][module.RECORD_MEMORY_FIELD] == "", (
            f"老表头没有 {module.RECORD_MEMORY_FIELD}，那一格应按空读，实际 "
            f"{rows[0][module.RECORD_MEMORY_FIELD]!r}")
        assert rows[0]["kernel"] == "legacy-row", rows[0]
        assert rows[0]["schema_version"] == str(module.LEGACY_SCHEMA_VERSION), (
            f"没有 schema_version 的老行按 LEGACY_SCHEMA_VERSION 处理，实际 "
            f"{rows[0]['schema_version']!r}")


# ── 6. 开关关闭时不写 ───────────────────────────────────────────────────────

def test_disabled_switch_writes_nothing() -> None:
    """``PYQCD_EINSUM_PROFILE=0``：``write_records`` 返回 0，记录表文件根本不存在。

    为什么："关掉配置表"必须是**真的零 IO**（不读、不写、连目录都不建）。返回 0 只是
    给人看的；磁盘上真的没有文件才是用户能验证的那一半。
    """
    module = _profile_cache
    with _isolated(disabled=True) as work:
        directory = os.path.join(work, "cache")
        module.configure(directory)
        path = module.record_path()
        assert module.enabled() is False, "开关未关闭（PYQCD_EINSUM_PROFILE=0 应先被读到）"
        written = module.write_records([_record(), _record(kernel="other")])
        assert written == 0, f"关闭时 write_records 必须返回 0，实际 {written}"
        assert not os.path.exists(path), (
            f"关闭时不得落任何文件，实际存在 {path!r}（内容="
            f"{_read_rows(path) if os.path.exists(path) else None!r}）")
        assert not os.path.exists(directory), (
            f"关闭时连目录都不该建：{os.listdir(work)}")
        # 记录表根本没被写出来，所以读口也拿不到东西（"读了不存在文件"这一路，不抛）。
        assert module.read_records() == [], (
            f"关闭且没有记录表时，读口应返回空表（不抛），实际 {module.read_records()!r}")


# ── 7. 旧位置只读回退 ───────────────────────────────────────────────────────

def test_legacy_location_is_a_read_only_fallback() -> None:
    """家目录没表、``<CWD>/.cache`` 有表：读路径命中旧表，旧文件字节与 ``mtime`` 不变。

    为什么：默认目录从"相对 CWD 的 ``./.cache``"改成家目录缓存时，老用户的表必须还能
    用一次。兼容只做**只读**回退：不复制、不移动、不删——改用户家目录或删用户文件都得
    先问过用户。所以这里同时钉住"能读到"与"旧文件一个字节都没动、新位置连目录都没建"。

    ``DEFAULT_DIR`` 是导入时解析好的绝对路径，直接改 ``HOME`` / ``XDG_CACHE_HOME`` 不会
    影响它；为了不碰用户真实的 ``~/.cache/pyqcd.einsum``，本用例在一个临时作用域里把
    ``DEFAULT_DIR`` 指到临时目录（退出即还原），其余逻辑（``_is_default_dir`` /
    ``_legacy_path`` / 只读回退）走的都是生产代码。
    """
    module = _profile_cache
    if not hasattr(module, "LEGACY_DIR_NAME"):
        raise unittest.SkipTest("旧位置回退未实现（模块里没有 LEGACY_DIR_NAME）")
    with _isolated() as work:
        home = os.path.join(work, "home")
        xdg = os.path.join(work, "xdg")
        os.makedirs(home)
        os.makedirs(xdg)
        os.environ["HOME"] = home
        os.environ["XDG_CACHE_HOME"] = xdg
        # 生效目录 = 缺省那一档（DEFAULT_DIR），且那里**没有**表；旧位置在 CWD 下。
        # 缺省路径按模块自己的解析器现算（它读 XDG_CACHE_HOME）：这样钉住的既是临时目录，
        # 又不绕过生产代码——解析器没了才退回"XDG + 最后一段"的拼法。
        resolver = getattr(module, "_default_dir_text", None)
        pinned_default = (resolver() if callable(resolver)
                          else os.path.join(xdg, module.DEFAULT_HOME_SUBDIR))
        assert os.path.abspath(pinned_default) == os.path.abspath(
            os.path.join(xdg, module.DEFAULT_HOME_SUBDIR)), (
            f"缺省目录没跟着 XDG_CACHE_HOME：{pinned_default!r} != "
            f"{os.path.join(xdg, module.DEFAULT_HOME_SUBDIR)!r}")
        legacy = os.path.join(work, module.LEGACY_DIR_NAME, module.PROFILE_FILENAME)
        os.makedirs(os.path.dirname(legacy))
        _write_csv(legacy,
                   list(module.PROFILE_COLUMNS),
                   [[str({"shape": "2x3|3x4", "dtype": "complex128", "device": "cpu",
                          "equation": "ij,jk->ik", "backend": "numpy",
                          "kernel": "legacy-kernel", "optimize": "greedy",
                          "wall_ms": "4.000", "schema_version": "3"}.get(name, ""))
                     for name in module.PROFILE_COLUMNS]])
        snapshot = (_read_bytes(legacy), os.stat(legacy).st_mtime_ns)

        with _patch_default_dir(module, pinned_default):
            assert module.configure(None) == os.path.abspath(pinned_default), (
                f"configure(None) 应落到临时 DEFAULT_DIR：{module.configure(None)!r}")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                row = module.read_winner(_key())
            note = module.source_note()

        assert row is not None, (
            f"家目录没有表、旧位置有表时必须回退命中；last_error="
            f"{module.last_error()!r}, 旧表={legacy!r}")
        assert row["kernel"] == "legacy-kernel", row
        assert note and legacy in note, (
            f"source_note 必须说明来源是旧位置：{note!r}（旧表={legacy!r}）")
        assert any("旧位置" in str(item.message) for item in caught), (
            "读旧表时应同时警告一次（明明读的不是家目录的表）"
            f"，实际警告={[str(item.message)[:80] for item in caught]}")
        assert (_read_bytes(legacy), os.stat(legacy).st_mtime_ns) == snapshot, (
            f"旧位置的表必须原样留着（不复制、不移动、不重写）：{legacy!r}")
        assert not os.path.exists(pinned_default), (
            f"回退是只读的：不得在新位置建目录/复制表，实际存在 {pinned_default!r}"
            f"（{os.listdir(pinned_default) if os.path.exists(pinned_default) else []}）")
        # 记录表**不**走旧位置回退：它只是 tune 的参考快照，旧目录里从来没有它。
        assert module.record_path() == os.path.join(os.path.abspath(pinned_default),
                                                    module.RECORD_FILENAME), (
            f"record_path() 只报生效目录下的路径：{module.record_path()!r}")


def _read_bytes(path: str) -> bytes:
    """文件的原始字节（用于"旧文件一个字节都没动"的断言）。"""
    with open(path, "rb") as handle:
        return handle.read()


@contextlib.contextmanager
def _patch_default_dir(module: Any, directory: str) -> Iterator[None]:
    """临时把 ``DEFAULT_DIR`` 指到 ``directory``（退出即还原，绝不碰真实家目录）。"""
    saved = module.DEFAULT_DIR
    module.DEFAULT_DIR = os.path.abspath(directory)
    try:
        yield
    finally:
        module.DEFAULT_DIR = saved


TESTS = (
    test_record_table_shares_the_profile_directory,
    test_record_columns_are_a_profile_superset,
    test_write_records_rewrites_the_whole_table,
    test_records_never_reach_the_hit_path,
    test_bad_rows_are_skipped_and_missing_columns_read_as_empty,
    test_disabled_switch_writes_nothing,
    test_legacy_location_is_a_read_only_fallback,
)


# ── 汇总入口 ────────────────────────────────────────────────────────────────

def _run() -> tuple[int, int, list[str]]:
    """跑全部断言；返回 ``(passed, skipped, failures)``。"""
    passed = skipped = 0
    failures: list[str] = []
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except Exception as exc:                      # noqa: BLE001 - 报告用
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    return passed, skipped, failures


def test_einsum_record_table_contracts() -> None:
    """注册入口：记录表"只作参考、绝不参与命中"的七条契约。

    失败时抛 :class:`AssertionError` 并列出每条失败（带具体值）；全部跳过时抛
    :class:`unittest.SkipTest`——不许留一个"看起来通过"的空套件。
    """
    saved = _snapshot(_profile_cache)
    try:
        passed, skipped, failures = _run()
    finally:
        _restore(_profile_cache, saved)
    if failures:
        raise AssertionError(
            f"{len(failures)} 条记录表契约失败（{passed} passed, {skipped} skipped）："
            + "; ".join(failures))
    if passed == 0:
        raise unittest.SkipTest(
            f"记录表契约全部跳过（{skipped} skipped）：没有可验证的来源")


def main() -> int:
    """独立运行入口：``python -B -m pyqcd.testing._einsum_record_table_contract``。"""
    passed, skipped, failures = _run()
    print(f"{passed} passed, {skipped} skipped, {len(failures)} failed", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
