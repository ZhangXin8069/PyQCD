"""契约：配置画像缓存（``pyqcd.einsum._profile_cache``）的读写口径。

这一层**不参与数值计算**，所以普通数值测试永远碰不到它：配置选错了、旧行没被换掉、
或者"关掉开关却还在建目录"，结果照样逐位正确，只是慢、只是刷屏、只是把用户的工作
目录搞脏。本模块把下面八条固化成能独立失败的断言：

1. **CSV 往返**：表头与 :data:`PROFILE_COLUMNS` 逐列一致；行数、方程、归一化形状、
   毫秒列格式都对得上；追加写不重复表头；``load(force=True)`` 能按文件重建索引。
2. **选优规则**：``compute_ms`` 最小 → 同值取 ``calls`` 大者 → 再同取 ``updated_at``
   新者；返回值是副本，改不动索引。
3. **稀疏写入**：新键必写；同键同配置不快过"当前最优 × 0.95"（严格小于）就不落盘，
   只更新内存；快过阈值才再写一行。同时钉住"内存取更小值、磁盘只留够格的那行"。
4. **compare 只写不读**：``record_compare`` 真写文件且表头正确；同一 (键, 配置, 参照)
   在 60 s 窗口内只写一次；``load`` / ``best`` / ``summary`` / ``notice`` **一次都不读**
   该文件（跟踪 ``builtins.open`` 的路径），``summary`` 也没有它的读计数。
5. **关闭零 IO**：``PYQCD_EINSUM_PROFILE=0`` 时开关为假、写入全返回 ``False``，
   且临时目录里**什么都不产生**（连目录都不建）。
6. **不可写降级**：目录 ``chmod 500`` 后不抛异常，``summary()["last_error"]`` 非空、
   ``enabled`` 为假，写入返回 ``False``，也没有 CSV 被造出来。
7. **``format_shape``**：轴间 ``x``、操作数间 ``|``，已规范化字符串原样返回，
   无法解析给空串。
8. **目录优先级**：``PYQCD_EINSUM_PROFILE_DIR`` 压过默认 ``./.cache``；``configure``
   的参数再压过环境变量。默认那档只在临时 CWD 下验证，**不在仓库根留下 ``.cache``**。

``configure`` / ``summary()["requires_configure"]`` 是并行改动的产物：它尚未落地时
本模块用环境变量走同一条代码路径（``_use_dir``），只有"参数压过环境"那条用例
``SkipTest`` 并写明原因——不许静默通过。

不做任何计时断言：本模块只关心"写了什么、读了什么"，不关心它多快。

可直接运行::

    python -B -m pyqcd.testing._einsum_profile_cache_contract
"""
from __future__ import annotations

import builtins
import contextlib
import csv
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from ..einsum import _profile_cache

#: ``configure(directory=None) -> str`` 是否已经落地（未落地时按环境变量走同一路径）。
_HAS_CONFIGURE = hasattr(_profile_cache, "configure")

#: 真实的 ``open``：跟踪包装要委托给它，否则 ``csv`` / ``with`` 语句全都不能用。
_REAL_OPEN = builtins.open

#: ``summary()`` 文档里承诺的键（``requires_configure`` 是随 ``configure`` 一起加的；
#: ``skipped_rows`` / ``truncated_fields`` / ``dropped_rows`` / ``max_rows_per_key``
#: 是随规模上限与逐行降级一起加的——"字段集合"必须跟着实现扩容，否则新增计数器
#: 会在"集合不等"上误报，而漏掉的计数器又再没人盯）。
SUMMARY_KEYS = {"dir", "file", "enabled", "requires_configure",
                "rows", "keys", "loaded", "skipped_rows", "truncated_fields",
                "dropped_rows", "max_rows_per_key", "last_error"}

#: ``_profile_cache`` 的**计数器/磁盘行数**状态：不是配置，是跨用例会累积的量。
#: ``_disk_rows`` / ``_rewrite_rows`` 还参与 ``flush`` 的"要不要整表重写"判断，
#: 留着上一条用例的数会让下一条的"写了几行"变成别人的结果。值即模块里声明的初值。
_COUNTER_FIELDS = {
    "_skipped_rows": 0,       # 最近一次读盘跳过的坏行
    "_truncated_fields": 0,   # 累计截断的超长字段
    "_dropped_rows": 0,       # 累计被"每键最快 K 条"裁剪的行
    "_disk_rows": -1,         # -1 = 还不知道，用到时再数
    "_rewrite_rows": -1,      # -1 = 本进程还没整表重写过
}


#: ``_api`` 的进程级状态：本模块不直接调用它，但 ``_PROFILE_READY`` 缓存的是
#: **已加载的绝对目录**（``str|None``，不是布尔）。只清 ``_profile_cache`` 而把
#: ``_api._PROFILE_READY`` 留在上一条用例的目录上，下一条用例的 ``contract()``
#: 会以为"这个目录我读过"，直接拿到别人的索引——顺序污染就是这么来的。
_API_SET_FIELDS = ("_NATIVE_DISABLED", "_NATIVE_WARNED_REASONS",
                   "_SPECIALIZED_KERNEL_WARNED", "_KERNEL_SKIP_WARNED")
_API_VALUE_FIELDS = ("_PROFILE_READY", "_LAST_KERNEL", "_FUSED_ENV_OWNED")


def _snapshot_api() -> dict[str, Any]:
    """``_api`` 状态快照：集合拷成新集合，标量原样（``_PROFILE_READY`` 是字符串）。"""
    from ..einsum import _api

    snapshot: dict[str, Any] = {
        name: set(getattr(_api, name)) for name in _API_SET_FIELDS}
    snapshot.update({name: getattr(_api, name) for name in _API_VALUE_FIELDS})
    snapshot["_PROFILE_NOTICED"] = set(_api._PROFILE_NOTICED)
    return snapshot


def _restore_api(snapshot: dict[str, Any]) -> None:
    """按 :func:`_snapshot_api` 的副本原样放回（原地更新集合，不换对象）。"""
    from ..einsum import _api

    for name in _API_SET_FIELDS:
        getattr(_api, name).clear()
        getattr(_api, name).update(snapshot[name])
    _api._PROFILE_NOTICED.clear()
    _api._PROFILE_NOTICED.update(snapshot["_PROFILE_NOTICED"])
    for name in _API_VALUE_FIELDS:
        setattr(_api, name, snapshot[name])


class _OpenTrace:
    """记录 ``builtins.open`` 收到的每个路径，再原样委托给真实 ``open``。"""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def __call__(self, file: Any, *args: Any, **kwargs: Any):
        try:
            self.paths.append(os.fspath(file))
        except TypeError:
            self.paths.append(str(file))
        return _REAL_OPEN(file, *args, **kwargs)

    def touched(self, name: str) -> list[str]:
        """被打开过的、以 ``name`` 结尾的路径。"""
        return [path for path in self.paths if path.endswith(name)]


def _reset_state() -> None:
    """把模块的进程内状态清到"没配置、没加载、没建目录"的初始态。

    用例之间不串味的关键：``_configured_dir`` 是 ``configure`` 钉住的目录，**不**跟着
    环境变量走；漏掉它会让上一条用例的目录（例如 chmod 500 的那个）漏进下一条。
    ``_notice_done`` 是"不可用只提示一次"的闸门，同样必须复位。

    计数器一并归零：``_disk_rows`` / ``_rewrite_rows`` 参与 ``flush`` 的整表重写判断，
    别处留下的盘上行数会让本模块的"写了几行"读到别人的结果；``_skipped_rows`` 等三个
    累计量只增不减，不复位就永远在涨。
    """
    module = _profile_cache
    module._index.clear()
    module._pending.clear()
    module._compare_recent.clear()
    module._configured_dir = ""
    module._ready = False
    module._dir = ""
    module._dir_ok = False
    module._last_error = None
    module._notice_done = False
    module._loaded = False
    module._loaded_rows = 0
    for name, initial in _COUNTER_FIELDS.items():
        setattr(module, name, initial)


@contextlib.contextmanager
def _isolated(disabled: bool = False):
    """临时 CWD + 干净的环境变量与进程内状态；退出时全部还原。

    缓存目录默认相对 CWD，所以"不污染仓库根"靠的是把 CWD 换到临时树里；默认那档
    用例因此能安全地验 ``./.cache``。

    ``_api`` 的进程级状态也快照/复原：它自己不看目录，但 ``_PROFILE_READY`` 记的是
    "已读过的绝对目录"，把别人的路径留在那儿就等于把别人的索引借给下一条用例。
    """
    module = _profile_cache
    saved_cwd = os.getcwd()
    saved_env = {name: os.environ.get(name)
                 for name in (module.PROFILE_ENV, module.DIR_ENV)}
    saved_api = _snapshot_api()
    saved = {
        "index": {key: {config: dict(row) for config, row in slot.items()}
                  for key, slot in module._index.items()},
        "pending": [dict(row) for row in module._pending],
        "compare": dict(module._compare_recent),
        "configured_dir": module._configured_dir,
        "ready": module._ready,
        "dir": module._dir,
        "dir_ok": module._dir_ok,
        "last_error": module._last_error,
        "notice_done": module._notice_done,
        "loaded": module._loaded,
        "loaded_rows": module._loaded_rows,
        "counters": {name: getattr(module, name) for name in _COUNTER_FIELDS},
    }
    with tempfile.TemporaryDirectory(prefix="pyqcd-profile-") as root:
        work = os.path.join(root, "work")
        os.makedirs(work)
        try:
            os.chdir(work)
            os.environ.pop(module.DIR_ENV, None)
            if disabled:
                os.environ[module.PROFILE_ENV] = "0"
            else:
                os.environ.pop(module.PROFILE_ENV, None)
            _reset_state()
            yield work
        finally:
            os.chdir(saved_cwd)
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            module._index.clear()
            module._index.update(saved["index"])
            module._pending[:] = saved["pending"]
            module._compare_recent.clear()
            module._compare_recent.update(saved["compare"])
            module._configured_dir = saved["configured_dir"]
            module._ready = saved["ready"]
            module._dir = saved["dir"]
            module._dir_ok = saved["dir_ok"]
            module._last_error = saved["last_error"]
            module._notice_done = saved["notice_done"]
            module._loaded = saved["loaded"]
            module._loaded_rows = saved["loaded_rows"]
            for name, value in saved["counters"].items():
                setattr(module, name, value)
            _restore_api(saved_api)


def _use_dir(directory: str) -> str:
    """让缓存落到 ``directory``：有 ``configure`` 就用它，否则设环境变量。"""
    if _HAS_CONFIGURE:
        return _profile_cache.configure(str(directory))
    os.environ[_profile_cache.DIR_ENV] = str(directory)
    return _profile_cache._dir_text()


def _activate(directory: str) -> str:
    """指向 ``directory`` 并**先读完一次**，把 ``_loaded`` 定下来。

    这一步不是可选的：``best`` / ``summary`` / ``notice`` 首次调用会走一遍
    ``_load_locked``，那条路径会 ``_index.clear()`` 后只把 ``_pending`` 合并回来。
    所以"只进内存、不够格落盘"的观测必须先有 ``_loaded = True`` 才能在这条用例里
    稳定读到——否则验的就不是选优规则，而是重载语义。
    """
    resolved = _use_dir(directory)
    _profile_cache.load()
    return resolved


def _key(**overrides: Any) -> dict[str, Any]:
    """主表键的稀疏写法（``host`` 缺省由模块补本机名，两边一致）。"""
    key = {
        "backend": "numpy",
        "dtype": "complex128",
        "equation": "ij,jk->ik",
        "shape": [(2, 3), (3, 4)],
        "device": "cpu",
        "user_device": "cpu",
    }
    key.update(overrides)
    return key


def _row(**overrides: Any) -> dict[str, Any]:
    """主表行的稀疏写法：键 + 配置 + 观测值。"""
    row = {
        **_key(),
        "kernel": "kupl_dedicated",
        "optimize": "auto",
        "compute_ms": 1.0,
        "calls": 1,
    }
    row.update(overrides)
    return row


def _column(name: str) -> int:
    """``PROFILE_COLUMNS`` 里的列序（避免测试里写死数字）。"""
    return _profile_cache.PROFILE_COLUMNS.index(name)


def _read_rows(path: str) -> list[list[str]]:
    """整份 CSV（含表头行）。"""
    with _REAL_OPEN(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


# ── 1. CSV 往返 ─────────────────────────────────────────────────────────────

def test_record_filters_non_replayable_kernels():
    """``equivalence`` 不可重放 → 拒写；``generic``/``backend`` 可重放 → 允许写。

    ``equivalence`` 是"本次由改写跑成"的诊断标签：在原方程上强制具体内核必然
    "未尝试"，所以它给不出可用建议。``generic`` 恰恰相反——``kernel_force="generic"``
    合法且实测常最快（某 10 操作数方程上 generic 385 ms vs topology 2735 ms），
    若把它拒之门外，缓存就把最大的一笔收益丢掉了。
    """
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _activate(directory)
        assert _profile_cache.record(_row(
            equation="zz->zz", shape=[(2, 2)],
            kernel="equivalence", compute_ms=0.1, calls=1)) is False
        assert _profile_cache.record(_row(
            equation="yy->yy", shape=[(2, 2)],
            kernel="generic", compute_ms=0.5, calls=1)) is True
        assert _profile_cache.record(_row(
            equation="xx->xx", shape=[(2, 2)],
            kernel="backend", compute_ms=0.4, calls=1)) is True
        assert _profile_cache.flush() == 2
        table = _read_rows(os.path.join(directory, _profile_cache.PROFILE_NAME))
        assert [row[_column("kernel")] for row in table[1:]] == ["generic", "backend"]


def test_profile_csv_roundtrip_preserves_columns_and_rows():
    """临时目录 → 写 3 行 → flush → 表头/行数/内容/追加不重写表头/读回。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _activate(directory)
        rows = [
            _row(equation="ij,jk->ik", shape=[(2, 3), (3, 4)],
                 kernel="kupl_dedicated", compute_ms=1.5, calls=2),
            _row(equation="ab,bc->ac", shape=[(4, 5), (5, 6)],
                 kernel="kupl_topology", compute_ms=0.25, calls=3),
            # 第三行刻意用另一个**可重放**的内核 + 另一个 optimize：``generic``
            # 不可重放，``record()`` 现在会拒绝它（见下面的断言），不能拿来当往返样本。
            _row(equation="abc,cd->abd", shape=[(2, 2, 2), (2, 2)],
                 kernel="kupl_dedicated", optimize="greedy",
                 compute_ms=7.0, calls=1),
        ]
        for row in rows:
            assert _profile_cache.record(row) is True, (
                "新键必须立即排队落盘", row["equation"])
        assert _profile_cache.flush() == 3

        path = os.path.join(directory, _profile_cache.PROFILE_NAME)
        table = _read_rows(path)
        # 表头必须与常量**逐列**一致（顺序即写序，读侧按名字取值）。
        assert table[0] == list(_profile_cache.PROFILE_COLUMNS), table[0]
        assert len(table) == 4, ("3 行数据 + 1 行表头", len(table))
        assert [row[_column("equation")] for row in table[1:]] == [
            "ij,jk->ik", "ab,bc->ac", "abc,cd->abd"]
        # shape 落盘前必须归一化：x 连轴、| 连操作数。
        assert [row[_column("shape")] for row in table[1:]] == [
            "2x3|3x4", "4x5|5x6", "2x2x2|2x2"]
        # 毫秒列固定三位小数（否则同一个数会写出多种字面量）。
        assert [row[_column("compute_ms")] for row in table[1:]] == [
            "1.500", "0.250", "7.000"]
        assert [row[_column("calls")] for row in table[1:]] == ["2", "3", "1"]

        # 读回：一次顺序读重建索引，best() 能取到对应配置（键含归一化 shape）。
        assert _profile_cache.load(force=True) == 3
        picked = _profile_cache.best(_key(equation="ab,bc->ac",
                                         shape=[(4, 5), (5, 6)]))
        assert picked is not None, "读回后必须命中"
        assert picked["kernel"] == "kupl_topology", picked
        assert picked["compute_ms"] == "0.250", picked
        message = _profile_cache.notice()
        assert message is not None and message.startswith("einsum 配置画像：已加载"), message
        assert "3 行" in message, message

        # 追加写：表头只写一次，新行直接接在后面。
        assert _profile_cache.record(
            _row(equation="ij->ji", shape=[(3, 3)], compute_ms=2.0)) is True
        assert _profile_cache.flush() == 1
        appended = _read_rows(path)
        assert appended[0] == list(_profile_cache.PROFILE_COLUMNS), appended[0]
        assert len(appended) == 5, ("追加不得重复表头", len(appended))


# ── 2. 选优规则 ─────────────────────────────────────────────────────────────

def test_profile_best_prefers_lowest_compute_then_calls_then_newest():
    """compute_ms 最小 → 同值 calls 大 → 再同值 updated_at 新；返回值是副本。"""
    with _isolated() as work:
        _activate(os.path.join(work, "cache"))
        key = _key()
        rows = [
            _row(kernel="k_slow", compute_ms=5.0, calls=1,
                 updated_at="2026-01-01T00:00:00+00:00"),
            _row(kernel="k_best_calls_low", compute_ms=2.0, calls=1,
                 updated_at="2026-01-01T00:00:01+00:00"),
            _row(kernel="k_best_calls_high", compute_ms=2.0, calls=9,
                 updated_at="2026-01-02T00:00:00+00:00"),
        ]
        for row in rows:
            _profile_cache.record(row)          # 返回值是稀疏口径，这里不关心

        picked = _profile_cache.best(key)
        assert picked is not None
        # 主判据：compute_ms 最小（5.0 那行必须被淘汰，哪怕它 calls 也小）。
        assert float(picked["compute_ms"]) == 2.0, picked
        # 并列判据一：calls 大者胜（9 > 1）。
        assert picked["kernel"] == "k_best_calls_high", picked
        assert picked["calls"] == "9", picked

        # 并列判据二：compute_ms 与 calls 都相同 → updated_at 新者胜。
        _profile_cache.record(
            _row(kernel="k_tie_newest", compute_ms=2.0, calls=9,
                 updated_at="2026-03-01T00:00:00+00:00"))
        newest = _profile_cache.best(key)
        assert newest is not None and newest["kernel"] == "k_tie_newest", newest
        assert newest["updated_at"] == "2026-03-01T00:00:00+00:00", newest

        # 副本语义：改返回值不得污染索引。
        newest["kernel"] = "hacked"
        assert _profile_cache.best(key)["kernel"] == "k_tie_newest"

        # 未命中返回 None（而不是空字典）。
        assert _profile_cache.best(_key(equation="zz->z")) is None


# ── 3. 稀疏写入 ─────────────────────────────────────────────────────────────

def test_profile_record_is_sparse():
    """同键同配置不快过"当前最优 × 0.95"→ False 且不落盘；快过 → True 再写一行。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _activate(directory)
        path = os.path.join(directory, _profile_cache.PROFILE_NAME)
        base = _row(compute_ms=100.0, calls=1)

        # 新键：必写。
        assert _profile_cache.record(base) is True
        # 只是排队，还没落盘。
        assert not os.path.exists(path), "record() 只该更新内存/队列，不该直接写盘"

        # 同样耗时：不够格。
        assert _profile_cache.record(_row(compute_ms=100.0, calls=1)) is False, "同耗时不该再写"
        # 更慢：不够格。
        assert _profile_cache.record(_row(compute_ms=140.0, calls=1)) is False, "更慢不该写"
        # 边界：恰好 95%（严格小于才算改进），不够格。
        assert _profile_cache.record(_row(compute_ms=95.0, calls=1)) is False, "95% 不算改进"
        assert not os.path.exists(path), "不够格的观测不得落盘"

        # 内存取更小值：这些"不够格写盘"的观测仍然参与选优。
        assert _profile_cache.best(_key())["compute_ms"] == "95.000", _profile_cache.best(_key())
        assert _profile_cache.flush() == 1
        table = _read_rows(path)
        assert len(table) == 2, ("表头 + 1 行", table)
        # 磁盘只留"够格"的那行——写入口径与内存口径本来就是两回事。
        assert [row[_column("compute_ms")] for row in table[1:]] == ["100.000"], table

        # 快过 5%（89.0 < 95.0 * 0.95 = 90.25）：值得再写一行。
        assert _profile_cache.record(_row(compute_ms=89.0, calls=1)) is True, "快过 5% 必须写"
        assert _profile_cache.flush() == 1
        table = _read_rows(path)
        assert len(table) == 3, ("表头 + 2 行", table)
        assert table[0] == list(_profile_cache.PROFILE_COLUMNS), table[0]
        assert [row[_column("compute_ms")] for row in table[1:]] == [
            "100.000", "89.000"], table
        assert _profile_cache.best(_key())["compute_ms"] == "89.000"


# ── 4. compare 只写不读 ─────────────────────────────────────────────────────

def test_profile_compare_is_write_only():
    """compare 真写且表头正确；窗口内去重；load/best/summary/notice 一次都不读它。"""
    with _isolated() as work:
        directory = os.path.join(work, "cache")
        _activate(directory)
        first = {
            **_key(equation="ab,bc->ac"),
            "kernel": "kupl_dedicated",
            "optimize": "auto",
            "compare": "ij,jk->ik",
            "ref_ms": 3.0,
            "compute_ms": 1.5,
            "equivalence": {"kind": "permute", "equation": "ij,jk->ik",
                            "operand_order": [1, 0]},
        }
        second = {
            **_key(equation="abc->acb"),
            "kernel": "kupl_topology",
            "optimize": "greedy",
            "compare": "abc->acb",
            "ref_ms": 2.0,
            "compute_ms": 1.0,
        }
        assert _profile_cache.record_compare(first) is True
        assert _profile_cache.record_compare(second) is True

        compare_path = os.path.join(directory, _profile_cache.COMPARE_NAME)
        assert os.path.exists(compare_path), "compare 行必须即时落盘（不进队列）"
        table = _read_rows(compare_path)
        assert table[0] == list(_profile_cache.COMPARE_COLUMNS), table[0]
        assert len(table) == 3, ("表头 + 2 行", len(table))
        equations = [row[_profile_cache.COMPARE_COLUMNS.index("equation")]
                     for row in table[1:]]
        assert equations == ["ab,bc->ac", "abc->acb"], equations
        # equivalence 报告（explain 的字典）可以直接塞进来，字段被摊平落盘。
        kinds = [row[_profile_cache.COMPARE_COLUMNS.index("equivalence_kind")]
                 for row in table[1:]]
        assert kinds[0] == "permute", kinds

        # 同一 (键, 配置, 参照) 在 60 s 窗口内只写一次（写入侧的稀疏口径）。
        assert _profile_cache.record_compare(dict(first)) is False, "窗口内重复应被去重"
        assert len(_read_rows(compare_path)) == 3, "窗口内重复行不得再写一次"

        # 只写不读：跟踪 open，四个读入口都不许碰 compare 文件。
        trace = _OpenTrace()
        with patch("builtins.open", trace):
            _profile_cache.load(force=True)
            _profile_cache.best(_key(equation="ab,bc->ac"))
            state = _profile_cache.summary()
            _profile_cache.notice()
        assert not trace.touched(_profile_cache.COMPARE_NAME), (
            "compare 表必须一次都不读", trace.touched(_profile_cache.COMPARE_NAME))
        # 主表确实被读过（否则上面的"没读 compare"可能只是整个 IO 都没发生）。
        assert trace.touched(_profile_cache.PROFILE_NAME), trace.paths

        # summary 里也没有 compare 的读取计数（没有"读了多少 compare 行"这种字段），
        # 文档承诺的键必须齐全。
        assert not [name for name in state if "compare" in name], list(state)
        assert SUMMARY_KEYS <= set(state), (SUMMARY_KEYS - set(state), list(state))
        # 目录可显式配置，但**不需要**先配置：这个提示位恒为假。
        assert state["requires_configure"] is False, state


# ── 5. 关闭零 IO ────────────────────────────────────────────────────────────

def test_profile_disabled_does_no_io():
    """``PYQCD_EINSUM_PROFILE=0``：开关假、写入假、临时目录里不产生任何文件。"""
    with _isolated(disabled=True) as work:
        assert _profile_cache.enabled() is False
        # 显式参数只回答开关本身（压过环境里的 0）。
        assert _profile_cache.enabled(True) is True
        assert _profile_cache.enabled(False) is False

        assert _profile_cache.record(_row()) is False
        assert _profile_cache.record_compare({**_key(), "compare": "ij->ji"}) is False
        assert _profile_cache.load() == 0
        assert _profile_cache.flush() == 0
        state = _profile_cache.summary()
        assert state["enabled"] is False, state
        assert state["rows"] == 0 and state["keys"] == 0, state
        assert state["loaded"] is False, state
        assert _profile_cache.notice() is None

        # 关闭时连目录都不建：临时 CWD 里必须还是空的。
        assert os.listdir(work) == [], os.listdir(work)
        assert not os.path.exists(os.path.join(work, _profile_cache.DEFAULT_DIR))


# ── 6. 不可写降级 ───────────────────────────────────────────────────────────

def test_profile_unwritable_directory_degrades():
    """chmod 500 的目录：不抛、``last_error`` 非空、写入返回 False、不造 CSV。"""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise unittest.SkipTest("以 root 运行，chmod 挡不住写入，本契约不可验")

    with _isolated() as work:
        directory = os.path.join(work, "readonly")
        os.makedirs(directory)
        os.chmod(directory, 0o500)
        try:
            # 定位/建目录不得把异常抛给调用方。
            _use_dir(directory)
            _profile_cache.load()

            state = _profile_cache.summary()
            assert state["last_error"], ("不可写必须留下原因", state)
            assert state["enabled"] is False, ("有效可用性必须为假", state)
            assert state["dir"] == os.path.abspath(directory), state

            assert _profile_cache.record(_row()) is False
            assert _profile_cache.record_compare({**_key(), "compare": "ij->ji"}) is False
            assert _profile_cache.flush() == 0
            # 一行提示，且必须说明"不可用"（不刷屏、也不假装有缓存）。
            message = _profile_cache.notice()
            assert message is not None and "不可用" in message, message

            # 目录本身可读：确认没有任何 CSV 被造出来。
            assert sorted(os.listdir(directory)) == [], os.listdir(directory)
        finally:
            os.chmod(directory, 0o700)


# ── 7. format_shape ────────────────────────────────────────────────────────

def test_profile_format_shape():
    """``[(2,3),(3,4)]`` → ``2x3|3x4``；已规范化字符串原样返回；怪输入不抛。"""
    format_shape = _profile_cache.format_shape
    assert format_shape([(2, 3), (3, 4)]) == "2x3|3x4"
    assert format_shape([[2, 3], [3, 4]]) == "2x3|3x4"
    assert format_shape([(2, 3)]) == "2x3"
    # 已规范化的字符串只去空白（幂等：写回读回都落到同一个键）。
    assert format_shape(" 2x3|3x4 ") == "2x3|3x4"
    assert format_shape(format_shape([(2, 3), (3, 4)])) == "2x3|3x4"
    # 无法解析给空串，归一化不因为怪输入把调用方拖崩。
    assert format_shape(None) == ""
    assert format_shape(7) == ""
    # 语义提醒：入参必须是"操作数列表"，裸轴长列表 ``[2, 3]`` 会被读成两个 0 维操作数。
    assert format_shape([2, 3]) == "2|3"


# ── 8. 目录优先级 ───────────────────────────────────────────────────────────

def test_profile_directory_priority_env_over_default():
    """环境变量压过默认 ``./.cache``；默认那档只落在临时 CWD 下。"""
    with _isolated() as work:
        module = _profile_cache
        default_dir = os.path.abspath(os.path.join(work, module.DEFAULT_DIR))
        # 1) 都没设：默认是 CWD 下的 .cache（不是仓库根）。
        state = module.summary()
        assert state["dir"] == default_dir, state
        assert state["file"] == os.path.join(default_dir, module.PROFILE_NAME), state
        assert state["dir"].startswith(os.path.abspath(work)), (
            "默认目录必须落在临时 CWD 下，仓库根不许出现 .cache", state)

        # 2) 环境变量覆盖默认：改完即时生效（重启目录解析）。
        env_dir = os.path.join(work, "env_cache")
        os.environ[module.DIR_ENV] = env_dir
        _reset_state()
        state = module.summary()
        assert state["dir"] == os.path.abspath(env_dir), state
        assert state["file"] == os.path.join(os.path.abspath(env_dir),
                                            module.PROFILE_NAME), state

        # 3) 真的写到环境变量指定的目录里（路径解析与落盘是同一份口径）。
        assert module.record(_row()) is True
        assert module.flush() == 1
        assert os.path.exists(os.path.join(env_dir, module.PROFILE_NAME))
        assert not os.path.exists(os.path.join(default_dir, module.PROFILE_NAME)), (
            "环境变量生效时不得再往默认目录写")


def test_profile_configure_parameter_beats_environment():
    """``configure(directory)`` 的参数压过 ``PYQCD_EINSUM_PROFILE_DIR``。"""
    if not _HAS_CONFIGURE:
        raise unittest.SkipTest(
            "_profile_cache.configure 尚未落地（模块 __all__ 里没有它）；"
            "参数优先级不可验，其余七条已用环境变量走同一条代码路径")

    with _isolated() as work:
        module = _profile_cache
        env_dir = os.path.join(work, "env_cache")
        param_dir = os.path.join(work, "param_cache")
        os.environ[module.DIR_ENV] = env_dir
        _reset_state()

        returned = module.configure(param_dir)
        assert isinstance(returned, str), type(returned).__name__
        assert returned == os.path.abspath(param_dir), returned

        # 参数压过环境：落盘真的去了 param_dir，而不是 env_dir。
        assert module.record(_row()) is True
        assert module.flush() == 1
        assert os.path.exists(os.path.join(param_dir, module.PROFILE_NAME))
        assert not os.path.exists(os.path.join(env_dir, module.PROFILE_NAME)), (
            "configure 的参数必须压过环境变量")

        # configure(None) 走回环境变量/默认值；至少必须给出一个绝对路径。
        fallback = module.configure(None)
        assert isinstance(fallback, str) and os.path.isabs(fallback), fallback


TESTS = (
    test_profile_csv_roundtrip_preserves_columns_and_rows,
    test_record_filters_non_replayable_kernels,
    test_profile_best_prefers_lowest_compute_then_calls_then_newest,
    test_profile_record_is_sparse,
    test_profile_compare_is_write_only,
    test_profile_disabled_does_no_io,
    test_profile_unwritable_directory_degrades,
    test_profile_format_shape,
    test_profile_directory_priority_env_over_default,
    test_profile_configure_parameter_beats_environment,
)


def test_einsum_profile_cache_contracts():
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
            f"{len(failures)} 条配置画像缓存契约失败：" + "; ".join(failures))
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
