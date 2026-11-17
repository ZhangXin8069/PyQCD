"""qcd-einsum 的配置画像缓存：用少量 IO 换最优配置（**只给建议，不参与数值计算**）。

为什么单独抽一层
----------------
同一个 ``(方程, 形状, 设备)`` 在长时间批处理里会被反复问"该用哪个 kernel、哪个
收缩序"，而答案只随机器与配置变化。把历史观测落成一张 CSV，下次开工用**一次顺序读**
换掉重复的试跑与匹配时间。本模块只回答"上次哪个配置最快"：它不 import 执行层
（``_api`` / ``_kernel_*`` / ``_shape_paths``）、不解析方程、不碰张量，也**不做计时**
——``init_ms`` / ``compute_ms`` 全部由调用方测好给进来。

两份表，读写口径不同
--------------------
* :data:`PROFILE_NAME`（``profile.csv``）——**读 + 写**。进程内由 :func:`load` 一次
  读进内存索引，:func:`best` 是纯字典查找；:func:`record` 默认只更新内存，把"值得
  落盘"的行攒进队列，由 :func:`flush`（或进程退出 ``atexit``）一次追加写。
* :data:`COMPARE_NAME`（``profile_compare.csv``）——**只写不读**。它是"下次同类方程
  可以直接套用等价改写、省掉匹配时间"的旁证，写出去就有人看，因此**不为它建索引、
  不为它做加载**，一次都不读。

主表的键与选优
--------------
键 = ``(host, user_device, backend, dtype, equation, shape, device)``
（:data:`KEY_FIELDS`）。``shape`` 统一规范成 ``2x3|3x4``——``x`` 连轴、``|`` 连操作数
（见 :func:`format_shape`），否则 ``[2,3]`` / ``2x3`` / ``(2,3)`` 三种写法会把同一个
形状拆成三个键。

一个键下挂若干**配置行**（由 ``kernel`` + ``optimize`` 区分），选优口径：
``compute_ms`` 最小 → 同值取 ``calls`` 大者 → 再同取 ``updated_at`` 新者。

稀疏写入
--------
:func:`record` 只在两种情况下真的排队写盘：**新键**，或**比该键当前最优快 >5%**
（:data:`_IMPROVE_RATIO`，严格小于）。其余观测只进内存——否则每次命中都追加一行，
读回时反而要为成堆的重复行做合并，日志也不再能一眼看出"配置是什么时候被换掉的"。

降级与线程安全
--------------
目录不可写、写盘失败、读盘失败**一律不抛给调用方**（原因只存在**运行时内部变量**
里，不改 ``PYQCD_EINSUM_PROFILE`` 的语义）。两种降级严格分开：**目录不可用**
（建不出、写不进）才让 :func:`enabled` 转 ``False``；**读侧**的一行坏数据（长度/
编码/列数不对）只跳过并计数（``summary()["skipped_rows"]``），读失败只记
``last_error``——一行坏文件不该让整张表作废。:func:`notice` 只给**一行**提示
（本模块从不主动打印，也不重复刷屏——不可用的说明只说一次）。索引与待写队列共用
**一个** :class:`threading.Lock`，不是每行一把锁；``fork`` 出的子进程用
``os.register_at_fork`` 重建它——``threading.Lock`` 不随 ``fork`` 重置，父进程某线程
持锁时 fork 会让子进程第一次 :func:`summary` / :func:`record` 就永久阻塞。退出阶段
的兜底 flush **带超时**：拿不到锁就丢弃待写行，宁可少几行记录也不卡住进程。

规模上限
--------
每键只保留 ``compute_ms`` 最快的 :data:`_MAX_ROWS_PER_KEY` 条配置行（内存与落盘
同一口径，裁剪掉的条数见 ``summary()["dropped_rows"]``）；``profile.csv`` 的数据行
超过 :data:`_MAX_ROWS` 时，:func:`flush` 先整表重写成"每键最快 K 条"。``note`` /
``equivalence_equation`` 写盘前按 :data:`_CLIP_LIMITS` 截断，截断次数见
``summary()["truncated_fields"]``。

配置目录
--------
:func:`configure` 是唯一入口，分辨率优先级**依次**为：显式参数 >
``PYQCD_EINSUM_PROFILE_DIR`` > ``DEFAULT_DIR``（``./.cache``，按当前工作目录取绝对
路径）；空白与 ``None`` 都算"未给"，逐级下退。没调用过 :func:`configure` 时，
第一次 IO 按同一口径自动解析一次。目录不存在就建；不可写只降级、不外抛。

环境变量
--------
``PYQCD_EINSUM_PROFILE_DIR``
    缓存目录；未设时用 ``./.cache``（相对当前工作目录）。不存在就
    ``mkdir(parents=True, exist_ok=True)``。
``PYQCD_EINSUM_PROFILE``
    总开关，缺省 ``"1"``（开启）；``0/false/off/no`` 关闭，关闭时**不读也不写、
    连目录都不创建**。
"""

from __future__ import annotations

import atexit
import csv
import datetime
import io
import os
import socket
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "COMPARE_COLUMNS",
    "COMPARE_NAME",
    "DEFAULT_DIR",
    "DIR_ENV",
    "KEY_FIELDS",
    "PROFILE_COLUMNS",
    "PROFILE_ENV",
    "PROFILE_NAME",
    "best",
    "configure",
    "enabled",
    "ensure_configured",
    "flush",
    "format_shape",
    "load",
    "notice",
    "record",
    "record_compare",
    "summary",
]

#: 配置画像缓存总开关：``0/false/off/no`` 关闭；缺省 ``"1"``（开启）。
PROFILE_ENV = "PYQCD_EINSUM_PROFILE"

#: 缓存目录覆盖；未设时用 :data:`DEFAULT_DIR`（相对当前工作目录）。
DIR_ENV = "PYQCD_EINSUM_PROFILE_DIR"

#: 缺省目录（相对 CWD）。
DEFAULT_DIR = ".cache"

#: 主表（读 + 写）：一个键下若干配置行。
PROFILE_NAME = "profile.csv"

#: 旁证表（只写不读）：等价改写/参照耗时。
COMPARE_NAME = "profile_compare.csv"

#: 主表键字段（顺序即键序）；``shape`` 必须先过 :func:`format_shape` 归一化。
KEY_FIELDS = (
    "host",
    "user_device",
    "backend",
    "dtype",
    "equation",
    "shape",
    "device",
)

#: 主表表头（顺序即写序，读侧按名字取值）。
PROFILE_COLUMNS = (
    "host",
    "user_device",
    "real_device",
    "backend",
    "dtype",
    "equation",
    "shape",
    "device",
    "kernel",
    "optimize",
    "init_ms",
    "compute_ms",
    "calls",
    "updated_at",
    "pyqcd_version",
    "note",
)

#: compare 表表头（同样顺序即写序）。
COMPARE_COLUMNS = (
    "host",
    "user_device",
    "real_device",
    "backend",
    "dtype",
    "equation",
    "shape",
    "device",
    "kernel",
    "optimize",
    "compare",
    "compare_optimize",
    "ref_ms",
    "ratio",
    "equivalence_kind",
    "equivalence_equation",
    "operand_order",
    "operand_permutations",
    "output_permutation",
    "compute_ms",
    "updated_at",
    "pyqcd_version",
)

# 新观测必须快过该键当前最优的这个比例才值得落盘（严格小于才写）。
_IMPROVE_RATIO = 0.95
#: **不可重放**的 kernel 取值：``equivalence`` 只说明"这次是改写跑成的"，在原方程上
#: 强制具体内核必然"未尝试"，不能当建议。``generic``/``backend`` 都**可**重放，
#: 必须参与选优（实测某 10 操作数方程 generic 385 ms vs topology 2735 ms）。
_NON_REPLAYABLE = frozenset({"", "equivalence"})

# 待写队列攒到这个长度就顺手冲一次，避免长跑进程里无限涨。
_FLUSH_THRESHOLD = 512

# 同一 (键, 配置, 参照) 的 compare 行 60 s 内只写一次。
_COMPARE_WINDOW_S = 60.0

# compare 去重表的容量上限（超了先清掉过期的）。
_COMPARE_MEMORY = 4096

#: 每键最多保留的配置行数（只留 ``compute_ms`` 最快的几条）。
_MAX_ROWS_PER_KEY = 4

#: 主表数据行的上限；超过就在 :func:`flush` 前整表重写成"每键最快 K 条"。
_MAX_ROWS = 50000

#: 整表重写后若仍高于上限（键太多，K 条也压不下去），要再涨这么多行才重写一次——
#: 否则每次 flush 都会整表重写，把缓存变成写放大源。
_REWRITE_GROWTH = _MAX_ROWS // 10

#: 超长文本字段的写盘上限（撑坏整张表的是这类字段，不是方程本身）。
_CLIP_LIMITS = {"note": 200, "equivalence_equation": 512}

#: 退出阶段拿锁的最长等待（秒）：拿不到就丢弃待写行，绝不卡住进程。
_EXIT_LOCK_TIMEOUT_S = 2.0

#: 单次读盘上限（病态大文件不该把内存吃光；超出的尾巴按坏行跳过）。
_MAX_LOAD_BYTES = 64 << 20

#: 连续这么多行解析失败，就认为 csv 流已不可用，停止读取。
_MAX_CONSECUTIVE_ERRORS = 8

#: 索引与待写队列的唯一一把锁（不要为单行加锁；``fork`` 后在子进程重建）。
_LOCK = threading.Lock()

#: 键 → {配置键: 该配置的最优行}；:func:`load` 一次填满，之后只查不改结构。
_index: dict[tuple[str, ...], dict[tuple[str, str], dict[str, str]]] = {}

#: 攒着准备追加写的行（只在内存里，冲盘后清空）。
_pending: list[dict[str, str]] = []

#: compare 的 60 s 去重表：键 → 上次写盘时刻（``time.monotonic``）。
_compare_recent: dict[tuple[str, ...], float] = {}

#: 是否已尝试加载过（``force=True`` 可重来）。
_loaded = False

#: 最近一次从 CSV 真正读进来的行数（``load`` 的返回值）。
_loaded_rows = 0

#: 最近一次读盘跳过了多少坏行（长度/编码/列数不对）。
_skipped_rows = 0

#: 累计截断了多少个超长字段（``note`` / ``equivalence_equation``）。
_truncated_fields = 0

#: 累计有多少配置行被"每键最快 K 条"裁剪掉。
_dropped_rows = 0

#: 盘上大约有多少数据行（``-1`` 表示不知道，用到时再数一次）。
_disk_rows = -1

#: 上一次整表重写时的行数（``-1`` = 本进程还没重写过；配合 :data:`_REWRITE_GROWTH`
#: 防止反复重写）。
_rewrite_rows = -1

#: 目录是否已解析/建好（只做一次；:func:`configure` 会重新置位）。
_ready = False

#: :func:`configure` 显式钉住的目录（绝对路径）；空串表示"没钉过，按 env/默认现算"。
_configured_dir = ""

#: 解析后的缓存目录。
_dir = ""

#: 目录是否可用（建得出、写得进）；不可用即"不缓存"。
_dir_ok = False

#: 最近一次降级原因（写失败/目录不可写/读失败）。
_last_error: str | None = None

#: 不可用的说明是否已经给过（:func:`notice` 只说一次）。
_notice_done = False

#: 主机名与真实设备型号都只探测一次。
_host: str | None = None
_real_device: str | None = None


# -- 对外接口 ---------------------------------------------------------------
def enabled(value: bool | None = None) -> bool:
    """配置画像缓存是否**可用**：总开关打开，且目录没在运行时被判不可用。

    无参调用是有效可用性：``PYQCD_EINSUM_PROFILE`` 打开（缺省 ``"1"``；
    ``0/false/off/no``，大小写不敏感、去首尾空白）**并且**目录没被判死（不可写/
    写失败/读失败，原因见 :func:`summary` 的 ``last_error``）。目录尚未解析时按
    "可用"处理——真正的探测留给第一次 IO，所以单纯查询开关不会建目录。
    与 :func:`._cache.enabled` 一样在**调用时**求值，同进程内改环境变量即时生效。

    显式 ``value`` 只回答**开关本身**（调用方覆盖开关用），不看目录状态。
    """
    if value is not None:
        return bool(value)
    return _switch_on() and not _blocked()


def configure(directory=None) -> str:
    """钉住缓存目录并立即就位，返回绝对路径；**不抛异常**。

    分辨率优先级与 CLI/API 同口径：显式参数 > ``PYQCD_EINSUM_PROFILE_DIR`` >
    :data:`DEFAULT_DIR`（``./.cache``）；``None`` 与空白都算"未给"，逐级下退。
    目录不存在就建；不可写时不外抛，只记下运行时原因（:func:`enabled` 转 ``False``、
    :func:`record` 返回 ``False``、:func:`summary` 的 ``last_error`` 给出原因），
    ``PYQCD_EINSUM_PROFILE`` 的语义不受影响。

    换目录等于换一张表：索引、载入标记、compare 去重表一并清空（否则 :func:`best`
    会把旧目录的行当成新目录的）；切走之前先把旧目录里还没落盘的观测冲掉（失败即
    放弃）。总开关关着时不建目录，但路径仍然记下，以后打开开关直接用它。
    """
    global _configured_dir, _dir, _ready, _dir_ok, _last_error, _notice_done
    global _loaded, _loaded_rows, _disk_rows, _rewrite_rows
    path = _resolve_dir_text(directory)
    with _LOCK:
        previous = _dir
        if previous and previous != path and _dir_ok:
            _flush_locked()                      # 旧目录的待写行先落盘，再换目录
        _configured_dir = path
        _dir = path
        _notice_done = False
        _last_error = None
        if previous != path:                     # 换目录 = 换一张表，索引不能沿用
            _index.clear()
            _compare_recent.clear()
            _loaded, _loaded_rows = False, 0
            _disk_rows = -1
            _rewrite_rows = -1
        # 先复位再探测：并发的 :func:`enabled` 此刻看到"没判过"，不会把探测中的
        # 半成品状态当成结论（它们随后会阻塞在这把锁上，拿到的是最终结果）。
        _dir_ok = False
        _ready = False
        if not _switch_on():
            return path                          # 关着就不碰磁盘，等真要用时再探测
        _probe_locked(path)
        _ready = True
        return path


def ensure_configured() -> str:
    """确保缓存目录已就位并返回绝对路径；**已钉住的目录保持不变**。

    与 :func:`configure` 的分工：后者总按"参数 > ``PYQCD_EINSUM_PROFILE_DIR`` >
    默认 ``./.cache``"重算，所以 ``configure(None)`` 会把调用方先前钉住的目录改回
    默认；本函数只"补齐"，不"改写"——已有目录就沿用它，没有才按同一优先级解析。

    调用方（``_api._profile_begin``）因此在 ``profile_dir is None`` 时不至于把
    测试/上层脚本用 :func:`configure` 钉好的目录悄悄换掉，也不会在仓库根凭空建
    ``./.cache``。
    """
    with _LOCK:
        _ensure_locked()
        return os.path.abspath(_dir_text())


def format_shape(shapes: Any) -> str:
    """把输入形状规范化成 ``2x3|3x4``（``x`` 连轴、``|`` 连操作数）。

    入参可以是**已规范化的字符串**（去空白后原样返回），也可以是一串操作数形状，
    例如 ``[(2, 3), (3, 4)]``。无法解析时返回空串——归一化不该因为一个怪形状把
    调用方拖崩。

    统一格式是键的一部分：``[2,3]`` / ``2x3`` / ``(2, 3)`` 必须落到同一个键上。
    """
    if isinstance(shapes, str):
        return shapes.strip()
    if shapes is None:
        return ""
    try:
        operands = list(shapes)
    except TypeError:
        return ""
    parts: list[str] = []
    for operand in operands:
        try:
            axes = list(operand)
        except TypeError:
            axes = [operand]
        parts.append("x".join(str(axis).strip() for axis in axes))
    return "|".join(parts)


def load(force: bool = False) -> int:
    """把 ``profile.csv`` 一次读进内存索引（进程内只读一次），返回读入行数。

    ``force=True`` 重读一遍：索引按文件重建，再把尚未落盘的内存行合并回去
    （否则刚 ``record`` 进来、还没 ``flush`` 的更优配置会凭空消失）。
    坏行是**逐行**跳过的（长度/编码/列数不对只丢那一行，计数见
    ``summary()["skipped_rows"]``），读失败只记 ``last_error``、不关掉缓存。
    关闭、目录不可用、文件不存在都返回 ``0``，**不抛异常**。
    """
    if not enabled():
        return 0
    with _LOCK:
        _ensure_locked()
        if not _dir_ok:
            return 0
        return _load_locked(force)


def best(key: Mapping[str, Any]) -> dict[str, str] | None:
    """按键查最优配置行（整行，值全是字符串）；未命中/关闭返回 ``None``。

    纯字典查找，不做 IO：索引由 :func:`load` 一次建好。选优口径见模块文档
    （``compute_ms`` 最小 → ``calls`` 大 → ``updated_at`` 新）。返回的是**副本**，
    调用方改了也不会污染索引。
    """
    if not enabled() or not isinstance(key, Mapping):
        return None
    with _LOCK:
        _ensure_locked()
        if _dir_ok:
            _load_locked(False)
        slot = _index.get(_key_of(key))
        if not slot:
            return None
        row = _pick(slot.values())
        return None if row is None else dict(row)


def record(row: Mapping[str, Any]) -> bool:
    """记一次观测；返回是否**真的排队写盘**（稀疏口径，见模块文档）。

    两种才写：该键第一次出现，或 ``compute_ms`` 比该键当前最优严格快过 5%。
    其余只更新内存索引（``best`` 立刻能看到），不占磁盘。缺省字段自动补齐：
    ``host`` / ``real_device`` / ``updated_at`` / ``pyqcd_version``；``calls``
    不给则该 (键, 配置) 已有值 +1、否则 ``1``。每个键最多保留 ``compute_ms``
    最快的 :data:`_MAX_ROWS_PER_KEY` 条配置行（内存与落盘同一口径，丢掉的条数
    见 ``summary()["dropped_rows"]``）。

    关闭、目录不可用、行不可解析一律返回 ``False``，**不抛异常**。
    """
    if not enabled() or not isinstance(row, Mapping):
        return False
    with _LOCK:
        _ensure_locked()
        if not _dir_ok:
            return False
        try:
            prepared = _profile_row(row)
        except Exception as exc:                 # noqa: BLE001 - 画像缓存绝不外抛
            _degrade_locked(exc)
            return False

        key = _key_of(prepared)
        config = (prepared["kernel"], prepared["optimize"])
        if str(prepared["kernel"]).strip().lower() in _NON_REPLAYABLE:
            # 只观测不落盘：它给不出下次可用的建议，留下来只会占名额、误导 best()。
            return False
        slot = _index.get(key)
        new_key = slot is None
        if slot is None:
            slot = _index[key] = {}
        previous = slot.get(config)
        if not prepared["calls"]:
            prepared["calls"] = (str(_int(previous.get("calls")) + 1)
                                 if previous else "1")

        current = _pick(slot.values())
        worth = new_key
        if not worth and current is not None:
            worth = (_ms_value(prepared.get("compute_ms"))
                     < _ms_value(current.get("compute_ms")) * _IMPROVE_RATIO)

        canonical = _merge(slot, config, prepared)
        if worth:
            _pending.append(dict(canonical))
        if _prune_slot(slot):                    # 每键只留最快的几条，内存/落盘同一口径
            _drop_orphans_locked()
        if worth and len(_pending) >= _FLUSH_THRESHOLD:
            _flush_locked()
        return worth


def record_compare(row: Mapping[str, Any]) -> bool:
    """追加一行 compare 旁证（**只写不读**）；返回是否真的写了。

    同一 (键, 配置, 参照, 参照优化器) 60 s 内只写一次，避免同一批量里刷屏。
    ``equivalence_*`` 可以直接给字段，也可以给一份 :func:`._equivalence.explain`
    的报告字典（读它的 ``kind`` / ``equation`` / 三个置换字段，只读不改）。

    关闭、目录不可用、写失败一律返回 ``False``，**不抛异常**。
    """
    if not enabled() or not isinstance(row, Mapping):
        return False
    with _LOCK:
        _ensure_locked()
        if not _dir_ok:
            return False
        try:
            prepared = _compare_row(row)
        except Exception as exc:                 # noqa: BLE001 - 画像缓存绝不外抛
            _degrade_locked(exc)
            return False

        token = _key_of(prepared) + (
            prepared["kernel"],
            prepared["optimize"],
            prepared["compare"],
            prepared["compare_optimize"],
        )
        now = time.monotonic()
        last = _compare_recent.get(token)
        if last is not None and now - last < _COMPARE_WINDOW_S:
            return False
        _compare_recent[token] = now
        if len(_compare_recent) > _COMPARE_MEMORY:
            _prune_compare_locked(now)

        try:
            _append(_compare_path(), COMPARE_COLUMNS, [prepared])
        except Exception as exc:                 # noqa: BLE001 - 写失败即降级
            _degrade_locked(exc)
            return False
        return True


def flush() -> int:
    """把攒下的主表行一次追加写盘，返回写出的行数；失败降级、不抛异常。

    进程退出时由 ``atexit`` 自动调一次；``compare`` 行是即时追加的，不在这里。
    """
    if not enabled():
        return 0
    with _LOCK:
        return _flush_locked()


def summary() -> dict[str, Any]:
    """当前状态快照：``{"dir","file","enabled","requires_configure","rows","keys",
    "loaded","skipped_rows","truncated_fields","dropped_rows","max_rows_per_key",
    "last_error"}``。

    ``dir`` 是生效目录的**绝对路径**，``enabled`` 是有效可用性（开关 AND 目录可用），
    ``requires_configure`` 恒为 ``False``（目录可显式配置，但**不需要**先配置）；
    ``rows`` 是索引里的 (键, 配置) 行数、``keys`` 是不同键数、``loaded`` 是是否已
    尝试加载；``skipped_rows`` 是最近一次读盘跳过的坏行数、``truncated_fields`` 是
    累计截断的超长字段数、``dropped_rows`` 是累计被"每键最快 K 条"裁剪掉的行数、
    ``max_rows_per_key`` 是那个 K；任何降级原因都在 ``last_error``（没有就是
    ``None``）。开关关着时不建目录、不读文件。
    """
    on = _switch_on()
    with _LOCK:
        if on:
            _ensure_locked()
            if _dir_ok:
                _load_locked(False)
        return {
            "dir": _dir or _dir_text(),
            "file": _profile_path(),
            "enabled": bool(on and _dir_ok),
            "requires_configure": False,
            "rows": sum(len(slot) for slot in _index.values()) if on else 0,
            "keys": len(_index) if on else 0,
            "loaded": _loaded if on else False,
            "skipped_rows": _skipped_rows if on else 0,
            "truncated_fields": _truncated_fields if on else 0,
            "dropped_rows": _dropped_rows if on else 0,
            "max_rows_per_key": _MAX_ROWS_PER_KEY,
            "last_error": _last_error,
        }


def notice() -> str | None:
    """启动时的一行提示：无缓存 / 已加载 N 行；关闭或已提示过不可用则 ``None``。

    只给一行文本，**不打印、不重复刷屏**：目录不可用的说明只在第一次调用时给出
    （原因与 :func:`summary` 的 ``last_error`` 同源）。加载沿用 :func:`load` 的
    "只读一次"口径，因此代价是一次顺序读。
    """
    global _notice_done
    if not _switch_on():
        return None
    with _LOCK:
        _ensure_locked()
        if not _dir_ok:
            if _notice_done:
                return None
            _notice_done = True
            reason = _last_error or "原因未知"
            return f"einsum 配置画像：无缓存（{_dir} 不可用：{reason}）"
        loaded = _load_locked(False)
        rows = max(loaded, sum(len(slot) for slot in _index.values()))
        keys = len(_index)
        if rows <= 0:
            return f"einsum 配置画像：无缓存记录（{_dir}）"
        return f"einsum 配置画像：已加载 {rows} 行 / {keys} 键（{_dir}）"


# -- 内部：行装配 -----------------------------------------------------------
def _profile_row(source: Mapping[str, Any]) -> dict[str, str]:
    """把调用方给的稀疏字典补齐成一整行（列序见 :data:`PROFILE_COLUMNS`）。"""
    row = {name: _text(source.get(name)) for name in PROFILE_COLUMNS}
    row["host"] = row["host"] or _host_name()
    row["real_device"] = row["real_device"] or _real_device_name()
    row["shape"] = format_shape(source.get("shape")) or row["shape"]
    row["init_ms"] = _number(source.get("init_ms"))
    row["compute_ms"] = _number(source.get("compute_ms"))
    row["calls"] = _count(source.get("calls"))
    row["updated_at"] = row["updated_at"] or _now()
    row["pyqcd_version"] = row["pyqcd_version"] or _version()
    return _clip_row(row)


def _compare_row(source: Mapping[str, Any]) -> dict[str, str]:
    """把调用方给的稀疏字典补齐成一行 compare 旁证。"""
    row = {name: _text(source.get(name)) for name in COMPARE_COLUMNS}
    row["host"] = row["host"] or _host_name()
    row["real_device"] = row["real_device"] or _real_device_name()
    row["shape"] = format_shape(source.get("shape")) or row["shape"]
    row["ref_ms"] = _number(source.get("ref_ms"))
    row["compute_ms"] = _number(source.get("compute_ms"))
    row["ratio"] = _ratio(source.get("ratio"))
    row["updated_at"] = row["updated_at"] or _now()
    row["pyqcd_version"] = row["pyqcd_version"] or _version()
    row["operand_order"] = _compact(source.get("operand_order"))
    row["operand_permutations"] = _compact(source.get("operand_permutations"))
    row["output_permutation"] = _compact(source.get("output_permutation"))

    info = source.get("equivalence")
    if isinstance(info, Mapping):                # explain() 的报告可直接整份塞进来
        row["equivalence_kind"] = (row["equivalence_kind"]
                                   or _text(info.get("kind")))
        row["equivalence_equation"] = (row["equivalence_equation"]
                                       or _text(info.get("equation")))
        row["operand_order"] = (row["operand_order"]
                                or _compact(info.get("operand_order")))
        row["operand_permutations"] = (row["operand_permutations"]
                                       or _compact(info.get("operand_permutations")))
        row["output_permutation"] = (row["output_permutation"]
                                     or _compact(info.get("output_permutation")))
    return _clip_row(row)


def _clip_row(row: dict[str, str]) -> dict[str, str]:
    """按 :data:`_CLIP_LIMITS` 截断超长文本字段（读写两侧共用同一口径）。"""
    for name in _CLIP_LIMITS:
        if name in row:
            row[name] = _clip(name, row[name])
    return row


def _key_of(key: Mapping[str, Any]) -> tuple[str, ...]:
    """按键字段取键元组；``shape`` 归一化、``host`` 缺省补本机名（与写侧一致）。"""
    values: list[str] = []
    for name in KEY_FIELDS:
        value = key.get(name)
        if name == "shape":
            values.append(format_shape(value))
        elif name == "host":
            values.append(_text(value) or _host_name())
        else:
            values.append(_text(value))
    return tuple(values)


def _merge(slot: dict[tuple[str, str], dict[str, str]],
           config: tuple[str, str], row: dict[str, str]) -> dict[str, str]:
    """把一行并进同配置槽位：耗时取小、``calls`` 取大、``updated_at`` 取新。"""
    current = slot.get(config)
    if current is None:
        slot[config] = row
        return row
    calls = max(_int(current.get("calls")), _int(row.get("calls")))
    keep = (row if _ms_value(row.get("compute_ms"))
            <= _ms_value(current.get("compute_ms")) else current)
    merged = dict(keep)
    merged["calls"] = str(calls)
    newest = (row if _stamp(row.get("updated_at"))
              >= _stamp(current.get("updated_at")) else current)
    merged["updated_at"] = newest.get("updated_at", "")
    slot[config] = merged
    return merged


def _rank(row: Mapping[str, Any]) -> tuple[float, int, float]:
    """选优排序键：``compute_ms`` 小 → ``calls`` 大 → ``updated_at`` 新（越小越好）。"""
    return (_ms_value(row.get("compute_ms")),
            -_int(row.get("calls")),
            -_stamp(row.get("updated_at")))


def _pick(rows: Iterable[dict[str, str]]) -> dict[str, str] | None:
    """选优：``compute_ms`` 最小 → ``calls`` 大 → ``updated_at`` 新（纯比较）。

    **不可重放的建议不参与选优**：``equivalence`` 只说明"这次是改写跑成的"
    （在原方程上强制具体内核必然"未尝试"），无法据此改道。``generic`` **是**可
    重放的（``kernel_force="generic"`` 合法且实测常最快，例如某 10 操作数方程
    generic 385 ms vs topology 2735 ms），必须参与选优——把它排除等于把 7 倍
    收益丢掉。
    """
    chosen: dict[str, str] | None = None
    chosen_rank: tuple[float, int, float] | None = None
    for row in rows:
        if str(row.get("kernel", "")).strip().lower() in _NON_REPLAYABLE:
            continue
        rank = _rank(row)
        if chosen_rank is None or rank < chosen_rank:
            chosen, chosen_rank = row, rank
    return chosen


def _prune_slot(slot: dict[tuple[str, str], dict[str, str]]) -> int:
    """每键只留 ``compute_ms`` 最快的 :data:`_MAX_ROWS_PER_KEY` 条，返回丢掉的条数。"""
    global _dropped_rows
    extra = len(slot) - _MAX_ROWS_PER_KEY
    if extra <= 0:
        return 0
    for config, _row in sorted(slot.items(), key=lambda item: _rank(item[1]))[
            _MAX_ROWS_PER_KEY:]:
        slot.pop(config, None)
    _dropped_rows += extra
    return extra


def _drop_orphans_locked() -> None:
    """把已被裁剪掉的配置行从待写队列剔除：内存与落盘必须是同一批行（持锁）。"""
    kept: list[dict[str, str]] = []
    for row in _pending:
        slot = _index.get(_key_of(row))
        if slot is not None and (row.get("kernel"), row.get("optimize")) in slot:
            kept.append(row)
    _pending[:] = kept


# -- 内部：加载与落盘 -------------------------------------------------------
def _load_locked(force: bool) -> int:
    """:func:`load` 的本体（持锁）。

    坏行是**逐行**处理的：某一行长度/编码/列数不对只跳过并计入
    ``summary()["skipped_rows"]``，其余行照常进索引；只有整份文件都打不开才算读失败，
    而读失败也只记 ``last_error``、**不**把目录判死（下一行代码不该因为一次坏读就
    失去缓存）。
    """
    global _loaded, _loaded_rows, _skipped_rows, _disk_rows
    if _loaded and not force:
        return _loaded_rows
    _index.clear()
    try:
        with open(_profile_path(), "rb") as handle:
            raw = handle.read(_MAX_LOAD_BYTES)
    except FileNotFoundError:
        raw = b""
    except Exception as exc:                     # noqa: BLE001 - 读失败只记原因
        _error_locked(exc)
        _merge_pending_locked()                  # 读失败也不能丢掉内存里的观测
        _loaded, _loaded_rows, _skipped_rows, _disk_rows = True, 0, 0, -1
        return 0
    text, skipped = _decode_table(raw)
    count, bad = _parse_table(text)
    skipped += bad
    for slot in _index.values():                 # 旧文件可能攒了重复行：读入即裁剪
        _prune_slot(slot)
    _merge_pending_locked()                      # 未落盘的更优观测不能被重读冲掉
    _loaded = True
    _loaded_rows = count
    _skipped_rows = skipped
    _disk_rows = count + skipped                 # 盘上实际有这么多数据行
    return count


def _merge_pending_locked() -> None:
    """把还没落盘的观测并回索引（持锁调用）。"""
    for row in _pending:
        _merge(_index.setdefault(_key_of(row), {}),
               (row["kernel"], row["optimize"]), dict(row))


def _decode_table(raw: bytes) -> tuple[str, int]:
    """整份字节 → 文本；整份解不开就逐行解码并丢掉坏行（返回跳过的行数）。"""
    try:
        return raw.decode("utf-8"), 0
    except UnicodeDecodeError:
        pass
    kept: list[str] = []
    skipped = 0
    for line in raw.split(b"\n"):
        try:
            kept.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            skipped += 1                         # 编码坏行：只丢这一行
    return "\n".join(kept), skipped


def _parse_table(text: str) -> tuple[int, int]:
    """逐行解析进索引（持锁调用）；返回 ``(读入行数, 跳过行数)``。"""
    expected = list(PROFILE_COLUMNS)
    count = skipped = streak = 0
    reader = csv.reader(io.StringIO(text))
    while True:
        try:
            fields = next(reader)
        except StopIteration:
            break
        except Exception as exc:                 # noqa: BLE001 - 坏行只跳过
            skipped += 1
            streak += 1
            if streak >= _MAX_CONSECUTIVE_ERRORS:
                _error_locked(exc)               # csv 流已不可用，停止读取
                break
            continue
        streak = 0
        if fields == expected:                   # 表头（重复出现的也只当表头）
            continue
        if not any(field.strip() for field in fields):
            continue                             # 空行
        if len(fields) != len(expected):
            skipped += 1                         # 列数不对：这一行坏了
            continue
        mapped = dict(zip(expected, fields))
        row = {name: _text(mapped.get(name)) for name in PROFILE_COLUMNS}
        row["shape"] = format_shape(row.get("shape"))
        _merge(_index.setdefault(_key_of(row), {}),
               (row["kernel"], row["optimize"]), _clip_row(row))
        count += 1
    return count, skipped


def _flush_locked() -> int:
    """把待写队列一次追加写盘（持锁）；失败即降级，队列丢弃。

    盘上数据行已经超过 :data:`_MAX_ROWS` 时先整表重写成"每键最快 K 条"——否则
    ``(键, 配置)`` 的乘法（``shape`` × ``Mom_curr`` × Nev 扫描）会把文件撑成
    无人能读的规模。重写后若仍超限（键太多，K 条压不下去），要等再涨
    :data:`_REWRITE_GROWTH` 行才重写：每次 flush 都整表重写等于写放大。
    """
    global _disk_rows
    if not _dir_ok or not _pending:
        return 0
    count = _disk_rows if _disk_rows >= 0 else _count_raw_rows()
    if (count + len(_pending) > _MAX_ROWS
            and count >= _rewrite_rows + _REWRITE_GROWTH):
        return _rewrite_locked()
    rows, _pending[:] = list(_pending), []
    try:
        _append(_profile_path(), PROFILE_COLUMNS, rows)
    except Exception as exc:                     # noqa: BLE001 - 写失败即降级
        _degrade_locked(exc)
        return 0
    _disk_rows = count + len(rows)
    return len(rows)


def _rewrite_locked() -> int:
    """整表重写：读回全表 → 每键只留最快 K 条 → 原子替换（持锁）。

    返回本次真正落盘的**待写行数**（文件里活下来的行数见 ``_loaded_rows`` /
    ``summary()["rows"]``——重写后两者相等）。
    """
    global _disk_rows, _loaded_rows, _rewrite_rows
    pending = len(_pending)
    _load_locked(True)                           # 索引 = 文件 ∪ 待写行
    for slot in _index.values():
        _prune_slot(slot)
    try:
        written = _write_table(_index_rows())
    except Exception as exc:                     # noqa: BLE001 - 写失败即降级
        _degrade_locked(exc)
        return 0
    _pending.clear()
    _disk_rows = _loaded_rows = written
    _rewrite_rows = written
    return pending


def _index_rows() -> list[dict[str, str]]:
    """索引里的全部行（每键已裁剪）：整表重写的写出内容。"""
    return [dict(row) for slot in _index.values() for row in slot.values()]


def _write_table(rows: Iterable[Mapping[str, Any]]) -> int:
    """整表重写：先写临时文件再原子替换（中途失败不会留下半张表）。"""
    path = _profile_path()
    temp = f"{path}.tmp{os.getpid()}"
    written = 0
    try:
        with open(temp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PROFILE_COLUMNS),
                                    extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "")
                                 for name in PROFILE_COLUMNS})
                written += 1
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return written


def _count_raw_rows() -> int:
    """盘上大约有多少数据行（按换行数近似、减掉表头）；读不到按 0。"""
    try:
        with open(_profile_path(), "rb") as handle:
            raw = handle.read(_MAX_LOAD_BYTES)
    except OSError:
        return 0
    lines = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
    return max(lines - 1, 0)


def _append(path: str, columns: tuple[str, ...],
            rows: Iterable[Mapping[str, Any]]) -> None:
    """追加写；文件不存在或是空文件时先写表头（表头只写一次）。"""
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns),
                                extrasaction="ignore", lineterminator="\n")
        if fresh:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def _flush_at_exit() -> None:
    """``atexit`` 兜底：**带超时**拿锁，拿不到就丢弃待写行、退出阶段不抛异常。

    退出瞬间还有线程持锁时，死等会把整个进程拖住（``flush()`` 本身无超时，那是
    调用方显式要求落盘的口径）；这里宁可少几行记录。
    """
    try:
        lock = _LOCK
        if lock is None or not _switch_on():
            return
        if not lock.acquire(timeout=_EXIT_LOCK_TIMEOUT_S):
            _pending.clear()                     # 拿不到锁：放弃待写行
            return
        try:
            _flush_locked()
        finally:
            lock.release()
    except Exception:                            # noqa: BLE001 - 退出阶段静默
        pass


def _reset_after_fork() -> None:
    """``fork`` 出的子进程重建锁与状态（父进程的锁可能正被某个线程持有）。

    ``threading.Lock`` 不随 ``fork`` 重置：父进程里某线程持锁时 fork，子进程第一次
    进 :func:`summary` / :func:`record` 就会**永久阻塞**。子进程也不该替父进程
    flush，所以待写队列直接丢弃；索引与目录探测标记一并复位（fork 可能发生在
    ``record`` 改到一半的时刻，继承来的内存视图不可信），首次使用时按文件重建。
    """
    global _LOCK, _index, _pending, _ready, _dir_ok, _disk_rows, _loaded, _loaded_rows
    _LOCK = threading.Lock()
    _index.clear()
    _pending.clear()
    _ready = False
    _dir_ok = False
    _disk_rows = -1
    _loaded, _loaded_rows = False, 0


atexit.register(_flush_at_exit)

if hasattr(os, "register_at_fork"):              # POSIX；Windows 没有这个钩子
    os.register_at_fork(after_in_child=_reset_after_fork)


# -- 内部：目录、设备与格式 -------------------------------------------------
def _switch_on() -> bool:
    """总开关本身（只看 ``PYQCD_EINSUM_PROFILE``，不含目录状态的运行时降级）。"""
    return os.environ.get(PROFILE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def _blocked() -> bool:
    """目录是否已在运行时被判不可用（判过才算，没判过不拦第一次 IO）。

    两个模块级布尔分开读：:func:`_probe_locked` 先落 ``_dir_ok`` 再由
    :func:`_ensure_locked` 落 ``_ready``，所以这里永远不会看到"判过但还没结论"。
    """
    return _ready and not _dir_ok


def _resolve_dir_text(directory: Any = None) -> str:
    """目录分辨率：显式参数 > :data:`DIR_ENV` > :data:`DEFAULT_DIR`，给绝对路径。

    空白与 ``None`` 都算"未给"——``""`` 不会挤掉环境变量、空环境变量不会挤掉默认值，
    这样 CLI 传空串与不传的行为一致。只算字符串，不建目录。
    """
    for candidate in (directory, os.environ.get(DIR_ENV)):
        text = _text(candidate)
        if text:
            return os.path.abspath(text)
    return os.path.abspath(DEFAULT_DIR)


def _dir_text() -> str:
    """当前生效目录的绝对路径（:func:`configure` 钉过就用它，否则现算；不建目录）。"""
    return _configured_dir or _resolve_dir_text(None)


def _profile_path() -> str:
    """``profile.csv`` 的绝对路径。"""
    return os.path.join(_dir or _dir_text(), PROFILE_NAME)


def _compare_path() -> str:
    """``profile_compare.csv`` 的绝对路径。"""
    return os.path.join(_dir or _dir_text(), COMPARE_NAME)


def _ensure_locked() -> None:
    """解析/创建缓存目录并**实写一次**探针文件确认可写（没 configure 时只做一次）。"""
    global _ready
    if _ready:
        return
    _probe_locked(_dir_text())
    _ready = True


def _probe_locked(path: str) -> None:
    """建目录 + 实写探针确认可写；不可写即降级、把原因留给 :func:`summary`（持锁）。"""
    global _dir, _dir_ok
    _dir = path
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f".pyqcd_profile_probe_{os.getpid()}")
        with open(probe, "a", encoding="utf-8"):
            pass
        try:
            os.unlink(probe)
        except OSError:
            pass
    except Exception as exc:                     # noqa: BLE001 - 不可写即降级
        _degrade_locked(exc)
        return
    _dir_ok = True


def _degrade_locked(exc: BaseException) -> None:
    """降级成"不缓存"，把原因留给 :func:`summary`（持锁调用）。"""
    global _dir_ok, _last_error
    _dir_ok = False
    _last_error = f"{type(exc).__name__}: {exc}"


def _error_locked(exc: BaseException) -> None:
    """只记原因、**不**改可用性：读侧的坏行/坏文件不该把整个缓存关掉（持锁）。"""
    global _last_error
    _last_error = f"{type(exc).__name__}: {exc}"


def _clip(name: str, value: str) -> str:
    """按 :data:`_CLIP_LIMITS` 截断超长字段，并累计一次截断。"""
    global _truncated_fields
    limit = _CLIP_LIMITS.get(name)
    if limit is None or len(value) <= limit:
        return value
    _truncated_fields += 1
    return value[:limit]


def _host_name() -> str:
    """本机名（只探测一次，失败给空串）。"""
    global _host
    if _host is None:
        try:
            _host = socket.gethostname()
        except Exception:                        # noqa: BLE001 - 拿不到就空
            _host = ""
    return _host


def _real_device_name() -> str:
    """真实计算设备型号：**廉价且失败安全**，取不到就空串。

    顺序：``CUDA_VISIBLE_DEVICES`` 显式隐藏（空串）时直接放弃；否则**只在 torch
    已经导入时**（``sys.modules``，绝不主动 import、不初始化 CUDA）问一次
    ``torch.cuda.get_device_name(0)``。任何异常都吞掉，结果进程内缓存一次。
    """
    global _real_device
    if _real_device is not None:
        return _real_device
    name = ""
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        hidden = visible is not None and not visible.strip()
        torch = sys.modules.get("torch")
        if torch is not None and not hidden:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None:
                try:
                    available = bool(cuda.is_available())
                except Exception:                # noqa: BLE001 - 探测失败按无设备
                    available = False
                if available:
                    name = str(cuda.get_device_name(0) or "").strip()
    except Exception:                            # noqa: BLE001 - 绝不因探测抛错
        name = ""
    _real_device = name
    return _real_device


def _version() -> str:
    """本包的 einsum 版本号（懒 import，取不到给空串）。"""
    try:
        from ._version import __version__

        return str(__version__)
    except Exception:                            # noqa: BLE001 - 版本只是元数据
        return ""


def _now() -> str:
    """带本地时区的 ISO 时间戳（可被 :func:`_stamp` 解析回来比较新旧）。"""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _stamp(text: Any) -> float:
    """``updated_at`` → 可比较的时间戳；解析失败按"最旧"处理。"""
    try:
        return datetime.datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError, OSError):
        return float("-inf")


def _ms_value(value: Any) -> float:
    """毫秒列的数值（缺失/坏值/NaN 按"无穷慢"处理，选优时永远排最后）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if number != number or number == float("inf"):   # NaN / inf 不参与比较
        return float("inf")
    return number


def _int(value: Any) -> int:
    """整型列的安全转换（坏值按 0）。"""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> str:
    """毫秒列写盘格式：三位小数（浮点）；非数值留空。"""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _ratio(value: Any) -> str:
    """比值列写盘格式：四位小数；非数值留空。"""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _count(value: Any) -> str:
    """计次列写盘格式：整数；缺失留空（由 :func:`record` 补缺省）。"""
    if value is None or value == "":
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def _compact(value: Any) -> str:
    """置换/次序的文本编码：操作数间 ``;``、轴间 ``:``、``None`` 记 ``-``。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if item is None:
                items.append("-")
            elif isinstance(item, (list, tuple)):
                items.append(":".join(str(axis) for axis in item))
            else:
                items.append(str(item))
        return ";".join(items)
    return str(value)


def _text(value: Any) -> str:
    """通用列写盘格式：字符串去空白、``None`` 空串、浮点三位小数。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _prune_compare_locked(now: float) -> None:
    """清掉 compare 去重表里过期的条目（持锁调用）。"""
    stale = [token for token, when in _compare_recent.items()
             if now - when >= _COMPARE_WINDOW_S]
    for token in stale:
        _compare_recent.pop(token, None)
