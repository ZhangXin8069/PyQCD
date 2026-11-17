"""qcd-einsum 的配置表（``pyqcd.einsum.profile.csv``）：**只有 tune 能写，普通调用只读**。

为什么单独一层
-------------
同一个 ``(shape, dtype, device, equation, backend)`` 在长时间批处理里会被反复问
"该用哪个 kernel、哪个收缩序"，而答案只随机器与配置变化。把 tune 的结论落成一张
CSV，普通 ``contract()`` 就用**一次顺序读**（进程内再加一层按 ``mtime``/``size``
失效的解析缓存）换掉重复的试跑与匹配时间。本模块不 import 执行层（``_api`` /
``_kernel_*`` / ``_shape_paths``）、不解析方程、不碰张量，也**不做计时**：``wall_ms``
由写者（tune）测好给进来。

唯一写者（冻结）
---------------
配置表**只有 tune 能写**（:func:`write_winner`）。普通 ``contract()`` 只走
:func:`read_winner`：不追加、不合并、不落盘——模块里也没有待写队列与 ``flush``。
总开关关闭时整层零 IO（不建目录、不读文件）。

``wall_ms`` 的口径（冻结）
------------------------
``wall_ms`` = **整个 ``pyqcd.einsum.contract(***)`` 的墙钟时间**：含等价转换（若有）、
缓存命中与读写（若有）、未命中时的规划-匹配。它**不是**"只包收缩执行"那一段，本模块
也不提供任何可以与它相加或混比的第二列计时。选优在 tune 侧只按它比较。

键与"同键一行"
-------------
键 = ``(shape, dtype, device, equation, backend)``（:data:`KEY_FIELDS`）；``host``
记在行里但**不参与匹配**——同一张表可以被复制到别的机器上继续用。``shape`` 统一
规范成 ``2x3|3x4``（``x`` 连轴、``|`` 连操作数，见 :func:`format_shape`），否则
``[2,3]`` / ``2x3`` / ``(2,3)`` 三种写法会把同一个形状拆成三个键；``equation`` 去掉
全部空白。``kernel`` / ``optimize`` / ``wall_ms`` / 三个置换字段都是这条记录的
**载荷**，不是键。

**同键只允许 1 行**，而且是硬约束：读侧遇到同键多行不猜、不选最优，直接抛
:class:`DuplicateProfileEntry` 并带上全部候选行（见 :func:`read_winner`）。写侧
（:func:`write_winner`）先删掉同键所有行再写 1 行，所以"同键多行"只可能来自手工
编辑或旧文件。

读侧（普通调用）
---------------
:func:`read_winner` 按键**精确匹配**：

* 0 行 → ``None``（本次无命中，按现有逻辑执行）；
* 1 行 → 该行的副本（另带 ``line``：它是配置表的第几条有效数据行，1-based）；
* 多行 → :class:`DuplicateProfileEntry`（消息与 ``rows`` 里是**全部**候选行）；
* 文件不存在 → :class:`ProfileMissing`（``cli_hint`` 是一条现成可跑的 tune 命令）。

它**绝不写**：不建目录、不写探针、没有 append、没有 merge、没有内存索引与待写队列。
读失败（权限、坏文件）只记 :func:`last_error` 并按"无命中"返回——一行坏文件不该让
调用方拿不到结果。

写侧（只有 tune）
----------------
:func:`write_winner` 是**唯一**写入口：读整表 → 删同键所有行 → 追加新行 → 临时文件
+ ``os.replace`` 原子替换。目录不存在时由**写侧**创建（读侧不建）。

* ``equivalence=None``（缺省）→ 写**性能配置**：``kernel`` / ``optimize`` /
  ``wall_ms``（以及 ``note``）。
* ``equivalence`` 非空 → 这一行**只含等价字段**（``target_equation``、
  ``operand_order`` / ``operand_permutations`` / ``output_permutation``）+ ``note``，
  性能字段留空。``target_equation`` 是重建改写必需的方程侧，另外三列是**置换**载荷
  （JSON 字符串，``null`` = 这一步不需要动）。``note`` 为空时默认写
  :data:`EQUIVALENCE_NOTE`——"走等价转换后的配置性能更佳"。给了 ``equivalence`` 却
  没有可执行的改写（缺 ``target_equation`` 或三个置换字段全空）会抛
  :class:`ProfileError`，不写一行既无配置又无载荷的垃圾行。

:func:`tune_session` 是遍历期间的"暂时关闭"：进入后 :func:`read_winner` 一律当无命中
（**不读文件**）、:func:`write_winner` **不写**（也不排队推迟）。所以写胜出配置必须
发生在 ``with`` 块**之外**——那只该是遍历结束之后的一次调用。

降级与线程安全
-------------
写盘失败**如实外抛**（唯一调用方是 tune，它负责打印并继续）；读失败只记
:func:`last_error` 并按无命中返回；坏行（列数不对、编码坏）逐行跳过。解析缓存共用
**一把** :class:`threading.Lock`；``fork`` 出的子进程用 ``os.register_at_fork`` 重置它
（``threading.Lock`` 不随 ``fork`` 重置，父进程某线程持锁时 fork 会让子进程第一次
读表就永久阻塞）。

配置目录
--------
:func:`configure` 是唯一入口，分辨率优先级**依次**为：显式参数 >
``PYQCD_EINSUM_PROFILE_DIR`` > :data:`DEFAULT_DIR`（``./.cache``，按当前工作目录取
绝对路径）；空白与 ``None`` 都算"未给"，逐级下退。没调用过 :func:`configure` 时，
第一次 IO 按同一口径自动解析一次。:func:`configure` 只记路径，**不碰磁盘**。

环境变量
--------
``PYQCD_EINSUM_PROFILE_DIR``
    配置表目录；未设时用 ``./.cache``（相对当前工作目录）。
``PYQCD_EINSUM_PROFILE``
    总开关，缺省 ``"1"``（开启）；``0/false/off/no`` 关闭，关闭时**不读也不写、
    连目录都不建**（零 IO）。
"""

from __future__ import annotations

import contextlib
import csv
import datetime
import io
import json
import os
import socket
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

__all__ = [
    "DEFAULT_DIR",
    "DIR_ENV",
    "DuplicateProfileEntry",
    "EQUIVALENCE_NOTE",
    "KEY_FIELDS",
    "PROFILE_COLUMNS",
    "PROFILE_ENV",
    "PROFILE_FILENAME",
    "ProfileConflict",
    "ProfileError",
    "ProfileMissing",
    "best",
    "configure",
    "enabled",
    "ensure_configured",
    "equivalence_payload",
    "format_shape",
    "last_error",
    "profile_path",
    "read_winner",
    "summary",
    "tune_session",
    "write_winner",
]

#: 配置表总开关：``0/false/off/no`` 关闭；缺省 ``"1"``（开启）。
PROFILE_ENV = "PYQCD_EINSUM_PROFILE"

#: 配置表目录覆盖；未设时用 :data:`DEFAULT_DIR`（相对当前工作目录）。
DIR_ENV = "PYQCD_EINSUM_PROFILE_DIR"

#: 缺省目录（相对 CWD）。
DEFAULT_DIR = ".cache"

#: 配置表文件名（**冻结**；旧名 ``profile.csv`` 已废除）。
PROFILE_FILENAME = "pyqcd.einsum.profile.csv"

#: 键字段（顺序即键序）；``shape`` 必须先过 :func:`format_shape` 归一化。
KEY_FIELDS = ("shape", "dtype", "device", "equation", "backend")

#: 性能载荷列：只有"性能行"带它们的值，等价行一律留空。
PERFORMANCE_FIELDS = ("kernel", "optimize", "wall_ms")

#: 置换载荷列（JSON 字符串；``null`` = 这一步不需要动）。
PERMUTATION_FIELDS = ("operand_order", "operand_permutations", "output_permutation")

#: 等价行需要的全部列：``target_equation``（方程侧）+ 三个置换字段。
EQUIVALENCE_FIELDS = ("target_equation",) + PERMUTATION_FIELDS

#: 等价行 ``note`` 列的缺省文案（tune 判出"走等价转换后性能更佳"时写这一句）。
EQUIVALENCE_NOTE = "走等价转换后的配置性能更佳"

#: 表头（顺序即写序，读侧按**名字**取值，所以容忍列序不同与多余列）。
PROFILE_COLUMNS = (KEY_FIELDS + PERFORMANCE_FIELDS + EQUIVALENCE_FIELDS
                   + ("note", "host", "updated_at", "pyqcd_version"))

#: 超长文本字段的写盘上限（撑坏整张表的是这类字段，不是方程本身）。
_CLIP_LIMITS = {"note": 200}

#: 单次读盘上限（病态大文件不该把内存吃光）。
_MAX_LOAD_BYTES = 64 << 20

#: "这个单元格解不出可用值"的哨兵——与"解出来正好是 ``None``"（不需要动）区分开。
_BAD_CELL = object()

#: 解析缓存与 tune 计数共用的一把锁（不要为单行加锁；``fork`` 后在子进程重建）。
_LOCK = threading.Lock()

#: 最近一次读盘解析结果的缓存：``(路径, mtime_ns, size, 行列表)``。
_TABLE: tuple[str, int, int, list[dict[str, str]]] | None = None

#: :func:`tune_session` 的嵌套计数（``> 0`` = 遍历进行中：读当无命中、写不落盘）。
_TUNING = 0

#: :func:`configure` 显式钉住的目录（绝对路径）；空串表示"没钉过，按 env/默认现算"。
_CONFIGURED_DIR = ""

#: 最近一次读失败的原因（写失败如实外抛，不记在这里）。
_LAST_ERROR: str | None = None

#: 主机名只探测一次。
_HOST: str | None = None


# -- 异常 -------------------------------------------------------------------
class ProfileError(RuntimeError):
    """配置表读写的可报告错误基类（调用方据此区分"表有问题"与"执行有问题"）。"""


class ProfileMissing(ProfileError):
    """配置表文件不存在（此前从没 tune 过）。

    ``cli_hint`` 是一条**已按该键填好**的 tune 命令，调用方可以直接打给用户；
    ``path`` 是期望的文件路径。要不要把它当错误由调用方决定——普通调用只该警告
    一句然后继续执行。
    """

    def __init__(self, cli_hint: str = "", path: str = "") -> None:
        self.cli_hint = str(cli_hint)
        self.path = str(path)
        detail = f"{path} 不存在" if path else "配置表不存在"
        hint = f"；需要先 tune：\n  {self.cli_hint}" if self.cli_hint else ""
        super().__init__(f"einsum 配置表缺失：{detail}{hint}")


class DuplicateProfileEntry(ProfileError):
    """同一个键在配置表里出现了多行（同键只允许 1 行）。

    ``rows`` 是**全部**候选行（每行带 ``line``：它在文件里的数据行号），调用方据此
    原样打印给用户；不替用户挑一行。
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]] = (),
                 key: Mapping[str, Any] | None = None, path: str = "") -> None:
        self.rows = [dict(row) for row in rows]
        self.key = dict(key) if isinstance(key, Mapping) else {}
        self.path = str(path)
        super().__init__(_duplicate_message(self.key, self.rows, self.path))


class ProfileConflict(ProfileError):
    """用户**显式**配置与命中行冲突（由 ``_api`` 抛出，向上冒泡到 CLI）。

    ``field`` 是冲突的维度（``kernel_force`` / ``optimize`` / ``as_equivalence``），
    ``recorded`` / ``requested`` 分别是表里的值与用户给的值。
    """

    def __init__(self, message: str = "", *, field: str = "", recorded: str = "",
                 requested: str = "", key: Mapping[str, Any] | None = None) -> None:
        self.field = str(field)
        self.recorded = str(recorded)
        self.requested = str(requested)
        self.key = dict(key) if isinstance(key, Mapping) else {}
        super().__init__(message)


# -- 对外接口 ---------------------------------------------------------------
def enabled(value: bool | None = None) -> bool:
    """配置表是否**开启**（只看 ``PYQCD_EINSUM_PROFILE``，调用时求值）。

    缺省 ``"1"``（开启）；``0/false/off/no``（大小写不敏感、去首尾空白）关闭。
    显式 ``value`` 只回答**开关本身**（调用方覆盖开关用）。

    目录状态**不再**参与：读路径不建目录、不写探针，所以"目录不可用"不是一种降级，
    而是"表不存在"（:class:`ProfileMissing`）或读失败（:func:`last_error`）——前者该
    提示先 tune，后者只该慢一点。
    """
    if value is not None:
        return bool(value)
    return _switch_on()


def configure(directory=None) -> str:
    """钉住配置表目录，返回绝对路径；**不抛异常、不碰磁盘**。

    分辨率优先级与 CLI/API 同口径：显式参数 > ``PYQCD_EINSUM_PROFILE_DIR`` >
    :data:`DEFAULT_DIR`（``./.cache``）；``None`` 与空白都算"未给"，逐级下退。

    换目录等于换一张表：解析缓存一并清空（否则会把旧目录的行当成新目录的）。目录
    不存在也不建——读路径不写；真正建目录的是 :func:`write_winner`。
    """
    global _CONFIGURED_DIR, _LAST_ERROR, _TABLE
    path = _resolve_dir_text(directory)
    with _LOCK:
        if _CONFIGURED_DIR != path:
            _TABLE = None
            _LAST_ERROR = None
        _CONFIGURED_DIR = path
    return path


def ensure_configured() -> str:
    """确保目录已解析并返回绝对路径；**已钉住的目录保持不变**。

    与 :func:`configure` 的分工：后者总按"参数 > ``PYQCD_EINSUM_PROFILE_DIR`` >
    默认 ``./.cache``"重算，所以 ``configure(None)`` 会把调用方先前钉住的目录改回
    默认；本函数只"补齐"，不"改写"——已有目录就沿用它，没有才按同一优先级解析。
    """
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


def profile_path() -> str:
    """配置表的绝对路径（**不建目录、不读文件**）。"""
    return os.path.join(_dir_text(), PROFILE_FILENAME)


def last_error() -> str | None:
    """最近一次读失败的原因（写失败如实外抛，不记在这里）；没有就是 ``None``。"""
    return _LAST_ERROR


def read_winner(key: Mapping[str, Any]) -> dict[str, str] | None:
    """按键**精确匹配**取那唯一一行；未命中返回 ``None``。**绝不写**。

    同键多行 → 抛 :class:`DuplicateProfileEntry`（``rows`` 是全部候选行）；文件不存在
    → 抛 :class:`ProfileMissing`（带一条现成的 tune 命令）。其他读失败（权限、坏文件）
    只记 :func:`last_error` 并按"无命中"返回——配置表读不出来只该慢一点，不该让调用
    方拿不到结果。

    总开关关闭、:func:`tune_session` 进行中一律返回 ``None``（不读文件：tune 量到的
    必须是被测配置本身的耗时）。返回的是**副本**，额外带 ``line``（该行在配置表里的
    数据行号，1-based），只用于把命中提示钉到具体记录。
    """
    if not enabled():
        return None
    token = _key_of(key)
    with _LOCK:
        if _TUNING > 0:
            return None                              # 遍历期间：读当无命中
        path = profile_path()
        try:
            rows = _table_locked(path)
        except FileNotFoundError:
            raise ProfileMissing(_cli_hint(token), path) from None
        except (OSError, csv.Error) as exc:
            _error_locked(exc)
            return None
        matches = [row for row in rows if _row_key(row) == token]
    if not matches:
        return None
    if len(matches) > 1:
        raise DuplicateProfileEntry(matches, key=key, path=path)
    return dict(matches[0])


def write_winner(key: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                 equivalence: Any = None, note: str = "") -> None:
    """写**唯一**获胜配置：先删掉同键所有行，再只写 1 行。**只有 tune 调用它**。

    ``equivalence`` 为空 → 写性能配置（``kernel`` / ``optimize`` / ``wall_ms``）；
    非空 → 该行只含等价字段（``target_equation`` + 三个置换字段）+ ``note``，性能
    字段留空（``note`` 为空时写 :data:`EQUIVALENCE_NOTE`）。``equivalence`` 可以是
    映射，也可以是带 ``equation``/``operand_order``/``operand_permutations``/
    ``output_permutation`` 属性的对象（例如 :class:`._equivalence.Equivalence`）。

    写盘是"临时文件 + ``os.replace``"的原子替换，中途失败不会留下半张表。目录不存在
    时在这里创建。写失败（含无权限、磁盘满）**如实外抛**——唯一调用方是 tune，它负责
    打印"写配置表失败"并继续。

    总开关关闭、:func:`tune_session` 进行中：**静默不写**（遍历期间表一个字都不变）。
    因此胜出配置必须在 ``with tune_session()`` **之外**写。
    """
    token = _key_of(key)
    with _LOCK:
        if not enabled() or _TUNING > 0:
            return None
        row = _compose_row(token, config, equivalence, note)
        _store_locked(token, row)
    return None


@contextlib.contextmanager
def tune_session() -> Iterator[None]:
    """遍历期间的"暂时关闭"：进入后不读表、不写表，退出即恢复。

    进入 → :func:`read_winner` 一律当无命中、:func:`write_winner` 静默不写；退出（含
    异常退出）→ 计数器归零，读写回到正常语义。嵌套安全（按计数）。

    为什么连写也关掉：遍历要量的是"被测配置本身"的耗时，照表施加或落盘都会污染它，
    而且没被选中的 29 组只会留下垃圾行。胜出配置由调用方在窗口外写一次。
    """
    global _TUNING
    with _LOCK:
        _TUNING += 1
    try:
        yield
    finally:
        with _LOCK:
            _TUNING = max(_TUNING - 1, 0)


def best(key: Mapping[str, Any]) -> dict[str, str] | None:
    """按键取那一行（报告层的便利读法）：缺表当未命中，同键多行照旧抛错。

    "每键一行"让"最优"这个词没有意义——留下的就是唯一答案。报告层（``--why`` /
    ``cache line``）只想区分"命中/未命中"，不想把"还没 tune 过"当成错误，所以这里
    把 :class:`ProfileMissing` 收敛成 ``None``；数据错误（多行）仍然抛出，不猜。
    """
    try:
        return read_winner(key)
    except ProfileMissing:
        return None


def summary() -> dict[str, Any]:
    """当前状态快照：``{"dir","file","enabled","rows","keys","last_error"}``。

    ``rows`` 是配置表里的有效数据行数、``keys`` 是不同键数（两者在"每键一行"下相等，
    分开报只是为了同键多行时能一眼看出异常）。读失败按 0 行报，原因进 ``last_error``。
    **只读**：不建目录、不写文件。
    """
    path = profile_path()
    if not _switch_on():
        return {"dir": _dir_text(), "file": path, "enabled": False,
                "rows": 0, "keys": 0, "last_error": _LAST_ERROR}
    rows: list[dict[str, str]] = []
    try:
        with _LOCK:
            rows = _table_locked(path)
    except FileNotFoundError:
        pass
    except (OSError, csv.Error) as exc:
        _error_locked(exc)
    return {
        "dir": _dir_text(),
        "file": path,
        "enabled": True,
        "rows": len(rows),
        "keys": len({_row_key(row) for row in rows}) if rows else 0,
        "last_error": _LAST_ERROR,
    }


def equivalence_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """等价行 → **可直接执行**的改写载荷；不是等价行/坏了返回 ``None``。

    返回 ``{"target_equation", "operand_order", "operand_permutations",
    "output_permutation", "note"}``，三个置换字段是元组（``None`` / 空元组 =
    "这一步不需要动"）。只做**结构**校验：``target_equation`` 非空且带 ``->``，
    三个置换字段必须是 JSON 数组或 ``null``，且至少有一个字段真的要求动张量
    （否则没有可执行的改写）。"数组长度对不对得上张量秩"需要张量本身，是调用方的
    责任（``_api`` 施加时还会再查一遍）。

    坏 JSON、缺列一律返回 ``None``——调用方据此退回重新匹配，**绝不抛**。
    """
    if not isinstance(row, Mapping):
        return None
    equation = _text(row.get("target_equation"))
    if not equation or "->" not in equation:
        return None
    order = _int_sequence(_cell_value(row.get("operand_order")))
    permutations = _int_matrix(_cell_value(row.get("operand_permutations")))
    output = _int_sequence(_cell_value(row.get("output_permutation")))
    for value in (order, permutations, output):
        if value is _BAD_CELL:
            return None
    if order is None and not permutations and output is None:
        return None                      # 没有可执行的改写（identity / 只有 note）
    return {
        "target_equation": equation,
        "operand_order": order,
        "operand_permutations": permutations or (),
        "output_permutation": output,
        "note": _text(row.get("note")),
    }


def _duplicate_message(key: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                       path: str) -> str:
    """同键多行的报错正文：键、文件、**全部**候选行（不挑、不猜）。"""
    where = path or "配置表"
    fields = " ".join(f"{name}={key.get(name, '')!r}" for name in KEY_FIELDS)
    lines = [f"einsum 配置表同键多行（{where}）：同一键只允许 1 行，无法判断该信哪一条。",
             f"  键: {fields}"]
    for row in rows:
        line = str(row.get("line") or "?").strip()
        lines.append(
            f"  候选 {where}:{line}"
            f"  kernel={row.get('kernel', '')!r}"
            f"  optimize={row.get('optimize', '')!r}"
            f"  wall_ms={row.get('wall_ms', '')!r}"
            f"  target_equation={row.get('target_equation', '')!r}"
            f"  note={row.get('note', '')!r}")
    lines.append("  处理：删到只剩 1 行，或重新 tune 那个键（tune 会先删同键所有行）。")
    return "\n".join(lines)


# -- 内部：行装配 -----------------------------------------------------------
def _compose_row(token: tuple[str, ...], config: Mapping[str, Any] | None,
                 equivalence: Any, note: str) -> dict[str, str]:
    """把键 + 载荷装配成一整行（列序见 :data:`PROFILE_COLUMNS`）。

    ``equivalence`` 非空就是**等价行**：只写等价字段 + ``note``，性能字段留空。给了
    等价载荷却解不出可执行的改写（没有目标方程，或三个置换字段全空）时抛
    :class:`ProfileError`——写一行两不靠的垃圾行比不写更坏。
    """
    row = {name: "" for name in PROFILE_COLUMNS}
    for name, value in zip(KEY_FIELDS, token):
        row[name] = value
    row["host"] = _host_name()
    row["updated_at"] = _now()
    row["pyqcd_version"] = _version()

    if _has_equivalence(equivalence):
        fields = _equivalence_fields(equivalence)
        if not fields:
            raise ProfileError(
                "equivalence 载荷缺少可执行的改写：需要 target_equation，且 "
                "operand_order / operand_permutations / output_permutation 至少一个非空")
        for name in EQUIVALENCE_FIELDS:
            value = fields.get(name)
            row[name] = (_payload_cell(value)
                         if name in PERMUTATION_FIELDS else _text(value))
        row["note"] = _clip("note", _text(note) or EQUIVALENCE_NOTE)
        return row

    source = config if isinstance(config, Mapping) else {}
    row["kernel"] = _text(source.get("kernel"))
    row["optimize"] = _text(source.get("optimize"))
    row["wall_ms"] = _number(source.get("wall_ms"))
    row["note"] = _clip("note", _text(note))
    return row


def _has_equivalence(equivalence: Any) -> bool:
    """``equivalence`` 是否要求写一行等价记录（``None`` / 空映射 = 不要求）。"""
    if equivalence is None:
        return False
    if isinstance(equivalence, Mapping):
        return bool(equivalence)
    return True


def _equivalence_fields(equivalence: Any) -> dict[str, Any]:
    """``equivalence`` → 四个载荷字段；空字典 = "解不出可执行的改写"。"""
    def look(name: str) -> Any:
        if isinstance(equivalence, Mapping):
            return equivalence.get(name)
        return getattr(equivalence, name, None)

    target = _text(look("target_equation") or look("equation"))
    fields = {"target_equation": target,
              "operand_order": look("operand_order"),
              "operand_permutations": look("operand_permutations"),
              "output_permutation": look("output_permutation")}
    executable = (fields["operand_order"] is not None
                  or fields["operand_permutations"]
                  or fields["output_permutation"] is not None)
    if not target or not executable:
        return {}
    return fields


def _key_of(key: Mapping[str, Any]) -> tuple[str, ...]:
    """按键字段取键元组（``shape`` 归一化、``equation`` 去空白）。"""
    if not isinstance(key, Mapping):
        raise TypeError(f"配置表键必须是映射，得到 {type(key).__name__}")
    return tuple(_canonical(name, key.get(name)) for name in KEY_FIELDS)


def _row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """已解析行的键元组（与 :func:`_key_of` 同一口径）。"""
    return tuple(_canonical(name, row.get(name)) for name in KEY_FIELDS)


def _canonical(name: str, value: Any) -> str:
    """单个键字段的规范写法（读写两侧共用，少一处不一致就少一种"幽灵未命中"）。

    ``shape`` 直接交给 :func:`format_shape`：调用方既可以给已规范化的字符串，也可以
    给 ``[(2, 3), (3, 4)]`` 这种操作数形状列表，两种写法必须落到同一把键上。
    """
    if name == "shape":
        return format_shape(value)
    text = _text(value)
    if name == "equation":
        return "".join(text.split())
    return text


# -- 内部：读 ---------------------------------------------------------------
def _table_locked(path: str) -> list[dict[str, str]]:
    """读整张表并缓存解析结果（持锁）；文件不存在抛 ``FileNotFoundError``。

    缓存按 ``(路径, mtime_ns, size)`` 失效：一次 ``stat`` 就能发现 tune 写了新表，
    不必每次调用都把整份 CSV 解析一遍。**只读**。
    """
    global _TABLE
    info = os.stat(path)
    stamp = (path, int(info.st_mtime_ns), int(info.st_size))
    if _TABLE is not None and _TABLE[:3] == stamp:
        return _TABLE[3]
    with open(path, "rb") as handle:
        raw = handle.read(_MAX_LOAD_BYTES)
    rows = _parse_rows(raw)
    _TABLE = (stamp[0], stamp[1], stamp[2], rows)
    return rows


def _parse_rows(raw: bytes) -> list[dict[str, str]]:
    """整份字节 → 行列表（表头按**名字**取值；列数不对的行跳过）。"""
    if not raw.strip():
        return []
    text = raw.decode("utf-8", "replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:                            # pragma: no cover - 空文件已提前返回
        return []
    columns = [_text(name) for name in header]
    rows: list[dict[str, str]] = []
    for fields in reader:
        if not any(field.strip() for field in fields):
            continue                                 # 空行
        if len(fields) != len(columns):
            continue                                 # 列数不对：这一行坏了，只丢它
        mapped = dict(zip(columns, fields))
        row = {name: _text(mapped.get(name)) for name in PROFILE_COLUMNS}
        row["shape"] = format_shape(row["shape"])
        row["equation"] = "".join(row["equation"].split())
        row["line"] = str(len(rows) + 1)             # 该表第几条**有效**数据行
        rows.append(row)
    return rows


# -- 内部：写 ---------------------------------------------------------------
def _store_locked(token: tuple[str, ...], row: dict[str, str]) -> None:
    """删掉同键所有行 + 追加这一行，整表原子重写（持锁调用）。

    只有 tune 能走到这里：先读回整表（别把别的键的行冲掉），把同键行全部剔除，再追加
    新行。读不出旧表就**不写**（宁可这次 tune 的结论丢掉，也不能把别人的记录清空）。
    """
    global _TABLE
    path = profile_path()
    try:
        rows = _table_locked(path)
    except FileNotFoundError:
        rows = []
    except (OSError, csv.Error) as exc:
        raise ProfileError(
            f"读不出配置表 {path}，为免清空其它键的记录，本次不写："
            f"{type(exc).__name__}: {exc}") from exc
    kept = [dict(item) for item in rows if _row_key(item) != token]
    kept.append(dict(row))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _write_rows(path, kept)
    _TABLE = None                                    # 下一次读重新解析


def _write_rows(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """整表重写：先写临时文件再原子替换（中途失败不会留下半张表）。"""
    temp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    try:
        with open(temp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PROFILE_COLUMNS),
                                    extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in PROFILE_COLUMNS})
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


# -- 内部：目录与格式 -------------------------------------------------------
def _switch_on() -> bool:
    """总开关本身（只看 ``PYQCD_EINSUM_PROFILE``）。"""
    return os.environ.get(PROFILE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


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
    return _CONFIGURED_DIR or _resolve_dir_text(None)


def _error_locked(exc: BaseException) -> None:
    """只记原因、**不**改可用性：读侧的坏行/坏文件不该把整张表关掉（持锁）。"""
    global _LAST_ERROR
    _LAST_ERROR = f"{type(exc).__name__}: {exc}"


# -- 内部：单元格编解码与设备探测 -------------------------------------------
def _payload_cell(value: Any) -> str:
    """置换/次序载荷的单元格编码：``None`` → ``"null"``，其余一律 JSON 文本。"""
    if value is None:
        return "null"
    try:
        return json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError):
        return ""


def _cell_value(cell: Any) -> Any:
    """单元格 → Python 值（``None`` = 不需要动）；坏 JSON 返回 :data:`_BAD_CELL`。"""
    text = _text(cell)
    if not text or text == "null":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return _BAD_CELL


def _int_sequence(value: Any) -> tuple[int, ...] | None:
    """JSON 值 → 整型元组；``None`` 原样返回，类型不符返回 :data:`_BAD_CELL`。"""
    if value is None or value is _BAD_CELL:
        return value
    if not isinstance(value, (list, tuple)):
        return _BAD_CELL
    items: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return _BAD_CELL
        items.append(int(item))
    return tuple(items)


def _int_matrix(value: Any) -> tuple[tuple[int, ...] | None, ...] | None:
    """``operand_permutations`` 的 JSON 值 → 元组的元组（元素可以是 ``None``）。"""
    if value is None or value is _BAD_CELL:
        return value
    if not isinstance(value, (list, tuple)):
        return _BAD_CELL
    rows: list[tuple[int, ...] | None] = []
    for item in value:
        row = _int_sequence(item)
        if row is _BAD_CELL:
            return _BAD_CELL
        rows.append(row)
    return tuple(rows)


def _cli_hint(token: tuple[str, ...]) -> str:
    """该键的 tune CLI 命令（可直接复制执行；缺文件时打给用户）。"""
    shape, dtype, device, equation, _backend = token
    return " ".join((
        "python -m pyqcd.einsum.test",
        f"--device {_cli_device(device)}",
        f"--dtype {_cli_dtype(dtype)}",
        f'--equation "{equation}"',
        f'--shape "{_cli_shapes(shape)}"',
        "--tune",
        "--compare opt_einsum",
    ))


def _cli_device(device: str) -> str:
    """键里的 ``device`` → ``--device`` 的合法取值（``cpu`` / ``gpu`` / ``cuda``）。"""
    text = _text(device).lower()
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("gpu"):
        return "gpu"
    return "cpu"


def _cli_dtype(dtype: str) -> str:
    """键里的 ``dtype`` → ``--dtype`` 写法（``torch.complex128`` → ``complex128``）。"""
    text = _text(dtype)
    return text.rsplit(".", 1)[-1] if text else "float64"


def _cli_shapes(shape: str) -> str:
    """规范形状 ``2x3|3x4`` → ``--shape`` 的 ``[2,3],[3,4]`` 写法（输出段可省略）。"""
    parts: list[str] = []
    for operand in _text(shape).split("|"):
        axes = [axis for axis in operand.split("x") if axis.strip()]
        parts.append("[" + ",".join(axes) + "]")
    return ",".join(parts)


def _host_name() -> str:
    """本机名（只探测一次，失败给空串）。"""
    global _HOST
    if _HOST is None:
        try:
            _HOST = socket.gethostname()
        except OSError:
            _HOST = ""
    return _HOST


def _version() -> str:
    """本包的 einsum 版本号（懒 import，取不到给空串）。"""
    try:
        from ._version import __version__

        return str(__version__)
    except Exception:                                # noqa: BLE001 - 版本只是元数据
        return ""


def _now() -> str:
    """带本地时区的 ISO 时间戳。"""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _number(value: Any) -> str:
    """毫秒列写盘格式：三位小数（浮点）；非数值留空。"""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _clip(name: str, value: str) -> str:
    """按 :data:`_CLIP_LIMITS` 截断超长文本字段。"""
    limit = _CLIP_LIMITS.get(name)
    if limit is None or len(value) <= limit:
        return value
    return value[:limit]


def _text(value: Any) -> str:
    """通用列写盘格式：字符串去空白、``None`` 空串、浮点三位小数。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _reset_after_fork() -> None:
    """``fork`` 出的子进程重建锁与状态（父进程的锁可能正被某个线程持有）。

    ``threading.Lock`` 不随 ``fork`` 重置：父进程里某线程持锁时 fork，子进程第一次
    读表就会**永久阻塞**。子进程也不该继承父进程的"遍历中"标记（fork 可能发生在
    tune 窗口里），解析缓存同理——首次使用时按文件重建。
    """
    global _LOCK, _TABLE, _TUNING
    _LOCK = threading.Lock()
    _TABLE = None
    _TUNING = 0


if hasattr(os, "register_at_fork"):                  # POSIX；Windows 没有这个钩子
    os.register_at_fork(after_in_child=_reset_after_fork)
