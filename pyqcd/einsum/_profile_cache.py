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
全部空白。``kernel`` / ``optimize`` / ``wall_ms`` / 三个置换字段都是这条记录的**载荷**，不是键。

``optimize`` 列（7 叶子，冻结）
------------------------------
只认这 **7 个叶子**（:data:`OPTIMIZE_LEAVES`）：

* ``greedy`` / ``optimal`` / ``dp`` / ``branch_1`` / ``branch_2`` / ``branch_all`` /
  ``none``。

留空 = 没有单独结论，按 :data:`DEFAULT_OPTIMIZE`（``greedy``）处理。

历史上还有第二列 ``optimize_topology_v2``（"topology 回落到 v2 通用执行器时采用的
收缩序"）。v2 执行器在 0.5.4 删除后那一列就没有任何写入者与读取者，0.5.12 把它从
表里一并删掉（``schema_version`` 3 → 4）；老表照旧能读——缺列按空处理。

读写口径：

* 大小写不敏感（:func:`read_winner` 读回时已归一成小写）；
* **连字符旧写法**（``branch-1`` / ``branch-all`` / ``branch-2``）读入时归一成下划线，
  **不视为非法**——老表照旧能用；
* ``auto``（这个词已从规划器删除）与其它任何拼写都**非法**：读侧**警告一次**（同一
  路径 + 列 + 值只报一遍，:data:`_WARNED_CELLS`），并且**该行不参与匹配**——绝不静默
  把它当成 ``greedy``。写侧遇到非法值不写那一格（留空 = 缺省），同样只警告一次；
  写进去的永远是 7 个叶子之一或空。

``schema_version`` 列
----------------------
:data:`PROFILE_SCHEMA_VERSION` 记**写这一行时的列集合版本**（本版 ``4``：``optimize``
起 **删除**了 ``optimize_topology_v2`` 那一列；``3`` 及以前有它）。老文件没有这一列，读回按
:data:`LEGACY_SCHEMA_VERSION`（``0``）处理；老行**既不降级也不删除**，缺的列按空读，
于是"老表 + 新列"能一起用。

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

配置目录（默认在**用户家目录**，不再相对 CWD）
---------------------------------------------
:func:`configure` 是唯一入口，分辨率优先级**依次**为：显式参数 >
``PYQCD_EINSUM_PROFILE_DIR`` > :data:`DEFAULT_DIR`；空白与 ``None`` 都算"未给"，
逐级下退。没调用过 :func:`configure` 时，第一次 IO 按同一口径自动解析一次。
:func:`configure` 只记路径，**不碰磁盘**。

:data:`DEFAULT_DIR` 是**导入时解析好的绝对路径**（不再是"相对当前工作目录的目录名"）：
``$XDG_CACHE_HOME/pyqcd.einsum``，没设 ``XDG_CACHE_HOME`` 时是
``~/.cache/pyqcd.einsum``（:data:`DEFAULT_HOME_SUBDIR` 是最后那一段）。为什么不再相对
CWD：相对 CWD 的默认值会让"在 A 目录 tune 出来的表，在 B 目录读不到"，``pip install``
的用户每换一个工作目录就得重新 tune 一次，从仓库根跑测试还会往工作树里写 ``.cache``。
显式目录（``--profile-dir`` / ``PYQCD_EINSUM_PROFILE_DIR``）不受影响，仍按
"``~`` 先展开、再取绝对路径"解析。

旧位置（``<CWD>/.cache/pyqcd.einsum.profile.csv``，本版之前的缺省）
-----------------------------------------------------------------
向后兼容是**一次性、只读**的：生效目录就是 :data:`DEFAULT_DIR`、家目录里还没有表、
而旧位置**有**表时，读路径把旧表原样当来源（不复制、不移动、不写），并把这件事如实
记进 :func:`source_note`（读侧同时警告一次）。写路径（:func:`write_winner`）永远落
**家目录**：它先按同一来源读旧表做底稿（否则别的键会被冲掉），再把整表写到新位置；
旧文件原样留在那儿等用户自己处理——改用户家目录/删用户文件都得先问过用户。

记录表（``pyqcd.einsum.record.csv``，**只作参考、绝不参与命中**）
--------------------------------------------------------------
tune 的**全部测试组合**（不只是胜出的那一组）另存一张记录表：列集合是
:data:`RECORD_COLUMNS` = :data:`PROFILE_COLUMNS` 的全部列 + 性能段末尾的
``peak_memory_mb``，与配置表同目录（:func:`record_path`）。写口 :func:`write_records`
**整表重写**（每次 tune 开始时清空、结束时写全量快照——追加会让同一把键留下几十行
历史噪音）；读口 :func:`read_records` **只给报告/工具用**，``_api`` / 选路 / 命中提示
一律不碰它。命中只有 :func:`read_winner` 一条路。

环境变量
--------
``PYQCD_EINSUM_PROFILE_DIR``
    配置表/记录表目录；未设时用 :data:`DEFAULT_DIR`（家目录缓存，见上）。
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
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

__all__ = [
    "DEFAULT_DIR",
    "DEFAULT_HOME_SUBDIR",
    "DEFAULT_OPTIMIZE",
    "DIR_ENV",
    "DuplicateProfileEntry",
    "EQUIVALENCE_NOTE",
    "KEY_FIELDS",
    "LEGACY_SCHEMA_VERSION",
    "OPTIMIZE_FIELDS",
    "OPTIMIZE_LEAVES",
    "PROFILE_COLUMNS",
    "PROFILE_ENV",
    "PROFILE_FILENAME",
    "PROFILE_SCHEMA_VERSION",
    "ProfileConflict",
    "ProfileError",
    "ProfileMissing",
    "RECORD_COLUMNS",
    "RECORD_FILENAME",
    "RECORD_SCHEMA_VERSION",
    "best",
    "clear_cli_hint_context",
    "configure",
    "enabled",
    "ensure_configured",
    "equivalence_payload",
    "format_shape",
    "last_error",
    "profile_path",
    "read_records",
    "read_winner",
    "record_path",
    "set_cli_hint_context",
    "source_note",
    "summary",
    "tune_session",
    "write_records",
    "write_winner",
]

#: 配置表总开关：``0/false/off/no`` 关闭；缺省 ``"1"``（开启）。
PROFILE_ENV = "PYQCD_EINSUM_PROFILE"

#: 配置表目录覆盖；未设时用 :data:`DEFAULT_DIR`（家目录缓存，见下）。
DIR_ENV = "PYQCD_EINSUM_PROFILE_DIR"

#: 缺省目录在缓存根下的那一段名字（``$XDG_CACHE_HOME/pyqcd.einsum`` 或
#: ``~/.cache/pyqcd.einsum``）。
DEFAULT_HOME_SUBDIR = "pyqcd.einsum"

#: 旧缺省目录的目录名（本版之前是**相对当前工作目录**的 ``./.cache``）。只用于只读
#: 回退（见 :func:`_legacy_path`），绝不作为写入目标。
LEGACY_DIR_NAME = ".cache"


def _default_dir_text() -> str:
    """缺省配置目录的绝对路径：``$XDG_CACHE_HOME/pyqcd.einsum`` 或 ``~/.cache/...``。

    遵循 XDG 口径（``XDG_CACHE_HOME`` 设了就用它，否则 ``~/.cache``）；两者都先
    ``expanduser`` 再 ``abspath``，解析结果是**绝对路径**，与当前工作目录无关——这正是
    本版要修的东西：相对 CWD 的缺省目录会让"在一个目录 tune 出来的表，换个目录读不到"。
    """
    root = os.environ.get("XDG_CACHE_HOME")
    root = root.strip() if isinstance(root, str) else ""
    if root:
        root = os.path.abspath(os.path.expanduser(root))
    else:
        root = os.path.join(os.path.expanduser("~"), LEGACY_DIR_NAME)
    return os.path.abspath(os.path.join(root, DEFAULT_HOME_SUBDIR))


#: 缺省配置目录：**导入时解析好的绝对路径**（:data:`DEFAULT_HOME_SUBDIR` 是最后一段）。
#: 语义已从"相对 CWD 的目录名"变成"绝对路径"：``pip install`` 的用户不会每换一个工作
#: 目录就重新 tune 一次，从仓库根跑测试也不会往工作树里写 ``.cache``。
#: 仍然公开、仍在 ``__all__`` 里——外部代码可以拿它做"表在哪"的展示与断言。
DEFAULT_DIR = _default_dir_text()

#: 配置表文件名（**冻结**；旧名 ``profile.csv`` 已废除）。
PROFILE_FILENAME = "pyqcd.einsum.profile.csv"

#: 记录表文件名：tune 的**全量测试组合**快照，只作参考，**绝不参与命中**。
RECORD_FILENAME = "pyqcd.einsum.record.csv"

#: 键字段（顺序即键序）；``shape`` 必须先过 :func:`format_shape` 归一化。
KEY_FIELDS = ("shape", "dtype", "device", "equation", "backend")

#: 配置表 schema 版本：**列集合**每变一次 +1。本版 ``4`` = **删除**了
#: ``optimize_topology_v2`` 那一列（``3`` 及以前有它）。写侧把它记进 ``schema_version``；
#: 老行（``<=3``）按"缺列"读，不降级也不删除。
PROFILE_SCHEMA_VERSION = 4

#: 老行的 ``schema_version``：本版之前写的行没有那一列（或那一格为空）。
LEGACY_SCHEMA_VERSION = 0

#: 两列 ``optimize`` 的 **7 个合法叶子**（字符串存储；大小写与连字符写法读入时归一）。
#: ``auto`` 不在这里——它已从规划器删除，出现在表里属于非法值（读侧警告 + 该行不匹配）。
OPTIMIZE_LEAVES = ("greedy", "optimal", "dp", "branch_1", "branch_2", "branch_all", "none")

#: ``optimize`` 留空（没有单独结论）时的缺省值。
DEFAULT_OPTIMIZE = "greedy"

#: 必须落在 :data:`OPTIMIZE_LEAVES` 里的列（其余拼写一律非法）。
#:
#: 历史注记：这里曾经是两列（``optimize`` 与 ``optimize_topology_v2``）。后者记的是
#: "topology 回落到 v2 通用执行器时采用的收缩序"，而 v2 执行器在 0.5.4 就删除了；
#: 0.5.12 把这一列**从表里一并删除**（`schema_version` 因此 +1，老表按"缺列"读）。
OPTIMIZE_FIELDS = ("optimize",)

#: 性能载荷列：只有"性能行"带它们的值，等价行一律留空。
PERFORMANCE_FIELDS = ("kernel", "optimize", "wall_ms")

#: 置换载荷列（JSON 字符串；``null`` = 这一步不需要动）。
PERMUTATION_FIELDS = ("operand_order", "operand_permutations", "output_permutation")

#: 等价行需要的全部列：``target_equation``（方程侧）+ 三个置换字段。
EQUIVALENCE_FIELDS = ("target_equation",) + PERMUTATION_FIELDS

#: 等价行 ``note`` 列的缺省文案（tune 判出"走等价转换后性能更佳"时写这一句）。
EQUIVALENCE_NOTE = "走等价转换后的配置性能更佳"

#: 表头（顺序即写序，读侧按**名字**取值，所以容忍列序不同与多余列）。
#: ``schema_version`` 在最后一列。
PROFILE_COLUMNS = (KEY_FIELDS + PERFORMANCE_FIELDS + EQUIVALENCE_FIELDS
                   + ("note", "host", "updated_at", "pyqcd_version", "schema_version"))

#: 记录表独有的那一列（峰值内存 MB），插在性能段末尾（``wall_ms`` 之后）。
RECORD_MEMORY_FIELD = "peak_memory_mb"

#: 记录表表头 = :data:`PROFILE_COLUMNS` 的全部列 + :data:`RECORD_MEMORY_FIELD`
#: （插在 ``wall_ms`` 之后、等价载荷之前，也就是"性能段末尾"）。
#:
#: 为什么不直接把 ``peak_memory_mb`` 塞进 :data:`PERFORMANCE_FIELDS`：那会改掉
#: :data:`PROFILE_COLUMNS`，而 ``profile.csv`` 的读侧、``write_winner`` 与多条冻结契约
#: （表头逐字、``PERFORMANCE_FIELDS`` 只含 wall_ms 等）都依赖它；记录表**不参与命中**，
#: 让它自带一份"profile 列 + 一列"的列元组是最小风险的选择。公共列逐字相同，所以两张表
#: 的格式仍然一致（记录表是配置表的超集）。
RECORD_COLUMNS = (KEY_FIELDS + PERFORMANCE_FIELDS + (RECORD_MEMORY_FIELD,)
                  + EQUIVALENCE_FIELDS
                  + ("note", "host", "updated_at", "pyqcd_version", "schema_version"))

#: 记录表的 ``schema_version``：列集合每变一次 +1。``5`` = 在配置表 ``4`` 的列集合上
#: 多一列 :data:`RECORD_MEMORY_FIELD`；更早的记录行缺这一格，按空读。
RECORD_SCHEMA_VERSION = 5

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

#: "本次读的表其实在**旧位置**"的说明；不是这种情况就是 ``None``（见 :func:`source_note`）。
_SOURCE_NOTE: str | None = None

#: 本次读到的表的真实路径（旧位置回退时是旧位置；否则 ``None`` = 生效目录）。
_SOURCE_PATH: str | None = None

#: 主机名只探测一次。
_HOST: str | None = None

#: 待打印的非法单元格：``(去重签名, 文案)``。解析/装配在锁里跑，警告只在**出锁后**发。
_PENDING_WARNINGS: list[tuple[tuple[str, str, str], str]] = []

#: 已经警告过的 ``(路径, 列, 值)``：同一件事只该说一遍（读盘解析会被重复触发）。
_WARNED_CELLS: set[tuple[str, str, str]] = set()

#: 行上的私有标记（不是列）：这一行的 ``optimize`` 两列有非法值，**不参与匹配**。
_SKIP_FIELD = "_skip"

# ``csv.reader`` rejects NUL at the stream level. Replace it with a private
# in-band marker so CSV can still determine the logical record boundary; any
# record containing this marker is discarded after parsing.
_NUL_MARKER = "\x01PYQCD_NUL\x01"


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
    :data:`DEFAULT_DIR`（**家目录缓存**的绝对路径，``$XDG_CACHE_HOME/pyqcd.einsum`` 或
    ``~/.cache/pyqcd.einsum``，见模块 docstring）；``None`` 与空白都算"未给"，逐级下退。

    换目录等于换一张表：解析缓存与"来源说明"一并清空（否则会把旧目录的行当成新目录的）。
    目录不存在也不建——读路径不写；真正建目录的是 :func:`write_winner`。

    显式给了目录（参数或环境变量）就**完全不碰家目录**：只有落到缺省那一档时才会走
    旧位置（``<CWD>/.cache``）的只读回退。
    """
    global _CONFIGURED_DIR, _LAST_ERROR, _SOURCE_NOTE, _TABLE
    path = _resolve_dir_text(directory)
    with _LOCK:
        if _CONFIGURED_DIR != path:
            _TABLE = None
            _LAST_ERROR = None
            _SOURCE_NOTE = None
        _CONFIGURED_DIR = path
    return path


def ensure_configured() -> str:
    """确保目录已解析并返回绝对路径；**已钉住的目录保持不变**。

    与 :func:`configure` 的分工：后者总按"参数 > ``PYQCD_EINSUM_PROFILE_DIR`` >
    默认 :data:`DEFAULT_DIR`（家目录缓存，不再是 ``./.cache``）"重算，所以
    ``configure(None)`` 会把调用方先前钉住的目录改回默认；本函数只"补齐"，不"改写"
    ——已有目录就沿用它，没有才按同一优先级解析。
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
    """配置表的绝对路径（**不建目录、不读文件**）。

    这是**生效目录**下的路径，与"这一次读到的到底是哪张表"是两件事：生效目录是缺省
    （家目录）而家目录里还没有表、旧位置（``<CWD>/.cache``）却有表时，读路径会回退到
    旧表（见 :func:`source_note`），但本函数仍报家目录的路径——``write_winner`` 也永远
    写这里。
    """
    return os.path.join(_dir_text(), PROFILE_FILENAME)


def record_path() -> str:
    """记录表（``pyqcd.einsum.record.csv``）的绝对路径：与 :func:`profile_path` **同目录**。

    **不建目录、不读文件**。记录表只作 tune 的全量参考，绝不参与命中。
    """
    return os.path.join(_dir_text(), RECORD_FILENAME)


def last_error() -> str | None:
    """最近一次读失败的原因（写失败如实外抛，不记在这里）；没有就是 ``None``。"""
    return _LAST_ERROR


def source_note() -> str | None:
    """"本次读的表其实在**旧位置**"的说明；不是这种情况（或还没读过）就是 ``None``。

    向后兼容只做**一次性、只读**回退：生效目录是缺省（家目录）、那里还没有表、而旧位置
    （``<CWD>/.cache/pyqcd.einsum.profile.csv``，本版之前的缺省目录）有表时，读路径把旧表
    原样当来源。函数名如实说明这件事——**不搬、不复制、不写**：改用户家目录或删用户文件
    都得先问过用户，所以旧文件会一直留在那儿，直到用户自己处理。

    读侧命中旧表时同时发一条 :class:`RuntimeWarning`（同一路径只报一次）。
    """
    return _SOURCE_NOTE


def read_winner(key: Mapping[str, Any]) -> dict[str, str] | None:
    """按键**精确匹配**取那唯一一行；未命中返回 ``None``。**绝不写**。

    同键多行 → 抛 :class:`DuplicateProfileEntry`（``rows`` 是全部候选行）；文件不存在
    → 抛 :class:`ProfileMissing`（带一条现成的 tune 命令）。其他读失败（权限、坏文件）
    只记 :func:`last_error` 并按"无命中"返回——配置表读不出来只该慢一点，不该让调用
    方拿不到结果。

    总开关关闭、:func:`tune_session` 进行中一律返回 ``None``（不读文件：tune 量到的
    必须是被测配置本身的耗时）。返回的是**副本**，额外带 ``line``（该行在配置表里的
    数据行号，1-based），只用于把命中提示钉到具体记录。

    文件不存在时**只读**回退到旧位置（``<CWD>/.cache``，见 :func:`source_note`）：
    生效目录不是缺省时不回退，所以显式 ``--profile-dir`` 的调用完全碰不到别的目录。

    ``optimize`` 已归一成 7 叶子之一或空（空 = 缺省
    :data:`DEFAULT_OPTIMIZE`）；**有非法值的那一行不参与匹配**（``auto`` 这类旧值会触发
    一次警告，绝不静默当成 ``greedy``）。
    """
    if not enabled():
        return None
    token = _key_of(key)
    with _LOCK:
        if _TUNING > 0:
            return None                              # 遍历期间：读当无命中
        path = profile_path()
        try:
            rows = _source_rows_locked(path)
        except FileNotFoundError:
            raise ProfileMissing(_cli_hint(token), path) from None
        except (OSError, csv.Error) as exc:
            _error_locked(exc)
            return None
        matches = [row for row in rows
                   if not _skipped(row) and _row_key(row) == token]
        source = _source_path_locked()
    _flush_warnings()
    if not matches:
        return None
    if len(matches) > 1:
        raise DuplicateProfileEntry(matches, key=key, path=source)
    row = dict(matches[0])
    row.pop(_SKIP_FIELD, None)
    return row


def write_winner(key: Mapping[str, Any], config: Mapping[str, Any] | None = None,
                 equivalence: Any = None, note: str = "") -> None:
    """写**唯一**获胜配置：先删掉同键所有行，再只写 1 行。**只有 tune 调用它**。

    ``equivalence`` 为空 → 写性能配置（``kernel`` / ``optimize`` / ``wall_ms``）；
    非空 → 该行只含等价字段（``target_equation`` + 三个置换字段）+
    ``note``，性能字段留空（``note`` 为空时写 :data:`EQUIVALENCE_NOTE`）。
    ``equivalence`` 可以是映射，也可以是带 ``equation``/``operand_order``/
    ``operand_permutations``/``output_permutation`` 属性的对象（例如
    :class:`._equivalence.Equivalence`）。

    两列 ``optimize`` 只写 7 个叶子之一（:data:`OPTIMIZE_LEAVES`）：连字符旧写法归一成
    下划线，其余拼写（含 ``auto``）**不写那一格**（留空 = 缺省，警告一次）——写一行
    永远匹配不上的记录比空着更坏。``schema_version`` 一律写 :data:`PROFILE_SCHEMA_VERSION`。

    写盘是"临时文件 + ``os.replace``"的原子替换，中途失败不会留下半张表。目录不存在
    时在这里创建。写失败（含无权限、磁盘满）**如实外抛**——唯一调用方是 tune，它负责
    打印"写配置表失败"并继续。

    写入位置永远是 :func:`profile_path`（缺省就是**家目录**）：旧位置（``<CWD>/.cache``）
    的表只当**只读底稿**（见 :func:`_store_locked`），既不搬也不删。

    总开关关闭、:func:`tune_session` 进行中：**静默不写**（遍历期间表一个字都不变）。
    因此胜出配置必须在 ``with tune_session()`` **之外**写。
    """
    token = _key_of(key)
    with _LOCK:
        if not enabled() or _TUNING > 0:
            return None
        row = _compose_row(token, config, equivalence, note)
        _store_locked(token, row)
    _flush_warnings()
    return None


def write_records(rows: Sequence[Mapping[str, Any]], note: str = "") -> int:
    """把本次 tune 的**全量测试组合**整表写进记录表（:func:`record_path`），返回行数。

    记录表**只作参考**：``_api`` / 选路 / 命中提示一律不读它，命中只有
    :func:`read_winner` 一条路。它的列集合是 :data:`RECORD_COLUMNS`
    （:data:`PROFILE_COLUMNS` + 性能段末尾的 ``peak_memory_mb``），公共列逐字相同，所以
    "格式与 ``pyqcd.einsum.profile.csv`` 相同"。

    **整表重写**，不是追加：``rows`` 就是本次 tune 的全量快照（每次 tune 开始时清空、
    结束时写全部行）。用户要的是"这一次跑出来的所有组"，追加只会让同一把键留下几十行
    历史噪音。``rows`` 里的行按列名取值（缺列写空），``schema_version`` 一律写
    :data:`RECORD_SCHEMA_VERSION`（``4``）。

    ``note`` 不是一列：它是给调用方打日志用的说明（本函数不落盘它）；``note`` 列的取值
    来自每一行自己的 ``note`` 字段。

    写盘与 :func:`write_winner` 同口径：目录不存在就建、临时文件 + ``os.replace``
    原子替换、**写失败如实外抛**（不静默）。

    只在总开关关闭时静默不写（返回 ``0``）。:func:`tune_session` **不**拦它：遍历窗口
    要保护的是"被测配置本身的耗时"，而记录表从不会被任何人读回（命中只走
    :func:`read_winner`）——拦下来只会让"tune 结束时写全量快照"静默丢掉整张表。
    """
    del note                                          # 只用于调用方日志，不进表
    row_list = [_record_row(row) for row in rows]
    with _LOCK:
        if not enabled():
            return 0
        path = record_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _write_rows(path, row_list, columns=RECORD_COLUMNS)
    return len(row_list)


def read_records() -> list[dict[str, str]]:
    """读记录表（:func:`record_path`）的全部行；**只给报告/工具用**。

    **本函数不得被 ``_api`` / 选路 / 命中提示调用**：记录表是"这一次 tune 的全量参考"，
    不是缓存。命中判定只走 :func:`read_winner`——读记录表去匹配会让"没被选中的组"也
    变成可选答案，那正好是本表存在的反面。

    缺文件/读失败/坏行一律返回空表或跳过该行（**不抛**）：它只是给报告添一列证据，
    坏了不该让调用方拿不到结果。返回的每一行都补齐 :data:`RECORD_COLUMNS` 全部列
    （缺列按空；``4`` 之前的记录行没有 ``peak_memory_mb``，读作空串）。
    """
    path = record_path()
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_MAX_LOAD_BYTES)
    except OSError:
        return []
    return _parse_rows(raw, path, columns=RECORD_COLUMNS)


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
    """当前状态快照：``{"dir","file","enabled","rows","keys","invalid","last_error"}``。

    ``rows`` 是配置表里的数据行数、``keys`` 是不同键数（两者在"每键一行"下相等，
    分开报只是为了同键多行时能一眼看出异常）；``invalid`` 是 ``optimize`` 两列含非法值、
    因此**不参与匹配**的行数（``read_winner`` 只认 ``rows - invalid`` 那些行）。读失败按
    0 行报，原因进 ``last_error``。**只读**：不建目录、不写文件。

    计数口径与 :func:`read_winner` **同源**：家目录里没有表而旧位置（``<CWD>/.cache``）
    有表时，这里数的也是那张旧表的行（:func:`source_note` 说明来源）；``file`` 仍是
    生效目录下的路径（写入位置）。
    """
    path = profile_path()
    if not _switch_on():
        return {"dir": _dir_text(), "file": path, "enabled": False,
                "rows": 0, "keys": 0, "invalid": 0, "last_error": _LAST_ERROR}
    rows: list[dict[str, str]] = []
    try:
        with _LOCK:
            rows = _source_rows_locked(path)
    except FileNotFoundError:
        pass
    except (OSError, csv.Error) as exc:
        _error_locked(exc)
    _flush_warnings()
    usable = [row for row in rows if not _skipped(row)]
    return {
        "dir": _dir_text(),
        "file": path,
        "enabled": True,
        "rows": len(rows),
        "keys": len({_row_key(row) for row in usable}) if usable else 0,
        "invalid": len(rows) - len(usable),
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

    ``equivalence`` 非空就是**等价行**：只写等价字段 + ``note``，性能字段留空（含
    结论——等价行不写任何性能列）。给了等价载荷却
    解不出可执行的改写（没有目标方程，或三个置换字段全空）时抛 :class:`ProfileError`
    ——写一行两不靠的垃圾行比不写更坏。

    两列 ``optimize`` 走 :func:`_optimize_cell`：只落 7 叶子之一，非法拼写留空并记一条
    待发警告。``schema_version`` 一律写 :data:`PROFILE_SCHEMA_VERSION`。
    """
    row = {name: "" for name in PROFILE_COLUMNS}
    for name, value in zip(KEY_FIELDS, token):
        row[name] = value
    row["host"] = _host_name()
    row["updated_at"] = _now()
    row["pyqcd_version"] = _version()
    row["schema_version"] = str(PROFILE_SCHEMA_VERSION)

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
    for name in OPTIMIZE_FIELDS:
        row[name] = _optimize_cell(name, source.get(name))
    row["wall_ms"] = _number(source.get("wall_ms"))
    row["note"] = _clip("note", _text(note))
    return row


def _record_row(source: Mapping[str, Any]) -> dict[str, str]:
    """记录表的一行：按 :data:`RECORD_COLUMNS` 取值、归一，并写记录表自己的 schema 版本。

    与配置表共用同一套单元格口径：``shape`` / ``equation`` 走 :func:`_canonical`
    （两张表的键列逐字相同才能直接对照），两列 ``optimize`` 走
    :func:`_normalize_optimize`（只落 7 叶子之一，非法拼写留空——记录表**不参与命中**，
    所以不为它发"配置表写盘"那类警告），``note`` 走 :func:`_clip`。
    ``schema_version`` 一律写 :data:`RECORD_SCHEMA_VERSION`——记录表的列集合是
    "配置表的列 + ``peak_memory_mb``"，所以它的版本号**永远**比配置表大 1。

    ``peak_memory_mb`` 不是数值列：调用方给什么就写什么（``_text``），解不出来的留空；
    记录表不参与命中，多一列只是为了让人看懂"这一组吃了多少内存"。
    """
    row = {name: "" for name in RECORD_COLUMNS}
    if not isinstance(source, Mapping):
        return row
    for name in KEY_FIELDS:
        row[name] = _canonical(name, source.get(name))
    row["kernel"] = _text(source.get("kernel"))
    for name in OPTIMIZE_FIELDS:
        row[name] = _normalize_optimize(source.get(name))
    row["wall_ms"] = _number(source.get("wall_ms"))
    row[RECORD_MEMORY_FIELD] = _text(source.get(RECORD_MEMORY_FIELD))
    for name in EQUIVALENCE_FIELDS:
        value = source.get(name)
        if name in PERMUTATION_FIELDS:
            row[name] = _payload_cell(value) if value is not None else ""
        else:
            row[name] = _text(value)
    row["note"] = _clip("note", _text(source.get("note")))
    row["host"] = _text(source.get("host")) or _host_name()
    row["updated_at"] = _text(source.get("updated_at")) or _now()
    row["pyqcd_version"] = _text(source.get("pyqcd_version")) or _version()
    row["schema_version"] = str(RECORD_SCHEMA_VERSION)
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


def _skipped(row: Mapping[str, Any]) -> bool:
    """该行是否**不参与匹配**（``optimize`` 两列含非法值，见 :func:`_parse_rows`）。"""
    return bool(row.get(_SKIP_FIELD))


def _normalize_optimize(value: Any) -> str:
    """``optimize`` 单元格 → 7 叶子之一；非法拼写（含 ``auto``）返回空串。

    大小写与首尾空白不敏感；**连字符旧写法归一成下划线**（``branch-1`` → ``branch_1``），
    所以老表读上来仍然可用。空单元格（"这一列没有结论"）也返回空串。
    """
    text = _text(value).lower().replace("-", "_")
    return text if text in OPTIMIZE_LEAVES else ""


def _normalize_optimize_row(row: dict[str, str], path: str) -> None:
    """把读出来的一行的两列 ``optimize`` 归一；有非法值就标记该行不参与匹配。

    为什么整行作废而不是只丢那一格：读侧无法判断"``optimize=auto`` + 某个内核"这条记录
    原本是被谁写的、按什么语义写的，静默改成 ``greedy`` 会把一个**不是** tune 结论的值
    当成结论施加。警告说清原因，行留着（不删老行），只是不参与匹配。

    非法单元格**保持原样**写回（不做归一）：tune 改写别的键时整表会重写，把 ``auto``
    悄悄洗成空串就等于替用户改了记录——它该一直坏着，直到有人真的重新 tune 这个键。
    """
    for name in OPTIMIZE_FIELDS:
        raw = row.get(name, "")
        normalized = _normalize_optimize(raw)
        if raw and not normalized:
            row[_SKIP_FIELD] = "1"
            _queue_warning(
                (path, name, raw),
                f"einsum 配置表 {path or '（内存）'} 的数据行 {row.get('line', '?')} 里 "
                f"{name}={raw!r} 不是合法取值：这一列只认 "
                f"{'/'.join(OPTIMIZE_LEAVES)}。该行**不参与匹配**（也不会被静默当成 "
                f"{DEFAULT_OPTIMIZE}）；改成合法叶子或重新 tune 即可。")
            continue
        row[name] = normalized


def _optimize_cell(name: str, value: Any) -> str:
    """写盘用的叶子单元格：连字符归一成下划线，非法值留空并记一条待发警告。

    空单元格 = "这一列没有结论"（读侧按 :data:`DEFAULT_OPTIMIZE` 处理）。把非法拼写原样
    写进去只会造出一行**永远不参与匹配**的记录，所以宁可不写。
    """
    text = _normalize_optimize(value)
    if text:
        return text
    raw = _text(value)
    if raw:
        _queue_warning(
            (f"{profile_path()}|write", name, raw),
            f"einsum 配置表的 {name}={raw!r} 不是合法取值：这一列只认 "
            f"{'/'.join(OPTIMIZE_LEAVES)}，本次**没有写进**那一格（留空 = 缺省 "
            f"{DEFAULT_OPTIMIZE}）。")
    return ""


def _queue_warning(signature: tuple[str, str, str], message: str) -> None:
    """登记一条待发警告（持锁调用）：警告本身留到**出锁后**再打。"""
    _PENDING_WARNINGS.append((signature, message))


def _flush_warnings() -> None:
    """把登记的警告逐条发出，**同一 (路径, 列, 值) 只发一次**。

    为什么不在解析里直接 ``warnings.warn``：解析跑在 :data:`_LOCK` 里，而调用方可能给
    ``warnings`` 装了会重入本模块的钩子——那把锁不可重入，重入就是死锁。
    """
    with _LOCK:
        pending, _PENDING_WARNINGS[:] = list(_PENDING_WARNINGS), []
        fresh = [(signature, message) for signature, message in pending
                 if signature not in _WARNED_CELLS]
        _WARNED_CELLS.update(signature for signature, _ in fresh)
    for _signature, message in fresh:
        warnings.warn(message, RuntimeWarning, stacklevel=3)


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
    rows = _parse_rows(raw, path)
    _TABLE = (stamp[0], stamp[1], stamp[2], rows)
    return rows


def _parse_rows(raw: bytes, path: str = "",
                columns: Sequence[str] = PROFILE_COLUMNS) -> list[dict[str, str]]:
    """整份字节 → 行列表（表头按**名字**取值；列数不对的行跳过）。

    ``columns`` 是要补齐的列集合（配置表用 :data:`PROFILE_COLUMNS`，记录表用
    :data:`RECORD_COLUMNS`；两者共用这一份解析口径）。每一行都补齐全部列（**缺列老文件
    按空读**，``shape`` / ``equation`` 按同一口径归一），并做两件与 schema 有关的事：

    * ``schema_version`` 为空 → :data:`LEGACY_SCHEMA_VERSION`（老行照样可用）；
    * ``optimize`` 两列归一成 7 叶子或空（:func:`_normalize_optimize`）。出现非法值
      （``auto`` 等）的行打上 :data:`_SKIP_FIELD`——``read_winner`` 不认它——并把一条
      待发警告登记到 :data:`_PENDING_WARNINGS`（同一路径 + 列 + 值只报一次）。
    """
    if not raw.strip():
        return []
    names = [str(name) for name in columns]
    text = raw.decode("utf-8", "replace").replace("\x00", _NUL_MARKER)
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:                            # pragma: no cover - 空文件已提前返回
        return []
    header_names = [_text(name) for name in header]
    rows: list[dict[str, str]] = []
    for fields in reader:
        if any(_NUL_MARKER in field for field in fields):
            continue                             # 包含 NUL：丢掉整条逻辑记录
        if not any(field.strip() for field in fields):
            continue                                 # 空行
        if len(fields) != len(header_names):
            continue                                 # 列数不对：这一行坏了，只丢它
        mapped = dict(zip(header_names, fields))
        row = {name: _text(mapped.get(name)) for name in names}
        row["shape"] = format_shape(row["shape"])
        row["equation"] = "".join(row["equation"].split())
        row["schema_version"] = (row["schema_version"]
                                 or str(LEGACY_SCHEMA_VERSION))
        row["line"] = str(len(rows) + 1)             # 该表第几条**有效**数据行
        _normalize_optimize_row(row, path)
        rows.append(row)
    return rows


# -- 内部：写 ---------------------------------------------------------------
def _store_locked(token: tuple[str, ...], row: dict[str, str]) -> None:
    """删掉同键所有行 + 追加这一行，整表原子重写（持锁调用）。

    只有 tune 能走到这里：先读回整表（别把别的键的行冲掉），把同键行全部剔除，再追加
    新行。读不出旧表就**不写**（宁可这次 tune 的结论丢掉，也不能把别人的记录清空）。

    底稿的来源与读路径同源（:func:`_source_rows_locked`）：家目录里还没有表、而旧位置
    有表时，旧表就是底稿——**只读**它，然后把整表写到 :func:`profile_path`（家目录）。
    旧文件原样留着：搬/删用户的文件得先问过用户。
    """
    global _TABLE
    path = profile_path()
    try:
        rows = _source_rows_locked(path)
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


def _write_rows(path: str, rows: Sequence[Mapping[str, Any]], *,
                columns: Sequence[str] = PROFILE_COLUMNS) -> None:
    """整表重写：先写临时文件再原子替换（中途失败不会留下半张表）。

    ``columns`` 是这张表的表头（默认 :data:`PROFILE_COLUMNS`；记录表传
    :data:`RECORD_COLUMNS`）——写盘口径（临时文件 + ``os.replace``、``extrasaction``、
    换行）只有这一份，两张表共用。
    """
    names = [str(name) for name in columns]
    temp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    try:
        with open(temp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names,
                                    extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in names})
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


# -- 内部：目录与格式 -------------------------------------------------------
def _switch_on() -> bool:
    """总开关本身（只看 ``PYQCD_EINSUM_PROFILE``）。

    显式解析而不是 ``bool(text)``：``bool("false")`` 是 ``True``，把环境变量原样交给
    布尔判定会让"关掉配置表"变成"强制打开"。同一个坑在 :func:`enabled` 的显式
    ``value`` 上也存在（那里直接 ``bool(value)``，所以**只该传布尔**）。
    """
    return os.environ.get(PROFILE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def _resolve_dir_text(directory: Any = None) -> str:
    """目录分辨率：显式参数 > :data:`DIR_ENV` > :data:`DEFAULT_DIR`，给绝对路径。

    空白与 ``None`` 都算"未给"——``""`` 不会挤掉环境变量、空环境变量不会挤掉默认值，
    这样 CLI 传空串与不传的行为一致。只算字符串，不建目录。

    缺省那档就是 :data:`DEFAULT_DIR` 本身（**导入时已解析好的绝对路径**：家目录缓存，
    与当前工作目录无关）。显式路径仍先 ``expanduser`` 再取绝对路径：``os.path.abspath``
    会把 ``~`` 当字面量，``--profile-dir ~/.cache`` 于是解析成 ``<cwd>/~/.cache``，与
    ``_api`` 侧（``DIR_ENV`` / 家目录缺省）不是同一张表，CLI 的目录参数**静默失效**。
    """
    for candidate in (directory, os.environ.get(DIR_ENV)):
        text = _text(candidate)
        if text:
            return os.path.abspath(os.path.expanduser(text))
    return os.path.abspath(DEFAULT_DIR)


def _dir_text() -> str:
    """当前生效目录的绝对路径（:func:`configure` 钉过就用它，否则现算；不建目录）。"""
    return _CONFIGURED_DIR or _resolve_dir_text(None)


def _is_default_dir() -> bool:
    """生效目录是否就是缺省（家目录）那一档——只有它才走旧位置的只读回退。"""
    try:
        return os.path.normpath(_dir_text()) == os.path.normpath(DEFAULT_DIR)
    except (TypeError, ValueError):                   # pragma: no cover - 路径千奇百怪
        return False


def _legacy_path() -> str | None:
    """旧位置配置表的绝对路径；不适用（生效目录不是缺省）时返回 ``None``。

    旧位置 = **当前工作目录**下的 ``.cache/pyqcd.einsum.profile.csv``（本版之前的缺省
    目录，相对 CWD）。只在"生效目录是缺省目录"时才有意义：显式 ``--profile-dir`` 或
    ``PYQCD_EINSUM_PROFILE_DIR`` 的调用不该被别的目录的表影响。
    """
    if not _is_default_dir():
        return None
    legacy = os.path.abspath(os.path.join(LEGACY_DIR_NAME, PROFILE_FILENAME))
    if os.path.normpath(legacy) == os.path.normpath(profile_path()):
        return None                                   # 两者重合：没有第二个来源
    return legacy


def _note_source_locked(path: str) -> None:
    """登记"本次读的是旧位置的表"（持锁）：写 :func:`source_note`，并排一条警告。"""
    global _SOURCE_NOTE, _SOURCE_PATH
    note = (f"本次读的是旧位置的配置表 {path}（本版缺省目录已改到 {DEFAULT_DIR}）；"
            "只读复用，没有复制、没有移动——要用新位置请重新 tune，或自己把文件搬过去。")
    _SOURCE_NOTE = note
    _SOURCE_PATH = path
    _queue_warning(
        (path, "source", "legacy"),
        f"einsum 配置表：{note}")


def _source_rows_locked(path: str) -> list[dict[str, str]]:
    """读整张表：生效目录优先，缺文件时**只读**回退旧位置（持锁）。

    回退只发生在"生效目录是缺省（家目录）、且那里还没有表"的时候（见
    :func:`_legacy_path`）。两边都没有就抛 ``FileNotFoundError``——调用方据此报
    :class:`ProfileMissing` 或按 0 行计数。旧表**原样读**：不复制、不移动、不写。
    """
    global _SOURCE_NOTE, _SOURCE_PATH
    _SOURCE_NOTE = None
    _SOURCE_PATH = None
    try:
        rows = _table_locked(path)
    except FileNotFoundError:
        legacy = _legacy_path()
        if legacy is None:
            raise
        try:
            rows = _table_locked(legacy)
        except FileNotFoundError:
            raise FileNotFoundError(path) from None
        _note_source_locked(legacy)
        return rows
    return rows


def _source_path_locked() -> str:
    """本次读到的表的**真实**路径（旧位置回退时是旧位置的路径）；持锁读。"""
    return _SOURCE_PATH or profile_path()


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


#: **本次命令行**的真实参数（由 ``pyqcd.einsum.test`` 装上，见 :func:`set_cli_hint_context`）。
#: ``None`` = 没有 CLI 上下文（库内直调缺表）：此时 :func:`_cli_hint` 退回按键重建。
_CLI_HINT: dict[str, Any] | None = None

_CLI_HINT_LOCK = threading.Lock()


def set_cli_hint_context(*, device: str = "", dtype: str = "", equation: str = "",
                        shape: Any = None, backend: str = "",
                        compare: Any = None) -> None:
    """记录**本次调用**的真实参数，供缺表提示逐字复现这条命令。

    为什么需要它：键里没有"用户给的 ``--shape``（含输出段）"与"用户给没给
    ``--compare``"这两件事，而 GPU 下键里的 ``device`` 是 ``cupy:0`` 而不是命令行上的
    ``gpu``。只按键重建，提示就会变成"在 CPU 上重跑一遍、形状还少一段"的命令
    （2026-11-18 用户实测）。装上上下文后提示取实参；没有上下文时才退回键重建。

    ``shape``/``compare`` 传 ``None`` 表示**用户没给**：命令里也就不写这两个选项。
    """
    global _CLI_HINT
    context = {
        "device": _text(device),
        "dtype": _text(dtype),
        "equation": "".join(_text(equation).split()),
        "shape": None if shape is None else _text(shape),
        "backend": _text(backend),
        "compare": None if compare is None else _text(compare),
    }
    with _CLI_HINT_LOCK:
        _CLI_HINT = context


def clear_cli_hint_context() -> None:
    """丢掉 CLI 上下文（一次 CLI 调用结束时调用，避免串到同进程的下一次调用）。"""
    global _CLI_HINT
    with _CLI_HINT_LOCK:
        _CLI_HINT = None


def _cli_hint_context(token: tuple[str, ...]) -> dict[str, Any] | None:
    """与这把键**同一条方程**的 CLI 上下文（没有上下文/方程不同则返回 ``None``）。"""
    context = _CLI_HINT
    if not context:
        return None
    if context["equation"] and token[3] and context["equation"] != token[3]:
        return None
    return dict(context)


def _cli_hint(token: tuple[str, ...]) -> str:
    """该键的 tune CLI 命令（可直接复制执行；缺文件时打给用户）。

    有 CLI 上下文时**逐字复现用户的命令**：``--device`` / ``--dtype`` / ``--equation``
    / ``--shape``（含输出段）都取实参，``--backend`` 只在 ``cupy``/``numpy``/``torch``
    时写，``--compare`` 只在用户原命令给过时写（``--tune`` 缺省不跑参照基准）。

    没有上下文（库内直调、缺表）时才按键重建：键里没有输出段与 ``--compare`` 的显式
    位，所以这条模板保持旧口径（``--compare opt_einsum``），形状按方程**推**输出段。
    """
    context = _cli_hint_context(token)
    if context is not None:
        return _cli_hint_from_context(context, token)
    shape, dtype, device, equation, backend = token
    parts = [
        "python -m pyqcd.einsum.test",
        f"--device {_cli_device(device)}",
        f"--dtype {_cli_dtype(dtype)}",
        f'--equation "{equation}"',
        f'--shape "{_cli_shapes(shape, equation)}"',
    ]
    backend_flag = _cli_backend(backend)
    if backend_flag:
        parts.append(backend_flag)
    parts.extend(["--tune", "--compare opt_einsum"])
    return " ".join(parts)


def _cli_hint_from_context(context: dict[str, Any],
                           token: tuple[str, ...]) -> str:
    """CLI 上下文 → 可直接复制执行的命令（用户怎么跑的就怎么给）。

    ``--backend`` 取用户的显式值；用户没给（``auto``）时退回**键里实际生效**的后端
    （``cupy``/``numpy``/``torch``）——后端影响选路，省略它等于让下一次跑在另一条路上。
    """
    parts = [
        "python -m pyqcd.einsum.test",
        f"--device {_cli_device(context.get('device'))}",
        f"--dtype {_cli_dtype(context.get('dtype'))}",
        f'--equation "{context.get("equation") or ""}"',
    ]
    shape = context.get("shape")
    if shape:
        parts.append(f'--shape "{shape}"')
    backend_flag = _cli_backend(context.get("backend")) or _cli_backend(token[4])
    if backend_flag:
        parts.append(backend_flag)
    compare = context.get("compare")
    if compare:
        parts.append(f"--compare {compare}")
    parts.append("--tune")
    return " ".join(parts)


def _cli_device(device: str) -> str:
    """键里的 ``device`` → ``--device`` 的合法取值（``cpu`` / ``gpu`` / ``cuda``）。

    ``cupy:0``（cupy 只给设备编号）与 ``torch.cuda:0`` 都是 GPU：只认 ``cuda`` 前缀会
    把 ``cupy:0`` 落成 ``cpu``，于是 GPU 用户收到一条"在 CPU 上重 tune"的命令
    （2026-11-18 实测）。
    """
    text = _text(device).lower()
    if "cuda" in text or text.startswith(("cupy", "gpu")):
        return "gpu"
    return "cpu"


def _cli_dtype(dtype: str) -> str:
    """键里的 ``dtype`` → ``--dtype`` 写法（``torch.complex128`` → ``complex128``）。"""
    text = _text(dtype)
    return text.rsplit(".", 1)[-1] if text else "float64"


def _cli_backend(backend: Any) -> str:
    """``--backend`` 片段：只有具名后端（``cupy``/``numpy``/``torch``）才写。

    ``auto``（与空串）是"按操作数判"的缺省，写进命令只会多一个不改变选路的参数。
    """
    text = _text(backend).lower()
    return f"--backend {text}" if text in ("cupy", "numpy", "torch") else ""


def _cli_shapes(shape: str, equation: str = "") -> str:
    """规范形状 ``2x3|3x4`` → ``--shape`` 的 ``[2,3],[3,4]`` 写法（**带输出段**）。

    输出段按方程的输出指标从**同一把键的**操作数尺寸里取：``ab,bc->ac`` + ``40x4|4x400``
    → ``->[40,400]``。推不出来（含 ``...``、指标数与轴数不符、输出指标没有尺寸）就
    不写输出段——宁可让用户自己补，也不猜一个数字。
    """
    text = _text(shape)
    if not text:
        return ""
    axes_per_operand: list[list[str]] = []
    parts: list[str] = []
    for operand in text.split("|"):
        axes = [axis.strip() for axis in operand.split("x") if axis.strip()]
        axes_per_operand.append(axes)
        parts.append("[" + ",".join(axes) + "]")
    output = _cli_output_axes(axes_per_operand, equation)
    joined = ",".join(parts)
    return f"{joined}->[{','.join(output)}]" if output else joined


def _cli_output_axes(axes_per_operand: Sequence[Sequence[str]],
                     equation: str) -> list[str]:
    """方程的输出指标 → 各轴尺寸（按操作数里的同一指标取值）。推不出来返回空表。"""
    text = "".join(_text(equation).split())
    if not text or "..." in text or text.count("->") > 1:
        return []
    left, arrow, right = text.partition("->")
    terms = left.split(",")
    if len(terms) != len(axes_per_operand):
        return []
    sizes: dict[str, str] = {}
    for term, axes in zip(terms, axes_per_operand):
        if len(term) != len(axes):
            return []
        for label, axis in zip(term, axes):
            if not label.isalpha():
                return []
            if sizes.setdefault(label, axis) != axis:
                return []                 # 同一指标两种长度：方程与形状本就不符，不猜
    if arrow:
        labels = list(right)
    else:
        # einsum 隐含输出：只出现一次的指标，按字母序（numpy 的同一规则）。
        counts: dict[str, int] = {}
        for label in left:
            if label.isalpha():
                counts[label] = counts.get(label, 0) + 1
        labels = sorted(label for label, count in counts.items() if count == 1)
    if not labels or any(label not in sizes for label in labels):
        return []
    return [sizes[label] for label in labels]


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
    tune 窗口里），解析缓存同理——首次使用时按文件重建。待发警告与"已警告过"的集合
    也一起清空：子进程的输出是独立的，父进程发过的警告不该在这里被吞掉。
    """
    global _LOCK, _TABLE, _TUNING, _SOURCE_NOTE, _SOURCE_PATH
    _LOCK = threading.Lock()
    _TABLE = None
    _TUNING = 0
    _SOURCE_NOTE = None
    _SOURCE_PATH = None
    _PENDING_WARNINGS.clear()
    _WARNED_CELLS.clear()


if hasattr(os, "register_at_fork"):                  # POSIX；Windows 没有这个钩子
    os.register_at_fork(after_in_child=_reset_after_fork)
