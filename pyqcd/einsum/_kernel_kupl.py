"""KUPL 原生 kernel 的 ctypes 绑定与形态判定。

本模块只保留**通用形态**入口，两条路径都按 dtype / 形状 / 连续性 / backend
严格校验，任一不满足即返回 ``None`` 让上层回退通用路径，绝不猜测或放宽：

* ``try_topology_apply``：一次原生调用跑完整个 version-2 计划。dtype 契约是
  ``TOPOLOGY_DTYPES`` 的四类 dtype，**完全平权**（没有主精度）：每一类都必须
  由原生 2x2 实测通过后才启用。未启用的 dtype 一律返回 ``None`` 回落，**绝不降级成
  单精度**，而且"探测没过"与"契约外 dtype"必须给出可区分的原因。每次成功下发都会
  把 plan 字节数、操作数个数、`out` 模式、被复制的操作数下标与 offsets 记进
  ``LAST_INFO``；其中 ``specialized`` 是路线：``True`` + ``path='kupl_topology'``
  表示真的命中了编译期 v3 步表，``False`` + ``path='kupl_topology_v2'`` 表示未命中
  步表、由 v2 通用执行器完成。**两条路的返回码与数值都相同**，只有性能含义不同，
  所以任何"拓扑性能"结论都必须先看这个字段。
  **能力与选路是两层**：``TOPOLOGY_DTYPES`` / ``topology_dtypes()`` 声明"原生能跑
  哪四类"，``_TOPOLOGY_DECLINED_DTYPES`` 才是"分派要不要真的走它"。实数档当前被
  策略挡在分派层（见 ``_topology_dtype_declined``），设
  ``PYQCD_EINSUM_TOPOLOGY_REAL=1`` 可强制重新启用做对照实验。
* ``try_dedicated``：把一次 greedy 收缩展开成固定 GEMM 步表后下发。
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import struct
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .kupl import einsum as kupl

_DISABLE_ENV = 'PYQCD_EINSUM_KERNEL_KUPL'


# 最近一次"该跑 kernel 却没跑"的原因，供上层给出可行动的告警。
LAST_FAILURE: dict[str, object] = {"reason": "", "status": None}


def _fail(reason: str, status=None):
    LAST_FAILURE["reason"] = reason
    LAST_FAILURE["status"] = status
    return None


def enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, '1').strip().lower() not in (
        '0', 'false', 'off', 'no')


def _normalize(subscripts: str) -> str:
    return ''.join(subscripts.split())


_NEGATIVE_CACHE_ENV = "PYQCD_EINSUM_NEGATIVE_CACHE"
_DEDICATED_PLAN_ENV = "PYQCD_EINSUM_DEDICATED_PLAN"
_NEGATIVE_CACHE_LIMIT = 16_384
_NEGATIVE_CACHE: OrderedDict[
    tuple[Any, ...], tuple[str, int]] = OrderedDict()
_NEGATIVE_CACHE_LOCK = threading.Lock()
_NEGATIVE_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
}


# ── 负缓存旁路（强制内核专用）───────────────────────────────────────────────
# 负缓存只服务"自动分派时不必反复重试确定性拒绝"，而**强制**内核的语义是
# "强制即真试一次"：用户明确要求跑某个 label 时，上一次的 status=4 不能变成
# ``negative-cache hit: ...`` 把他挡回去。旁路用模块内计数器表达（0 = 不旁路，
# 支持嵌套），**不改 ``os.environ``** —— 环境变量是调用方与并发线程共享的状态，
# 为一次调用改它会把 ``PYQCD_EINSUM_NEGATIVE_CACHE`` 的取值泄漏出去。
# 读取是一次整数比较（不加锁；GIL 下读到旧值只会让旁路晚一次生效，不会死锁），
# 写入在 ``_NEGATIVE_CACHE_BYPASS_LOCK`` 内。**不能复用**
# ``_NEGATIVE_CACHE_LOCK``：``negative_cache_info`` 就是在该锁内调
# ``_negative_cache_enabled``，同一把非重入锁会自锁死。
_NEGATIVE_CACHE_BYPASS = 0
_NEGATIVE_CACHE_BYPASS_LOCK = threading.Lock()


@contextlib.contextmanager
def bypass_negative_cache():
    """上下文内忽略负缓存；用于强制内核时"强制即真试一次"。

    进入只把计数器 +1，退出（含异常路径）在 ``finally`` 里 -1，因此嵌套进入/退出
    天然配平：内层退出后仍处于旁路状态，外层退出才恢复。

    旁路期间**既不读也不写**负缓存。读被跳过，同一个（方程, 形状, dtype, 优化器）
    会真的再下发一次原生；写也被跳过，失败不会以"以后都别试"的形式留下，退出后
    ``negative_cache_info()`` 与进入前逐项相同（连 ``hits``/``misses``/``stores``
    都不动）。
    """
    global _NEGATIVE_CACHE_BYPASS
    with _NEGATIVE_CACHE_BYPASS_LOCK:
        _NEGATIVE_CACHE_BYPASS += 1
    try:
        yield
    finally:
        with _NEGATIVE_CACHE_BYPASS_LOCK:
            _NEGATIVE_CACHE_BYPASS -= 1


def _negative_cache_enabled() -> bool:
    """Whether deterministic native rejections may be reused this call.

    ``bypass_negative_cache()`` 期间恒为 ``False``：旁路标志**先于**环境变量判定，
    所以强制内核时即使 ``PYQCD_EINSUM_NEGATIVE_CACHE=1`` 也会真试一次；两者互不
    覆盖 —— 旁路只在本上下文内生效，退出后回到环境变量原本的取值。
    """
    if _NEGATIVE_CACHE_BYPASS:
        return False
    return os.environ.get(_NEGATIVE_CACHE_ENV, "1").strip().lower() not in (
        "0", "false", "off", "no")


def _effective_dedicated_optimizer(optimize: str) -> str:
    """Apply the explicit plan switch only when the caller kept the default."""
    if optimize != "auto":
        return optimize
    override = os.environ.get(_DEDICATED_PLAN_ENV, "").strip()
    return override or optimize


def _negative_cache_key(subscripts: str, shapes: Sequence[Sequence[int]],
                        dtype: Any, optimize: Any,
                        backend: str) -> tuple[Any, ...]:
    """Stable key for deterministic unsupported results.

    Strides, device ids and output-buffer identity are deliberately absent: those
    are validated before this cache is consulted and can make the same shape valid
    or invalid on a later call.
    """
    return (
        _normalize(subscripts),
        tuple(tuple(int(extent) for extent in shape) for shape in shapes),
        np.dtype(dtype).str,
        str(optimize),
        str(backend),
    )


def _negative_cache_lookup(key: tuple[Any, ...]) -> tuple[str, int] | None:
    if not _negative_cache_enabled():
        return None
    with _NEGATIVE_CACHE_LOCK:
        entry = _NEGATIVE_CACHE.get(key)
        if entry is None:
            _NEGATIVE_CACHE_STATS["misses"] += 1
            return None
        _NEGATIVE_CACHE.move_to_end(key)
        _NEGATIVE_CACHE_STATS["hits"] += 1
        return entry


def _negative_cache_store(key: tuple[Any, ...], reason: str,
                          status: int = 4) -> None:
    """Cache only deterministic ``UNSUPPORTED`` rejections.

    旁路期间不写（见 :func:`bypass_negative_cache`）：强制内核的那次失败不能被
    记成"以后都别试"，否则退出旁路后缓存状态与进入前不一致。
    """
    if status != 4 or not _negative_cache_enabled():
        return
    with _NEGATIVE_CACHE_LOCK:
        _NEGATIVE_CACHE.pop(key, None)
        _NEGATIVE_CACHE[key] = (str(reason), int(status))
        _NEGATIVE_CACHE_STATS["stores"] += 1
        while len(_NEGATIVE_CACHE) > _NEGATIVE_CACHE_LIMIT:
            _NEGATIVE_CACHE.popitem(last=False)
            _NEGATIVE_CACHE_STATS["evictions"] += 1


def _negative_cache_clear() -> None:
    with _NEGATIVE_CACHE_LOCK:
        _NEGATIVE_CACHE.clear()
        for name in _NEGATIVE_CACHE_STATS:
            _NEGATIVE_CACHE_STATS[name] = 0


def negative_cache_info() -> dict[str, Any]:
    """Expose bounded-cache state for contracts and production diagnostics."""
    with _NEGATIVE_CACHE_LOCK:
        return {
            "enabled": _negative_cache_enabled(),
            "size": len(_NEGATIVE_CACHE),
            "maxsize": _NEGATIVE_CACHE_LIMIT,
            **_NEGATIVE_CACHE_STATS,
        }


def _cached_unsupported(key: tuple[Any, ...]) -> tuple[str, int] | None:
    cached = _negative_cache_lookup(key)
    if cached is None:
        return None
    reason, status = cached
    return f"negative-cache hit: {reason}", status


# 多操作数拓扑入口：一次原生调用跑完整条 version-2 计划。计划字节流按
# (方程, 各操作数形状, dtype) 缓存，重复调用不再重新规划。
# 计划缓存按 (方程, 各操作数形状, dtype) 键控。旧上限 32 项是 FIFO：整个语料
# 扫一遍会反复建计划（冷 2.7 ms/条），单遍扫描耗时不能当稳态。改为真 LRU +
# 字节预算，口径与 dedicated 侧一致（计划 0.7-3.2 KB/条，64 MiB 覆盖全语料）。
_TOPOLOGY_CACHE_LIMIT = 16_384
_TOPOLOGY_CACHE_BUDGET = 64 << 20
_TOPOLOGY_PLAN_CACHE: "OrderedDict[tuple[Any, ...], bytes]" = OrderedDict()
_TOPOLOGY_CACHE_BYTES = 0
_TOPOLOGY_LOCK = threading.Lock()
_entry_topology: Any = None
_failed_topology = False

# ── dtype 契约 ──────────────────────────────────────────────────────────────
# 四类 dtype **完全平权**：没有"默认精度"或"主精度"，因此这里不能写成
# "单精度恒可下发、双精度另议"那种层级。每一类都必须由同一个 2x2 实测
# （``_probe_dtype``）自证可用才允许下发；没通过的那一类只回落它自己，不影响
# 其它三类，更不会降级成另一类精度。顺序与 ``QCD_DTYPE_*`` 同值。
TOPOLOGY_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.complex128), np.dtype(np.complex64),
    np.dtype(np.float64), np.dtype(np.float32))
_TOPOLOGY_DTYPE_SUPPORT: dict[np.dtype, bool] = {}
_TOPOLOGY_PROBE_LOCK = threading.Lock()
# 探测方程必须是**语料内**的拓扑：拓扑入口只服务 wick 语料的编译期步表，
# 语料外的方程一律 UNSUPPORTED（不再兜底）。用 ``ij,jk->ik`` 探会在"查不到步表"
# 上失败，于是探测判负 → dtype 永不启用 → 语料方程也被门控拦住，形成死锁。
# 这里取语料第一条（arity=6，输出为标量），轴长用 2 保持与旧口径同样的低成本。
_PROBE_EQUATION = "cadb,menf,ac,em,bd,fn->"
_REBUILD_HINT = "bash cpp/kupl/einsum/make.sh"
# 探测容差按元素精度分档：单精度在 6 算子链上对自身就有 ~1e-7 的舍入，用 1e-9
# 会把**正确**的单精度实现判成不支持（实测 complex64 rel=3.6e-7）。双精度仍取
# 1e-9，足以抓住"返回 0/写错"的实现。
_PROBE_TOLERANCE = {"complex128": 1e-9, "float64": 1e-9,
                    "complex64": 1e-4, "float32": 1e-4}

# ── 分派策略层（与上面的 dtype 能力契约**正交**）─────────────────────────────
# 为什么要有这一层：``TOPOLOGY_DTYPES`` 回答的是"原生能不能正确执行这类 dtype"，
# 与"分派时该不该把它交给拓扑入口"是两件事。
#
# 四类 dtype 平权后的实测（2026-11-15，``distill`` 口径即 Nev=100/Mom=1，轴长由语料
# roles 反推，热态 3 轮中位；脚本
# ``python -m pyqcd.testing.einsum.bench_topology_vs_dedicated``，产物在
# ``pyqcd/testing/einsum/artifacts/topo_vs_dedicated_*``）：
#
#   dtype        n   中位    最优    最差   快于 dedicated
#   complex128  19  8.18x  0.17x  24.63x      5/19
#   complex64   19  6.56x  0.09x  48.95x      6/19
#   float64     19  6.80x  0.30x  56.22x      5/19
#   float32     20  6.73x  0.48x  62.19x      5/20
#
# 三点结论：
#   1. **复数与实数没有系统差异**（中位 6.6x vs 6.7–6.8x）——这正是复数接通 v3
#      （cgemm/zgemm）生效的证据。此前"复数 6–7x、seq=1700 complex64 0.15x"的旧注释
#      测的是 **v2**（当时 ``run_specialized`` 对复数直接 miss），已作废。
#   2. 小规模格子上 6–9x 的来源是拓扑入口的固定开销（签名哈希 + plan 缓存查找，
#      约 1–2 ms），**不是** GEMM 形状退化；绝对差很小。
#   3. **16 算子的大规模语料上拓扑反超**（最优 0.09–0.48x，共 21/77 格快于 dedicated）：
#      v3 的静态步表省掉了 dedicated 每步的路径规划，规模一大就摊薄了入口开销。
#
# 因此实数档仍走 dedicated：它没有那 ~2 ms 固定开销，且小规模是本库日常用例的主体。
# 契约仍四类平权，``topology_dtypes()`` 仍返回四类，探测（``_probe_dtype``）语义不变。
# 若将来要让大规模场景自动改走拓扑，正确做法是按 ``seq``/元素数选路，而不是按 dtype。
_TOPOLOGY_REAL_ENV = "PYQCD_EINSUM_TOPOLOGY_REAL"
_TOPOLOGY_DECLINED_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.float64), np.dtype(np.float32))


def _topology_real_override() -> bool:
    """``PYQCD_EINSUM_TOPOLOGY_REAL=1`` 时忽略下面的回落策略（对照实验用）。"""
    return os.environ.get(_TOPOLOGY_REAL_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


def _topology_dtype_declined(dtype: np.dtype) -> bool:
    """分派策略：该 dtype 是否被**有意**排除在拓扑路径之外（不是能力问题）。"""
    if _topology_real_override():
        return False
    return np.dtype(dtype) in _TOPOLOGY_DECLINED_DTYPES


# 最近一次成功下发时实际交给原生的参数：plan 字节数/步骤数、操作数个数、输出
# 形状、out 是调用方提供还是新分配、哪些操作数被复制成连续、raw offsets。
# 失败（回落）时为空 —— 调用方据此区分"跑过原生"与"根本没跑"。
LAST_INFO: dict[str, object] = {}


def _probe_dtype(dtype: np.dtype) -> bool:
    """实测原生拓扑入口能否**正确**执行 ``dtype``（四类 dtype 同一把尺子）。

    探测是 dtype 下发的**唯一**开关，因此它自己必须 fail closed：构造操作数、
    规划、原生下发与参考比较全部在 ``try`` 里。任何一步抛异常都只等价于"该
    dtype 不可用"并返回 ``False``，绝不把异常抛给调用方 —— 否则一次探测失败会
    升级成用户可见的崩溃（"complex128 不被支持"），而且失败结果无法缓存、每次
    调用都会重探一次。缓存由 :func:`_topology_dtype_enabled` 负责，每进程每
    dtype 只探测一次。

    规模固定为 ``dim=2`` 的语料方程（``_PROBE_EQUATION``），与用户数据无关；结果
    必须与 numpy 一致才算支持 —— 返回 0 却写错的原生实现不能被当成"已实现该 dtype"。

    方程必须来自语料：拓扑入口只认编译期步表，语料外一律 UNSUPPORTED，用
    ``ij,jk->ik`` 之类探会把"没查表"误判成"dtype 不支持"，进而把语料方程也拦住。
    """
    try:
        from ._api import contract_path
        from ._native_plan import encode_execution_plan

        dimension = 2
        rng = np.random.default_rng(20261114)
        operands = []
        for term in _PROBE_EQUATION.split("->")[0].split(","):
            shape = (dimension,) * len(term)
            values = rng.standard_normal(shape)
            if dtype.kind == "c":
                values = values + 1j * rng.standard_normal(shape)
            operands.append(np.ascontiguousarray(values.astype(dtype)))
        _, path_info = contract_path(_PROBE_EQUATION, *operands)
        plan = encode_execution_plan(path_info.contraction_list, tuple(operands))
        actual = kupl.topology_apply(plan, tuple(operands))
        expected = np.einsum(_PROBE_EQUATION, *operands)
        if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != dtype:
            return False
        scale = float(np.max(np.abs(expected))) or 1.0
        tolerance = _PROBE_TOLERANCE[dtype.name]
        return bool(float(np.max(np.abs(actual - expected))) <= tolerance * scale)
    except Exception:  # noqa: BLE001 - 探测失败 = 该 dtype 不可用，一律回落
        return False


def topology_dtypes() -> tuple[np.dtype, ...]:
    """契约内的 dtype（四类平权；只读，不触发探测）。"""
    return TOPOLOGY_DTYPES


def topology_enabled_dtypes() -> tuple[np.dtype, ...]:
    """本进程**已实测通过**的 dtype（会按需探测，每 dtype 一次）。"""
    return tuple(dtype for dtype in TOPOLOGY_DTYPES
                 if _topology_dtype_enabled(dtype))


def _topology_dtype_enabled(dtype: np.dtype) -> bool:
    if dtype not in TOPOLOGY_DTYPES:
        return False
    support = _TOPOLOGY_DTYPE_SUPPORT.get(dtype)
    if support is None:
        with _TOPOLOGY_PROBE_LOCK:
            support = _TOPOLOGY_DTYPE_SUPPORT.get(dtype)
            if support is None:
                try:
                    support = _probe_dtype(dtype)
                except Exception:  # noqa: BLE001 - 探测器契约是不抛异常，这里再兜一层
                    support = False
                _TOPOLOGY_DTYPE_SUPPORT[dtype] = support
    return support


def _topology_dtype_reason(dtype: np.dtype) -> str | None:
    """``None`` 表示该 dtype 可以下发；否则给出**真实**的拒绝原因。

    两种拒绝必须能分辨，否则用户会把"探测没过"读成"这个 dtype 永远不支持"：

    * ``outside the topology dtype contract`` —— 契约外 dtype（如 int64），
      只能改代码；
    * ``native dtype probe failed`` —— 契约内 dtype，只是原生没通过 2×2 实测，
      重建原生库即可恢复。

    这里只答"原生能不能跑"，不答"分派要不要跑"；后者见
    :func:`_topology_dtype_declined`。
    """
    if dtype not in TOPOLOGY_DTYPES:
        declared = ", ".join(str(item) for item in TOPOLOGY_DTYPES)
        return (f"dtype {dtype} is outside the topology dtype contract "
                f"(declared: {declared}); no implicit downcast is performed")
    if _topology_dtype_enabled(dtype):
        return None
    return (f"dtype {dtype} is inside the topology dtype contract but its native "
            f"probe failed (2x2 matmul disagreed with numpy); falling back to the "
            f"generic path with no implicit downcast. Rebuild the native library "
            f"({_REBUILD_HINT}) to re-enable it")


# 入口绑定（dlopen + ctypes argtypes/restype 赋值）必须是**幂等单次**的：
# 没有锁时并发首调会各自 dlopen 一遍、交叉改写同一份 argtypes，实测能让进程
# 静默终止（无 traceback）。这不是"性能优化"，是正确性要求。
_ENTRY_LOCK = threading.Lock()


def _label_product(labels: Sequence[Any], sizes: dict[Any, int]) -> int:
    """一张按 ``labels`` 张开的张量有多少元素（``sizes`` 缺项返回 0）。"""
    total = 1
    for label in labels:
        extent = sizes.get(label)
        if extent is None:
            return 0
        total *= int(extent)
    return total


def _side_input_kind(axes: Sequence[int], batch_rank: int,
                     first_rank: int, second_rank: int) -> str:
    """一侧输入的布局类别，判据与 ``v3_run_step`` 逐行同口径。

    原生在 ``topology_gemm.h::v3_run_step`` 里对每一侧独立判两件事：

    * ``plain``：``a_axes[axis] == axis``，源本身即规范序，直接交给 BLAS；
    * ``trans``：源是「组交换」序——batch 轴保持在前，后两组整体互换
      （A 是 ``[batch][K][M]``、B 是 ``[batch][N][K]``），由 ``kCblasTrans`` 吸收。

    两者都不满足就是 ``pack``：必须 gather 成规范序，那才是唯一要计的 pack 流量。
    ``first_rank``/``second_rank`` 是该侧规范序里前/后两组的秩：A 侧为
    ``(M, K)``、B 侧为 ``(K, N)``，所以两侧共用这一个实现（原生的两段循环也只差
    这两个秩）。返回的字符串与原生诊断行的 ``a=``/``b=`` 字段逐字相同。
    """
    plain = True
    trans = True
    for axis, source in enumerate(axes):
        if source != axis:
            plain = False
        if axis < batch_rank:
            swapped = axis
        elif axis < batch_rank + first_rank:
            swapped = axis + second_rank
        else:
            swapped = axis - first_rank
        if source != swapped:
            trans = False
    if plain:
        return "plain"
    return "trans" if trans else "pack"


def _topology_step_records(plan: Any) -> list[dict[str, Any]] | None:
    """解 plan → 逐步的 GEMM 分组、两侧直连判定与输出是否规范序。

    这是**唯一**把 :func:`_gemm_layout`（轴序分组，与 dedicated/步表同口径）与
    :func:`_side_input_kind`（``v3_run_step`` 的直连判据）组合起来的地方：
    :func:`_topology_pack_bound_reason` 只做记账，契约用例把同一批记录与原生
    ``PYQCD_EINSUM_TOPOLOGY_TRACE=1`` 的逐步诊断行逐行对账，因此两边的判定
    不可能各写一套。字段与原生诊断行一一对应：

    ``batch``/``m``/``k``/``n``    —— ``batch=.. M=.. K=.. N=..``
    ``a_kind``/``b_kind``          —— ``a=plain|trans|pack``（仅轴序判据）
    ``path``                       —— ``path=direct|direct-a|direct-b|pack``
                                      （含越界兜底后的最终结论）
    ``output_canonical``           —— ``run_typed_v3`` 的 ``direct``：GEMM 直接落
                                      进目的缓冲区、没有 unpack。

    一元步没有 GEMM，直接跳过；任何结构推不出来（plan 解不开、分组对不上）就
    返回 ``None`` —— 不猜，交给原生自己判断。
    """
    from ._native_plan import decode_execution_plan

    try:
        decoded = decode_execution_plan(plan)
    except Exception:                     # noqa: BLE001 - 诊断失败不阻断下发
        return None
    sizes = {index: int(extent) for index, extent in enumerate(decoded.sizes)}
    records: list[dict[str, Any]] = []
    for step in decoded.steps:
        if len(step.inputs) != 2:
            continue                      # 一元步没有 GEMM，按通用路径计
        left, right = step.inputs
        result = step.output
        layout = _gemm_layout(left, right, result, sizes)
        if layout is None:
            return None                   # 结构推不出来：不猜，交给原生判断
        batch_rank = layout["batch_rank"]
        row_rank, inner_rank = layout["m_rank"], layout["k_rank"]
        col_rank = len(right) - batch_rank - inner_rank
        count_batch = layout["batch"]
        a_block = layout["m"] * layout["k"]
        b_block = layout["k"] * layout["n"]
        a_total = _label_product(left, sizes)
        b_total = _label_product(right, sizes)
        a_kind = _side_input_kind(layout["a_axes"], batch_rank, row_rank, inner_rank)
        b_kind = _side_input_kind(layout["b_axes"], batch_rank, inner_rank, col_rank)
        # 轴序判据之外还有原生的越界兜底：纯轴置换下 ``count_batch * block`` 应
        # 恰好等于该侧元素数，不等就说明推演与源不一致 —— 原生据此 fail closed
        # 转 pack，这里同步按"要 pack"计，绝不低估。
        a_direct = (a_kind != "pack" and 0 < a_block <= a_total
                    and count_batch <= a_total // a_block)
        b_direct = (b_kind != "pack" and 0 < b_block <= b_total
                    and count_batch <= b_total // b_block)
        if a_direct and b_direct:
            path = "direct"
        elif a_direct:
            path = "direct-a"
        elif b_direct:
            path = "direct-b"
        else:
            path = "pack"
        records.append({
            "batch": count_batch,
            "m": layout["m"], "k": layout["k"], "n": layout["n"],
            "a_kind": a_kind,
            "b_kind": b_kind,
            "path": path,
            "output_canonical": layout["c_axes"] == list(range(len(result))),
            "a_direct": a_direct,
            "b_direct": b_direct,
            "a_elements": count_batch * a_block,
            "b_elements": count_batch * b_block,
            "c_elements": count_batch * layout["m"] * layout["n"],
        })
    return records


# topology 侧的 ``pack/flops`` 阈值。判据形态与 dedicated 的
# ``_HEAVY_PACK_RATIO_LIMIT`` 相同——比值**与** ``pack_bytes >= 1 GiB`` 并列、不是
# 独立生效——但数值必须单独标定：本口径的流量是**按步表重算的下界**（逐侧判直连、
# 输出恒等序不计 unpack），与 dedicated 的 ``_plan_statistics`` 只在逐步判定恰好
# 一致时才给出同一个数（seq=856 两口径逐字节相同：8,281,146,368 B pack /
# 2,096,413,216 FLOPs / 3.95），多步计划上本口径只会更小。
#
# 为什么要这条门：seq=856 在 ``distill_curr`` 尺寸下 pack 8.28 GB、FLOPs 2.10e9，
# 绝对判据（``flops <= 1e9``）放行；dedicated 被比值拒后，路由链的下一站就是
# topology，实测 2675 ms，而通用路径 2330 ms —— 缺这条门就把它停在"比最慢的好、
# 比最快的差"的中间位置。
#
# 标定（2026-11-15，全语料 8638 条 × ``distill``/``distill_curr`` 两档、
# complex128；只统计同时满足 ``pack_bytes >= 1 GiB`` 与 ``flops > 1e9`` 的格子，
# 因为绝对判据已拒的那一侧根本到不了这条门）：
#
#   ``distill``       3802 格：比值 0.0016..0.3222，**无一格 >= 1.0**；被绝对判据
#                     拒掉的 688 格比值全部 >= 14.69。两峰空档 0.322..14.69，
#                     1.0 / 2.0 / 4.0 / 8.0 在该口径下新拒**都是 0 格**。
#   ``distill_curr``  5873 格：比值跨 2.5e-5..15.93，**没有单一空档**——观测到的
#                     最大空档是 4.07..11.57，但那一段取阈值就放过了 seq=856 的
#                     3.95（4.0 只新拒 131 格，seq=856 恰好落选）。次宽的两处是
#                     2.15..2.77 与 3.34..3.94；实测各阈值新拒格数：
#                     1.0 → 991（17%，含 [1.0,2.15) 那 776 格）、2.0 → 215、
#                     2.5 → 213（全部 pack >= 7.65 GiB）、3.9 → 190。
#
# 故取 2.5：落在实测空档 2.15..2.77 内（不切开任何一族），对 seq=856 的 3.95 留
# 1.45 的余量（3.9 只有 0.011 余量，plan 一改就可能翻），且不去碰 [1.0,2.15) 那
# 776 格——它们的比值只比 dedicated 的 1.0 高不到 1 倍，没有实测支持判死。
#
# **已知代价（2026-11-15 实测，如实记录）**：``distill_curr`` 上被这条门新拒的
# 那一族（比值 ≈ 3.95、pack 8.28 GB、FLOPs 2.10e9）**并不齐性**。同一份原始代码上
# 配对测（``python -m pyqcd.einsum.test``，``--rounds 5`` 中位，kernel 开=topology
# vs ``PYQCD_EINSUM_KERNEL_KUPL=0``=通用），6 格的"拒掉能省多少"：
#
#   seq    topology ms     generic ms    拒掉的效果
#   856    3014 / 6296     699           快 4.3–9.0×
#   14     3030–3224       736–780       快 4.1×
#   20     2659–2713       830           快 3.3×
#   9      291–317         698–706       慢 2.3×
#   3      144–169         756–819       慢 5.0×
#   5      138–179         792–878       慢 5.3×
#
# 3 快 3 慢、量级相当（对数口径大致打平），所以这条门是"用三格换三格"，不是纯赚。
# 根因不在比值本身：**非 ``distill`` 尺寸下原生步表的收缩序与运行期 plan 不是同一条
# 路径**（``PYQCD_EINSUM_TOPOLOGY_TRACE=1`` 实测：seq=3 原生
# ``M=9 K=400 N=640000`` 而 Python 推演是 ``M=4 K=640000 N=400``，seq=9/856 同样
# 对不上）—— 步表在生成期按 ``distill`` 尺寸定型，尺寸一变 Python 的 ``auto`` 会换
# 序。也就是说上面那个比值在 ``distill_curr`` 上度量的是"另一条计划"，本来就不可
# 全信；``distill`` 口径（模型与步表在 3/855/2 上逐 step 对账通过）才是它的有效域，
# 而那里这条门一格都不拒。真正的修法在原生侧（按尺寸重生成步表，或像 CUDA 侧
# §6.3 那样加一道"收缩序可接受"守卫），见 ``AGENTS.md`` §6.2。
# 比值门在 topology 口径上**已撤销**（2026-11-15，实测后回退），见
# :func:`_topology_pack_bound_reason` 里的完整记录。标定时用的阈值是 2.5
# （``distill`` 零误拒、``distill_curr`` 拒 213 格），但那一批里好坏混杂
# （3 快 3 慢），故不再作为判据；``_topology_pack_ratio`` 仍暴露该量供诊断。


def _topology_pack_bound_reason(plan: Any, itemsize: int) -> str | None:
    """topology 路径的 CPU 预算守卫：pack 流量压倒 GEMM 时劝回落。

    两条判据，语义与 :func:`_cpu_plan_fallback_reason` 相同：

    * 绝对：``pack_bytes >= 1 GiB`` 且 ``flops <= 1e9``；
    * 比值：``pack_bytes >= 1 GiB`` 且 ``pack_bytes >= LIMIT * flops``——每做一次
      浮点运算就搬 LIMIT 字节的布局流量，算术强度已经低到没有意义。LIMIT 是本
      口径自己的阈值（不是 dedicated 的 1.0，理由与
      标定见该常量）。

    但**流量模型必须与原生执行器同口径**，否则结构上本来零拷贝的计划会被误拒。
    三点逐条对齐 ``topology_gemm.h``：

    1. **步序来自真正下发的 plan**：``decode_execution_plan`` 解的就是
       :func:`_topology_plan` 编码、交给原生同一份字节流，而不是另跑一遍
       ``_simulate_plan``。v3 步表与这份 plan 同源（``_topology_kernel_gen.plan_steps``
       用同一个 ``contract_path(optimize="auto")`` 在生产尺寸上求序，再 ``decompose``
       出 ``a_axes``/``b_axes``/``c_axes``）。**必须按生产尺寸**：步表的收缩序在
       生成期定型，等长小 dim 下 greedy 的次序会不同（实测 seq=379 在 dim=3 就换了
       一条路径），而生产尺寸下两者逐步一致 —— 契约用例对 seq=3/855/2 的生产尺寸
       逐 step 与原生 ``PYQCD_EINSUM_TOPOLOGY_TRACE=1`` 对账（本修复期间对
       seq=379/646 也做过同样的全 13 步对账）。**"生产尺寸"指 ``distill``**：
       ``distill_curr`` 下步表与运行期 plan 会换序（实测 seq 3/9/856 都对不上，
       见 :func:`_topology_pack_bound_reason` 的"已撤销"段），那时本模型量的是另一条
       计划，只能当下界用。
    2. **输入侧逐侧判直连**（``v3_run_step``）：恒等序或组交换序的一侧零拷贝，
       只有另一侧真的要 gather。旧实现按"两侧都 pack"估上界，而语料里大量步骤
       是单侧直连（``_topology_kernel_gen`` 的静态统计：121724 步中 65053 步
       "一侧可直连、另一侧不可"）。
    3. **输出侧**（``run_typed_v3`` 的 ``direct = step_output_is_canonical(step)``）：
       ``c_axes`` 恒等于规范序时 GEMM 直接落进目的缓冲区，没有 unpack 流量。
       旧实现无条件给最终输出记 2×，于是 seq=379/845 这类"输入全直连 + 输出全
       规范序"的计划被算成 3.04 GiB 判死，而原生实测流量只有 0.0007 GiB。
       这一项与 dedicated 的 ``_plan_statistics`` 同口径（它也只在 ``c_axes``
       非恒等时计 ``c_elements``）。
    4. **FLOPs 按真实 GEMM 循环计**：``count_batch * rows * cols * inner`` 次乘加
       —— ``v3_run_step`` 对每个 batch 调一次 ``M×N×K`` 的 gemm。旧写法里 ``m``/
       ``n`` 已经含了 batch，再乘一次 batch，高 batch 计划因此被高估、门槛等于被
       架空；对齐后 6 条高 batch 的附录 seq 在 ``distill_curr`` 下重新落进判据内
       （``distill`` 下 0 条）。

    为什么需要这道门：seq=646（``amwq`` 四轴各 100，distill 尺寸）的输出本身就是
    1e8 元素且末步 ``c_axes=[2,3,1,0]``（必须 unpack），只用 1.08e8 次乘加。任何
    实现都得写下这 800 MB，但 topology 还会在中间体上整块 pack/unpack：绕过守卫
    直调原生实测 **8.9 s**，而通用路径 0.39 s（23×）。dedicated 早就被同一个判据
    拒掉，topology 缺这道门就会把它当成"命中最优路径"——修复后它仍然被拒
    （末步 3.05 GiB unpack 流量）。seq=369/845 同属这一类：它们的**输入侧**确实
    全可直连（因此新口径把估算从 3.36 GB 降到 3.20 GB），但末步同样是 1e8 元素的
    非规范序输出，原生实测 7.8/8.0 s vs 通用路径 0.57 s —— 只修输入侧就放行会把
    13–14× 的回归当成收益。

    口径的边界：本模型描述的是 **v3 步表执行器**。未命中步表时走 v2 通用执行器
    （``run_step`` 两侧都 gather），那条路线的流量只会更多，因此这里是它的下界。
    两档生产尺寸下逐 seq 实测（**只有绝对判据时**的 32 seq 抽样）：
    ``distill`` 放行 32 条（全部是 v3 命中的基础 seq，0 条转为拒绝）；
    ``distill_curr`` 放行 62 条（2 条基础 seq + 60 条附录 v2），另有 6 条附录 seq
    转为拒绝（全部来自上面第 4 点）。放行的附录 seq 走 v2，实测与通用路径持平
    （seq=7822 155.5 vs 149.3 ms，seq=7989 130.5 vs 127.6 ms），没有因此变慢。
    加上比值判据后的全语料（8638 条）放行计数是 ``distill`` 3802/3802（新拒 0）、
    ``distill_curr`` 5660/5873（新拒 213），标定明细见
    诊断函数 :func:`_topology_pack_ratio`。
    """
    records = _topology_step_records(plan)
    if records is None:
        return None                       # 推不出结构：不猜，交给原生判断
    flops = 0
    pack_bytes = 0
    for record in records:
        flops += 2 * record["batch"] * record["m"] * record["n"] * record["k"]
        if not record["a_direct"]:
            pack_bytes += 2 * itemsize * record["a_elements"]
        if not record["b_direct"]:
            pack_bytes += 2 * itemsize * record["b_elements"]
        if not record["output_canonical"]:
            pack_bytes += 2 * itemsize * record["c_elements"]
    if pack_bytes >= _HEAVY_PACK_MIN_BYTES and flops <= _HEAVY_PACK_MAX_FLOPS:
        return (f"pack-bound on CPU ({pack_bytes} pack bytes, {flops} FLOPs): "
                f"the layout traffic dominates the arithmetic; "
                f"the generic path handles it")
    # 比值判据在此口径上**已撤销**（2026-11-15，实测后回退）。
    #
    # 曾加过 ``pack_bytes >= _TOPOLOGY_PACK_RATIO_LIMIT * flops``（2.5，与 1 GiB
    # 底并列）。标定时它在 ``distill`` 上零误拒、只在 ``distill_curr`` 拒 213 格，
    # 但那 213 格里既有真病态（seq=856：topology 3014 ms vs 通用 699 ms，拒掉
    # 快 4.3×），也有**好计划**（seq=9：291 ms vs 698 ms；seq=3：144 vs 756；
    # seq=5：138 vs 792——拒掉反而慢 2.3–5.3×）。3 快 3 慢、量级相当，对数口径
    # 大致打平，属"用三格换三格"，不是净收益。
    #
    # 根因不在比值：**非 ``distill`` 尺寸下编译期步表的收缩序与运行期 plan 不是
    # 同一条路径**（``PYQCD_EINSUM_TOPOLOGY_TRACE=1`` 实测：seq=3 原生是
    # ``M=9 K=400 N=640000``，Python 推演是 ``M=4 K=640000 N=400``；seq=9/856
    # 同样对不上）。步表在生成期按 ``distill`` 定型，尺寸一变 Python 的 ``auto``
    # 就换序——本函数在那个口径下量的是**另一条计划**，比值不可信。而在模型可信
    # 的 ``distill`` 口径上它一格都不拒（3802/3802 全放行），也就是这条门实际只
    # 在不可信的口径上开火，正是最该避免的情形。
    #
    # 因此这里**不拒绝**；比值改用 :func:`_topology_pack_ratio` 暴露给诊断与契约。
    # 真正的修法在原生侧：按尺寸重生成步表，或像 CUDA 侧 §6.3 那样加一道
    # "收缩序可接受"守卫，让步表与运行期 plan 同源后再启用比值门。
    return None


def _topology_pack_ratio(plan: Any, itemsize: int) -> float | None:
    """``pack_bytes / flops``（诊断用，**不参与拒绝**）。

    记账与 :func:`_topology_pack_bound_reason` **逐行同源**（同样逐侧判直连、同样
    只在 ``c_axes`` 非恒等时计 unpack），所以两个数永远互相自洽。保留它是为了让
    ``pyqcd.einsum.test --verbose`` 与契约能看到这个量，并在原生侧收缩序修好之后
    重新评估阈值。结构解不开或 ``flops <= 0`` 时返回 ``None``。
    """
    records = _topology_step_records(plan)
    if records is None:
        return None
    flops = 0
    pack_bytes = 0
    for record in records:
        flops += 2 * record["batch"] * record["m"] * record["n"] * record["k"]
        if not record["a_direct"]:
            pack_bytes += 2 * itemsize * record["a_elements"]
        if not record["b_direct"]:
            pack_bytes += 2 * itemsize * record["b_elements"]
        if not record["output_canonical"]:
            pack_bytes += 2 * itemsize * record["c_elements"]
    if flops <= 0:
        return None
    return pack_bytes / flops

def _function_topology():
    global _entry_topology, _failed_topology
    if _entry_topology is not None or _failed_topology:
        return _entry_topology
    with _ENTRY_LOCK:
        if _entry_topology is not None or _failed_topology:
            return _entry_topology
        try:
            _entry_topology = kupl.load_topology()
        except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
            _failed_topology = True
            return None
    return _entry_topology


# ── 路线查询：本次下发到底走了 v3 步表还是 v2 通用执行器 ─────────────────────
# 命中编译期步表（v3，per-拓扑 GEMM 分步）与未命中后交给 version-2 通用执行器，
# 原生返回码都是 0、数值也都正确，**返回值区分不了两者**。旧实现因此无条件写
# ``LAST_INFO['path'] = 'kupl_topology'``：2026-11-15 的 77 格蒸馏基准里 52 格
# （语料附录 seq>2582）实际跑的是 v2，却被当成 v3 的性能结论。
# 原生为此提供 ``qcd_kupl_topology_last_specialized()``（线程内、语义为"最近一次
# apply"）。缺这个符号就**无法证实**路线，入口 fail closed：宁可回落通用路径，也
# 不给一个无法证实的 ``kupl_topology`` 标签。
_TOPOLOGY_ROUTE_V3 = 1
_TOPOLOGY_ROUTE_V2 = 0
_TOPOLOGY_ROUTE_PATHS = {True: "kupl_topology", False: "kupl_topology_v2"}
_entry_topology_route: Any = None
_failed_topology_route = False


def _function_topology_route():
    """绑定 ``qcd_kupl_topology_last_specialized``；失败返回 ``None``。"""
    global _entry_topology_route, _failed_topology_route
    if _entry_topology_route is not None or _failed_topology_route:
        return _entry_topology_route
    with _ENTRY_LOCK:                       # 同 _function_topology：绑定必须单次
        if _entry_topology_route is not None or _failed_topology_route:
            return _entry_topology_route
        try:
            entry = kupl._load_library().qcd_kupl_topology_last_specialized
        except Exception:  # noqa: BLE001 - 旧库没有这个符号
            _failed_topology_route = True
            return None
        entry.argtypes = []
        entry.restype = ctypes.c_int
        _entry_topology_route = entry
    return _entry_topology_route


def _topology_specialized(entry) -> bool | None:
    """把原生路线码翻成 ``True``(v3)/``False``(v2)；``None`` = 没有结论。

    必须是**紧接 apply 之后**的同一线程调用：原生按线程保存"最近一次"的路线。
    """
    try:
        value = int(entry())
    except Exception:  # noqa: BLE001 - 查询失败等价于"没有结论"
        return None
    if value == _TOPOLOGY_ROUTE_V3:
        return True
    if value == _TOPOLOGY_ROUTE_V2:
        return False
    return None


def _topology_key(subscripts: str, operands: Sequence[Any]) -> tuple[Any, ...]:
    return (
        _normalize(subscripts),
        tuple(tuple(int(extent) for extent in operand.shape) for operand in operands),
        operands[0].dtype.str,
    )


def _topology_plan(key: tuple[Any, ...], subscripts: str,
                   operands: Sequence[Any]) -> bytes:
    global _TOPOLOGY_CACHE_BYTES
    with _TOPOLOGY_LOCK:
        plan = _TOPOLOGY_PLAN_CACHE.get(key)
        if plan is not None:
            _TOPOLOGY_PLAN_CACHE.move_to_end(key)      # 真 LRU：命中即刷新
            return plan
    from ._api import contract_path
    from ._native_plan import encode_execution_plan

    _, info = contract_path(subscripts, *operands)
    plan = encode_execution_plan(info.contraction_list, operands)
    with _TOPOLOGY_LOCK:
        previous = _TOPOLOGY_PLAN_CACHE.pop(key, None)
        if previous is not None:
            _TOPOLOGY_CACHE_BYTES -= len(previous)
        _TOPOLOGY_PLAN_CACHE[key] = plan
        _TOPOLOGY_CACHE_BYTES += len(plan)
        while (len(_TOPOLOGY_PLAN_CACHE) > _TOPOLOGY_CACHE_LIMIT
               or _TOPOLOGY_CACHE_BYTES > _TOPOLOGY_CACHE_BUDGET):
            _, dropped = _TOPOLOGY_PLAN_CACHE.popitem(last=False)
            _TOPOLOGY_CACHE_BYTES -= len(dropped)
            if not _TOPOLOGY_PLAN_CACHE:
                break
    return plan


def try_topology_apply(subscripts: Any, operands: Sequence[Any],
                       out: Any = None,
                       offsets: Sequence[int] | None = None
                       ) -> np.ndarray | None:
    """把整条收缩计划一次性交给原生拓扑入口；不满足条件即返回 ``None`` 回落。

    与 ``try_dedicated`` 同一契约：命中且执行成功才返回结果数组，否则记录一次
    ``LAST_FAILURE`` 并让上层走通用路径。

    dtype 契约（见 ``TOPOLOGY_DTYPES`` / ``topology_dtypes``）：只下发本模块
    契约内（四类、平权）且原生 2x2 实测通过的 dtype。未启用的 dtype
    直接返回 ``None``：**绝不把 float64/complex128 转成单精度**，也绝不把异常
    抛给调用方。

    分派策略（见 ``_TOPOLOGY_DECLINED_DTYPES`` / ``_topology_dtype_declined``）：
    实数档 ``float64`` / ``float32`` 虽仍在契约内、探测也能通过，但**分派层主动
    拒绝**（``status=4``，原因含 ``declined by policy``）——实测它们的拓扑入口比
    dedicated 步表慢 52.9×/53.1×，而 dedicated 在 ``_api`` 的分派顺序里更靠前。
    这是选路而不是能力：设 ``PYQCD_EINSUM_TOPOLOGY_REAL=1``（``true``/``on``/
    ``yes`` 亦可）即忽略该策略，实数档重新参与下发，便于做对照实验。复数档不受
    该开关影响。

    ``offsets`` 只能是 ``None``：numpy 视图的 ``ctypes.data`` 已经带着字节偏移，
    再叠加 offsets 会双重偏移、静默读到错位数据，因此非 None 一律 fail closed。

    路线（见 ``LAST_INFO``）：``specialized=True`` 且 ``path='kupl_topology'`` 才
    表示真的走了编译期 v3 步表；未命中步表而由 v2 通用执行器完成时写成
    ``specialized=False`` / ``path='kupl_topology_v2'``。**两者数值都正确**，只有
    性能含义不同 —— 把后者当成"拓扑特化"就是这条契约要防的错误结论。
    """
    LAST_INFO.clear()
    if not enabled() or not isinstance(subscripts, str):
        return _fail("kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)")
    operands = list(operands)
    if not operands or any(not isinstance(operand, np.ndarray) for operand in operands):
        return _fail("operands must be numpy arrays")
    dtype = operands[0].dtype
    if any(operand.dtype != dtype for operand in operands[1:]):
        return _fail("mixed operand dtypes are not supported by the native path")
    # 先判库、再判 dtype：库缺失的原因是"没有原生入口"，不该被报成"dtype 未启用"。
    if _function_topology() is None:
        return _fail("topology entry point is missing from the native library")
    # 策略层先于探测层：被策略挡掉的 dtype 即使探测通过也一律回落，且原因必须是
    # "策略拒绝"而不是"探测没过"，否则用户会去重建一份本来就没问题的原生库。
    if _topology_dtype_declined(np.dtype(dtype)):
        return _fail(
            f"real dtype topology is declined by policy: dedicated is dispatched "
            f"first and carries no step-table overhead (after the 2026-11-15 order "
            f"fix the gap is 1.09x at seq=500 scale, and the remaining 6-9x on small "
            f"shapes is a fixed ~2 ms entry cost, not GEMM degradation). Set "
            f"{_TOPOLOGY_REAL_ENV}=1 to force it back on for an A/B comparison",
            4)
    dtype_reason = _topology_dtype_reason(np.dtype(dtype))
    if dtype_reason is not None:
        return _fail(dtype_reason)
    if out is not None:
        if not isinstance(out, np.ndarray):
            return _fail("out must be a numpy array")
        if out.dtype != dtype:
            return _fail(f"out dtype {out.dtype} does not match operand dtype {dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            return _fail("out buffer must be C-contiguous")
    if offsets is not None:
        return _fail(
            "operand_offsets must stay unused: a numpy view's data pointer already "
            "carries the byte offset, so a non-zero offset would double it")
    # 路线查询必须在执行前就绪：拿不到它就只能给出无法证实的标签，那正是本契约要防的。
    route_query = _function_topology_route()
    if route_query is None:
        return _fail(
            "the native library cannot report whether the v3 step table ran "
            "(qcd_kupl_topology_last_specialized missing), so a v2 fallback would be "
            f"indistinguishable from a v3 hit; rebuild it: {_REBUILD_HINT}")
    shapes = tuple(tuple(int(extent) for extent in operand.shape)
                   for operand in operands)
    negative_key = _negative_cache_key(
        subscripts, shapes, dtype, "topology", "kupl_topology")
    cached_unsupported = _cached_unsupported(negative_key)
    if cached_unsupported is not None:
        return _fail(*cached_unsupported)
    try:
        plan = _topology_plan(_topology_key(subscripts, operands), subscripts, operands)
        heavy = _topology_pack_bound_reason(plan, np.dtype(dtype).itemsize)
        if heavy is not None:
            _negative_cache_store(negative_key, f"topology declined: {heavy}", 4)
            return _fail(f"topology declined: {heavy}", 4)
        result = kupl.topology_apply(plan, operands, out, info=LAST_INFO)
        specialized = _topology_specialized(route_query)
        if specialized is None:
            # 原生返回 0 却给不出路线：这次输出不能贴任何"拓扑特化"标签。丢弃结果
            # 并让调用方走通用路径，比留下一个无法证实的 kupl_topology 更安全。
            LAST_INFO.clear()
            return _fail(
                "the native topology entry returned success but did not report its "
                "route (v3 step table vs v2 fallback), so the result cannot be "
                "labelled; falling back to the generic path")
    except Exception as exc:  # noqa: BLE001 - 失败即回落，绝不抛给调用方
        LAST_INFO.clear()
        status = getattr(exc, "status", None)
        reason = f"topology execution unavailable: {type(exc).__name__}: {exc}"
        if status == 4:
            _negative_cache_store(negative_key, reason, 4)
        return _fail(reason, status)
    LAST_INFO["specialized"] = specialized
    LAST_INFO["path"] = _TOPOLOGY_ROUTE_PATHS[specialized]
    return result


# ── 固定步表专用入口（dedicated） ───────────────────────────────────────────
#
# 与拓扑入口的区别：拓扑入口把整条 version-2 计划交给原生、由原生按 label 还原
# 轴序；这里把一次 greedy 收缩**展开成 GEMM 步表**（槽位 + 轴序 + batch/M/N/K），
# 原生只做 pack → cblas GEMM → unpack，不再做任何符号解析或路径推导。
#
# 步表格式见 cpp/kupl/einsum/include/dedicated_apply.h（v1）。本模块是唯一的
# Python 生成端：``_dedicated_plan`` 负责编码，``try_dedicated`` 负责下发。
# 精度契约：接受 float32 / float64 / complex64 / complex128，原生分别走
# sgemm / dgemm / cgemm / zgemm（复数元素按 numpy 的交错实虚布局直传，累加在
# 原生侧按各精度 BLAS 口径进行）。其它 dtype 一律返回 ``None`` —— **绝不隐式
# 降精度**，也不把异常抛给调用方。

_DEDICATED_MAGIC = 0x31414451
_DEDICATED_VERSION = 1
_DEDICATED_HEADER = struct.Struct("=8i")
_DEDICATED_STEP_HEADER = struct.Struct("=9i4q")
_DEDICATED_TOTAL_OFFSET = 2 * 4
# 与 cpp/kupl/einsum/include/dedicated_apply.h 的 QCD_DEDICATED_MAX_* 逐项相同：
# 原生超过任一上限就返回 4（UNSUPPORTED），这里提前拒绝可以省掉一次注定失败的
# 下发，也让 LAST_FAILURE 说明真实原因。
_DEDICATED_MAX_RANK = 32
_DEDICATED_MAX_SLOTS = 256
_DEDICATED_MAX_OPERANDS = 4096
_DEDICATED_MAX_STEPS = 4096
_DEDICATED_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.complex128), np.dtype(np.complex64),
    np.dtype(np.float64), np.dtype(np.float32))
# 与 pyqcd/einsum/_native_plan.py::_DTYPE_IDS、einsum_define.h 的 QCD_DTYPE_* 同序：
# complex128=0, complex64=1, float64=2, float32=3。
_DEDICATED_DTYPE_IDS = {
    np.dtype(np.complex128): 0, np.dtype(np.complex64): 1,
    np.dtype(np.float64): 2, np.dtype(np.float32): 3,
}
# 计划缓存按 (去空白方程, 各操作数形状, dtype, 优化器) 键控。语料级扫描（wick
# 2582–8638 条拓扑，每条一个方程）里条目数远超 32：旧的 32 项 FIFO 会让**每次
# 调用**都重跑 greedy 规划（实测 2.7 ms/条，是内核执行时间的 4–9 倍，dim≤8
# 也一样）。这里改成真实 LRU + 字节预算：命中 move_to_end，超限/超预算淘汰最久
# 未使用项。预算按 plan 字节的两倍计（Python bytes + ctypes 副本）；实测计划
# 0.7–3.2 KB。完整 8638 条 cold sweep 实测 4096 项时每轮仍全 miss，
# 3.553 ms/call、30.61 s；64 MiB 同时覆盖条目数上限与实际字节占用。
_DEDICATED_CACHE_LIMIT = 16_384
_DEDICATED_CACHE_BUDGET = 64 << 20
_DEDICATED_PLAN_CACHE: OrderedDict[
    tuple[Any, ...], tuple[bytes, tuple[int, ...], Any] | None] = OrderedDict()
_DEDICATED_CACHE_BYTES = 0
_DEDICATED_LOCK = threading.Lock()

# branch-2 only replaces the current candidate for plans whose static arena is
# already very large, and only when all three measured cost dimensions improve
# materially. This avoids promoting its lower-FLOPs but sometimes slower plans
# for ordinary contractions.
_HEAVY_BRANCH2_MIN_ARENA_BYTES = 8 << 30
_HEAVY_BRANCH2_MAX_FLOPS_PERCENT = 80
_HEAVY_BRANCH2_MAX_ARENA_PERCENT = 65
_HEAVY_BRANCH2_MAX_PACK_PERCENT = 65
_HEAVY_BRANCH2_DTYPE_IDS = (0,)  # complex128 only; other dtypes remain unverified.
_CPU_PLAN_ARENA_LIMIT = 8 << 30
_CPU_PLAN_MAX_STEP_MACS_LIMIT = 100_000_000_000
_HEAVY_PACK_MIN_BYTES = 1 << 30
_HEAVY_PACK_MAX_FLOPS = 1_000_000_000
# pack_bytes / flops：每做 1 次浮点运算要搬多少字节的布局流量（读+写各计一次）。
# 为什么不只看绝对值：绝对判据（≥1 GiB 且 ≤1e9 FLOPs）漏掉"活干得也多"的病态
# 计划——实测 seq=856（`ambn,cgdh,ekfl,oipj,ce,mo,gi,bdf,np,hjl->ak`，形状
# `[400,4,1,1],[4,4,9,1],[4,400,400,400],...`）pack 8.2 GB、FLOPs 2.1e9，绝对判据
# 放行，dedicated 实测 8662.7 ms，而 `specialize=False` 的通用路径 2330 ms（干净进程
# 5 轮中位）、改后实际回落的 `kupl_topology` 2675 ms（3.24×）；它 step 0 就要把
# `ekfl`(3.9 GiB) 整块重排成扁平 GEMM 布局，流量/算术 = 3.95 B/FLOP。
# 为什么取 1.0：CPU 上 1 B/FLOP 已经等于"算术强度 1"，任何 GEMM 都是带宽支配；
# 实测两档生产尺寸（complex128）的比值是双峰分布，1.0 落在峰间空档——
# `distill` 放行格最高 0.313、被拒格最低 4.976；`distill_curr` 放行格最高 0.937、
# 被拒格最低 1.748。取 0.5 会开始切进 `distill_curr` 的正常格一侧，取 2.0 会放过
# 1.748 那一族，故 1.0 是当前语料上最宽的空档。
# 已知副作用（如实记录，默认路径 `contract()` 实测改前→改后）：比值不看绝对规模，
# `distill` 下 pack 只有 KB–MB 的小计划也会超限——32 格里新拒 18 格，实测 10 个小格
# 变慢 1.9–4.1×（0.11→0.38、0.13→0.44、0.15→0.60、0.33→0.88 ms，绝对差 ≤0.55 ms），
# 因为回落目标 `kupl_topology` 自带入口开销。`distill_curr` 侧：seq=855
# 41.7→5.2 ms（8.0× 快）、seq=3431 34.9→30.3 ms（1.15×）、seq=2583 持平，
# 但 seq=880（比值 1.748）4.4→10.3 ms（2.3× 慢）。
# 若要消除这类误伤，把本条与 `pack_bytes >= 1 GiB` 并列即可；实测新拒会降到
# `distill` 0 格 / `distill_curr` 3 格（全是 7.9 GiB 级），seq=856 仍被拒。
_HEAVY_PACK_RATIO_LIMIT = 1.0
_entry_dedicated: Any = None
_failed_dedicated = False


def dedicated_dtypes() -> tuple[np.dtype, ...]:
    """本入口可执行的 dtype（只读，不触发加载）。"""
    return _DEDICATED_DTYPES


def _function_dedicated():
    global _entry_dedicated, _failed_dedicated
    if _entry_dedicated is not None or _failed_dedicated:
        return _entry_dedicated
    with _ENTRY_LOCK:                       # 同 _function_topology：绑定必须单次
        if _entry_dedicated is not None or _failed_dedicated:
            return _entry_dedicated
        try:
            library = kupl._load_library()
            entry = library.qcd_kupl_dedicated_apply
        except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
            _failed_dedicated = True
            return None
        entry.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                          ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                          ctypes.c_void_p, ctypes.c_int]
        entry.restype = ctypes.c_int
        _entry_dedicated = entry
    return _entry_dedicated


def _extent_map(terms: Sequence[str],
                shapes: Sequence[Sequence[int]]) -> dict[str, int] | None:
    """label → 轴长；同名 label 轴长必须一致（广播不在本入口能力内）。"""
    sizes: dict[str, int] = {}
    for term, shape in zip(terms, shapes):
        for label, extent in zip(term, shape):
            value = int(extent)
            if value < 0:
                return None
            known = sizes.get(label)
            if known is None:
                sizes[label] = value
            elif known != value:
                return None
    return sizes


def _gemm_layout(left: Sequence[Any], right: Sequence[Any], result: Sequence[Any],
                 sizes: dict[Any, int]):
    """把一次二元 einsum 步拆成 batch/M/K/N 分组与三张轴序表（**唯一**实现）。

    与 ``pyqcd.testing._topology_kernel_gen.decompose`` 同一分解口径，区别是
    这里带**真实轴长**（各轴不必等长），且不做广播：两侧同名轴长必须相等。

    ``left``/``right``/``result`` 只按"可哈希、可比对、可 ``.index``"使用，
    ``sizes`` 是 label → 轴长。于是同一套次序规则既能服务 ``_dedicated_step``
    （label 是字符、轴长来自操作数形状），也能服务 :func:`_topology_pack_bound_reason`
    （label 是 plan 的 symbol id、轴长来自 plan 的 size 表）——两条路径的轴序
    必须逐位相同，否则 pack 估算会与原生执行对不上。
    """
    left_set, right_set, out_set = set(left), set(right), set(result)
    if len(left_set) != len(left) or len(right_set) != len(right):
        return None                       # 单操作数内重复下标：不支持
    batch = [ch for ch in left if ch in right_set and ch in out_set]
    inner = [ch for ch in left if ch in right_set and ch not in out_set]
    rows = [ch for ch in left if ch not in right_set]
    cols = [ch for ch in right if ch not in left_set]
    if not set(rows) <= out_set or not set(cols) <= out_set:
        return None
    if set(batch) | set(inner) | set(rows) != left_set:
        return None
    if set(batch) | set(inner) | set(cols) != right_set:
        return None
    if set(batch) | set(rows) | set(cols) != out_set:
        return None
    if len(out_set) != len(result):
        return None
    if any(label not in sizes for label in left_set | right_set):
        return None                       # 轴长缺项：不猜

    # 组内次序自由（batch/rows/inner/cols 组内谁先谁后不影响 GEMM 语义），但对
    # **打包代价**有决定性影响：某一侧的最终轴序若恰好等于它的源轴序，那次 pack
    # 就退化成恒等排列，原生执行器直接跳过拷贝。
    #
    # batch 与 inner 是两侧共用的组，只能选一个顺序，因此选**整张量元素数更大
    # 那一侧的原序**：大张量保持恒等（零拷贝），由小张量去付置换的账。
    # 实测（n=32，`cadb,ac->db`，m_rank=0）：原先按左操作数升序排 inner 得到
    # `[a, c]`，令 1M 元素的 `cadb` 走非连续 pack，耗时 7.94 ms；改取右操作数
    # 原序 `[c, a]` 后 `b_axes` 恒等，pack 归零，整条收缩 11.8 ms → 1.6 ms。
    def element_count(labels: Sequence[Any]) -> int:
        total = 1
        for label in labels:
            total *= sizes[label]
        return total

    shared = set(batch) | set(inner)
    left_shared = [ch for ch in left if ch in shared]
    right_shared = [ch for ch in right if ch in shared]
    prefer_right = element_count(right) > element_count(left)
    ordered_shared = right_shared if prefer_right else left_shared
    batch = [ch for ch in ordered_shared if ch in batch]
    inner = [ch for ch in ordered_shared if ch in inner]
    rows.sort(key=left.index)
    cols.sort(key=right.index)
    a_order = batch + rows + inner
    b_order = batch + inner + cols
    c_order = batch + rows + cols

    def product(labels: Sequence[Any]) -> int:
        value = 1
        for label in labels:
            value *= sizes[label]
        return value

    return {
        "a_axes": [left.index(ch) for ch in a_order],
        "b_axes": [right.index(ch) for ch in b_order],
        "c_axes": [c_order.index(ch) for ch in result],
        "batch_rank": len(batch),
        "m_rank": len(rows),
        "k_rank": len(inner),
        "batch": product(batch),
        "m": product(rows),
        "n": product(cols),
        "k": product(inner),
        "result_shape": tuple(sizes[ch] for ch in result),
    }


def _dedicated_step(left: str, right: str, result: str,
                    left_shape: Sequence[int], right_shape: Sequence[int]):
    """``_gemm_layout`` 的操作数形状入口（dedicated 步表用）。"""
    sizes = _extent_map((left, right), (left_shape, right_shape))
    if sizes is None:
        return None
    return _gemm_layout(left, right, result, sizes)


def _encode_shape(buffer: bytearray, shape: Sequence[int]) -> None:
    buffer += struct.pack("=i", len(shape))
    if shape:
        buffer += struct.pack("=%dq" % len(shape), *[int(extent) for extent in shape])


def _dedicated_plan(subscripts: str, shapes: Sequence[Sequence[int]],
                    dtype_id: int,
                    optimize: str = "greedy") -> tuple[bytes, tuple[int, ...]] | None:
    """把一条收缩路径编码成原生步表；任何不支持的结构都返回 ``None``。

    ``optimize`` 原样交给 :func:`pyqcd.einsum.contract_path`。默认 ``"greedy"``
    只为兼容 ``_family_plan`` 的既有口径；``try_dedicated`` 传 ``"auto"``，与通用
    路径用同一个优化器（实测两者在部分式子上差 2.9× FLOPs，见 local-sizing.md
    的 ``[Q2b]``）。
    """
    text = _normalize(subscripts)
    if "..." in text or "->" not in text:
        return None
    lhs, rhs = text.split("->", 1)
    terms = lhs.split(",")
    shapes = [tuple(int(extent) for extent in shape) for shape in shapes]
    if len(terms) != len(shapes):
        return None
    if any(len(term) != len(shape) for term, shape in zip(terms, shapes)):
        return None
    for term in terms + [rhs]:
        if len(set(term)) != len(term) or len(term) > _DEDICATED_MAX_RANK:
            return None
    sizes = _extent_map(terms, shapes)
    if sizes is None or any(label not in sizes for label in rhs):
        return None

    from ._api import contract_path

    try:
        _, info = contract_path(text, *shapes, shapes=True, optimize=optimize)
    except Exception:  # noqa: BLE001 - 规划失败即回落
        return None
    contraction_list = list(info.contraction_list)
    if not contraction_list:
        return None

    pool_terms = list(terms)
    pool_shapes = [tuple(shape) for shape in shapes]
    pool_slots = list(range(len(terms)))
    slot_shapes: list[tuple[int, ...]] = []
    steps: list[tuple[int, int, int, dict]] = []
    for number, (inds, _removed, einsum_str, _remaining, _blas) in enumerate(
            contraction_list):
        if len(inds) != 2:
            return None                   # 一元步不在本入口能力内
        left_text, _arrow, result_text = einsum_str.partition("->")
        pair = left_text.split(",")
        if len(pair) != 2 or [pool_terms[index] for index in inds] != pair:
            return None
        if len(result_text) > _DEDICATED_MAX_RANK:
            return None
        step = _dedicated_step(pair[0], pair[1], result_text,
                               pool_shapes[inds[0]], pool_shapes[inds[1]])
        if step is None:
            return None
        final = number + 1 == len(contraction_list)
        dst_slot = -1 if final else len(slot_shapes)
        # 槽位号必须在删除池元素之前取走。
        a_slot = pool_slots[inds[0]]
        b_slot = pool_slots[inds[1]]
        steps.append((a_slot, b_slot, dst_slot, step))
        if not final:
            slot_shapes.append(step["result_shape"])
        for index in sorted(inds, reverse=True):
            del pool_terms[index]
            del pool_shapes[index]
            del pool_slots[index]
        pool_terms.append(result_text)
        pool_shapes.append(step["result_shape"])
        # 原生槽位号：0..num_operands-1 是操作数，num_operands+s 才是中间槽 s。
        pool_slots.append(-1 if final else len(terms) + dst_slot)
    if pool_terms != [rhs]:
        return None                       # 末步结果必须就是方程输出

    out_shape = tuple(sizes[label] for label in rhs)
    if steps[-1][2] != -1 or steps[-1][3]["result_shape"] != out_shape:
        return None

    # 原生上限（dedicated_apply.h 的 QCD_DEDICATED_MAX_*）：超限时原生返回 4，
    # 这里提前拒绝，省掉一次注定失败的下发，也让 LAST_FAILURE 说清真实原因。
    if not (1 <= len(shapes) <= _DEDICATED_MAX_OPERANDS):
        return None
    if len(slot_shapes) > _DEDICATED_MAX_SLOTS or len(steps) > _DEDICATED_MAX_STEPS:
        return None

    buffer = bytearray()
    buffer += _DEDICATED_HEADER.pack(_DEDICATED_MAGIC, _DEDICATED_VERSION, 0,
                                     len(shapes), len(slot_shapes), len(steps),
                                     len(rhs), int(dtype_id))
    for shape in list(shapes) + slot_shapes + [out_shape]:
        _encode_shape(buffer, shape)
    for a_slot, b_slot, dst_slot, step in steps:
        buffer += _DEDICATED_STEP_HEADER.pack(
            a_slot, b_slot, dst_slot,
            len(step["a_axes"]), len(step["b_axes"]), len(step["c_axes"]),
            step["batch_rank"], step["m_rank"], step["k_rank"],
            step["batch"], step["m"], step["n"], step["k"])
        for axes in (step["a_axes"], step["b_axes"], step["c_axes"]):
            if axes:
                buffer += struct.pack("=%di" % len(axes), *axes)
    struct.pack_into("=i", buffer, _DEDICATED_TOTAL_OFFSET, len(buffer))
    return bytes(buffer), out_shape


def _dedicated_entry_bytes(entry) -> int:
    """缓存项占用字节：plan 原串 + ctypes 副本；负结果（走不通）记 0。"""
    return 0 if entry is None else 2 * len(entry[0])


@lru_cache(maxsize=_DEDICATED_CACHE_LIMIT)
def _plan_statistics(plan: bytes) -> dict[str, int]:
    """Decode plan cost, intermediate size and estimated pack traffic once.

    执行器每步就是一次 GEMM，实测其耗时与 cblas 逐步一致（kernel/cblas =
    0.97 @dim=8、1.01 @dim=16），所以 FLOPs 是本执行器上最直接的计划代价。
    ``plan_pack_bytes`` 按读写双向计；恒等轴序不产生 pack/unpack 流量。
    """
    (_magic, _version, _total, n_operands, n_slots, n_steps,
     _out_rank, _dtype) = _DEDICATED_HEADER.unpack_from(plan, 0)
    position = _DEDICATED_HEADER.size
    shapes: list[tuple[int, ...]] = []
    for _ in range(n_operands + n_slots + 1):
        (rank,) = struct.unpack_from("=i", plan, position)
        position += 4
        shape = struct.unpack_from("=%dq" % rank, plan, position)
        position += 8 * rank
        shapes.append(tuple(int(extent) for extent in shape))
    itemsize = (16, 8, 8, 4)[_dtype]
    intermediate_shapes = shapes[n_operands:n_operands + n_slots]
    intermediate_elements = [
        int(np.prod(shape, dtype=np.int64)) for shape in intermediate_shapes
    ]
    flops = 0
    max_step_macs = 0
    pack_elements = 0
    max_a_elements = 0
    max_b_elements = 0
    max_c_elements = 0
    for _ in range(n_steps):
        fields = _DEDICATED_STEP_HEADER.unpack_from(plan, position)
        position += _DEDICATED_STEP_HEADER.size
        rank_a, rank_b, rank_c = fields[3:6]
        axes = []
        for rank in (rank_a, rank_b, rank_c):
            axes.append(struct.unpack_from("=%di" % rank, plan, position)
                        if rank else ())
            position += 4 * rank
        batch, m, n, k = fields[9:13]
        macs = batch * m * n * k
        flops += 2 * macs
        max_step_macs = max(max_step_macs, macs)
        a_elements = batch * m * k
        b_elements = batch * k * n
        c_elements = batch * m * n
        max_a_elements = max(max_a_elements, a_elements)
        max_b_elements = max(max_b_elements, b_elements)
        max_c_elements = max(max_c_elements, c_elements)
        if axes[0] != tuple(range(rank_a)):
            pack_elements += a_elements
        if axes[1] != tuple(range(rank_b)):
            pack_elements += b_elements
        if axes[2] != tuple(range(rank_c)):
            pack_elements += c_elements
    intermediate_bytes = sum(intermediate_elements) * itemsize
    max_intermediate_bytes = max(intermediate_elements, default=0) * itemsize
    arena_bytes = (
        sum(intermediate_elements)
        + max_a_elements + max_b_elements + max_c_elements
    ) * itemsize
    return {
        "plan_flops": int(flops),
        "plan_max_intermediate": int(max(intermediate_elements, default=0)),
        "plan_max_intermediate_bytes": int(max_intermediate_bytes),
        "plan_intermediate_bytes": int(intermediate_bytes),
        "plan_pack_bytes": int(2 * pack_elements * itemsize),
        "plan_arena_bytes": int(arena_bytes),
        "plan_max_step_macs": int(max_step_macs),
    }


def _plan_flops(plan: bytes) -> int:
    """Total ``Σ 2·batch·m·n·k`` for candidate-plan selection."""
    return int(_plan_statistics(plan)["plan_flops"])


def _cpu_plan_fallback_reason(stats: dict[str, int]) -> str | None:
    """Reject plans whose dedicated workspace, single GEMM or layout traffic is unsafe.

    Two independent pack criteria, because an absolute byte budget alone lets
    through plans that move gigabytes *and* do the arithmetic to hide it:

    * absolute: ``pack_bytes >= 1 GiB`` while ``flops <= 1e9`` — the layout
      traffic dwarfs the work regardless of scale;
    * ratio: ``pack_bytes / flops >= _HEAVY_PACK_RATIO_LIMIT`` — the plan spends
      at least one byte of layout traffic per FLOP, so it is bandwidth-bound
      even when the absolute FLOPs look large.  Calibrated on the corpus in both
      production scenes (see ``_HEAVY_PACK_RATIO_LIMIT``): the free/limited
      populations are separated by a gap that contains 1.0.

    ``flops == 0`` makes the ratio meaningless (zero-work plan), so only the
    absolute criterion applies there.
    """
    arena_bytes = int(stats.get("plan_arena_bytes", 0))
    max_step_macs = int(stats.get("plan_max_step_macs", 0))
    flops = int(stats.get("plan_flops", 0))
    pack_bytes = int(stats.get("plan_pack_bytes", 0))
    if arena_bytes >= _CPU_PLAN_ARENA_LIMIT:
        return (
            "plan arena exceeds the CPU dedicated budget "
            f"({arena_bytes} >= {_CPU_PLAN_ARENA_LIMIT} bytes)")
    if max_step_macs >= _CPU_PLAN_MAX_STEP_MACS_LIMIT:
        return (
            "single GEMM exceeds the CPU dedicated budget "
            f"({max_step_macs} >= {_CPU_PLAN_MAX_STEP_MACS_LIMIT} MACs)")
    if pack_bytes >= _HEAVY_PACK_MIN_BYTES and flops <= _HEAVY_PACK_MAX_FLOPS:
        return (
            "plan is pack-bound on CPU "
            f"({pack_bytes} pack bytes, {flops} FLOPs)")
    # 比值判据**与绝对底并列**，不是独立生效。单看比值会误杀大量小计划：
    # 实测（2026-11-15，语料 32 seq × 2 档、complex128）独立生效时新拒
    # distill 18/32、distill_curr 17/32（56%），其中大半 pack 只有数百字节；
    # 而极小/全 1 张量的比值恒在 8–26 且**与规模无关**（等长 dim 1→12 依次
    # 25.6 / 8.36 / 4.30 / 2.66 / 1.32 / 0.79 / 0.376），纯调阈值排除不掉，
    # 只能配绝对底。加上该底后新拒降到 0/32 与 3/32（3 格全是 7.9 GiB 级），
    # 而病态本例（pack 8.28 GB、flops 2.10e9、比值 3.95）仍被拒。
    if (flops > 0
            and pack_bytes >= _HEAVY_PACK_MIN_BYTES
            and pack_bytes >= _HEAVY_PACK_RATIO_LIMIT * flops):
        return (
            "plan pack traffic per FLOP exceeds the CPU dedicated budget "
            f"(pack/flops = {pack_bytes / flops:.3f} >= {_HEAVY_PACK_RATIO_LIMIT} "
            f"bytes per FLOP: {pack_bytes} pack bytes, {flops} FLOPs)")
    return None


def _cheapest_dedicated_plan(subscripts: str, shapes: Sequence[Sequence[int]],
                             dtype_id: int, allow_heavy_branch2: bool = False):
    """在 ``auto`` 与 ``greedy`` 之间取 FLOPs 更小的一步表（并列时取 auto）。

    只用一个优化器会踩坑：实测 14 元式子上 greedy 可多做 2.9× FLOPs
    （特化路径慢 1.76–2.28×），而 auto 在个别式子上也会更慢。两条都编码、
    按步表 FLOPs 择优，代价只是首次多一次规划（之后按 optimize="auto" 缓存）。

    CPU complex128 full-scene 的 ``seq=6585`` 还需要第三个收益条件：当基础计划
    静态 arena 已超过 8 GiB，且 branch-2 同时把 FLOPs 压到 ≤80%、arena 与 pack
    流量压到 ≤65% 时，branch-2 实测端到端快约 30%–41%。阈值避免对 ``seq=129``
    这类“arena 小但 FLOPs 变大”的计划做错误替换；其他 dtype 尚无同口径证据，
    继续由调用方显式选择 branch-2。
    """
    candidates = []
    for name in ("auto", "greedy"):
        candidate = _dedicated_plan(subscripts, shapes, dtype_id, name)
        if candidate is not None:
            candidates.append((_plan_flops(candidate[0]), candidate))
    if not candidates:
        return None
    best = min(candidates, key=lambda item: item[0])[1]
    if not allow_heavy_branch2 or dtype_id not in _HEAVY_BRANCH2_DTYPE_IDS:
        return best

    best_stats = _plan_statistics(best[0])
    if best_stats["plan_arena_bytes"] < _HEAVY_BRANCH2_MIN_ARENA_BYTES:
        return best
    branch2 = _dedicated_plan(subscripts, shapes, dtype_id, "branch-2")
    if branch2 is None:
        return best
    branch2_stats = _plan_statistics(branch2[0])
    best_pack = best_stats["plan_pack_bytes"]
    pack_ok = (
        branch2_stats["plan_pack_bytes"] == 0
        if best_pack == 0
        else branch2_stats["plan_pack_bytes"] * 100
        <= best_pack * _HEAVY_BRANCH2_MAX_PACK_PERCENT
    )
    if (
        branch2_stats["plan_flops"] * 100
        <= best_stats["plan_flops"] * _HEAVY_BRANCH2_MAX_FLOPS_PERCENT
        and branch2_stats["plan_arena_bytes"] * 100
        <= best_stats["plan_arena_bytes"] * _HEAVY_BRANCH2_MAX_ARENA_PERCENT
        and pack_ok
    ):
        return branch2
    return best


def _dedicated_cache_store(key: tuple[Any, ...], entry) -> None:
    """按 LRU + 字节预算写入；调用方必须已持有 ``_DEDICATED_LOCK``。"""
    global _DEDICATED_CACHE_BYTES
    previous = _DEDICATED_PLAN_CACHE.pop(key, None)
    if previous is not None:
        _DEDICATED_CACHE_BYTES -= _dedicated_entry_bytes(previous)
    _DEDICATED_PLAN_CACHE[key] = entry
    _DEDICATED_CACHE_BYTES += _dedicated_entry_bytes(entry)
    while _DEDICATED_PLAN_CACHE and (
            len(_DEDICATED_PLAN_CACHE) > _DEDICATED_CACHE_LIMIT
            or _DEDICATED_CACHE_BYTES > _DEDICATED_CACHE_BUDGET):
        _, evicted = _DEDICATED_PLAN_CACHE.popitem(last=False)
        _DEDICATED_CACHE_BYTES -= _dedicated_entry_bytes(evicted)


def _dedicated_cached(subscripts: str, shapes: Sequence[Sequence[int]],
                      dtype: np.dtype, optimize: str, backend: str = "kupl"):
    key = (_normalize(subscripts),
           tuple(tuple(int(extent) for extent in shape) for shape in shapes),
           dtype.str, optimize, str(backend))
    with _DEDICATED_LOCK:
        if key in _DEDICATED_PLAN_CACHE:
            entry = _DEDICATED_PLAN_CACHE[key]
            _DEDICATED_PLAN_CACHE.move_to_end(key)   # 真 LRU：命中即刷新
            return entry
    encoded = None
    dtype_id = _DEDICATED_DTYPE_IDS.get(dtype)
    if dtype_id is not None:
        # "auto"（默认）在 auto 与 greedy 两条计划里按步表 FLOPs 择优，避免单一
        # 优化器的坏计划（实测 greedy 可多做 2.9× FLOPs，特化路径因此慢 2×）。
        if optimize == "auto":
            encoded = _cheapest_dedicated_plan(
                subscripts, shapes, dtype_id, str(backend) == "kupl")
        else:
            encoded = _dedicated_plan(subscripts, shapes, dtype_id, optimize)
    if encoded is None:
        # 负结果同样入缓存：否则每次调用都要重跑一遍 greedy 规划（大式子上是
        # 毫秒级），而"这条路走不通"对同一 (式子, 形状, dtype, 优化器) 恒定。
        entry = None
    else:
        plan, output_shape = encoded
        # FLOPs 只在编码时算一次：它进 LAST_INFO 做诊断，不能在每次调用里重解码。
        entry = (plan, output_shape,
                 (ctypes.c_ubyte * len(plan)).from_buffer_copy(plan),
                 _plan_flops(plan))
    with _DEDICATED_LOCK:
        _dedicated_cache_store(key, entry)
    return entry


def try_dedicated(subscripts: Any, operands: Sequence[Any], out: Any = None,
                  *, optimize: str = "auto") -> np.ndarray | None:
    """把一次多算子收缩按下发前的固定步表交给原生执行；否则返回 ``None``。

    与 ``try_topology_apply`` 同一契约（fail-open、失败记录 ``LAST_FAILURE``、
    绝不抛异常），区别只在原生吃的是**已展开的 GEMM 步表**而不是 version-2
    计划：Python 只付一次规划 + 编码成本（按方程/形状/dtype/优化器缓存），执行期
    没有逐步骤的 Python 调度。

    ``optimize`` 默认 ``"auto"``：在 auto 与 greedy 两条计划里按步表 FLOPs 取小
    （并列取 auto）；仅当 CPU complex128 基础计划静态 arena 超过 8 GiB，且
    branch-2 同时把 FLOPs 压到 ≤80%、arena/pack 压到 ≤65% 时，才条件提升
    branch-2。这样既避免单一优化器的坏计划（greedy 在个别式子上多做 487×
    FLOPs），也不会把 branch-2 用于 ``seq=129`` 这类 arena 更小但 FLOPs 更大、
    实测慢 27.5% 的路径。

    dtype 契约：``float32`` / ``float64`` / ``complex64`` / ``complex128``（原生分别
    走 sgemm / dgemm / cgemm / zgemm）。其余 dtype 直接返回 ``None`` 并记录原因 ——
    **不做隐式降精度**。原生 BLAS 缺 cgemm/zgemm 时复数档收到状态 4（UNSUPPORTED）
    并回落通用路径，不会被当成错误。
    """
    LAST_INFO.clear()
    if not enabled() or not isinstance(subscripts, str):
        return _fail("kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)")
    operands = list(operands)
    if not operands or any(not isinstance(operand, np.ndarray) for operand in operands):
        return _fail("operands must be numpy arrays")
    dtype = np.dtype(operands[0].dtype)
    if any(np.dtype(operand.dtype) != dtype for operand in operands[1:]):
        return _fail("mixed operand dtypes are not supported by the native path")
    optimize = _effective_dedicated_optimizer(optimize)
    shapes = tuple(tuple(int(extent) for extent in operand.shape) for operand in operands)
    if any(not operand.flags["C_CONTIGUOUS"] for operand in operands):
        return _fail("dedicated kernel requires C-contiguous operands")
    negative_key = _negative_cache_key(
        subscripts, shapes, dtype, optimize, "kupl_dedicated")
    cached_unsupported = _cached_unsupported(negative_key)
    if cached_unsupported is not None:
        return _fail(*cached_unsupported)
    if dtype not in _DEDICATED_DTYPES:
        supported = ", ".join(str(item) for item in _DEDICATED_DTYPES)
        reason = (
            f"dtype {dtype} has no dedicated step-table path "
            f"(supported: {supported}); no implicit downcast is performed")
        _negative_cache_store(negative_key, reason, 4)
        return _fail(reason, 4)
    entry = _function_dedicated()
    if entry is None:
        return _fail("dedicated entry point is missing from the native library")

    cached = _dedicated_cached(subscripts, shapes, dtype, optimize)
    if cached is None:
        return _fail("the contraction path is not expressible as a fixed step table")
    plan, output_shape, plan_buffer, plan_flops = cached
    stats = _plan_statistics(plan)
    plan_gate = _cpu_plan_fallback_reason(stats)
    if plan_gate is not None:
        _negative_cache_store(negative_key, plan_gate, 4)
        return _fail(plan_gate, 4)

    if out is None:
        output = np.empty(output_shape, dtype=dtype)
    else:
        if not isinstance(out, np.ndarray):
            return _fail("out must be a numpy array")
        if np.dtype(out.dtype) != dtype:
            return _fail(f"out dtype {out.dtype} does not match operand dtype {dtype}")
        if tuple(output_shape) != tuple(out.shape):
            return _fail(f"out shape {out.shape} does not match plan shape {output_shape}")
        if not out.flags["C_CONTIGUOUS"]:
            return _fail("out buffer must be C-contiguous")
        output = out

    pointers = (ctypes.c_void_p * len(operands))(
        *[int(operand.ctypes.data) for operand in operands])
    try:
        status = int(entry(ctypes.addressof(plan_buffer), len(plan), pointers,
                           len(operands), int(output.ctypes.data),
                           _DEDICATED_DTYPE_IDS[dtype]))
    except Exception as exc:  # noqa: BLE001 - 失败即回落，绝不抛给调用方
        status = getattr(exc, "status", None)
        reason = f"dedicated execution unavailable: {type(exc).__name__}: {exc}"
        if status == 4:
            _negative_cache_store(negative_key, reason, 4)
        return _fail(reason, status)
    if status != 0:
        reason = f"dedicated kernel returned status {status}"
        if status == 4:
            _negative_cache_store(negative_key, reason, 4)
        return _fail(reason, status)
    LAST_INFO.update(
        path="kupl_dedicated",
        equation=_normalize(subscripts),
        dtype=str(dtype),
        dtype_id=_DEDICATED_DTYPE_IDS[dtype],
        n_operands=len(operands),
        plan_bytes=len(plan),
        plan_flops=plan_flops,
        plan_max_intermediate=stats["plan_max_intermediate"],
        plan_pack_bytes=stats["plan_pack_bytes"],
        plan_arena_bytes=stats["plan_arena_bytes"],
        plan_max_step_macs=stats["plan_max_step_macs"],
        optimize=optimize,
        output_shape=tuple(output_shape),
        out="allocated" if out is None else "provided",
    )
    return output


# ── 25 族「可套用模板」入口（family） ────────────────────────────────────────
#
# 与 dedicated 的区别只在**准入条件**：dedicated 接受任意形状的二元步表；这里
# 要求计划属于 docs/wick_kernel_families_20261111.md 的 25 族之一——按
# (算子元数, 各操作数 rank 的多重集, 输出 rank) 命中原生侧的一个模板实例化。
# 同族内实现是同一份模板（元数/rank/输出阶都是编译期常量），差异只在步表这份
# 运行期数据，所以 Python 侧不需要为每条式子生成代码，只需要一份缓存计划。
#
# 计划字节 = 40 字节族表头 + dedicated v1 的「形状表 + 步记录」尾部。尾部直接
# 复用 ``_dedicated_plan`` 的编码结果（同一份二分步表、同一套校验），族表头只是
# 换 magic/版本并补两个保留字，因此这里没有第二份收缩分解代码。
#
# 精度契约：族模板（family_apply.cpp）只实例化了实数两档，因此这里**仍只接受
# float32 / float64**（分别走 sgemm / dgemm），与 dedicated 入口放开的四种 dtype
# 不同——complex 直接返回 ``None`` 并记录原因，**绝不隐式降精度**。

_FAMILY_MAGIC = 0x31414651
_FAMILY_VERSION = 1
_FAMILY_HEADER = struct.Struct("=10i")
_FAMILY_TOTAL_OFFSET = 2 * 4
_FAMILY_DTYPES: tuple[np.dtype, ...] = (np.dtype(np.float32), np.dtype(np.float64))
_FAMILY_CACHE_LIMIT = 32
_FAMILY_PLAN_CACHE: dict[tuple[Any, ...], tuple[bytes, tuple[int, ...], Any] | None] = {}
_FAMILY_LOCK = threading.Lock()
_entry_family: Any = None
_failed_family = False


def family_dtypes() -> tuple[np.dtype, ...]:
    """本入口可执行的 dtype（只读，不触发加载）。"""
    return _FAMILY_DTYPES


def _function_family():
    global _entry_family, _failed_family
    if _entry_family is not None or _failed_family:
        return _entry_family
    try:
        library = kupl._load_library()
        entry = library.qcd_kupl_family_apply
    except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
        _failed_family = True
        return None
    entry.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                      ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_int]
    entry.restype = ctypes.c_int
    _entry_family = entry
    return _entry_family


def _family_plan(subscripts: str, shapes: Sequence[Sequence[int]],
                 dtype_id: int) -> tuple[bytes, tuple[int, ...]] | None:
    """去掉 dedicated 的 32 字节表头，换成 40 字节族表头（尾部逐字节不变）。"""
    if _DEDICATED_HEADER.size != 32:
        return None
    encoded = _dedicated_plan(subscripts, shapes, dtype_id)
    if encoded is None:
        return None
    base, output_shape = encoded
    _, _, _, n_ops, n_slots, n_steps, out_rank, _ = _DEDICATED_HEADER.unpack_from(base, 0)
    buffer = bytearray(_FAMILY_HEADER.pack(_FAMILY_MAGIC, _FAMILY_VERSION, 0,
                                           n_ops, n_slots, n_steps, out_rank,
                                           int(dtype_id), 0, 0))
    buffer += base[_DEDICATED_HEADER.size:]
    struct.pack_into("=i", buffer, _FAMILY_TOTAL_OFFSET, len(buffer))
    return bytes(buffer), output_shape


def _family_cached(subscripts: str, shapes: Sequence[Sequence[int]],
                   dtype: np.dtype, optimize: str):
    key = (_normalize(subscripts),
           tuple(tuple(int(extent) for extent in shape) for shape in shapes),
           dtype.str, optimize)
    with _FAMILY_LOCK:
        if key in _FAMILY_PLAN_CACHE:
            return _FAMILY_PLAN_CACHE[key]
    if optimize != "greedy":
        return None                       # 目前只编码 greedy 路径
    dtype_id = _DEDICATED_DTYPE_IDS.get(dtype)
    if dtype_id is None:
        return None
    encoded = _family_plan(subscripts, shapes, dtype_id)
    if encoded is None:
        return None
    plan, output_shape = encoded
    entry = (plan, output_shape,
             (ctypes.c_ubyte * len(plan)).from_buffer_copy(plan))
    with _FAMILY_LOCK:
        if len(_FAMILY_PLAN_CACHE) >= _FAMILY_CACHE_LIMIT:
            _FAMILY_PLAN_CACHE.pop(next(iter(_FAMILY_PLAN_CACHE)))
        _FAMILY_PLAN_CACHE[key] = entry
    return entry


def try_family_apply(subscripts: Any, operands: Sequence[Any], out: Any = None,
                     *, optimize: str = "greedy") -> np.ndarray | None:
    """把整条收缩交给「25 族模板」入口；任何不满足条件都返回 ``None`` 回落。

    与 ``try_dedicated`` 同一契约（fail-open、失败记录 ``LAST_FAILURE``、绝不抛
    异常），区别只在原生侧的准入：计划必须属于
    ``docs/wick_kernel_families_20261111.md`` 的 25 族之一（元数 + 各操作数 rank
    的多重集 + 输出 rank）。同族内由同一份 C++ 模板实例化执行，差异只在步表。

    原生返回 4 表示"这份计划不属于 25 族"或"dtype 无本入口的数值路径"，与 1
    （参数非法）分开记录，便于区分"正常回落"和"调用写错了"。

    dtype 契约：``float32`` / ``float64``。其余 dtype（含 complex）直接返回
    ``None`` 并记录原因——**不做隐式降精度**。
    """
    LAST_INFO.clear()
    if not enabled() or not isinstance(subscripts, str):
        return _fail("kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)")
    operands = list(operands)
    if not operands or any(not isinstance(operand, np.ndarray) for operand in operands):
        return _fail("operands must be numpy arrays")
    dtype = np.dtype(operands[0].dtype)
    if any(np.dtype(operand.dtype) != dtype for operand in operands[1:]):
        return _fail("mixed operand dtypes are not supported by the native path")
    if dtype not in _FAMILY_DTYPES:
        supported = ", ".join(str(item) for item in _FAMILY_DTYPES)
        return _fail(
            f"dtype {dtype} has no family-template path "
            f"(supported: {supported}); no implicit downcast is performed")
    if any(not operand.flags["C_CONTIGUOUS"] for operand in operands):
        return _fail("family kernel requires C-contiguous operands")
    entry = _function_family()
    if entry is None:
        return _fail("family entry point is missing from the native library")

    shapes = tuple(tuple(int(extent) for extent in operand.shape) for operand in operands)
    cached = _family_cached(subscripts, shapes, dtype, optimize)
    if cached is None:
        return _fail("the contraction path is not expressible as a family step table")
    plan, output_shape, plan_buffer = cached

    if out is None:
        output = np.empty(output_shape, dtype=dtype)
    else:
        if not isinstance(out, np.ndarray):
            return _fail("out must be a numpy array")
        if np.dtype(out.dtype) != dtype:
            return _fail(f"out dtype {out.dtype} does not match operand dtype {dtype}")
        if tuple(output_shape) != tuple(out.shape):
            return _fail(f"out shape {out.shape} does not match plan shape {output_shape}")
        if not out.flags["C_CONTIGUOUS"]:
            return _fail("out buffer must be C-contiguous")
        output = out

    pointers = (ctypes.c_void_p * len(operands))(
        *[int(operand.ctypes.data) for operand in operands])
    try:
        status = int(entry(ctypes.addressof(plan_buffer), len(plan), pointers,
                           len(operands), int(output.ctypes.data),
                           _DEDICATED_DTYPE_IDS[dtype]))
    except Exception as exc:  # noqa: BLE001 - 失败即回落，绝不抛给调用方
        return _fail(f"family execution unavailable: {type(exc).__name__}: {exc}")
    if status != 0:
        reason = ("the plan is not one of the 25 families"
                  if status == 4 else "invalid arguments or an internal error")
        return _fail(f"family kernel returned status {status} ({reason})", status)
    LAST_INFO.update(
        path="kupl_family",
        equation=_normalize(subscripts),
        dtype=str(dtype),
        dtype_id=_DEDICATED_DTYPE_IDS[dtype],
        n_operands=len(operands),
        plan_bytes=len(plan),
        output_shape=tuple(output_shape),
        out="allocated" if out is None else "provided",
    )
    return output
