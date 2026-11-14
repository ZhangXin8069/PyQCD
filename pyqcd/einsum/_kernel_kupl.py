"""KUPL 原生 kernel 的 ctypes 绑定与形态判定。

本模块只保留**通用形态**入口，两条路径都按 dtype / 形状 / 连续性 / backend
严格校验，任一不满足即返回 ``None`` 让上层回退通用路径，绝不猜测或放宽：

* ``try_topology_apply``：一次原生调用跑完整个 version-2 计划。dtype 契约是
  ``TOPOLOGY_DTYPES`` 的四类 dtype，**完全平权**（没有主精度）：每一类都必须
  由原生 2x2 实测通过后才启用。未启用的 dtype 一律返回 ``None`` 回落，**绝不降级成
  单精度**，而且"探测没过"与"契约外 dtype"必须给出可区分的原因。每次成功下发都会
  把 plan 字节数、操作数个数、`out` 模式、被复制的操作数下标与 offsets 记进
  ``LAST_INFO``。
  **能力与选路是两层**：``TOPOLOGY_DTYPES`` / ``topology_dtypes()`` 声明"原生能跑
  哪四类"，``_TOPOLOGY_DECLINED_DTYPES`` 才是"分派要不要真的走它"。实数档当前被
  策略挡在分派层（见 ``_topology_dtype_declined``），设
  ``PYQCD_EINSUM_TOPOLOGY_REAL=1`` 可强制重新启用做对照实验。
* ``try_dedicated``：把一次 greedy 收缩展开成固定 GEMM 步表后下发。
"""

from __future__ import annotations

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


def _negative_cache_enabled() -> bool:
    """Whether deterministic native rejections may be reused this call."""
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
    """Cache only deterministic ``UNSUPPORTED`` rejections."""
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
# 与"分派时该不该把它交给拓扑入口"是两件事。实测（seq=500，scene scale=8，3 轮中位）
# 实数档的拓扑入口慢于 dedicated 步表 52.9×（float64 246.8 ms vs 4.7 ms）与
# 53.1×（float32 140.5 ms vs 2.6 ms），根因是原生 v3 执行器的 GEMM 形状退化；而
# 复数档两者等价（1.05× / 0.99×），且 dedicated 在 ``_api`` 的分派顺序里本就靠前。
# 所以这里**只**把实数档挡在分派层：契约仍四类平权，``topology_dtypes()`` 仍返回
# 四类，探测（``_probe_dtype``）语义不变，复数档行为完全不动。
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
            f"real dtype topology is declined by policy: the step table is slower "
            f"than dedicated for real inputs (measured 53x at seq=500, scene "
            f"scale=8: 246.8 ms vs 4.7 ms for float64, 140.5 ms vs 2.6 ms for "
            f"float32); dedicated handles it and is dispatched first. Set "
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
    shapes = tuple(tuple(int(extent) for extent in operand.shape)
                   for operand in operands)
    negative_key = _negative_cache_key(
        subscripts, shapes, dtype, "topology", "kupl_topology")
    cached_unsupported = _cached_unsupported(negative_key)
    if cached_unsupported is not None:
        return _fail(*cached_unsupported)
    try:
        plan = _topology_plan(_topology_key(subscripts, operands), subscripts, operands)
        result = kupl.topology_apply(plan, operands, out, info=LAST_INFO)
    except Exception as exc:  # noqa: BLE001 - 失败即回落，绝不抛给调用方
        LAST_INFO.clear()
        status = getattr(exc, "status", None)
        reason = f"topology execution unavailable: {type(exc).__name__}: {exc}"
        if status == 4:
            _negative_cache_store(negative_key, reason, 4)
        return _fail(reason, status)
    LAST_INFO["path"] = "kupl_topology"
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


def _dedicated_step(left: str, right: str, result: str,
                    left_shape: Sequence[int], right_shape: Sequence[int]):
    """把一次二元 einsum 步拆成 batch/M/K/N 分组与三张轴序表。

    与 ``pyqcd.testing._topology_kernel_gen.decompose`` 同一分解口径，区别是
    这里带**真实轴长**（各轴不必等长），且不做广播：两侧同名轴长必须相等。
    """
    sizes = _extent_map((left, right), (left_shape, right_shape))
    if sizes is None:
        return None
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

    # 组内次序自由（batch/rows/inner/cols 组内谁先谁后不影响 GEMM 语义），但对
    # **打包代价**有决定性影响：某一侧的最终轴序若恰好等于它的源轴序，那次 pack
    # 就退化成恒等排列，原生执行器直接跳过拷贝。
    #
    # batch 与 inner 是两侧共用的组，只能选一个顺序，因此选**整张量元素数更大
    # 那一侧的原序**：大张量保持恒等（零拷贝），由小张量去付置换的账。
    # 实测（n=32，`cadb,ac->db`，m_rank=0）：原先按左操作数升序排 inner 得到
    # `[a, c]`，令 1M 元素的 `cadb` 走非连续 pack，耗时 7.94 ms；改取右操作数
    # 原序 `[c, a]` 后 `b_axes` 恒等，pack 归零，整条收缩 11.8 ms → 1.6 ms。
    def element_count(labels: Sequence[str]) -> int:
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

    def product(labels: Sequence[str]) -> int:
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
    """Reject plans whose dedicated workspace or single GEMM is unsafe on CPU."""
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
