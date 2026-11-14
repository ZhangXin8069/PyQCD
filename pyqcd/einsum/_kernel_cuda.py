"""CUDA 原生 kernel 的绑定与形态判定（拓扑 / 固定步表入口）。

两条路径的契约都按 dtype / 形状 / 连续性 / 设备严格校验，任一不满足即返回
``None`` 回落，绝不猜测或放宽：

* ``try_topology_apply``：多操作数拓扑入口，dtype 契约是 ``TOPOLOGY_DTYPES`` 与
  ``TOPOLOGY_DTYPES`` 的四类 dtype，**完全平权**（没有主精度）：每一类都必须由
  原生 2x2 实测通过后才启用。未启用的 dtype 返回
  ``None`` 回落且**绝不降级**，"探测没过"与"契约外 dtype"必须给出可区分的原因；
  成功下发时把 plan 字节数、操作数个数、``out`` 模式与复制下标记进 ``LAST_INFO``。
* ``try_dedicated``：与 KUPL 侧共用同一份 GEMM 步表字节流。
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np

_DISABLE_ENV = 'PYQCD_EINSUM_KERNEL_CUDA'


# 最近一次"该跑 kernel 却没跑"的原因，供上层给出可行动的告警。
LAST_FAILURE: dict[str, object] = {"reason": "", "status": None}


def _fail(reason: str, status=None):
    LAST_FAILURE["reason"] = reason
    LAST_FAILURE["status"] = status
    return None


def enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, '1').strip().lower() not in (
        '0', 'false', 'off', 'no')


# 多操作数拓扑入口：一次原生调用跑完整条计划。下发的是**拓扑描述符**（只带
# 结构：每个操作数的标签与形状、输出标签、签名摘要），收缩顺序来自原生库里的
# 预编译步表 `cpp/cuda/einsum/src/topology_steps.h` —— 这里不再做 path 搜索，
# 也不再逐步骤调度。描述符按 (方程, 各操作数形状, dtype) 缓存，重复调用零规划。
# 与 KUPL 侧同口径：真 LRU + 字节预算。32 项 FIFO 会让语料扫描每轮全 miss。
_TOPOLOGY_CACHE_LIMIT = 16_384
_TOPOLOGY_CACHE_BUDGET = 64 << 20
_TOPOLOGY_PLAN_CACHE: "OrderedDict[tuple[Any, ...], bytes]" = OrderedDict()
_TOPOLOGY_CACHE_BYTES = 0
_TOPOLOGY_LOCK = threading.Lock()

# 描述符字节流的魔数与版本，必须与 `cpp/cuda/einsum/src/apply_contract.cu` 同步。
_DESCRIPTOR_MAGIC = 0x51435444
_DESCRIPTOR_VERSION = 1
_DESCRIPTOR_HEADER_WORDS = 8
_DESCRIPTOR_MAX_EXTENT = 2 ** 31 - 1

# ── dtype 契约 ──────────────────────────────────────────────────────────────
# 四类 dtype **完全平权**：没有"默认精度"或"主精度"。每一类都必须由同一个 2x2
# 实测（``_probe_dtype``）自证可用才允许下发；没通过的那一类只回落它自己，不影响
# 其它三类，更不会降级成另一类精度。顺序与 ``QCD_DTYPE_*`` 同值。
TOPOLOGY_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.complex128), np.dtype(np.complex64),
    np.dtype(np.float64), np.dtype(np.float32))
_TOPOLOGY_DTYPE_SUPPORT: dict[np.dtype, bool] = {}
_TOPOLOGY_PROBE_LOCK = threading.Lock()
_PROBE_EQUATION = "ij,jk->ik"
_REBUILD_HINT = "bash cpp/cuda/einsum/make-nv.sh"
# 与 KUPL 侧同口径：单精度按 1e-4，双精度 1e-9（见 _kernel_kupl._PROBE_TOLERANCE）。
_PROBE_TOLERANCE = {"complex128": 1e-9, "float64": 1e-9,
                    "complex64": 1e-4, "float32": 1e-4}

# 最近一次成功下发时实际交给原生的参数（见 `_kernel_kupl.LAST_INFO`）。
LAST_INFO: dict[str, object] = {}


def _probe_dtype(dtype: np.dtype) -> bool:
    """实测设备侧拓扑入口能否**正确**执行 ``dtype``（四类 dtype 同一把尺子）。

    与 KUPL 侧同一口径：探测本身必须 fail closed。``import cupy``、构建设备侧
    操作数、规划、原生下发与参考比较全部在 ``try`` 里 —— 历史上 ``cupy.arange``
    在 ``try`` 之外，NVRTC 架构不匹配时会把 ``CompileException`` 抛给调用方，
    于是 complex128 的 dtype 决策变成硬崩溃，而且失败结果无法缓存、每次调用都
    重探。任何异常现在只等价于"该 dtype 不可用"。
    """
    try:
        import cupy

        from ._api import contract_path
        from ._native_plan import encode_execution_plan
        from .cuda._program import topology_apply as native_topology_apply

        base = cupy.arange(4, dtype=cupy.float64).reshape(2, 2) + 1.0
        if dtype.kind == "c":
            left = (base * (1.0 + 1.0j)).astype(dtype)
            right = ((base - 2.0) * (1.0 - 0.5j)).astype(dtype)
        else:
            left = base.astype(dtype)
            right = (base - 2.0).astype(dtype)
        _, path_info = contract_path(_PROBE_EQUATION, left, right)
        plan = encode_execution_plan(path_info.contraction_list, (left, right))
        actual = cupy.asnumpy(native_topology_apply(plan, (left, right)))
        expected = np.einsum(
            _PROBE_EQUATION, cupy.asnumpy(left), cupy.asnumpy(right))
        if actual.shape != expected.shape or actual.dtype != dtype:
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

    两种拒绝必须能分辨（与 KUPL 侧同一口径）：契约外 dtype 只能改代码；契约内
    但探测没过的 dtype 重建原生归档即可恢复。
    """
    if dtype not in TOPOLOGY_DTYPES:
        declared = ", ".join(str(item) for item in TOPOLOGY_DTYPES)
        return (f"dtype {dtype} is outside the topology dtype contract "
                f"(declared: {declared}); no implicit downcast is performed")
    if _topology_dtype_enabled(dtype):
        return None
    return (f"dtype {dtype} is inside the topology dtype contract but its native "
            f"probe failed (2x2 matmul disagreed with numpy); falling back to the "
            f"generic path with no implicit downcast. Rebuild the native archive "
            f"({_REBUILD_HINT}) to re-enable it")


def _topology_key(subscripts: str, operands: Sequence[Any]) -> tuple[Any, ...]:
    return (
        ''.join(subscripts.split()),
        tuple(tuple(int(extent) for extent in operand.shape) for operand in operands),
        operands[0].dtype.str,
    )


def _descriptor_words(
        signature: tuple, shapes: Sequence[Sequence[int]], dtype: Any) -> list[int]:
    """把规范签名 + 各操作数形状编码成描述符的 int32 字序列。

    签名的编号口径由 ``_specialize.canonical_signature`` 保证（全式首次出现顺序，
    与字母无关），因此描述符里没有任何方程字符串，只有整数标签。
    """
    from ._native_plan import dtype_id as native_dtype_id

    *terms, output = signature
    labels_seen = {label for term in terms for label in term} | set(output)
    if any(label < 0 for label in labels_seen):
        raise ValueError("broadcast (ellipsis) labels are not supported by the "
                         "topology descriptor")
    num_labels = max(labels_seen) + 1 if labels_seen else 0
    if len(terms) != len(shapes):
        raise ValueError(
            f"equation has {len(terms)} operands but {len(shapes)} shapes were given")
    words = [
        _DESCRIPTOR_MAGIC,
        _DESCRIPTOR_VERSION,
        0,                                  # total_bytes，最后回填
        int(native_dtype_id(dtype)),
        len(terms),
        num_labels,
        len(output),
        0,                                  # reserved
    ]
    from . import _specialize

    # 索引侧摘要已截到 96 位（源码更小、冷启动更快），而 CUDA 描述符固定按
    # 128 位（4×int32）打包，因此这里零填充到 16 字节。零填充是单射，
    # 不影响唯一性；C++ 侧格式不变。
    digest = bytes.fromhex(_specialize._signature_digest(signature)).ljust(16, b"\0")
    if len(digest) != 16:
        raise ValueError("topology digest must be 128 bits")
    words.extend(int(value) for value in np.frombuffer(digest, dtype='<i4'))
    for labels, shape in zip(terms, shapes):
        extents = [int(extent) for extent in shape]
        if len(extents) != len(labels):
            raise ValueError(
                f"operand rank {len(extents)} does not match term {labels}")
        if any(extent <= 0 or extent > _DESCRIPTOR_MAX_EXTENT for extent in extents):
            raise ValueError(f"operand shape {tuple(extents)} is out of range")
        words.append(len(labels))
        words.extend(int(label) for label in labels)
        words.extend(extents)
    words.extend(int(label) for label in output)
    words[2] = 4 * len(words)
    return words


def _topology_descriptor(subscripts: str, shapes: Sequence[Sequence[int]],
                         dtype: Any) -> bytes:
    """拓扑描述符字节流：结构 + 签名摘要，不含收缩顺序（顺序在原生步表里）。"""
    from . import _specialize

    signature = _specialize.canonical_signature(subscripts)
    if signature is None:
        raise ValueError(f"equation has no canonical topology signature: {subscripts!r}")
    words = _descriptor_words(signature, shapes, dtype)
    return np.asarray(words, dtype='<i4').tobytes()


def _topology_plan(key: tuple[Any, ...], subscripts: str,
                   operands: Sequence[Any]) -> bytes:
    """返回描述符字节流；首次遇到 (方程, 形状, dtype) 时构造，之后直接复用。"""
    global _TOPOLOGY_CACHE_BYTES
    with _TOPOLOGY_LOCK:
        plan = _TOPOLOGY_PLAN_CACHE.get(key)
        if plan is not None:
            _TOPOLOGY_PLAN_CACHE.move_to_end(key)      # 真 LRU：命中即刷新
            return plan
    plan = _topology_descriptor(
        subscripts, [tuple(int(extent) for extent in operand.shape)
                     for operand in operands], operands[0].dtype)
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
                       offsets: Sequence[int] | None = None) -> Any:
    """把整条收缩计划一次性交给 CUDA 拓扑入口；不满足条件即返回 ``None`` 回落。

    与 ``try_dedicated`` 同一契约：命中且执行成功才返回
    ``cupy`` 数组，否则记录一次 ``LAST_FAILURE`` 并让上层走通用路径。

    dtype 契约（见 ``TOPOLOGY_DTYPES`` / ``topology_dtypes``）：只下发契约内
    （四类、平权）且原生 2x2 实测通过的 dtype。未启用的 dtype 直接返回
    ``None``，**绝不把 float64/complex128 转成单精度**，也绝不抛异常。

    ``offsets`` 只能是 ``None``：``cupy`` 视图的 ``data.ptr`` 已经带着字节偏移，
    再叠加 offsets 会双重偏移、静默读到错位数据，因此非 None 一律 fail closed。
    """
    LAST_INFO.clear()
    if not enabled() or not isinstance(subscripts, str):
        return _fail("kernel disabled (PYQCD_EINSUM_KERNEL_CUDA = 0, or library missing)")
    operands = list(operands)
    if not operands:
        return _fail("no operands")
    try:
        import cupy
    except ImportError:
        return _fail("cupy is not installed")
    if any(not isinstance(operand, cupy.ndarray) for operand in operands):
        return _fail("operands must be cupy arrays")
    dtype = operands[0].dtype
    if any(operand.dtype != dtype for operand in operands[1:]):
        return _fail("mixed operand dtypes are not supported by the native path")
    # 先判库、再判 dtype：库缺失/缺符号的原因是"没有原生入口"，不该报成 dtype 问题。
    from .cuda._program import topology_apply as native_topology_apply
    from .cuda._program import topology_binding

    if topology_binding() is None:
        return _fail("qcd_cuda_topology_apply is not exported by the archive")
    dtype_reason = _topology_dtype_reason(np.dtype(dtype))
    if dtype_reason is not None:
        return _fail(dtype_reason)
    if out is not None:
        if not isinstance(out, cupy.ndarray):
            return _fail("out must be a cupy array")
        if out.dtype != dtype:
            return _fail(f"out dtype {out.dtype} does not match operand dtype {dtype}")
        if not out.flags['C_CONTIGUOUS']:
            return _fail("out buffer must be C-contiguous")
        if int(out.device.id) != int(operands[0].device.id):
            return _fail("out must live on the same CUDA device as the operands")
    if offsets is not None:
        return _fail(
            "operand_offsets must stay unused: a cupy view's data pointer already "
            "carries the byte offset, so a non-zero offset would double it")
    from . import _kernel_kupl

    shapes = tuple(tuple(int(extent) for extent in operand.shape)
                   for operand in operands)
    negative_key = _kernel_kupl._negative_cache_key(
        subscripts, shapes, np.dtype(dtype), "topology", "cuda_topology")
    cached_unsupported = _kernel_kupl._cached_unsupported(negative_key)
    if cached_unsupported is not None:
        return _fail(*cached_unsupported)
    try:
        plan = _topology_plan(_topology_key(subscripts, operands), subscripts, operands)
        result = native_topology_apply(plan, operands, out=out, info=LAST_INFO)
    except Exception as error:  # noqa: BLE001 - 失败即回落，绝不抛给调用方
        LAST_INFO.clear()
        status = getattr(error, "status", None)
        reason = f"topology execution unavailable: {type(error).__name__}: {error}"
        if status == 4:
            _kernel_kupl._negative_cache_store(negative_key, reason, 4)
        return _fail(reason, status)
    LAST_INFO["path"] = "cuda_topology"
    return result


# ── 固定步表专用入口（cuda_dedicated） ───────────────────────────────────────
#
# 与拓扑入口（``try_topology_apply``）的区别：拓扑入口把整条 version-2 计划 /
# 拓扑描述符交给原生、由原生按 label 还原轴序；这里把一次 greedy 收缩**展开成
# GEMM 步表**（槽位 + 轴序 + batch/M/N/K），原生只做 pack → cuBLAS GEMM →
# unpack，运行期没有符号解析、没有 path 搜索、也没有逐步骤的 Python 调度。
#
# 步表格式（v1）与生成端都**复用 KUPL 侧**：格式由
# ``cpp/kupl/einsum/include/dedicated_apply.h`` 固定，编码只在
# ``_kernel_kupl._dedicated_plan`` 实现一次（纯 Python，不需要 KUPL 原生库）。
# 两个后端吃同一份字节，因此槽位/轴序/可行性判定逐位一致。
#
# 精度契约：接受 ``float32`` / ``float64`` / ``complex64`` / ``complex128``（原生分别
# 走 cublasSgemm / Dgemm / Cgemm / Zgemm；复数元素按 cupy 的交错实虚布局直传）。
# 白名单由原生 ``qcd_cuda_dedicated_dtypes`` 的运行期能力掩码裁剪（见
# ``dedicated_dtypes``）：本机缺某个 cuBLAS 符号时该 dtype 直接不进白名单。
# 其它 dtype / 未启用的 dtype 一律返回 ``None`` 并记录原因 —— **绝不隐式降精度**，
# 也绝不抛异常。

_DEDICATED_DECLARED_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.complex128), np.dtype(np.complex64),
    np.dtype(np.float64), np.dtype(np.float32))
# 与 _kernel_kupl._DEDICATED_DTYPE_IDS 逐项相同（两个后端吃同一份步表字节）：
# complex128=0, complex64=1, float64=2, float32=3。
_DEDICATED_DTYPE_IDS = {
    np.dtype(np.complex128): 0, np.dtype(np.complex64): 1,
    np.dtype(np.float64): 2, np.dtype(np.float32): 3,
}
_CUDA_PLAN_ARENA_LIMIT = 3 << 30
_CUDA_PLAN_MAX_STEP_MACS_LIMIT = 40_000_000_000
_CUDA_HEAVY_PACK_MIN_BYTES = 1 << 30
_CUDA_HEAVY_PACK_MAX_FLOPS = 1_000_000_000
_entry_dedicated: Any = None
_failed_dedicated = False
_dedicated_dtypes: tuple[np.dtype, ...] | None = None
_dedicated_dtypes_lock = threading.Lock()


def dedicated_dtypes() -> tuple[np.dtype, ...]:
    """本入口在**本机**可执行的 dtype（首次调用时向原生查一次能力掩码）。

    白名单不是写死的常量：原生 ``qcd_cuda_dedicated_dtypes`` 按运行期 cuBLAS 符号
    是否齐备返回 bit 掩码，这里据此裁剪。缺符号的 dtype 不进白名单，
    ``try_dedicated`` 对它返回 ``None`` 并说明原因（fail open，回落通用路径）。
    库缺失或查询入口缺失时返回空元组 —— 此时本入口本来就不可用。
    """
    global _dedicated_dtypes
    if _dedicated_dtypes is not None:
        return _dedicated_dtypes
    with _dedicated_dtypes_lock:
        if _dedicated_dtypes is None:
            _dedicated_dtypes = _query_dedicated_dtypes()
    return _dedicated_dtypes


def _plan_fallback_reason(dtype: Any, stats: dict[str, int]) -> str | None:
    """Fail open for plans outside the measured A100 dedicated budget."""
    arena_bytes = int(stats.get("plan_arena_bytes", 0))
    max_step_macs = int(stats.get("plan_max_step_macs", 0))
    flops = int(stats.get("plan_flops", 0))
    pack_bytes = int(stats.get("plan_pack_bytes", 0))
    if arena_bytes >= _CUDA_PLAN_ARENA_LIMIT:
        return (
            "plan arena exceeds the measured CUDA dedicated budget "
            f"({arena_bytes} >= {_CUDA_PLAN_ARENA_LIMIT} bytes)")
    if max_step_macs >= _CUDA_PLAN_MAX_STEP_MACS_LIMIT:
        return (
            "single GEMM exceeds the measured CUDA dedicated budget "
            f"({max_step_macs} >= {_CUDA_PLAN_MAX_STEP_MACS_LIMIT} MACs)")
    if pack_bytes >= _CUDA_HEAVY_PACK_MIN_BYTES and flops <= _CUDA_HEAVY_PACK_MAX_FLOPS:
        return (
            "plan is pack-bound on CUDA "
            f"({pack_bytes} pack bytes, {flops} FLOPs)")
    return None


def _query_dedicated_dtypes() -> tuple[np.dtype, ...]:
    try:
        from ._native import _cuda_library_path
        from .cuda._program import _bind_host_cublas

        # 先绑宿主自己的 libcublas，避免归档的绝对 RUNPATH 拉进第二份（见
        # ``cuda._program._bind_host_cublas`` 的完整说明）。
        _bind_host_cublas()
        library = ctypes.CDLL(str(_cuda_library_path()))
        query = library.qcd_cuda_dedicated_dtypes
    except Exception:  # noqa: BLE001 - 缺库/缺符号即视为入口不可用
        return ()
    query.argtypes = []
    query.restype = ctypes.c_int
    try:
        mask = int(query())
    except Exception:  # noqa: BLE001
        return ()
    return tuple(dtype for dtype in _DEDICATED_DECLARED_DTYPES
                 if mask & (1 << _DEDICATED_DTYPE_IDS[dtype]))


# 与 KUPL 侧同因：dlopen + argtypes 必须是幂等单次的（见 _kernel_kupl._ENTRY_LOCK）。
_ENTRY_LOCK = threading.Lock()


def _function_dedicated():
    global _entry_dedicated, _failed_dedicated
    if _entry_dedicated is not None or _failed_dedicated:
        return _entry_dedicated
    with _ENTRY_LOCK:
        return _function_dedicated_locked()


def _function_dedicated_locked():
    global _entry_dedicated, _failed_dedicated
    if _entry_dedicated is not None or _failed_dedicated:
        return _entry_dedicated
    try:
        from ._native import _cuda_library_path
        from .cuda._program import _bind_host_cublas

        _bind_host_cublas()
        library = ctypes.CDLL(str(_cuda_library_path()))
        entry = library.qcd_cuda_dedicated_apply
    except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
        _failed_dedicated = True
        return None
    entry.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                      ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    entry.restype = ctypes.c_int
    _entry_dedicated = entry
    return _entry_dedicated


def try_dedicated(subscripts: Any, operands: Sequence[Any], out: Any = None,
                  *, optimize: str = "auto") -> Any:
    """把一次多算子收缩按下发前的固定步表交给 CUDA 原生执行；否则返回 ``None``。

    与 ``try_topology_apply`` 同一契约（fail-open、失败记录 ``LAST_FAILURE``、
    绝不抛异常），区别只在原生吃的是**已展开的 GEMM 步表**：Python 只付一次
    规划 + 编码成本（与 KUPL 侧共用 ``_kernel_kupl`` 的缓存），执行期没有逐步骤
    的 Python 调度。

    ``optimize`` 默认 ``"auto"``，与 ``_kernel_kupl.try_dedicated`` 及
    ``_dedicated_plan`` 的文档口径一致：在 auto / greedy 两条计划里按步表 FLOPs
    择优。默认值曾是 ``"greedy"``，实测语料 8%（dim=8 抽 100 条）的式子因此多做
    1.05–4.48× 乘加（全表 1.86×），且与 KUPL 侧缓存键错开、规划要跑两遍。

    dtype 契约：``float32`` / ``float64`` / ``complex64`` / ``complex128``（原生分别
    走 cublasSgemm / Dgemm / Cgemm / Zgemm）。其余 dtype 直接返回 ``None`` 并记录
    原因 —— **不做隐式降精度**。
    """
    LAST_INFO.clear()
    if not enabled() or not isinstance(subscripts, str):
        return _fail("kernel disabled (PYQCD_EINSUM_KERNEL_CUDA = 0, or library missing)")
    operands = list(operands)
    if not operands:
        return _fail("no operands")
    try:
        import cupy
    except ImportError:
        return _fail("cupy is not installed")
    if any(not isinstance(operand, cupy.ndarray) for operand in operands):
        return _fail("operands must be cupy arrays")
    dtype = np.dtype(operands[0].dtype)
    if any(np.dtype(operand.dtype) != dtype for operand in operands[1:]):
        return _fail("mixed operand dtypes are not supported by the native path")
    from . import _kernel_kupl

    optimize = _kernel_kupl._effective_dedicated_optimizer(optimize)
    shapes = tuple(tuple(int(extent) for extent in operand.shape) for operand in operands)
    if any(not operand.flags['C_CONTIGUOUS'] for operand in operands):
        return _fail("dedicated kernel requires C-contiguous operands")
    negative_key = _kernel_kupl._negative_cache_key(
        subscripts, shapes, dtype, optimize, "cuda_dedicated")
    cached_unsupported = _kernel_kupl._cached_unsupported(negative_key)
    if cached_unsupported is not None:
        return _fail(*cached_unsupported)
    supported_dtypes = dedicated_dtypes()
    if dtype not in supported_dtypes:
        supported = ", ".join(str(item) for item in supported_dtypes) or "none"
        reason = (
            f"dtype {dtype} has no dedicated step-table path on this machine "
            f"(supported: {supported}); no implicit downcast is performed")
        _kernel_kupl._negative_cache_store(negative_key, reason, 4)
        return _fail(reason, 4)
    entry = _function_dedicated()
    if entry is None:
        return _fail("qcd_cuda_dedicated_apply is not exported by the archive")

    # 步表生成端只有一份（KUPL 侧纯 Python 编码器），这里直接复用它的缓存。
    try:
        cached = _kernel_kupl._dedicated_cached(
            subscripts, shapes, dtype, optimize, "cuda")
    except Exception as error:  # noqa: BLE001 - 规划失败也一律回落，绝不抛给调用方
        return _fail(f"step-table planning failed: {type(error).__name__}: {error}")
    if cached is None:
        return _fail("the contraction path is not expressible as a fixed step table")
    # KUPL 侧缓存返回 4 元组 (plan, output_shape, plan_buffer, plan_flops)；
    # 按 3 元组解包会抛 ValueError（在 try 之外，不会被吞成回落）。
    plan, output_shape, plan_buffer, plan_flops = cached
    stats = _kernel_kupl._plan_statistics(plan)
    plan_gate = _plan_fallback_reason(dtype, stats)
    if plan_gate is not None:
        _kernel_kupl._negative_cache_store(negative_key, plan_gate, 4)
        return _fail(plan_gate, 4)

    if out is None:
        output = cupy.empty(output_shape, dtype=dtype)
    else:
        if not isinstance(out, cupy.ndarray):
            return _fail("out must be a cupy array")
        if np.dtype(out.dtype) != dtype:
            return _fail(f"out dtype {out.dtype} does not match operand dtype {dtype}")
        if tuple(output_shape) != tuple(out.shape):
            return _fail(f"out shape {out.shape} does not match plan shape {output_shape}")
        if not out.flags['C_CONTIGUOUS']:
            return _fail("out buffer must be C-contiguous")
        if int(out.device.id) != int(operands[0].device.id):
            return _fail("out must live on the same CUDA device as the operands")
        output = out

    pointers = (ctypes.c_void_p * len(operands))(
        *[int(operand.data.ptr) for operand in operands])
    try:
        status = int(entry(ctypes.addressof(plan_buffer), len(plan), pointers,
                           len(operands), int(output.data.ptr),
                           _DEDICATED_DTYPE_IDS[dtype], None))
    except Exception as error:  # noqa: BLE001 - 失败即回落，绝不抛给调用方
        status = getattr(error, "status", None)
        reason = f"dedicated execution unavailable: {type(error).__name__}: {error}"
        if status == 4:
            _kernel_kupl._negative_cache_store(negative_key, reason, 4)
        return _fail(reason, status)
    if status != 0:
        reason = ("the step table or this machine's cuBLAS capability is outside "
                  "the CUDA executor's range" if status == 4
                  else "invalid arguments or an internal error")
        full_reason = f"dedicated kernel returned status {status} ({reason})"
        if status == 4:
            _kernel_kupl._negative_cache_store(negative_key, full_reason, 4)
        return _fail(full_reason, status)
    LAST_INFO.update(
        path="cuda_dedicated",
        equation=''.join(subscripts.split()),
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
