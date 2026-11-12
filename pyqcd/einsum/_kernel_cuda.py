"""CUDA 专用 kernel 的绑定与形态判定（15 算子 3pt，P1/P2）。

契约见 `cpp/cuda/einsum/include/pattern_3pt.h`：
- 只支持 complex128；
- `dims` 长度 5：``[nev, mom, link, spin(4), proj(2)]``；
- 操作数与输出都是**设备**指针，必须 C 连续；
- 输出形状 ``(mom, mom, 4, link, 2, 2)``。

实测（2026-11-08，单卡 RTX 4060，nvec=50，10 次中位）：
kernel 30.74 ms vs 通用 CUDA 路径 74.13 ms（**2.41x**）vs opt_einsum 285.40 ms（9.29x），
数值 ``rel = 3.13e-15``。

另有一个**多操作数拓扑入口**（``try_topology_apply``），语义与 KUPL 侧同构：
按 ``TOPOLOGY_DTYPES`` 显式声明 dtype（单精度默认与首选，双精度须原生实测通过），
未启用的 dtype 返回 ``None`` 回落且**绝不降级**；成功下发时把 plan 字节数、
操作数个数、``out`` 模式与复制下标记进 ``LAST_INFO``。
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Any, Sequence

import numpy as np

QCD_DTYPE_COMPLEX128 = 0

THREE_PT_EQUATIONS = {
    'agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy': 0,
    'agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy': 1,
}

_DISABLE_ENV = 'PYQCD_EINSUM_KERNEL_CUDA'
_entry: Any = None
_failed = False


# 最近一次"该跑 kernel 却没跑"的原因，供上层给出可行动的告警。
LAST_FAILURE: dict[str, object] = {"reason": "", "status": None}


def _fail(reason: str, status=None):
    LAST_FAILURE["reason"] = reason
    LAST_FAILURE["status"] = status
    return None


def enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, '1').strip().lower() not in (
        '0', 'false', 'off', 'no')


NUCLEON_2PT_EQ = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'

# 标记：CUDA 2pt kernel 已接线，但与现有通用路径相比收益不稳定（见 try_nucleon_2pt
# 的 docstring 实测表），**需要进一步优化**。优化达标后应删除本标识。
NUCLEON_2PT_NEEDS_OPTIMIZATION = True
_nucleon_notice_shown = False

_entry_2pt: Any = None
_failed_2pt = False


def _function():
    global _entry, _failed
    if _entry is not None or _failed:
        return _entry
    try:
        from ._native import _cuda_library_path

        library = ctypes.CDLL(str(_cuda_library_path()))
        entry = library.qcd_cuda_pattern_3pt
    except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
        _failed = True
        return None
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                      ctypes.c_int, ctypes.c_void_p]
    entry.restype = ctypes.c_int
    _entry = entry
    return _entry


def _shape_ok(operands: Sequence[Any]) -> tuple[int, int, int] | None:
    """返回 ``(nvec, mom, link)``，不满足 P1/P2 契约时返回 ``None``。"""
    try:
        import cupy
    except ImportError:
        return None
    if len(operands) != 15:
        return None
    for operand in operands:
        if not isinstance(operand, cupy.ndarray):
            return None
        if operand.dtype != cupy.complex128:
            return None
        if not operand.flags['C_CONTIGUOUS']:
            return None
    perams = operands[:5]
    nvec = perams[0].shape[2] if perams[0].ndim == 4 else 0
    if nvec < 1 or any(peram.shape != (4, 4, nvec, nvec) for peram in perams):
        return None
    if operands[5].shape != (4, 4) or operands[6].shape != (4, 4, 4):
        return None
    if operands[7].shape != (4, 4) or operands[8].shape != (4, 4):
        return None
    sink, vdv, src_vdv, src_vvv = operands[9], operands[10], operands[11], operands[12]
    if sink.ndim != 4 or vdv.ndim != 4 or src_vdv.ndim != 3 or src_vvv.ndim != 4:
        return None
    mom = sink.shape[0]
    link = vdv.shape[1]
    if mom < 1 or link < 1:
        return None
    if sink.shape != (mom, nvec, nvec, nvec):
        return None
    if vdv.shape != (mom, link, nvec, nvec):
        return None
    if src_vdv.shape != (mom, nvec, nvec):
        return None
    if src_vvv.shape != (mom, nvec, nvec, nvec):
        return None
    if operands[13].shape != (4, 2) or operands[14].shape != (4, 2):
        return None
    return nvec, mom, link


def try_three_pt(subscripts: Any, operands: Sequence[Any]) -> Any | None:
    """匹配 P1/P2 且条件满足时用专用 kernel 计算，否则返回 ``None``。"""
    if not enabled() or not isinstance(subscripts, str):
        return _fail("kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)")
    variant = THREE_PT_EQUATIONS.get(''.join(subscripts.split()))
    if variant is None:
        return None
    shapes = _shape_ok(operands)
    if shapes is None:
        return _fail("operand shapes do not satisfy the kernel contract")
    entry = _function()
    if entry is None:
        return _fail("kernel entry point is missing from the native library")
    import cupy

    nvec, mom, link = shapes
    dims = (ctypes.c_int64 * 5)(nvec, mom, link, 4, 2)
    output = cupy.zeros((mom, mom, 4, link, 2, 2), dtype=cupy.complex128)
    pointers = (ctypes.c_void_p * len(operands))(
        *[int(operand.data.ptr) for operand in operands])
    status = entry(QCD_DTYPE_COMPLEX128, pointers, int(output.data.ptr), dims,
                   variant, None)
    if status != 0:
        return _fail(f"kernel returned status {status} "
                     "(wrong CUDA architecture or an internal error)", status)
    return output


def _function_2pt():
    global _entry_2pt, _failed_2pt
    if _entry_2pt is not None or _failed_2pt:
        return _entry_2pt
    try:
        from ._native import _cuda_library_path

        library = ctypes.CDLL(str(_cuda_library_path()))
        entry = library.qcd_cuda_nucleon_2pt
    except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
        _failed_2pt = True
        return None
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                      ctypes.c_void_p]
    entry.restype = ctypes.c_int
    _entry_2pt = entry
    return _entry_2pt


def _nucleon_shape_ok(operands: Sequence[Any]) -> tuple[int, int] | None:
    """返回 ``(nvec, nmom)``；不满足 CUDA 2pt 契约时返回 ``None``。"""
    try:
        import cupy
    except ImportError:
        return None
    if len(operands) != 8:
        return None
    for operand in operands:
        if not isinstance(operand, cupy.ndarray):
            return None
        if operand.dtype != cupy.complex128:
            return None
        # 设备端不做 stride 解释：非连续输入必须回退，否则会按连续内存错误读取。
        if not operand.flags['C_CONTIGUOUS']:
            return None
    perams = operands[:3]
    if any(peram.ndim != 4 or peram.shape[:2] != (4, 4) for peram in perams):
        return None
    nvec = perams[0].shape[2]
    if nvec < 1 or any(peram.shape != (4, 4, nvec, nvec) for peram in perams):
        return None
    for gamma in (operands[3], operands[4], operands[7]):
        if gamma.shape != (4, 4):
            return None
    vvv_a, vvv_b = operands[5], operands[6]
    if vvv_a.ndim != 4 or vvv_b.ndim != 4:
        return None
    nmom = vvv_a.shape[0]
    if nmom < 1 or vvv_a.shape != (nmom, nvec, nvec, nvec):
        return None
    if vvv_b.shape != (nmom, nvec, nvec, nvec):
        return None
    return nvec, nmom


def try_nucleon_2pt(subscripts: Any, operands: Sequence[Any]) -> Any | None:
    """匹配 E1 且条件满足时用 CUDA 专用 kernel 计算，否则返回 ``None``。

    **已接线，但标记为 NEEDS-OPTIMIZATION**（``NUCLEON_2PT_NEEDS_OPTIMIZATION``）。
    生产库实测（单卡 RTX 4060，10 次中位）：

    | Nvec | kernel | 通用路径 | opt_einsum | kernel/通用 |
    |---|---|---|---|---|
    | 50 | 86.5 ms | 74.0–89.3 ms | 88.9–151.7 ms | 0.86–1.05x |
    | 100 | 415.3 ms | 392.2–486.0 ms | 380.6–3083 ms | 0.91–1.17x |

    通用路径本身在同机多次测量间波动达 ±20%，故 kernel 相对收益不稳定；
    与 opt_einsum 相比通常领先（1.75x–7.4x）。按契约仍需继续优化。
    """
    if not enabled() or not isinstance(subscripts, str):
        return _fail("kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)")
    if ''.join(subscripts.split()) != NUCLEON_2PT_EQ:
        return None
    shapes = _nucleon_shape_ok(operands)
    if shapes is None:
        return _fail("operand shapes do not satisfy the kernel contract")
    entry = _function_2pt()
    if entry is None:
        return _fail("kernel entry point is missing from the native library")
    import cupy

    nvec, nmom = shapes
    # 契约顺序：{P1,P2,P3,G1,G2,pr,V1,V2}（投影在 V 之前）
    ordered = [operands[0], operands[1], operands[2], operands[3],
               operands[4], operands[7], operands[5], operands[6]]
    # dims(10) = [N, mx, my, mode, permA..permV2, matflags]，恒等 perm 码 = 123
    dims = (ctypes.c_int64 * 10)(nvec, nmom, nmom, 0, 123, 123, 123, 123, 123, 0)
    output = cupy.zeros((nmom,), dtype=cupy.complex128)
    pointers = (ctypes.c_void_p * len(ordered))(
        *[int(operand.data.ptr) for operand in ordered])
    status = entry(QCD_DTYPE_COMPLEX128, pointers, int(output.data.ptr), dims,
                   None)
    if status != 0:
        return _fail(f"kernel returned status {status} "
                     "(wrong CUDA architecture or an internal error)", status)
    global _nucleon_notice_shown
    if not _nucleon_notice_shown:
        _nucleon_notice_shown = True
        print('[pyqcd.einsum] 提示：CUDA 2pt 专用 kernel 已接线，但与通用路径相比'
              '收益不稳定（0.86–1.17x），标记 needs-optimization', flush=True)
    return output


# 多操作数拓扑入口：一次原生调用跑完整条计划。下发的是**拓扑描述符**（只带
# 结构：每个操作数的标签与形状、输出标签、签名摘要），收缩顺序来自原生库里的
# 预编译步表 `cpp/cuda/einsum/src/topology_steps.h` —— 这里不再做 path 搜索，
# 也不再逐步骤调度。描述符按 (方程, 各操作数形状, dtype) 缓存，重复调用零规划。
_TOPOLOGY_CACHE_LIMIT = 32
_TOPOLOGY_PLAN_CACHE: dict[tuple[Any, ...], bytes] = {}
_TOPOLOGY_LOCK = threading.Lock()

# 描述符字节流的魔数与版本，必须与 `cpp/cuda/einsum/src/apply_contract.cu` 同步。
_DESCRIPTOR_MAGIC = 0x51435444
_DESCRIPTOR_VERSION = 1
_DESCRIPTOR_HEADER_WORDS = 8
_DESCRIPTOR_MAX_EXTENT = 2 ** 31 - 1

# ── dtype 契约 ──────────────────────────────────────────────────────────────
# 入口只下发这里**显式声明**的 dtype。单精度是默认与首选，恒可下发。
TOPOLOGY_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.complex64), np.dtype(np.float32))
# 双精度只在原生**实测通过**后才启用：设备侧分步执行器尚未实例化双精度数值路径
# 时本入口返回 None 让上层回落通用路径 —— 绝不把 float64/complex128 降级成单精度
# （那会静默丢精度），也绝不抛异常。
TOPOLOGY_DOUBLE_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.complex128), np.dtype(np.float64))
_TOPOLOGY_DOUBLE_SUPPORT: dict[np.dtype, bool] = {}
_TOPOLOGY_PROBE_LOCK = threading.Lock()

# 最近一次成功下发时实际交给原生的参数（见 `_kernel_kupl.LAST_INFO`）。
LAST_INFO: dict[str, object] = {}


def _probe_double(dtype: np.dtype) -> bool:
    """实测设备侧拓扑入口能否**正确**执行 ``dtype``（每进程每 dtype 一次）。"""
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
    try:
        _, path_info = contract_path("ij,jk->ik", left, right)
        plan = encode_execution_plan(path_info.contraction_list, (left, right))
        actual = cupy.asnumpy(native_topology_apply(plan, (left, right)))
    except Exception:  # noqa: BLE001 - 探测失败 = 双精度未实现，一律回落
        return False
    expected = np.einsum(
        "ij,jk->ik", cupy.asnumpy(left), cupy.asnumpy(right))
    if actual.shape != expected.shape or actual.dtype != dtype:
        return False
    scale = float(np.max(np.abs(expected))) or 1.0
    return bool(float(np.max(np.abs(actual - expected))) <= 1e-9 * scale)


def topology_dtypes() -> tuple[np.dtype, ...]:
    """当前进程里拓扑入口可执行的 dtype（只读，不触发探测）。"""
    return TOPOLOGY_DTYPES + tuple(
        dtype for dtype in TOPOLOGY_DOUBLE_DTYPES
        if _TOPOLOGY_DOUBLE_SUPPORT.get(dtype) is True)


def _topology_dtype_enabled(dtype: np.dtype) -> bool:
    if dtype in TOPOLOGY_DTYPES:
        return True
    if dtype not in TOPOLOGY_DOUBLE_DTYPES:
        return False
    support = _TOPOLOGY_DOUBLE_SUPPORT.get(dtype)
    if support is None:
        with _TOPOLOGY_PROBE_LOCK:
            support = _TOPOLOGY_DOUBLE_SUPPORT.get(dtype)
            if support is None:
                support = _probe_double(dtype)
                _TOPOLOGY_DOUBLE_SUPPORT[dtype] = support
    return support


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
    with _TOPOLOGY_LOCK:
        plan = _TOPOLOGY_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    plan = _topology_descriptor(
        subscripts, [tuple(int(extent) for extent in operand.shape)
                     for operand in operands], operands[0].dtype)
    with _TOPOLOGY_LOCK:
        if len(_TOPOLOGY_PLAN_CACHE) >= _TOPOLOGY_CACHE_LIMIT:
            _TOPOLOGY_PLAN_CACHE.pop(next(iter(_TOPOLOGY_PLAN_CACHE)))
        _TOPOLOGY_PLAN_CACHE[key] = plan
    return plan


def try_topology_apply(subscripts: Any, operands: Sequence[Any],
                       out: Any = None,
                       offsets: Sequence[int] | None = None) -> Any:
    """把整条收缩计划一次性交给 CUDA 拓扑入口；不满足条件即返回 ``None`` 回落。

    与 ``try_three_pt`` / ``try_nucleon_2pt`` 同一契约：命中且执行成功才返回
    ``cupy`` 数组，否则记录一次 ``LAST_FAILURE`` 并让上层走通用路径。

    dtype 契约（见 ``TOPOLOGY_DTYPES`` / ``topology_dtypes``）：只下发显式声明
    的 dtype——单精度恒可下发，双精度需原生实测通过。未声明的 dtype 直接返回
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
    if not _topology_dtype_enabled(np.dtype(dtype)):
        supported = ", ".join(str(item) for item in topology_dtypes())
        return _fail(
            f"dtype {np.dtype(dtype)} is not enabled for the topology entry "
            f"(supported: {supported}); no implicit downcast is performed")
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
    try:
        plan = _topology_plan(_topology_key(subscripts, operands), subscripts, operands)
        result = native_topology_apply(plan, operands, out=out, info=LAST_INFO)
    except Exception as error:  # noqa: BLE001 - 失败即回落，绝不抛给调用方
        LAST_INFO.clear()
        return _fail(f"topology execution unavailable: {type(error).__name__}: {error}",
                     getattr(error, "status", None))
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
# 其它 dtype 一律返回 ``None`` 并记录原因 —— **绝不隐式降精度**，也绝不抛异常。

_DEDICATED_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.complex128), np.dtype(np.complex64),
    np.dtype(np.float64), np.dtype(np.float32))
# 与 _kernel_kupl._DEDICATED_DTYPE_IDS 逐项相同（两个后端吃同一份步表字节）：
# complex128=0, complex64=1, float64=2, float32=3。
_DEDICATED_DTYPE_IDS = {
    np.dtype(np.complex128): 0, np.dtype(np.complex64): 1,
    np.dtype(np.float64): 2, np.dtype(np.float32): 3,
}
_entry_dedicated: Any = None
_failed_dedicated = False


def dedicated_dtypes() -> tuple[np.dtype, ...]:
    """本入口可执行的 dtype（只读，不触发加载）。"""
    return _DEDICATED_DTYPES


def _function_dedicated():
    global _entry_dedicated, _failed_dedicated
    if _entry_dedicated is not None or _failed_dedicated:
        return _entry_dedicated
    try:
        from ._native import _cuda_library_path

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
    if dtype not in _DEDICATED_DTYPES:
        supported = ", ".join(str(item) for item in _DEDICATED_DTYPES)
        return _fail(
            f"dtype {dtype} has no dedicated step-table path "
            f"(supported: {supported}); no implicit downcast is performed")
    if any(not operand.flags['C_CONTIGUOUS'] for operand in operands):
        return _fail("dedicated kernel requires C-contiguous operands")
    entry = _function_dedicated()
    if entry is None:
        return _fail("qcd_cuda_dedicated_apply is not exported by the archive")

    shapes = tuple(tuple(int(extent) for extent in operand.shape) for operand in operands)
    # 步表生成端只有一份（KUPL 侧纯 Python 编码器），这里直接复用它的缓存。
    try:
        from . import _kernel_kupl

        cached = _kernel_kupl._dedicated_cached(subscripts, shapes, dtype, optimize)
    except Exception as error:  # noqa: BLE001 - 规划失败也一律回落，绝不抛给调用方
        return _fail(f"step-table planning failed: {type(error).__name__}: {error}")
    if cached is None:
        return _fail("the contraction path is not expressible as a fixed step table")
    # KUPL 侧缓存返回 4 元组 (plan, output_shape, plan_buffer, plan_flops)；
    # 按 3 元组解包会抛 ValueError（在 try 之外，不会被吞成回落）。
    plan, output_shape, plan_buffer, plan_flops = cached

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
        return _fail(f"dedicated execution unavailable: {type(error).__name__}: {error}",
                     getattr(error, "status", None))
    if status != 0:
        reason = ("the step table is outside the CUDA executor's range"
                  if status == 4 else "invalid arguments or an internal error")
        return _fail(f"dedicated kernel returned status {status} ({reason})", status)
    LAST_INFO.update(
        path="cuda_dedicated",
        equation=''.join(subscripts.split()),
        dtype=str(dtype),
        dtype_id=_DEDICATED_DTYPE_IDS[dtype],
        n_operands=len(operands),
        plan_bytes=len(plan),
        plan_flops=plan_flops,
        optimize=optimize,
        output_shape=tuple(output_shape),
        out="allocated" if out is None else "provided",
    )
    return output
