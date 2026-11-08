"""CUDA 专用 kernel 的绑定与形态判定（15 算子 3pt，P1/P2）。

契约见 `cpp/cuda/einsum/include/pattern_3pt.h`：
- 只支持 complex128；
- `dims` 长度 5：``[nev, mom, link, spin(4), proj(2)]``；
- 操作数与输出都是**设备**指针，必须 C 连续；
- 输出形状 ``(mom, mom, 4, link, 2, 2)``。

实测（2026-11-08，单卡 RTX 4060，nvec=50，10 次中位）：
kernel 30.74 ms vs 通用 CUDA 路径 74.13 ms（**2.41x**）vs opt_einsum 285.40 ms（9.29x），
数值 ``rel = 3.13e-15``。
"""

from __future__ import annotations

import ctypes
import os
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


def enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, '1').strip().lower() not in (
        '0', 'false', 'off', 'no')


NUCLEON_2PT_EQ = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'

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
        return None
    variant = THREE_PT_EQUATIONS.get(''.join(subscripts.split()))
    if variant is None:
        return None
    shapes = _shape_ok(operands)
    if shapes is None:
        return None
    entry = _function()
    if entry is None:
        return None
    import cupy

    nvec, mom, link = shapes
    dims = (ctypes.c_int64 * 5)(nvec, mom, link, 4, 2)
    output = cupy.zeros((mom, mom, 4, link, 2, 2), dtype=cupy.complex128)
    pointers = (ctypes.c_void_p * len(operands))(
        *[int(operand.data.ptr) for operand in operands])
    status = entry(QCD_DTYPE_COMPLEX128, pointers, int(output.data.ptr), dims,
                   variant, None)
    if status != 0:
        return None
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

    **当前未接线**：内核仍在优化中（N=100 时 415.19 ms vs 通用 419.40 ms，
    仅 1.01x；N=50 时 30.43 ms vs 25.05 ms，0.82x）。收敛后只需在
    ``_api.contract`` 加一行分派。
    """
    if not enabled() or not isinstance(subscripts, str):
        return None
    if ''.join(subscripts.split()) != NUCLEON_2PT_EQ:
        return None
    shapes = _nucleon_shape_ok(operands)
    if shapes is None:
        return None
    entry = _function_2pt()
    if entry is None:
        return None
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
        return None
    return output
