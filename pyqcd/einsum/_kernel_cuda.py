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


def _function():
    global _entry, _failed
    if _entry is not None or _failed:
        return _entry
    try:
        from .cuda import _library_path

        library = ctypes.CDLL(str(_library_path()))
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
