"""KUPL 专用 kernel 的 ctypes 绑定与形态判定。

当前只接一个形态：核子 2pt（E1）

    gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M

kernel 的契约由 `cpp/kupl/einsum/include/nucleon2pt.h` 定义；本模块负责
**严格校验**调用条件（下标、形状、dtype、连续性、backend），任一不满足即返回
``None`` 让上层回退通用路径，绝不猜测或放宽。

实测（2026-11-08，4 线程）：Nvec=100 时 kernel 464 ms vs 通用 653 ms（1.41x）
vs opt_einsum 3332 ms（7.17x）；数值 `rel = 3.85e-15`。
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Sequence

import numpy as np

from .kupl import einsum as kupl

QCD_DTYPE_COMPLEX128 = 0

NUCLEON_2PT_EQ = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'

THREE_PT_EQUATIONS = {
    'agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy': 0,
    'agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy': 1,
}

_DISABLE_ENV = 'PYQCD_EINSUM_KERNEL_KUPL'
_SMALL_OPERAND_ELEMENTS = 1024

# 标记：P1/P2 的 KUPL kernel 已接线，但实测与通用路径持平（0.94–1.01x），
# 且 N>=50 时相对 opt_einsum 慢约 1%，**需要进一步优化**。
THREE_PT_NEEDS_OPTIMIZATION = True
_three_pt_notice_shown = False
_entry: Any = None
_failed = False
_entry_3pt: Any = None
_failed_3pt = False


def enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, '1').strip().lower() not in (
        '0', 'false', 'off', 'no')


def _function():
    global _entry, _failed
    if _entry is not None or _failed:
        return _entry
    try:
        library = kupl._load_library()
        entry = library.qcd_kupl_nucleon_2pt
    except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
        _failed = True
        return None
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                      ctypes.c_int]
    entry.restype = ctypes.c_int
    _entry = entry
    return _entry


def _function_3pt():
    global _entry_3pt, _failed_3pt
    if _entry_3pt is not None or _failed_3pt:
        return _entry_3pt
    try:
        library = kupl._load_library()
        entry = library.qcd_kupl_pattern_3pt
    except Exception:  # noqa: BLE001 - 缺库/缺符号时静默回退
        _failed_3pt = True
        return None
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                      ctypes.c_int]
    entry.restype = ctypes.c_int
    _entry_3pt = entry
    return _entry_3pt


def _normalize(subscripts: str) -> str:
    return ''.join(subscripts.split())


def _shape_ok(operands: Sequence[Any]) -> tuple[int, int] | None:
    """返回 ``(nvec, nmom)``，不满足 E1 契约时返回 ``None``。"""
    if len(operands) != 8:
        return None
    for operand in operands:
        if type(operand).__module__.split('.')[0] != 'numpy':
            return None
        if not isinstance(operand, np.ndarray):
            return None
        if operand.dtype != np.complex128 or not operand.flags['C_CONTIGUOUS']:
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


def try_nucleon_2pt(subscripts: Any, operands: Sequence[Any]) -> np.ndarray | None:
    """匹配 E1 且条件满足时用专用 kernel 计算，否则返回 ``None``。"""
    if not enabled() or not isinstance(subscripts, str):
        return None
    if _normalize(subscripts) != NUCLEON_2PT_EQ:
        return None
    shapes = _shape_ok(operands)
    if shapes is None:
        return None
    entry = _function()
    if entry is None:
        return None
    nvec, nmom = shapes
    dims = (ctypes.c_int64 * 8)(nvec, nmom, 4, 0, 0, 0, 0, 0)
    output = np.zeros((nmom,), dtype=np.complex128)
    pointers = (ctypes.c_void_p * len(operands))(
        *[operand.ctypes.data for operand in operands])
    status = entry(QCD_DTYPE_COMPLEX128, pointers, output.ctypes.data, dims, 0)
    if status != 0:
        return None
    return output


def _three_pt_shape_ok(operands: Sequence[Any]) -> tuple[int, int, int, int] | None:
    """返回 ``(nvec, mom_source, mom_sink, link)``，不满足 P1/P2 契约时返回 ``None``。"""
    if len(operands) != 15:
        return None
    for operand in operands:
        if not isinstance(operand, np.ndarray):
            return None
        if operand.dtype != np.complex128:
            return None
        # 小数组（投影 4x2=8 元素等）允许非连续：由调用侧转连续，代价可忽略；
        # 大张量必须本身连续，避免静默整块拷贝。
        if operand.flags['C_CONTIGUOUS']:
            continue
        if operand.size > _SMALL_OPERAND_ELEMENTS:
            return None
    perams = operands[:5]
    if any(peram.ndim != 4 or peram.shape[:2] != (4, 4) for peram in perams):
        return None
    nvec = perams[0].shape[2]
    if nvec < 1 or any(peram.shape != (4, 4, nvec, nvec) for peram in perams):
        return None
    if operands[5].shape != (4, 4) or operands[7].shape != (4, 4):
        return None
    if operands[6].shape != (4, 4, 4) or operands[8].shape != (4, 4):
        return None
    sink, vdv, src_vdv, src_vvv = operands[9], operands[10], operands[11], operands[12]
    if sink.ndim != 4 or vdv.ndim != 4 or src_vdv.ndim != 3 or src_vvv.ndim != 4:
        return None
    mom_sink = sink.shape[0]
    mom_source = src_vdv.shape[0]
    link = vdv.shape[1]
    if mom_sink < 1 or mom_source < 1 or link < 1:
        return None
    if sink.shape != (mom_sink, nvec, nvec, nvec):
        return None
    if vdv.shape != (mom_sink, link, nvec, nvec):
        return None
    if src_vdv.shape != (mom_source, nvec, nvec):
        return None
    if src_vvv.shape != (mom_source, nvec, nvec, nvec):
        return None
    if operands[13].shape != (4, 2) or operands[14].shape != (4, 2):
        return None
    return nvec, mom_source, mom_sink, link


def try_three_pt(subscripts: Any, operands: Sequence[Any]) -> np.ndarray | None:
    """匹配 P1/P2 时用专用 kernel 计算，否则返回 ``None``。

    **已接线，但标记为 NEEDS-OPTIMIZATION**（见 ``THREE_PT_NEEDS_OPTIMIZATION``）。
    实测 2026-11-08、4 线程：

    | Nvec | kernel | 通用路径 | opt_einsum | kernel/通用 |
    |---|---|---|---|---|
    | 20 | 14.29 ms | 15.17 ms | 14.83 ms | 0.94x |
    | 50 | 138.23 ms | 136.71 ms | 137.47 ms | 1.01x |
    | 100 | 1837.88 ms | 1855.30 ms | 1825.43 ms | 0.99x |

    与现有通用路径基本持平，且 N>=50 时相对 opt_einsum 慢约 1%。
    按当前停用标准本不应启用，但按明确要求保持接线，并以该标识与一次性运行时
    提示标记"待进一步优化"；优化达标后应删除本标识。
    """
    if not enabled() or not isinstance(subscripts, str):
        return None
    variant = THREE_PT_EQUATIONS.get(_normalize(subscripts))
    if variant is None:
        return None
    shapes = _three_pt_shape_ok(operands)
    if shapes is None:
        return None
    entry = _function_3pt()
    if entry is None:
        return None
    nvec, mom_source, mom_sink, link = shapes
    operands = [operand if operand.flags['C_CONTIGUOUS']
                else np.ascontiguousarray(operand) for operand in operands]
    dims = (ctypes.c_int64 * 8)(nvec, mom_source, mom_sink, link, 4, 2, 0, 0)
    output = np.zeros((mom_source, mom_sink, 4, link, 2, 2),
                      dtype=np.complex128)
    pointers = (ctypes.c_void_p * len(operands))(
        *[operand.ctypes.data for operand in operands])
    status = entry(QCD_DTYPE_COMPLEX128, pointers, output.ctypes.data, dims,
                   variant)
    if status != 0:
        return None
    global _three_pt_notice_shown
    if not _three_pt_notice_shown:
        _three_pt_notice_shown = True
        print('[pyqcd.einsum] 提示：P1/P2 专用 kernel 已接线，但实测与通用路径'
              '持平（0.94–1.01x，N>=50 略慢于 opt_einsum），标记 '
              'needs-optimization', flush=True)
    return output
