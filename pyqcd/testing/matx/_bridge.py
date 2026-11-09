"""ctypes 绑定：`cpp/cuda/matx` 导出的 MatX einsum C ABI。

库格式镜像 `pyqcd/einsum/kupl/einsum.py`：`set_ptrs` 与 `params` 是平坦整数数组，
指针和计数都走整数槽位，因此 Python 与 C 之间没有结构体对齐约定。

本模块只负责**接口层**：加载归档、编组参数、把状态码翻译成异常。真正的 MatX
调用在 C++ 侧 `pyqcmatx_einsum()` 内完成，本机未编译过该路径时统一返回
`PYQCMATX_STATUS_UNAVAILABLE`，由 :func:`probe` 如实报告，绝不伪造成功。

导入本模块不触碰文件系统；第一次调用才做 `CDLL`。
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import _contract

ABI_VERSION = 1
MAX_OPERANDS = 2
MAX_RANK = 4

STATUS_OK = 0
STATUS_INVALID = 1
STATUS_MEMORY = 2
STATUS_INTERNAL = 3
STATUS_UNAVAILABLE = 4

STATUS_NAMES = {
    STATUS_OK: 'OK',
    STATUS_INVALID: 'INVALID',
    STATUS_MEMORY: 'MEMORY',
    STATUS_INTERNAL: 'INTERNAL',
    STATUS_UNAVAILABLE: 'UNAVAILABLE',
}

DTYPE_IDS = {
    np.dtype('complex128'): 0,
    np.dtype('complex64'): 1,
    np.dtype('float64'): 2,
    np.dtype('float32'): 3,
}

_PARAM_NUM_OPS = 0
_PARAM_INPUT_RANK = 1
_PARAM_DTYPE_ID = 2
_PARAM_OUT_NDIM = 3
_PARAM_SIZE = 8

_SET_OUTPUT = 0
_SET_INPUTS = 1
_SET_INPUT_SHAPES = 2
_SET_OUTPUT_SHAPE = 3
_SET_SIZE = 8

_REPO = Path(__file__).resolve().parents[3]
_REPO_LIB = _REPO / 'cpp' / 'cuda' / 'matx' / 'libpyqcmatx_cuda.so'

_LIBRARY: Any = None
_LOAD_FAILURE: str | None = None
_LIBRARY_LOCK = threading.Lock()


class NativeUnavailable(RuntimeError):
    """请求的 MatX 原生操作无法执行（缺库 / 缺 GPU 张量 / 未编 cuTENSOR）。"""


class MatxNativeError(RuntimeError):
    """原生桥返回非 OK 状态码。"""

    def __init__(self, status: int, message: str):
        self.status = int(status)
        self.status_name = STATUS_NAMES.get(self.status, f'STATUS_{self.status}')
        super().__init__(f'{self.status_name}({self.status}): {message}')


def library_path() -> Path:
    """返回本模块会加载的归档路径（`PYQCD_MATX_LIB` 优先）。"""
    requested = os.environ.get('PYQCD_MATX_LIB')
    if requested:
        return Path(requested).expanduser()
    return _REPO_LIB


def _load_library() -> Any:
    """加载归档并绑定四个导出符号；失败只记录一次。"""
    global _LIBRARY, _LOAD_FAILURE
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if _LOAD_FAILURE is not None:
            raise NativeUnavailable(_LOAD_FAILURE)
        path = library_path()
        if not path.is_file():
            _LOAD_FAILURE = (
                f'MatX einsum 桥未构建: {path} 不存在；'
                '运行 bash cpp/cuda/matx/make-nv.sh')
            raise NativeUnavailable(_LOAD_FAILURE)
        try:
            library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        except OSError as exc:
            _LOAD_FAILURE = f'MatX einsum 桥加载失败: {exc}'
            raise NativeUnavailable(_LOAD_FAILURE) from exc
        try:
            library.pyqcmatx_abi_version.argtypes = ()
            library.pyqcmatx_abi_version.restype = ctypes.c_int
            library.pyqcmatx_cutensor_enabled.argtypes = ()
            library.pyqcmatx_cutensor_enabled.restype = ctypes.c_int
            library.pyqcmatx_einsum.argtypes = (
                ctypes.c_char_p, ctypes.c_longlong, ctypes.c_longlong)
            library.pyqcmatx_einsum.restype = ctypes.c_int
            library.pyqcmatx_last_error.argtypes = ()
            library.pyqcmatx_last_error.restype = ctypes.c_char_p
        except AttributeError as exc:
            _LOAD_FAILURE = f'MatX einsum 桥缺少导出符号: {exc}'
            raise NativeUnavailable(_LOAD_FAILURE) from exc
        _LIBRARY = library
        return library


def _last_error() -> str:
    try:
        raw = _load_library().pyqcmatx_last_error()
    except NativeUnavailable:
        return ''
    if not raw:
        return ''
    return raw.decode('utf-8', 'replace')


def probe() -> dict[str, Any]:
    """只读探针：库是否可加载、ABI 版本、是否编入 cuTENSOR。

    永不抛异常；缺库或缺符号时 `available` 为 `False` 并给出可直接照做的原因。
    """
    path = library_path()
    info: dict[str, Any] = {
        'available': False,
        'library': str(path),
        'abi_version': None,
        'cutensor': False,
        'reason': '',
    }
    if not path.is_file():
        info['reason'] = (
            f'{path} 不存在；先运行 bash cpp/cuda/matx/make-nv.sh'
            f'（无 MatX 头文件时构建的是 UNAVAILABLE 桩库）')
        return info
    try:
        library = _load_library()
        abi_version = int(library.pyqcmatx_abi_version())
        cutensor = bool(int(library.pyqcmatx_cutensor_enabled()))
    except NativeUnavailable as exc:
        info['reason'] = str(exc)
        return info
    except Exception as exc:  # noqa: BLE001 - 探针只报告，不抛出
        info['reason'] = f'ABI 探测异常: {type(exc).__name__}: {exc}'
        return info
    info['abi_version'] = abi_version
    info['cutensor'] = cutensor
    info['available'] = abi_version >= ABI_VERSION
    if not info['available']:
        info['reason'] = (
            f'ABI 版本 {abi_version} 低于 Python 侧要求的 {ABI_VERSION}')
    elif not cutensor:
        info['reason'] = (
            '桥已构建但未编入 cuTENSOR：重新构建需 '
            'PYQCD_MATX_ROOT=<MatX 源码根> bash cpp/cuda/matx/make-nv.sh '
            '--cutensor（cuTENSOR>=1.7.0、cuTensorNet>=23.03.0.20）')
    else:
        info['reason'] = 'ABI 与 cuTENSOR 均已就绪；执行仍需 GPU executor'
    return info


def skip_reason() -> str:
    """返回一行可直接打印的 SKIP 原因。"""
    info = probe()
    return '' if info['available'] and info['cutensor'] else info['reason']


def _device_pointer(operand: Any) -> int:
    """取出 GPU 张量的设备指针；numpy 等主机数组显式拒绝。"""
    if isinstance(operand, int):
        return int(operand)
    interface = getattr(operand, '__cuda_array_interface__', None)
    if isinstance(interface, dict):
        return int(interface['data'][0])
    pointer = getattr(operand, 'data_ptr', None)
    if callable(pointer) and getattr(operand, 'is_cuda', False):
        return int(pointer())
    raise NativeUnavailable(
        f'MatX einsum 只在 GPU executor 上运行，收到主机数组 '
        f'{type(operand).__name__}；请传 cupy/torch CUDA 张量或设备指针整数')


def _flat_shapes(shapes: Sequence[Sequence[int]]) -> list[int]:
    flat: list[int] = []
    for shape in shapes:
        flat.extend(int(extent) for extent in shape)
    return flat


def native_status(subscripts: str, out_pointer: int,
                  input_pointers: Sequence[int],
                  input_shapes: Sequence[Sequence[int]],
                  out_shape: Sequence[int],
                  dtype_id: int) -> tuple[int, str]:
    """调用 `pyqcmatx_einsum()` 并返回 `(status, message)`，不抛异常。

    负例测试直接用它，这样可以在没有真实 GPU 张量的机器上验证 fail-closed。
    """
    library = _load_library()
    slots = (ctypes.c_longlong * _SET_SIZE)()
    controls = (ctypes.c_int * _PARAM_SIZE)()
    pointers = (ctypes.c_longlong * max(len(input_pointers), 1))(
        *[int(pointer) for pointer in input_pointers])
    shapes = (ctypes.c_longlong * max(len(_flat_shapes(input_shapes)), 1))(
        *_flat_shapes(input_shapes))
    out_dims = (ctypes.c_longlong * max(len(out_shape), 1))(
        *[int(extent) for extent in out_shape])
    slots[_SET_OUTPUT] = int(out_pointer)
    slots[_SET_INPUTS] = ctypes.addressof(pointers)
    slots[_SET_INPUT_SHAPES] = ctypes.addressof(shapes)
    slots[_SET_OUTPUT_SHAPE] = ctypes.addressof(out_dims)
    controls[_PARAM_NUM_OPS] = len(input_pointers)
    controls[_PARAM_INPUT_RANK] = len(input_shapes[0]) if len(input_shapes) else 0
    controls[_PARAM_DTYPE_ID] = int(dtype_id)
    controls[_PARAM_OUT_NDIM] = len(out_shape)
    status = int(library.pyqcmatx_einsum(
        str(subscripts).encode('utf-8'),
        int(ctypes.addressof(slots)),
        int(ctypes.addressof(controls))))
    return status, _last_error()


def einsum_into(out: Any, subscripts: str, *operands: Any) -> Any:
    """按 MatX 的 out-优先约定执行 `einsum` 并返回 `out`。

    只接受 GPU 张量；主机数组在进入 C ABI 之前就被拒绝，避免把 CPU 指针
    伪装成设备指针。
    """
    if not operands:
        raise _contract.MatXInterfaceError('至少需要一个输入张量')
    normalized = _contract.normalize_subscripts(subscripts)
    shapes = [tuple(int(extent) for extent in np.shape(operand))
              for operand in operands]
    expected = _contract.infer_output_shape(normalized, shapes)
    actual = tuple(int(extent) for extent in np.shape(out))
    if actual != tuple(expected):
        raise _contract.MatXShapeError(
            f'输出张量形状 {actual} 与推断结果 {tuple(expected)} 不一致'
            f'（{normalized!r}）')
    dtype = np.dtype(getattr(out, 'dtype', np.complex128))
    if dtype not in DTYPE_IDS:
        raise _contract.MatXInterfaceError(f'不支持的 dtype: {dtype}')
    for operand in operands:
        if np.dtype(getattr(operand, 'dtype', dtype)) != dtype:
            raise _contract.MatXInterfaceError(
                f'MatX einsum 要求操作数与输出 dtype 一致: '
                f'{np.dtype(getattr(operand, "dtype", dtype))} vs {dtype}')
    status, message = native_status(
        normalized,
        _device_pointer(out),
        [_device_pointer(operand) for operand in operands],
        shapes,
        actual,
        DTYPE_IDS[dtype])
    if status != STATUS_OK:
        raise MatxNativeError(status, message)
    return out
