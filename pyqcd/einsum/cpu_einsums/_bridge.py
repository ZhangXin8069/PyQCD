"""ctypes 绑定：``cpp/cpu/cpu-einsums`` 导出的 Einsums 字符串 einsum C ABI。

冻结的 ABI（Python 侧不可改动）::

    int  pyqceinsums_abi_version(void);        // 期望 1
    int  pyqceinsums_backend_enabled(void);    // 1=编入 Einsums，0=桩
    int  pyqceinsums_einsum(const char* spec, long long set_ptrs,
                            long long params);
    const char* pyqceinsums_last_error(void);

``set_ptrs`` 是 ``int64[8]``：``[0]=OUTPUT``、``[1]=INPUTS``（指向
``int64[nops]`` 指针表）、``[2]=INPUT_SHAPES``（指向 ``int64[nops*rank]``）、
``[3]=OUTPUT_SHAPE``（指向 ``int64[out_ndim]``）。``params`` 是 ``int32[8]``：
``[0]=NUM_OPS``、``[1]=INPUT_RANK``、``[2]=DTYPE_ID``、``[3]=OUT_NDIM``。
全部是**主机（CPU）指针**，numpy 数组直接取 ``.ctypes.data`` 即可。

本模块只负责接口层：加载归档、编组参数、把状态码翻译成异常。真正的 Einsums
调用在 C++ 侧完成；没构建或没编入 Einsums 时 C 侧返回
``pyqceinsums_*_UNAVAILABLE``，由 :func:`probe` 如实报告，**绝不伪造原生成功**。

导入本模块不触碰文件系统；第一次调用才 ``CDLL``，失败原因只记录一次（线程锁）。
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import _contract

ABI_VERSION = 1

MAX_OPERANDS = 2
MAX_RANK = 4
MAX_OUT_NDIM = 4

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

LIBRARY_ENV = 'PYQCD_EINSUMS_LIB'
LIBRARY_NAME = 'libpyqceinsums.so'

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
_REPO_LIB = _REPO / 'cpp' / 'cpu' / 'cpu-einsums' / LIBRARY_NAME

_BUILD_HINT = (
    '先运行 bash cpp/cpu/cpu-einsums/make.sh（无 Einsums 头文件时产出会 '
    'fail-closed 返回 UNAVAILABLE 的桩库；PYQCD_EINSUMS_ROOT 指向 Einsums '
    '源码/安装前缀可编入真后端）；产物应为 cpp/cpu/cpu-einsums/'
    f'{LIBRARY_NAME}，也可用环境变量 {LIBRARY_ENV} 指向别处')

_LIBRARY: Any = None
_LOAD_FAILURE = None
_LIBRARY_LOCK = threading.Lock()


class NativeUnavailable(RuntimeError):
    """请求的原生 Einsums 操作无法执行（缺库 / 桩库未编入 Einsums）。"""


class EinsumsNativeError(RuntimeError):
    """原生桥返回非 OK 状态码。"""

    def __init__(self, status, message='', name=None):
        self.status = int(status)
        self.name = name or STATUS_NAMES.get(self.status,
                                             f'STATUS_{self.status}')
        self.message = str(message)
        super().__init__(f'{self.name}({self.status}): {self.message}')


class EinsumsBridgeError(_contract.EinsumsError):
    """调用无法用冻结 ABI 表达（操作数个数/rank/dtype/指针越界）。"""


def library_path():
    """返回本模块会加载的归档路径（``PYQCD_EINSUMS_LIB`` 优先）。"""
    requested = os.environ.get(LIBRARY_ENV)
    if requested:
        return Path(requested).expanduser()
    return _REPO_LIB


def _load_library():
    """加载归档并绑定四个导出符号；失败原因只记录一次。"""
    global _LIBRARY, _LOAD_FAILURE
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if _LOAD_FAILURE is not None:
            raise NativeUnavailable(_LOAD_FAILURE)
        path = library_path()
        if not path.is_file():
            _LOAD_FAILURE = f'Einsums einsum 桥未构建: {path} 不存在；{_BUILD_HINT}'
            raise NativeUnavailable(_LOAD_FAILURE)
        try:
            library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        except OSError as exc:
            _LOAD_FAILURE = f'Einsums einsum 桥加载失败: {exc}'
            raise NativeUnavailable(_LOAD_FAILURE) from exc
        try:
            library.pyqceinsums_abi_version.argtypes = ()
            library.pyqceinsums_abi_version.restype = ctypes.c_int
            library.pyqceinsums_backend_enabled.argtypes = ()
            library.pyqceinsums_backend_enabled.restype = ctypes.c_int
            library.pyqceinsums_einsum.argtypes = (
                ctypes.c_char_p, ctypes.c_longlong, ctypes.c_longlong)
            library.pyqceinsums_einsum.restype = ctypes.c_int
            library.pyqceinsums_last_error.argtypes = ()
            library.pyqceinsums_last_error.restype = ctypes.c_char_p
        except AttributeError as exc:
            _LOAD_FAILURE = f'Einsums einsum 桥缺少导出符号: {exc}'
            raise NativeUnavailable(_LOAD_FAILURE) from exc
        _LIBRARY = library
        return library


def _last_error():
    """读取 ``pyqceinsums_last_error()``；不可用时返回空串。"""
    try:
        raw = _load_library().pyqceinsums_last_error()
    except NativeUnavailable:
        return ''
    if not raw:
        return ''
    return raw.decode('utf-8', 'replace')


def probe():
    """只读探针：库是否可加载、ABI 版本、是否编入 Einsums。

    返回 ``{'available', 'library', 'abi_version', 'backend_enabled',
    'reason'}``，永不抛异常。``available`` 只说明 ABI 可用；数值调用是否真的
    落到 Einsums 由 ``backend_enabled`` 表示。
    """
    path = library_path()
    info = {
        'available': False,
        'library': str(path),
        'abi_version': None,
        'backend_enabled': False,
        'reason': '',
    }
    if not path.is_file():
        info['reason'] = f'{path} 不存在；{_BUILD_HINT}'
        return info
    try:
        library = _load_library()
        abi_version = int(library.pyqceinsums_abi_version())
        backend_enabled = bool(int(library.pyqceinsums_backend_enabled()))
    except NativeUnavailable as exc:
        info['reason'] = str(exc)
        return info
    except Exception as exc:  # noqa: BLE001 - 探针只报告，不抛出
        info['reason'] = f'ABI 探测异常: {type(exc).__name__}: {exc}'
        return info
    info['abi_version'] = abi_version
    info['backend_enabled'] = backend_enabled
    info['available'] = abi_version >= ABI_VERSION
    if not info['available']:
        info['reason'] = (f'ABI 版本 {abi_version} 低于 Python 侧要求的 '
                          f'{ABI_VERSION}')
    elif not backend_enabled:
        info['reason'] = ('桥已加载但 backend_enabled=0：构建时未编入 Einsums，'
                          '数值入口会 fail-closed 返回 UNAVAILABLE(4)；'
                          f'{_BUILD_HINT}')
    else:
        info['reason'] = 'ABI 与 Einsums 后端均已就绪'
    return info


def skip_reason():
    """返回一行可直接打印的 SKIP 原因（可用时为空串）。"""
    info = probe()
    return '' if info['available'] else info['reason']


def check_abi_bounds(nops, ranks, out_ndim):
    """检查调用是否落在冻结 ABI 的上限内，返回 ``(ok, reason)``。

    ``ranks`` 是每个输入的 rank。>>> ``params[1]`` 只有单个 ``INPUT_RANK``
    槽位，因此两个输入的 rank 必须相同——这一点常被忽略，必须显式拒绝。
    """
    if int(nops) != 2:
        return False, (
            f'操作数个数 {nops} 不被支持：Einsums compute_graph::einsum 只有'
            '二元与三元以上 n 元重载（Operations.hpp / NaryEinsum.hpp），'
            '没有一元版本；转置请用 cg::permute，标量内积请用 cg::dot')
    ranks = [int(rank) for rank in ranks]
    if len(ranks) != int(nops):
        return False, f'rank 表长度 {len(ranks)} 与操作数个数 {nops} 不一致'
    too_deep = [rank for rank in ranks if rank > MAX_RANK]
    if too_deep:
        return False, f'输入 rank {too_deep} 超出桥上限 {MAX_RANK}'
    if len(set(ranks)) > 1:
        return False, (f'输入 rank {ranks} 不一致：冻结 ABI 只有单个 '
                       'INPUT_RANK 槽位（params[1]），无法表达异 rank 操作数')
    if int(out_ndim) < 1:
        return False, (
            'Einsums cg::einsum 不支持 rank-0 输出（本版本的 rank-0 张量不能'
            '作为 einsum 操作数或输出；标量内积请用 cg::dot）')
    if int(out_ndim) > MAX_OUT_NDIM:
        return False, f'输出 rank {out_ndim} 超出桥上限 {MAX_OUT_NDIM}'
    if int(out_ndim) < 0:
        return False, f'输出 rank 不能为负: {out_ndim}'
    return True, ''


def _pointer_of(value):
    """主机指针：整数原样使用，其余走 ``np.asarray(x).ctypes.data``。"""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return int(np.asarray(value).ctypes.data)


def _shape_of(value):
    """形状：整数指针没有形状信息，返回空元组。"""
    if isinstance(value, (int, np.integer)):
        return ()
    return tuple(int(extent) for extent in np.shape(value))


def _dtype_id(dtype):
    """dtype → dtype id；整数 id 原样透传（便于测 C 侧未知 id 的 fail-closed）。"""
    if isinstance(dtype, (int, np.integer)):
        return int(dtype)
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        raise EinsumsBridgeError(f'无法识别的 dtype: {dtype!r}') from None
    if resolved not in DTYPE_IDS:
        supported = sorted(str(item) for item in DTYPE_IDS)
        raise EinsumsBridgeError(
            f'不支持的 dtype {resolved}；只支持 {supported}')
    return DTYPE_IDS[resolved]


def native_status(spec, out, inputs, dtype):
    """调用 ``pyqceinsums_einsum()`` 并返回 ``(status, message)``，不抛异常。

    负例测试直接用它，这样可以在没有真实 Einsums 的机器上验证 fail-closed。
    ``out``/``inputs`` 可以是 numpy 数组（取 ``.ctypes.data``）或整数指针；
    ``dtype`` 可以是 numpy dtype 或已经是 dtype id 的整数。

    spec 原样透传（不做规范化）：Einsums 自己解析它。无法用冻结 ABI 表达的
    调用（操作数 rank 不一致）在 Python 侧直接报 ``INVALID``，与桥的
    fail-closed 语义一致，不做任何伪造。
    """
    try:
        return _native_status(spec, out, inputs, dtype)
    except NativeUnavailable as exc:
        return STATUS_UNAVAILABLE, str(exc)
    except EinsumsBridgeError as exc:
        return STATUS_INVALID, str(exc)
    except Exception as exc:  # noqa: BLE001 - 状态码接口不抛异常
        return STATUS_INTERNAL, f'{type(exc).__name__}: {exc}'


def _native_status(spec, out, inputs, dtype):
    library = _load_library()
    spec_text = spec if isinstance(spec, bytes) else str(spec)

    operands = list(inputs)
    pointers = [_pointer_of(item) for item in operands]
    shapes = [_shape_of(item) for item in operands]
    ranks = sorted({len(shape) for shape in shapes})
    if len(ranks) > 1:
        raise EinsumsBridgeError(
            f'输入 rank {[len(shape) for shape in shapes]} 不一致：'
            '冻结 ABI 只有单个 INPUT_RANK 槽位（params[1]）')
    rank = ranks[0] if ranks else 0

    out_pointer = _pointer_of(out)
    out_shape = _shape_of(out)
    dtype_id = _dtype_id(dtype)

    slots = (ctypes.c_longlong * _SET_SIZE)()
    controls = (ctypes.c_int * _PARAM_SIZE)()
    pointer_table = (ctypes.c_longlong * max(len(pointers), 1))(*pointers)
    flat_shapes = [extent for shape in shapes for extent in shape]
    shape_table = (ctypes.c_longlong * max(len(flat_shapes), 1))(*flat_shapes)
    out_dims = (ctypes.c_longlong * max(len(out_shape), 1))(*out_shape)

    slots[_SET_OUTPUT] = int(out_pointer)
    slots[_SET_INPUTS] = ctypes.addressof(pointer_table)
    slots[_SET_INPUT_SHAPES] = ctypes.addressof(shape_table)
    slots[_SET_OUTPUT_SHAPE] = ctypes.addressof(out_dims)

    controls[_PARAM_NUM_OPS] = len(pointers)
    controls[_PARAM_INPUT_RANK] = int(rank)
    controls[_PARAM_DTYPE_ID] = int(dtype_id)
    controls[_PARAM_OUT_NDIM] = len(out_shape)

    status = int(library.pyqceinsums_einsum(
        spec_text.encode('utf-8') if isinstance(spec_text, str) else spec_text,
        int(ctypes.addressof(slots)),
        int(ctypes.addressof(controls))))
    return status, _last_error()


def _check_writable(out):
    """输出必须是可写、C 连续、本机字节序的 ndarray（ABI 不带 strides）。"""
    if not isinstance(out, np.ndarray):
        raise EinsumsBridgeError(
            f'输出必须是 numpy 数组，收到 {type(out).__name__}')
    if not out.flags.writeable:
        raise EinsumsBridgeError('输出数组不可写')
    if not out.flags['C_CONTIGUOUS']:
        raise EinsumsBridgeError(
            '输出数组必须是 C 连续：冻结 ABI 不传 strides，'
            '非连续视图会写错位置')
    if out.dtype.byteorder not in ('=', '|'):
        raise EinsumsBridgeError(f'输出 dtype 字节序不受支持: {out.dtype}')


def einsum_into(out, spec, *operands):
    """按 Einsums 语义执行 ``einsum`` 并把结果写进 ``out``，返回 ``out``。

    先做 Python 侧校验（spec 规范化 + 形状/标签一致性 + ABI 上限 + dtype +
    连续性），再走 C ABI。原生返回非 OK 时抛 :class:`EinsumsNativeError`；
    库缺失时抛 :class:`NativeUnavailable`——两种情况都不会返回伪造结果。
    """
    if not operands:
        raise EinsumsBridgeError('至少需要一个输入张量')
    if len(operands) != 2:
        raise EinsumsBridgeError(
            f'Einsums compute_graph::einsum 没有 {len(operands)} 元重载；'
            '本桥只支持二元收缩，转置请用 cg::permute、标量内积请用 cg::dot')
    if int(np.ndim(out)) == 0:
        raise EinsumsBridgeError(
            'Einsums 本版本的 cg::einsum 不支持 rank-0 输出；'
            '标量内积请用 cg::dot')
    normalized = _contract.normalize_spec(spec)
    out_labels, in_labels = _contract.parse_einsums_spec(normalized)
    if len(in_labels) != len(operands):
        raise EinsumsBridgeError(
            f'spec {normalized!r} 需要 {len(in_labels)} 个输入，'
            f'收到 {len(operands)} 个')

    arrays = [np.asarray(operand) for operand in operands]
    shapes = [tuple(int(extent) for extent in array.shape)
              for array in arrays]
    sizes = _contract.check_shapes(out_labels, in_labels, shapes)
    expected = tuple(sizes[label] for label in out_labels)
    _check_writable(out)
    actual = tuple(int(extent) for extent in out.shape)
    if actual != expected:
        raise _contract.EinsumsShapeError(
            f'输出形状 {actual} 与推断结果 {expected} 不一致（{normalized!r}）')

    ranks = [len(shape) for shape in shapes]
    ok, reason = check_abi_bounds(len(arrays), ranks, len(expected))
    if not ok:
        raise EinsumsBridgeError(reason)

    dtype = np.dtype(out.dtype)
    if dtype not in DTYPE_IDS:
        raise EinsumsBridgeError(
            f'不支持的输出 dtype {dtype}；只支持 '
            f'{sorted(str(item) for item in DTYPE_IDS)}')
    if dtype.byteorder not in ('=', '|'):
        raise EinsumsBridgeError(f'输出 dtype 字节序不受支持: {dtype}')
    for index, array in enumerate(arrays, start=1):
        if array.dtype != dtype:
            raise EinsumsBridgeError(
                f'第 {index} 个输入 dtype {array.dtype} 与输出 {dtype} 不一致'
                '（混合精度 fail closed）')
        if not array.flags['C_CONTIGUOUS']:
            raise EinsumsBridgeError(
                f'第 {index} 个输入不是 C 连续：冻结 ABI 不传 strides')

    _load_library()
    status, message = native_status(normalized, out, arrays, DTYPE_IDS[dtype])
    if status != STATUS_OK:
        raise EinsumsNativeError(status, message)
    return out
