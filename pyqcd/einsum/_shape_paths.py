"""形状特化路径：语料之外的、按规范签名精确匹配的**手写固定算法**。

与另外两条特化链的分工：

* ``_topology_registry`` + 原生步表：wick_12cases 语料（8638 条）的通用拓扑；
* ``_dedicated_kernels``：任意方程的固定归并序（Python 逐对执行）；
* 本模块：**少量结构特殊的方程**，用一次性推导出的更优算法（通常是一次 GEMM）
  直接算完，不走规划器、不建步表。

契约（与专用内核一致）：结构完全对上才执行；任何前置条件不满足就返回 ``None``
让调用方回落通用路径。绝不猜测、绝不改 dtype、绝不抛异常。

第一条路径：``x,ax,bx,cx->abc``
------------------------------
.. code-block:: text

    out[a,b,c] = Σ_x w[x] · A[a,x] · B[b,x] · C[c,x]

规划器会拆成三步（``x,ax->xa``、``bx,cx->bxc``、``xa,bxc->abc``），中间体
``b×c×x`` 是 n³ 元素且第二步是逐元素乘。这里压成**一次 GEMM**：

.. code-block:: text

    Q[a·n+b, x] = A[a,x] · B[b,x]        # n³ 次乘，与输出同阶
    W[c, x]     = C[c,x] · w[x]
    out[a,b,c]  = reshape(Q @ Wᵀ)        # M=n², K=n, N=n 的单个 GEMM

实测（本机，float64，n=8…128）：相对 ``contract()`` 通用路径 ``0.06×–0.75×``
（n=8 快 16×，n=128 快 1.5×），相对 ``np.einsum(optimize="greedy")`` 在
``n≥100`` 也更快；数值偏差 ≤ 4e-16（同一 BLAS 口径）。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import numpy as np

from . import _specialize

__all__ = ["entry", "lookup", "try_execute", "registered"]

# n³ 中间体（与输出同阶）的字节上限：超过就走通用路径，别为了几十毫秒吃掉显存/内存。
_MAX_BYTES_ENV = "PYQCD_EINSUM_SHAPE_PATH_MAX_BYTES"
_DEFAULT_MAX_BYTES = 512 << 20


def _max_bytes() -> int:
    raw = os.environ.get(_MAX_BYTES_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_MAX_BYTES


def _abc_sandwich(operands: Sequence[Any]) -> Any:
    """``out[a,b,c] = Σ_x w[x] A[a,x] B[b,x] C[c,x]``，一次 GEMM 完成。"""
    if len(operands) != 4:
        return None
    weight, left, middle, right = operands
    shapes = []
    for operand in operands:
        shape = getattr(operand, "shape", None)
        if shape is None:
            return None
        shapes.append(tuple(int(extent) for extent in shape))
    vector_shape, *matrix_shapes = shapes
    if len(vector_shape) != 1 or any(len(shape) != 2 for shape in matrix_shapes):
        return None
    size = vector_shape[0]
    if size <= 0 or any(shape != (size, size) for shape in matrix_shapes):
        return None
    dtype = getattr(left, "dtype", None)
    if dtype is None or any(getattr(item, "dtype", None) != dtype for item in operands):
        return None
    free = float(np.dtype(dtype).itemsize) * size ** 3
    if free > _max_bytes():
        return None

    module = _module_of(weight)
    if module is None:
        return None
    if not all(item.flags["C_CONTIGUOUS"] for item in operands):
        operands = [module.ascontiguousarray(item) for item in operands]
        weight, left, middle, right = operands
    # Q[a*n+b, x] = A[a,x]·B[b,x]：广播乘一次成型，reshape 后是连续矩阵。
    scaled = (left[:, None, :] * middle[None, :, :]).reshape(size * size, size)
    combined = right * weight[None, :]
    return module.matmul(scaled, combined.T).reshape(size, size, size)


def _module_of(operand: Any) -> Any:
    """按操作数类型取数组模块（numpy / cupy）；其它类型一律不接管。"""
    if isinstance(operand, np.ndarray):
        return np
    if type(operand).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: PLC0415 - 只在 CUDA 操作数上才导入

        return cupy
    return None


def _entry(name: str, note: str, kernels: tuple[str, ...],
           run: Callable[[Sequence[Any]], Any]) -> dict[str, Any]:
    return {"name": name, "kind": "kernel(shape)", "note": note,
            "kernels": kernels, "run": run, "subscripts": ""}


_PATHS: dict[tuple, dict[str, Any]] = {}


def _register(equation: str, name: str, note: str,
              kernels: tuple[str, ...], run: Callable[[Sequence[Any]], Any]) -> None:
    signature = _specialize.canonical_signature(equation)
    if signature is None:
        raise ValueError(f"equation is not parsable: {equation!r}")
    entry = _entry(name, note, kernels, run)
    entry["subscripts"] = equation
    _PATHS[signature] = entry


_register(
    "x,ax,bx,cx->abc",
    "shape:abc-sandwich",
    "形状特化（非语料）：三路 Hadamard + 归约压成一次 (n²×n)·(n×n) GEMM",
    ("shape_abc_sandwich",),
    _abc_sandwich,
)


def registered() -> tuple[str, ...]:
    """已注册路径的代表写法（诊断用）。"""
    return tuple(entry["subscripts"] for entry in _PATHS.values())


def lookup(subscripts: Any) -> dict[str, Any] | None:
    """按规范签名查形状特化路径；标签改名/加空格同样命中。"""
    if not isinstance(subscripts, str):
        return None
    signature = _specialize.canonical_signature(subscripts)
    if signature is None:
        return None
    return _PATHS.get(signature)


def entry(subscripts: Any) -> dict[str, Any] | None:
    """查询接口用的只读条目（不含执行函数）。"""
    found = lookup(subscripts)
    if found is None:
        return None
    return {key: value for key, value in found.items() if key != "run"}


try_execute_announced: set[str] = set()


def try_execute(subscripts: Any, operands: Sequence[Any]) -> Any | None:
    """命中且前置条件全部成立时返回结果，否则 ``None``（调用方回落）。"""
    found = lookup(subscripts)
    if found is None:
        return None
    try:
        result = found["run"](operands)
    except Exception:  # noqa: BLE001 - 形状特化失败必须回落，绝不抛给调用方
        return None
    if result is not None and _notice_enabled():
        name = found["name"]
        if name not in try_execute_announced:
            try_execute_announced.add(name)
            print(f"[pyqcd.einsum] 形状特化: {name} [kernel: "
                  f"{found['kernels'][0]}] {found['note']}", flush=True)
    return result


def _notice_enabled() -> bool:
    return _specialize.notice_enabled()
