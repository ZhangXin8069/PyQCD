"""Execution core for :mod:`pyqcd.einsum`.

Consumes a contraction plan produced by the planning layer and performs the
contractions on the active backend. Mirrors ``opt_einsum.contract._core_contract``
semantics while avoiding redundant transposes and offering optional buffer reuse.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Iterable, Sequence

from ._sharing import (einsum_cache_wrap, tensordot_cache_wrap,
                       transpose_cache_wrap)

_FUNC_CACHE: dict[tuple[str, str], Any] = {}

_FALLBACK_MODULES = {
    "numpy": "numpy",
    "cupy": "cupy",
    "torch": "torch",
    "dask": "dask.array",
    "jax": "jax.numpy",
    "sparse": "sparse",
}

_POOL_ENABLED = os.environ.get("PYQCD_EINSUM_POOL", "1") != "0"


def resolve_backend(operands: Sequence[Any], backend: str = "auto") -> str:
    """Resolve ``backend='auto'`` from the first array-like operand."""
    if backend and backend != "auto":
        return backend
    for operand in operands:
        if operand is None:
            continue
        module = type(operand).__module__
        name = module.split(".")[0]
        if name in _FALLBACK_MODULES:
            return name
        return name
    return "numpy"


def get_backend_func(name: str, backend: str) -> Any:
    """Fetch ``backend.<name>`` lazily, falling back to importlib."""
    key = (name, backend)
    cached = _FUNC_CACHE.get(key)
    if cached is not None:
        return cached

    func = None
    try:
        from .backends import dispatch

        func = dispatch.get_func(name, backend, default=None)
    except Exception:
        func = None

    if func is None:
        module = importlib.import_module(_FALLBACK_MODULES.get(backend, backend))
        func = getattr(module, name)
    _FUNC_CACHE[key] = func
    return func


def has_func(name: str, backend: str) -> bool:
    try:
        get_backend_func(name, backend)
        return True
    except Exception:
        return False


def _transpose(array: Any, axes: tuple[int, ...], backend: str) -> Any:
    if axes == tuple(range(len(axes))):
        return array
    try:
        func = get_backend_func("transpose", backend)
    except Exception:
        return array.transpose(axes)
    return func(array, axes)


def _tensordot(left: Any, right: Any, axes, backend: str, **kwargs) -> Any:
    try:
        func = get_backend_func("tensordot", backend)
    except Exception:
        module = importlib.import_module(_FALLBACK_MODULES.get(backend, backend))
        func = module.tensordot
    return func(left, right, axes=axes, **kwargs)


def _einsum(equation: str, *operands: Any, backend: str, out: Any = None, **kwargs) -> Any:
    try:
        func = get_backend_func("einsum", backend)
    except Exception:
        module = importlib.import_module(_FALLBACK_MODULES.get(backend, backend))
        func = module.einsum
    if out is not None:
        return func(equation, *operands, out=out, **kwargs)
    return func(equation, *operands, **kwargs)


def _blas_axes(input_left: str, input_right: str, idx_removed: Iterable[str]):
    idx_rm = idx_removed
    if not idx_rm:
        return ((), ())
    left_pos = []
    right_pos = []
    for symbol in idx_rm:
        left_pos.append(input_left.find(symbol))
        right_pos.append(input_right.find(symbol))
    return tuple(zip(*sorted(zip(left_pos, right_pos))))


def step_axes(step, idx_removed: Iterable[str]):
    """Return ``(axes, tensor_result, result_index)`` for a BLAS-eligible step."""
    _, _, equation, _, _ = step
    input_str, results_index = equation.split("->")
    input_left, input_right = input_str.split(",")
    idx_rm = idx_removed
    tensor_result = "".join(s for s in input_left + input_right if s not in idx_rm)
    return _blas_axes(input_left, input_right, idx_rm), tensor_result, results_index


# 与 opt_einsum 一致：把三个后端原语包上共享缓存，使 `shared_intermediates()`
# 真正复用中间结果（未共享时包装器直接透传，无额外开销）。
_transpose = transpose_cache_wrap(_transpose)
_tensordot = tensordot_cache_wrap(_tensordot)
_einsum = einsum_cache_wrap(_einsum)


def _loses_imaginary_part(out: Any, result: Any) -> bool:
    """``out=`` 写入是否会丢掉结果的虚部（跨 numpy / torch / cupy 的判定）。

    只有"复数结果写进实数缓冲"算错误：numpy 会在那里静默丢掉虚部（仅一条
    ComplexWarning）。实数之间收窄（float64→float32）是有损舍入，与 numpy 自身
    ``out=`` 的语义一致，保持放行。torch 的 ``dtype`` 没有 ``kind``，因此按
    ``is_complex()`` / ``kind`` 两种能力探测。
    """
    def is_complex(dtype: Any) -> bool:
        kind = getattr(dtype, "kind", None)
        if isinstance(kind, str):
            return kind == "c"
        probe = getattr(dtype, "is_complex", None)
        if callable(probe):
            return bool(probe())
        name = str(dtype)
        return "complex" in name

    return is_complex(getattr(result, "dtype", None)) and not is_complex(
        getattr(out, "dtype", None))


def execute_plan(
    contraction_list: Sequence[tuple],
    operands: Sequence[Any],
    backend: str = "auto",
    out: Any = None,
    evaluate_constants: bool = False,
    **kwargs: Any,
):
    """Run a precomputed contraction plan.

    Returns the contracted array, or ``(operands, remaining_plan)`` when
    ``evaluate_constants`` stops at the first runtime operand.
    """
    specified_out = out is not None
    work = list(operands)
    backend = resolve_backend(work, backend)
    no_einsum = not has_func("einsum", backend)

    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, equation, _, blas_flag = contraction

        if evaluate_constants and any(work[x] is None for x in inds):
            return work, list(contraction_list[num:])

        tmp_operands = [work.pop(x) for x in inds]
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        if blas_flag and ("EINSUM" not in blas_flag or no_einsum):
            axes, tensor_result, results_index = step_axes(contraction, idx_rm)
            new_view = _tensordot(
                tmp_operands[0], tmp_operands[1], axes=axes, backend=backend, **kwargs
            )
            if tensor_result != results_index or handle_out:
                transpose = tuple(tensor_result.index(s) for s in results_index)
                new_view = _transpose(new_view, transpose, backend)
                if handle_out:
                    # ``out`` 的契约是"shape 与 dtype 必须同上"（见 contract 的
                    # docstring）。这里曾经直接 ``out[...] = new_view``：dtype 不符
                    # 时靠赋值语义静默 cast。**复数的虚部被丢掉**是真正的错误，
                    # 必须显式失败；实数之间收敛到更窄的浮点只是有损舍入，既有
                    # 用例（torch 后端用 float32 缓冲承接 float64 结果）依赖它，
                    # 而且 numpy 自己的 ``out=`` 也允许，所以保持放行。
                    if _loses_imaginary_part(out, new_view):
                        raise ValueError(
                            "out dtype must keep the imaginary part: result is "
                            f"{new_view.dtype} but out is {out.dtype}")
                    out[...] = new_view
        else:
            new_view = _einsum(
                equation,
                *tmp_operands,
                backend=backend,
                out=out if handle_out else None,
                **kwargs,
            )

        work.append(new_view)
        del tmp_operands, new_view

    return out if specified_out else work[0]


def execute_step(step, operands, backend: str = "auto", out: Any = None, **kwargs: Any):
    """Execute a single contraction step and return the produced array."""
    return execute_plan([step], list(operands), backend=backend, out=out, **kwargs)
