"""Dispatch array operations to the correct backend library.

Functions are imported lazily and memoized so that importing
:mod:`pyqcd.einsum` never pulls in optional libraries.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Callable, Sequence

from .._sharing import to_backend_cache_wrap

__all__ = [
    "get_func",
    "has_einsum",
    "has_tensordot",
    "infer_backend",
    "parse_backend",
    "build_expression",
    "evaluate_constants",
    "has_backend",
    "to_backend",
    "to_torch",
    "to_cupy",
    "to_jax",
    "to_dask",
]

# known non top-level imports
_aliases: dict[str, str] = {
    "dask": "dask.array",
    "theano": "theano.tensor",
    "jax": "jax.numpy",
    "jaxlib": "jax.numpy",
    "autograd": "autograd.numpy",
    "mars": "mars.tensor",
    "cupy": "cupy",
    "torch": "torch",
    "pytorch": "torch",
    "tensorflow": "tensorflow",
    "sparse": "sparse",
    # numpy handles object arrays itself, so ``backend="object"`` just uses it
    "object": "numpy",
}

_SYMBOLS_BASE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _ascii_equation(equation: str) -> str:
    """Rename non-ASCII index symbols, which most backends reject."""
    if equation.isascii():
        return equation
    symbols = sorted(set(equation) - set(",->"))
    replacer = {c: _SYMBOLS_BASE[i] for i, c in enumerate(symbols) if i < len(_SYMBOLS_BASE)}
    return "".join(replacer.get(c, c) for c in equation)


def _wrap_einsum(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt ``{backend}.einsum`` to accept non-ASCII index symbols.

    ``opt_einsum`` renames such symbols inside its own einsum wrapper; the
    execution layer dispatches straight to the backend, so the renaming has to
    happen here for unicode equations to work on every path.
    """

    def einsum(equation: Any, *operands: Any, **kwargs: Any) -> Any:
        if isinstance(equation, str) and not equation.isascii():
            equation = _ascii_equation(equation)
        return fn(equation, *operands, **kwargs)

    einsum.__name__ = getattr(fn, "__name__", "einsum")
    einsum.__doc__ = getattr(fn, "__doc__", None)
    return einsum


def _torch_module() -> Any:
    return importlib.import_module("torch")


def _torch_transpose(a: Any, axes: Sequence[int]) -> Any:
    return a.permute(*axes)


def _torch_einsum(equation: str, *operands: Any, **kwargs: Any) -> Any:
    # ``out``/``dtype``/``casting``/``order`` have no torch equivalent
    return _torch_module().einsum(_ascii_equation(equation), *operands)


def _torch_tensordot(x: Any, y: Any, axes: Any = 2) -> Any:
    torch = _torch_module()
    if hasattr(torch, "tensordot"):
        return torch.tensordot(x, y, dims=axes)

    xnd, ynd = x.dim(), y.dim()
    if isinstance(axes, int):
        axes = (range(xnd - axes, xnd), range(axes))
    if isinstance(axes[0], int):
        axes = (axes[0],), axes[1]
    if isinstance(axes[1], int):
        axes = axes[0], (axes[1],)

    x_ix: list[Any] = [None] * xnd
    y_ix: list[Any] = [None] * ynd
    out_ix: list[str] = []
    available = iter(_SYMBOLS_BASE)
    for ax1, ax2 in zip(*axes):
        repeat = next(available)
        x_ix[ax1] = repeat
        y_ix[ax2] = repeat
    for i in range(xnd):
        if x_ix[i] is None:
            leave = next(available)
            x_ix[i] = leave
            out_ix.append(leave)
    for i in range(ynd):
        if y_ix[i] is None:
            leave = next(available)
            y_ix[i] = leave
            out_ix.append(leave)
    return _torch_einsum("{},{}->{}".format(*map("".join, (x_ix, y_ix, out_ix))), x, y)


# numpy-compatible wrappers for backends whose top level API differs
_SHIMS: dict[str, dict[str, Callable[..., Any]]] = {
    "torch": {
        "transpose": _torch_transpose,
        "einsum": _torch_einsum,
        "tensordot": _torch_tensordot,
    },
    "pytorch": {
        "transpose": _torch_transpose,
        "einsum": _torch_einsum,
        "tensordot": _torch_tensordot,
    },
}

_cached_funcs: dict[tuple[str, str], Any] = {}

try:
    import numpy as np

    _cached_funcs[("tensordot", "numpy")] = np.tensordot
    _cached_funcs[("transpose", "numpy")] = np.transpose
except ModuleNotFoundError:  # pragma: no cover - numpy is a hard dependency
    pass


def _import_func(func: str, backend: str, default: Any = None) -> Any:
    """Import ``{backend}.{func}``, falling back to ``default`` if given."""
    shim = _SHIMS.get(backend.lower(), {}).get(func)
    if shim is not None:
        return shim

    try:
        lib = importlib.import_module(_aliases.get(backend, backend))
    except ModuleNotFoundError:
        if default is None:
            raise
        return default

    if default is not None:
        return getattr(lib, func, default)

    try:
        found = getattr(lib, func)
    except AttributeError:
        message = (
            "{} doesn't seem to provide the function {} - see the backend documentation "
            "for details on which functions are required for which contractions."
        )
        raise AttributeError(message.format(backend, func)) from None

    return _wrap_einsum(found) if func == "einsum" else found


def get_func(func: str, backend: str = "numpy", default: Any = None) -> Any:
    """Return ``{backend}.{func}``, e.g. ``numpy.einsum``, or ``default``."""
    try:
        return _cached_funcs[func, backend]
    except KeyError:
        fn = _import_func(func, backend, default)
        _cached_funcs[func, backend] = fn
        return fn


_has_einsum: dict[str, bool] = {}


def has_einsum(backend: str) -> bool:
    """Check if ``{backend}.einsum`` exists, cache result for performance."""
    try:
        return _has_einsum[backend]
    except KeyError:
        try:
            get_func("einsum", backend)
            _has_einsum[backend] = True
        except (AttributeError, ModuleNotFoundError):
            _has_einsum[backend] = False
        return _has_einsum[backend]


_has_tensordot: dict[str, bool] = {}


def has_tensordot(backend: str) -> bool:
    """Check if ``{backend}.tensordot`` exists, cache result for performance."""
    try:
        return _has_tensordot[backend]
    except KeyError:
        try:
            get_func("tensordot", backend)
            _has_tensordot[backend] = True
        except (AttributeError, ModuleNotFoundError):
            _has_tensordot[backend] = False
        return _has_tensordot[backend]


@lru_cache(maxsize=None)
def _infer_backend_class(cls: type) -> str:
    return cls.__module__.split(".")[0]


def infer_backend(x: Any) -> str:
    """Infer the backend name from the module that defines ``x``'s class."""
    return _infer_backend_class(x.__class__)


def parse_backend(arrays: Sequence[Any], backend: str | None = "auto") -> str:
    """Resolve ``backend="auto"`` by inspecting the first array."""
    if (backend != "auto") and (backend is not None):
        return backend
    if not len(arrays):
        return "numpy"
    backend = infer_backend(arrays[0])

    # some arrays are defined in modules that don't implement tensordot
    # etc. so instead default to numpy
    if not has_tensordot(backend):
        return "numpy"

    return backend


##### backend conversion (used by ``ContractExpression``)


@to_backend_cache_wrap
def to_torch(array: Any) -> Any:
    """Convert ``array`` to a torch tensor."""
    torch = _torch_module()
    if isinstance(array, torch.Tensor):
        return array
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.as_tensor(np.asanyarray(array)).to(device)


@to_backend_cache_wrap
def to_cupy(array: Any) -> Any:
    """Convert ``array`` to a cupy array."""
    return importlib.import_module("cupy").asarray(array)


@to_backend_cache_wrap
def to_jax(array: Any) -> Any:
    """Convert ``array`` to a jax array."""
    return importlib.import_module("jax.numpy").asarray(array)


@to_backend_cache_wrap
def to_dask(array: Any) -> Any:
    """Convert ``array`` to a dask array."""
    return importlib.import_module("dask.array").asarray(array)


def _from_torch(array: Any) -> Any:
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    return array.numpy()


def _from_cupy(array: Any) -> Any:
    return importlib.import_module("cupy").asnumpy(array)


def _from_jax(array: Any) -> Any:
    return np.asanyarray(array)


def _from_dask(array: Any) -> Any:
    return np.asanyarray(array.compute())


_TO_BACKEND: dict[str, Callable[[Any], Any]] = {
    "torch": to_torch,
    "pytorch": to_torch,
    "cupy": to_cupy,
    "jax": to_jax,
    "jaxlib": to_jax,
    "dask": to_dask,
}

_FROM_BACKEND: dict[str, Callable[[Any], Any]] = {
    "torch": _from_torch,
    "pytorch": _from_torch,
    "cupy": _from_cupy,
    "jax": _from_jax,
    "jaxlib": _from_jax,
    "dask": _from_dask,
}

CONVERT_BACKENDS = tuple(_TO_BACKEND)
EVAL_CONSTS_BACKENDS = tuple(_TO_BACKEND)


def has_backend(backend: str) -> bool:
    """Checks if the backend supports numpy conversion."""
    return str(backend).lower() in _TO_BACKEND


def build_expression(backend: str, arrays: Sequence[Any], expr: Any) -> Any:
    """Build a function evaluating ``expr`` on ``backend`` arrays.

    numpy arrays are converted to ``backend`` on the way in and the result is
    converted back to numpy, matching ``opt_einsum``'s conversion backends.
    """
    key = str(backend).lower()
    to_fn = _TO_BACKEND.get(key)
    if to_fn is None:
        raise KeyError(f"Backend '{backend}' does not support numpy conversion.")
    from_fn = _FROM_BACKEND[key]

    def converted_contract(*args: Any) -> Any:
        result = expr._contract([to_fn(x) for x in args], backend=backend)
        return from_fn(result)

    return converted_contract


def evaluate_constants(backend: str, arrays: Sequence[Any], expr: Any) -> Any:
    """Convert constant arrays to ``backend`` and contract as many as possible."""
    key = str(backend).lower()
    to_fn = _TO_BACKEND.get(key)
    if to_fn is None:
        raise KeyError(f"Backend '{backend}' does not support numpy conversion.")

    const_arrays = [None if array is None else to_fn(array) for array in arrays]
    return expr(*const_arrays, backend=backend, evaluate_constants=True)


def to_backend(array: Any, backend: str = "numpy") -> Any:
    """Convert ``array`` to ``backend`` (no-op for numpy)."""
    to_fn = _TO_BACKEND.get(str(backend).lower())
    return array if to_fn is None else to_fn(array)
