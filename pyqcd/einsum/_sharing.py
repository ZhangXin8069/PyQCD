"""Sharing of intermediate results between contractions.

Intermediate tensors are memoized inside a :func:`shared_intermediates`
context; the cache is keyed by object identity so arrays that happen to be
equal are never conflated.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import numbers
import threading
from collections import Counter, defaultdict
from typing import Any, Iterator

__all__ = [
    "currently_sharing",
    "get_sharing_cache",
    "shared_intermediates",
    "count_cached_ops",
    "transpose_cache_wrap",
    "tensordot_cache_wrap",
    "einsum_cache_wrap",
    "to_backend_cache_wrap",
]

_SHARING_STACK: dict[int, list[dict[Any, Any]]] = defaultdict(list)

_PARSER: dict[str, Any] = {}


def _parser() -> Any:
    try:
        return _PARSER["parser"]
    except KeyError:
        module = _PARSER["parser"] = importlib.import_module(".parser", __package__)
        return module


def _parse_einsum_input(operands: Any) -> tuple[str, str, list[Any]]:
    return _parser().parse_einsum_input(operands)


def _alpha_canonicalize(equation: str) -> str:
    get_symbol = _parser().get_symbol
    rename: dict[str, str] = {}
    for name in equation:
        if name in ".,->":
            continue
        if name not in rename:
            rename[name] = get_symbol(len(rename))
    return "".join(rename.get(x, x) for x in equation)


def currently_sharing() -> bool:
    """Check if we are currently sharing a cache -- thread specific."""
    return threading.get_ident() in _SHARING_STACK


def get_sharing_cache() -> dict[Any, Any]:
    """Return the most recent sharing cache -- thread specific."""
    return _SHARING_STACK[threading.get_ident()][-1]


def _add_sharing_cache(cache: dict[Any, Any]) -> None:
    _SHARING_STACK[threading.get_ident()].append(cache)


def _remove_sharing_cache() -> None:
    tid = threading.get_ident()
    _SHARING_STACK[tid].pop()
    if not _SHARING_STACK[tid]:
        del _SHARING_STACK[tid]


@contextlib.contextmanager
def shared_intermediates(cache: dict[Any, Any] | None = None) -> Iterator[dict[Any, Any]]:
    """Context in which contraction intermediates are shared.

    The yielded cache holds every intermediate produced inside the context, so
    those tensors stay alive until both the context and the cache are dropped.
    Pass an existing ``cache`` to resume sharing across contexts.
    """
    if cache is None:
        cache = {}
    _add_sharing_cache(cache)
    try:
        yield cache
    finally:
        _remove_sharing_cache()


def count_cached_ops(cache: dict[Any, Any]) -> Counter:
    """Returns a counter of the types of each op in the cache."""
    return Counter(key[0] for key in cache.keys())


def _save_tensors(*tensors: Any) -> None:
    """Save tensors in the cache to prevent their ids from being recycled."""
    cache = get_sharing_cache()
    for tensor in tensors:
        cache["tensor", id(tensor)] = tensor


def _memoize(key: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Memoize ``fn(*args, **kwargs)`` using the given ``key``."""
    cache = get_sharing_cache()
    if key in cache:
        return cache[key]
    result = fn(*args, **kwargs)
    cache[key] = result
    return result


def transpose_cache_wrap(transpose: Any) -> Any:
    """Decorate a ``transpose()`` implementation to be memoized when sharing."""

    @functools.wraps(transpose)
    def cached_transpose(a, axes, backend: str = "numpy"):
        if not currently_sharing():
            return transpose(a, axes, backend=backend)

        _save_tensors(a)
        axes = tuple(axes)
        key = "transpose", backend, id(a), axes
        return _memoize(key, transpose, a, axes, backend=backend)

    return cached_transpose


def tensordot_cache_wrap(tensordot: Any) -> Any:
    """Decorate a ``tensordot()`` implementation to be memoized when sharing."""

    @functools.wraps(tensordot)
    def cached_tensordot(x, y, axes=2, backend: str = "numpy"):
        if not currently_sharing():
            return tensordot(x, y, axes, backend=backend)

        _save_tensors(x, y)
        if isinstance(axes, numbers.Number):
            axes = (
                list(range(len(x.shape)))[len(x.shape) - axes :],
                list(range(len(y.shape)))[:axes],
            )
        axes = tuple(axes[0]), tuple(axes[1])
        key = "tensordot", backend, id(x), id(y), axes
        return _memoize(key, tensordot, x, y, axes, backend=backend)

    return cached_tensordot


def einsum_cache_wrap(einsum: Any) -> Any:
    """Decorate an ``einsum()`` implementation to be memoized when sharing."""

    @functools.wraps(einsum)
    def cached_einsum(*args: Any, **kwargs: Any):
        if not currently_sharing():
            return einsum(*args, **kwargs)

        backend = kwargs.pop("backend", "numpy")
        equation = args[0]
        inputs, output, operands = _parse_einsum_input(args)
        inputs = inputs.split(",")

        _save_tensors(*operands)

        # hash modulo commutativity by computing a canonical ordering and names
        canonical = sorted(zip(inputs, map(id, operands)), key=lambda x: x[1])
        canonical_ids = tuple(id_ for _, id_ in canonical)
        canonical_inputs = ",".join(input_ for input_, _ in canonical)
        canonical_equation = _alpha_canonicalize(canonical_inputs + "->" + output)

        key = "einsum", backend, canonical_equation, canonical_ids
        return _memoize(key, einsum, equation, *operands, backend=backend)

    return cached_einsum


def to_backend_cache_wrap(to_backend: Any = None, constants: Any = False) -> Any:
    """Decorate a ``to_backend()`` implementation to be memoized when sharing."""
    if to_backend is None:
        return functools.partial(to_backend_cache_wrap, constants=constants)

    if constants:

        @functools.wraps(to_backend)
        def cached_to_backend(array, constant: Any = False):
            if not currently_sharing():
                return to_backend(array, constant=constant)

            key = to_backend.__name__, id(array), constant
            return _memoize(key, to_backend, array, constant=constant)

    else:

        @functools.wraps(to_backend)
        def cached_to_backend(array):
            if not currently_sharing():
                return to_backend(array)

            key = to_backend.__name__, id(array)
            return _memoize(key, to_backend, array)

    return cached_to_backend
