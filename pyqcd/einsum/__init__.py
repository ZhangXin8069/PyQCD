"""Optimized Einstein summation for :mod:`pyqcd`.

This package is a self-contained reimplementation of ``opt_einsum``'s public
API: :func:`contract_path` finds a cost minimizing contraction order,
:func:`contract` evaluates it, and :func:`contract_expression` caches the
resulting contraction list for repeated evaluation.

Only re-exports live here; the implementation is in the private modules.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import backends, paths
from ._api import contract, contract_path
from ._expr import ContractExpression, contract_expression, format_const_einsum_str, shape_only
from ._pathinfo import PathInfo
from ._sharing import count_cached_ops, shared_intermediates
from .backends.dispatch import (
    get_func,
    has_backend,
    has_einsum,
    has_tensordot,
    infer_backend,
    parse_backend,
)
from .paths import PATH_OPTIONS, PathOptimizer, get_path_fn, register_path_fn

__all__ = [
    "backends",
    "blas",
    "ContractExpression",
    "contract",
    "contract_expression",
    "contract_path",
    "count_cached_ops",
    "format_const_einsum_str",
    "get_func",
    "get_path_fn",
    "get_symbol",
    "has_backend",
    "has_einsum",
    "has_tensordot",
    "helpers",
    "infer_backend",
    "parse_backend",
    "parser",
    "PATH_OPTIONS",
    "PathInfo",
    "PathOptimizer",
    "paths",
    "register_path_fn",
    "shape_only",
    "shared_intermediates",
]

# names served by sibling modules that may be written concurrently
_LAZY_MODULES = ("helpers", "blas", "parser")
_LAZY_PATH_ATTRS = (
    "BranchBound",
    "DynamicProgramming",
    "RandomGreedy",
    "linear_to_ssa",
    "ssa_to_linear",
)


def __getattr__(name: str) -> Any:
    """Lazily resolve modules and planner classes owned by sibling modules."""
    if name in _LAZY_MODULES:
        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module
        return module

    if name == "get_symbol":
        value = importlib.import_module(".parser", __package__).get_symbol
        globals()["get_symbol"] = value
        return value

    if name in _LAZY_PATH_ATTRS:
        try:
            value = getattr(paths, name)
        except AttributeError:
            pass
        else:
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
