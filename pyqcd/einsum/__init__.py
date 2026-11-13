"""Optimized Einstein summation for :mod:`pyqcd`.

This package is a self-contained reimplementation of ``opt_einsum``'s public
API: :func:`contract_path` finds a cost minimizing contraction order,
:func:`contract` evaluates it, and :func:`contract_expression` caches the
resulting contraction list for repeated evaluation.

Only re-exports live here; the implementation is in the private modules.

Stability: ``pyqcd`` 0.4.1 is a frozen qcd-einsum stable release (baseline tag
``stab31``). Behaviour-changing edits are disallowed by default until the next
qcd-einsum stable release; see ``AGENTS.md`` §0 for the three exceptions.
"""

from __future__ import annotations
import os

import importlib
from typing import Any

from . import backends, paths
from ._api import contract, contract_path
from ._expr import ContractExpression, contract_expression, format_const_einsum_str, shape_only
from ._pathinfo import PathInfo
from ._specialize import describe as specializations
from ._native import build_info
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

from . import _threads as _threads_module

# 默认按物理核数配置 OpenMP 与 BLAS 线程（可用环境变量覆盖，见 _threads 文档）。
_threads_module.apply()

__all__ = [
    "configure_threads",
    "thread_info",
    "thread_profiles",
    "probe",
    "format_report",
    "backends",
    "build_info",
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
    "query_specialization",
    "register_path_fn",
    "shape_only",
    "shared_intermediates",
    "specializations",
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


def thread_profiles() -> dict:
    """Return the auto thread profiles (``kernel`` / ``generic``) for this machine."""
    return _threads_module.profiles()


def thread_info() -> dict:
    """Return the effective OpenMP/BLAS thread configuration and its source."""
    return _threads_module.current()


def configure_threads(omp: int | None = None, blas: int | None = None,
                      force: bool = False) -> dict:
    """Override the thread configuration at runtime.

    ``configure_threads(4, 2)`` pins 4 OpenMP threads and 2 BLAS threads. Passing
    ``None`` keeps the auto-resolved value. The standard ``OMP_NUM_THREADS`` /
    ``OPENBLAS_NUM_THREADS`` variables still win when they are set explicitly.
    """
    if omp is not None:
        os.environ[_threads_module.OMP_ENV] = str(int(omp))
    if blas is not None:
        os.environ[_threads_module.BLAS_ENV] = str(int(blas))
    return _threads_module.apply(force=True)


def probe() -> dict:
    """Diagnose whether the CUDA fast path is really active in this install."""
    from . import _probe as probe_module

    return probe_module.probe()


def query_specialization(subscripts: str) -> dict:
    """Report which specialized path ``subscripts`` takes, without contracting.

    Pure query: no operands are built and no global state is touched. Returns the
    same mapping as :func:`pyqcd.einsum.verify_specialized.query`.
    """
    from .verify_specialized import query

    return query(subscripts)


def format_report(report: dict) -> str:
    """Render :func:`probe` output as human-readable text."""
    from . import _probe as probe_module

    return probe_module.format_report(report)
