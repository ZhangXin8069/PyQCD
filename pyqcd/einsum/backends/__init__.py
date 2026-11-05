"""Backend dispatch for :mod:`pyqcd.einsum`."""

from __future__ import annotations

from .dispatch import (
    CONVERT_BACKENDS,
    EVAL_CONSTS_BACKENDS,
    build_expression,
    evaluate_constants,
    get_func,
    has_backend,
    has_einsum,
    has_tensordot,
    infer_backend,
    parse_backend,
    to_backend,
    to_cupy,
    to_dask,
    to_jax,
    to_torch,
)

__all__ = [
    "CONVERT_BACKENDS",
    "EVAL_CONSTS_BACKENDS",
    "build_expression",
    "evaluate_constants",
    "get_func",
    "has_backend",
    "has_einsum",
    "has_tensordot",
    "infer_backend",
    "parse_backend",
    "to_backend",
    "to_cupy",
    "to_dask",
    "to_jax",
    "to_torch",
]
