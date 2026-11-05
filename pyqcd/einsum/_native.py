"""Backend dispatcher for the native einsum kernels.

Layout follows PyQCU: the CUDA backend is the Cython extension in
``pyqcd.einsum.cuda`` and the KUPL backend is the ctypes bridge in
``pyqcd.einsum.kupl``. ``auto`` prefers CUDA and falls back to KUPL, then to the
pure Python planner (``None`` return, never an exception).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

_BACKEND_ENV = "PYQCD_EINSUM_NATIVE"
_BACKENDS = ("kupl", "cuda")


def requested_backend() -> str:
    """Return the backend requested through ``PYQCD_EINSUM_NATIVE`` (``auto`` default)."""
    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    return value if value in _BACKENDS else "auto"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cuda_entry() -> Any:
    try:
        from .cuda import einsum as compiled
    except ImportError:
        return None
    return getattr(compiled, "applyPlanQcd", None)


def _kupl_entry() -> Any:
    from .kupl import einsum as kupl

    try:
        library = kupl._load_library()
    except Exception:
        return None
    return getattr(library, "applyPlanQcd", None)


def _order() -> tuple[str, ...]:
    requested = requested_backend()
    return ("cuda", "kupl") if requested == "auto" else (requested,)


def library_path(backend: str | None = None) -> Path:
    """Return the archive the given backend would load."""
    name = backend or requested_backend()
    if name == "auto":
        name = "cuda" if _cuda_entry() is not None else "kupl"
    if name == "cuda":
        return _repo_root() / "cpp" / "cuda" / "einsum" / "libpyqcdeinsum.so"
    from .kupl import einsum as kupl

    return kupl.library_path()


def backend_name() -> str:
    """Report which native backend is usable: ``cuda``, ``kupl`` or ``python``."""
    for name in _order():
        entry = _cuda_entry() if name == "cuda" else _kupl_entry()
        if entry is not None:
            return name
    return "python"


def available() -> bool:
    return backend_name() != "python"


def plan(inputs, output, sizes, planner_id=0, memory_limit=None):
    """Plan with the first usable native backend, or return ``None``."""
    from .kupl import einsum as kupl

    for name in _order():
        entry = _cuda_entry() if name == "cuda" else _kupl_entry()
        if entry is None:
            continue
        result = kupl._plan_with(
            entry, inputs, output, sizes, planner_id, memory_limit)
        if result is not None:
            return result
    return None
