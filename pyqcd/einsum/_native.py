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

from ._native_plan import (
    ExecutionPlanInfo,
    decode_execution_plan,
    dtype_from_id,
    dtype_id,
    encode_execution_plan,
)
from .kupl.einsum import NativeUnavailable

_BACKEND_ENV = "PYQCD_EINSUM_NATIVE"
_BACKENDS = ("kupl", "cuda")


def requested_backend() -> str:
    """Return the backend requested through ``PYQCD_EINSUM_NATIVE`` (``auto`` default)."""
    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    return value if value in _BACKENDS else "auto"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cuda_library_path() -> Path:
    package_path = Path(__file__).resolve().with_name("cuda") / "libpyqcdeinsum_cuda.so"
    if package_path.is_file():
        return package_path
    return _repo_root() / "cpp" / "cuda" / "einsum" / "libpyqcdeinsum_cuda.so"


def _cuda_module() -> Any:
    try:
        from .cuda import einsum as compiled
    except (ImportError, OSError):
        return None
    return compiled


def _cuda_entry(name: str = "applyPlanQcd") -> Any:
    compiled = _cuda_module()
    return None if compiled is None else getattr(compiled, name, None)


def _kupl_entry(name: str = "applyPlanQcd") -> Any:
    from .kupl import einsum as kupl

    try:
        return kupl.load_plan() if name == "applyPlanQcd" else kupl.load_contract()
    except NativeUnavailable:
        return None


def _order() -> tuple[str, ...]:
    requested = requested_backend()
    return ("cuda", "kupl") if requested == "auto" else (requested,)


def library_path(backend: str | None = None) -> Path:
    """Return the archive the given backend would load."""
    name = backend or requested_backend()
    if name == "auto":
        name = "cuda" if _cuda_entry() is not None else "kupl"
    if name == "cuda":
        return _cuda_library_path()
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


def load_plan_symbol(backend: str | None = None) -> Any:
    """Load the planning entry point for a backend, or raise NativeUnavailable."""
    selection = requested_backend() if backend is None or backend == "auto" else backend
    if selection not in (*_BACKENDS, "auto"):
        raise ValueError(f"unknown einsum backend: {selection!r}")
    for name in (("cuda", "kupl") if selection == "auto" else (selection,)):
        entry = _cuda_entry() if name == "cuda" else _kupl_entry()
        if entry is not None:
            return entry
    raise NativeUnavailable(f"native einsum planner is unavailable for {selection!r}")


def load_contract_symbol(backend: str | None = None) -> Any:
    """Load the native contract entry point, or raise NativeUnavailable."""
    selection = requested_backend() if backend is None or backend == "auto" else backend
    if selection not in (*_BACKENDS, "auto"):
        raise ValueError(f"unknown einsum backend: {selection!r}")
    for name in (("cuda", "kupl") if selection == "auto" else (selection,)):
        entry = (
            _cuda_entry("applyContractQcd")
            if name == "cuda"
            else _kupl_entry("applyContractQcd")
        )
        if entry is not None:
            return entry
    raise NativeUnavailable(f"native einsum contract is unavailable for {selection!r}")


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


def execute(plan_data, arrays, out=None, backend: str = "auto"):
    """Execute a plan with CUDA's ``contract_plan`` wrapper or the KUPL ctypes path."""
    backend = str(backend).lower()
    if backend in {"numpy", "cpu"}:
        backend = "kupl"
    elif backend in {"cupy", "gpu"}:
        backend = "cuda"
    selection = requested_backend() if backend == "auto" else backend
    if selection not in (*_BACKENDS, "auto"):
        raise ValueError(f"unknown einsum backend: {selection!r}")
    if selection in ("auto", "cuda"):
        compiled = _cuda_module()
        contract_plan_entry = (
            None if compiled is None else getattr(compiled, "contract_plan", None)
        )
        if contract_plan_entry is not None:
            return contract_plan_entry(plan_data, arrays, out=out)
        if selection == "cuda" or requested_backend() == "cuda":
            raise NativeUnavailable("CUDA einsum contract_plan is unavailable")

    from .kupl import einsum as kupl

    return kupl.execute(plan_data, arrays, out=out)


def compile(plan_data, backend: str = "auto"):
    """Return a reusable native program for a fixed execution plan."""
    backend = str(backend).lower()
    if backend in {"numpy", "cpu"}:
        backend = "kupl"
    elif backend in {"cupy", "gpu"}:
        backend = "cuda"
    selection = requested_backend() if backend == "auto" else backend
    if selection in ("auto", "kupl"):
        from .kupl import einsum as kupl

        try:
            return kupl.compile_program(plan_data, backend="numpy")
        except NativeUnavailable:
            if selection == "kupl":
                raise
    if selection in ("auto", "cuda"):
        compiled = _cuda_module()
        if compiled is not None and hasattr(compiled, "applyCompileQcd"):
            from .cuda._program import CudaProgram

            return CudaProgram(plan_data)
    raise NativeUnavailable(f"native einsum compiler is unavailable for {selection!r}")


def build_info() -> dict[str, Any]:
    """返回当前原生构建的可查询标识（后端 / 库路径）。"""
    backend = backend_name()
    path = str(library_path())
    try:
        from ..tools._backend import get_backend_name

        array_backend = get_backend_name()
    except Exception:  # noqa: BLE001 - 诊断信息，绝不抛错
        array_backend = 'unknown'
    return {
        'backend': backend,
        'native_available': backend != 'python',
        'array_backend': array_backend,
        'native_library': path,
    }
