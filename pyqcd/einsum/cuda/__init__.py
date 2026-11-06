"""CUDA binding for the native einsum library (Cython).

Mirrors ``pyqcu/cuda/qcu``: a compiled extension keeps pointer marshaling in C
and releases the GIL around the native call. Importing this package never
compiles anything; a missing extension raises ``ImportError`` carrying the build
hint so callers fall back to the KUPL or pure Python path.
"""

from __future__ import annotations

import importlib
from typing import Any

_BUILD_HINT = (
    "pyqcd.einsum.cuda.einsum is not built; run `bash install.sh cuda` "
    "(or `bash build-nv.sh`) from the repository root"
)


def _compiled_module():
    return importlib.import_module(f"{__name__}.einsum")


def compiled_available() -> bool:
    try:
        _compiled_module()
    except (ImportError, OSError):
        return False
    return True


def load():
    """Return the compiled module, raising ``ImportError`` with the build hint."""
    try:
        return _compiled_module()
    except (ImportError, OSError) as error:
        raise ImportError(_BUILD_HINT) from error


def backend_info() -> dict[str, Any] | None:
    """Return the compiled backend description, or ``None`` when unbuilt."""
    try:
        return _compiled_module().backend_info()
    except (ImportError, OSError):
        return None


def contract_mpi(*args: Any, **kwargs: Any) -> Any:
    """Run CUDA einsum with rank-local slicing through ``mpi4py``."""
    from .mpi import contract_mpi as execute

    return execute(*args, **kwargs)


def clear_mpi_cache() -> None:
    """Release cached rank-local CUDA MPI programs."""
    from .mpi import clear_mpi_cache as clear

    clear()


__all__ = [
    "backend_info",
    "clear_mpi_cache",
    "compiled_available",
    "contract_mpi",
    "load",
]
