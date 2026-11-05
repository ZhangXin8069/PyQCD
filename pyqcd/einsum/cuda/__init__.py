"""CUDA binding for the native einsum library (Cython).

Mirrors ``pyqcu/cuda/qcu``: a compiled extension keeps pointer marshaling in C
and releases the GIL around the native call. Importing this package never
compiles anything; a missing extension raises ``ImportError`` carrying the build
hint so callers fall back to the KUPL or pure Python path.
"""

from __future__ import annotations

from typing import Any

_BUILD_HINT = (
    "pyqcd.einsum.cuda.einsum is not built; run `bash install.sh cuda` "
    "(or `bash build-nv.sh`) from the repository root"
)


def compiled_available() -> bool:
    try:
        from . import einsum  # noqa: F401
    except ImportError:
        return False
    return True


def load():
    """Return the compiled module, raising ``ImportError`` with the build hint."""
    try:
        from . import einsum
    except ImportError as error:
        raise ImportError(_BUILD_HINT) from error
    return einsum


def backend_info() -> dict[str, Any] | None:
    """Return the compiled backend description, or ``None`` when unbuilt."""
    try:
        from . import einsum
    except ImportError:
        return None
    return einsum.backend_info()


__all__ = ["backend_info", "compiled_available", "load"]
