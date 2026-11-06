"""Type declarations for the compiled CUDA einsum bridge."""

from collections.abc import Sequence
from typing import Any, Final

QCD_STATUS_OK: Final[int]
QCD_STATUS_INVALID: Final[int]
QCD_STATUS_MEMORY: Final[int]
QCD_STATUS_INTERNAL: Final[int]
STATUS_NAMES: Final[tuple[str, ...]]


def applyPlanQcd(set_ptrs: int, params: int) -> int:
    """Plan through ``libpyqcdeinsum`` using CPU table addresses."""
    ...


def applyContractQcd(plan: int, tensors: int, out: int, params: int) -> int:
    """Apply a kernel plan using CPU pointer tables and device addresses."""
    ...


def applyCompileQcd(plan: int, params: int, out_context: int) -> int:
    """Compile a plan into a reusable native CUDA context."""
    ...


def applyCompiledQcd(context: int, tensors: int, out: int, params: int) -> int:
    """Execute through a context returned by ``applyCompileQcd``."""
    ...


def applyDestroyQcd(context: int) -> None:
    """Destroy a native CUDA context; zero is a no-op."""
    ...


def applyInitQcd(set_ptrs: int, params: int) -> None:
    """Initialise backend session state, if any."""
    ...


def applyEndQcd(set_ptrs: int, params: int) -> None:
    """Release backend session state, if any."""
    ...


def backend_info() -> dict[str, str | bool]:
    """Return the compiled backend identity."""
    ...


def contract_plan(
    plan_data: object,
    arrays: Sequence[object],
    out: object | None = None,
) -> Any:
    """Execute a version-2 plan using CuPy array pointers."""
    ...
