"""Registry of contraction path optimizers.

The built-in optimizers live in :mod:`pyqcd.einsum._plan_py`; this module only
maps their public names onto that module and lets callers add their own.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

__all__ = [
    "PathOptimizer",
    "PATH_OPTIONS",
    "get_path_fn",
    "register_path_fn",
]

# Public name -> candidate attribute names in ``_plan_py`` (first hit wins).
# Later candidates are documented fallbacks used when the planner does not
# provide the specialized variant.
_BUILTIN_ATTRS: dict[str, tuple[str, ...]] = {
    "optimal": ("optimal",),
    "greedy": ("greedy",),
    "branch": ("branch",),
    "branch-all": ("branch_all", "branch"),
    "branch-2": ("branch_2", "branch"),
    "branch-1": ("branch_1", "branch"),
    "dp": ("dynamic_programming", "dp", "optimal"),
    "dynamic-programming": ("dynamic_programming", "dp", "optimal"),
    "auto": ("auto", "greedy"),
    "auto-hq": ("auto_hq", "auto", "optimal"),
    "random-greedy": ("random_greedy", "greedy"),
    "random-greedy-128": ("random_greedy_128", "random_greedy", "greedy"),
    "eager": ("greedy",),
    "opportunistic": ("greedy",),
}

PATH_OPTIONS: set[str] = set(_BUILTIN_ATTRS)

_REGISTERED: dict[str, Callable[..., Any]] = {}

_PLAN_MODULES: dict[str, Any] = {}


def _plan_py() -> Any:
    try:
        return _PLAN_MODULES["_plan_py"]
    except KeyError:
        module = _PLAN_MODULES["_plan_py"] = importlib.import_module("._plan_py", __package__)
        return module


class _BasePathOptimizer:
    """Fallback base class, used only if the planner is unavailable."""

    def _check_args_against_first_call(
        self,
        inputs: list[Any],
        output: Any,
        size_dict: dict[str, int],
    ) -> None:
        args = (inputs, output, size_dict)
        if not hasattr(self, "_first_call_args"):
            self._first_call_args = args
        elif args != self._first_call_args:
            raise ValueError(
                "The arguments specifying the contraction that this path optimizer "
                "instance was called with have changed - try creating a new instance."
            )

    def __call__(
        self,
        inputs: list[Any],
        output: Any,
        size_dict: dict[str, int],
        memory_limit: int | None = None,
    ) -> Any:
        raise NotImplementedError


def _path_optimizer_base() -> type:
    """The planner's ``PathOptimizer`` if it exists, so ``isinstance`` works."""
    try:
        base = getattr(_plan_py(), "PathOptimizer", None)
    except Exception:  # pragma: no cover - planner missing or broken
        base = None
    return base if isinstance(base, type) else _BasePathOptimizer


# Subclasses of this class are recognised by ``opt_einsum`` style path dispatch
# and are instances of the planner's own ``PathOptimizer`` when it provides one.
PathOptimizer = _path_optimizer_base()


def is_builtin_path_name(name: str) -> bool:
    """Whether ``name`` is served by :mod:`pyqcd.einsum._plan_py`."""
    key = name.lower()
    return key in _BUILTIN_ATTRS and key not in _REGISTERED


def register_path_fn(name: str, fn: Callable[..., Any]) -> None:
    """Add path finding function ``fn`` as an option with ``name``."""
    key = name.lower()
    if key in _REGISTERED or key in _BUILTIN_ATTRS:
        raise KeyError(f"Path optimizer '{name}' already exists.")

    _REGISTERED[key] = fn
    PATH_OPTIONS.add(key)


def get_path_fn(name: str) -> Any:
    """Get the path finding function registered as ``name``."""
    key = name.lower()

    try:
        return _REGISTERED[key]
    except KeyError:
        pass

    candidates = _BUILTIN_ATTRS.get(key)
    if candidates is not None:
        plan = _plan_py()
        for attr in candidates:
            fn = getattr(plan, attr, None)
            if fn is not None:
                return fn

    raise KeyError(
        f"Path optimizer '{name}' not found, valid options are {sorted(PATH_OPTIONS)}."
    )


# Alias kept for parity with ``opt_einsum.paths._PATH_OPTIONS`` consumers that
# only iterate over the registered names.
_PATH_OPTIONS = PATH_OPTIONS


def __getattr__(name: str) -> Any:
    """Forward planner attributes such as ``optimal`` or ``DynamicProgramming``."""
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(_plan_py(), name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
