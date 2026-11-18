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
#
# The primary spelling uses **underscores** because it is the module attribute
# spelling (``_plan_py.branch_1``): a name that cannot be looked up exports a
# ``KeyError`` from ``get_path_fn`` for a perfectly reasonable input.
# ``branch_1`` first, its hyphen spelling as an alias below.
#
# 表里只有**七个叶子**，**拼写统一为下划线**（用户裁定 2026-11-18：
# "branch* 统一使用下划线，通用性更强"）。连字符拼写 ``branch-1`` 等**不再是可写名字**，
# 解析时抛 ``KeyError``——保留它只会让同一路径有两条拼写，README 里写哪个、配置表里
# 存哪个都会漂移。
#
# 已删除的名字一律抛 ``KeyError``：``auto``/``auto-hq``（缺省值，不是可写名字，见
# :data:`_IMPLICIT_DEFAULT_NAMES`）、``eager``/``opportunistic``/``random-greedy``/
# ``random-greedy-128``（静默等同 ``greedy``）、``branch``（全宽 ``branch_all`` 的
# 重复入口）、三个连字符 ``branch-*``。
_BUILTIN_ATTRS: dict[str, tuple[str, ...]] = {
    "optimal": ("optimal",),
    "greedy": ("greedy",),
    "branch_all": ("branch_all", "branch"),
    "branch_2": ("branch_2", "branch"),
    "branch_1": ("branch_1", "branch"),
    "dp": ("dynamic_programming", "dp", "optimal"),
}


#: ``optimize="none"`` = *no optimizer*. It is not a path search: the caller
#: sends the one n-ary contraction step (``optimize=False``). The sentinel is a
#: callable so it survives the "resolves to something callable" contract, and
#: ``_api`` recognises it instead of asking the planner for a step table.
def _no_path(
    inputs: list[Any],
    output: Any,
    size_dict: dict[str, int],
    memory_limit: int | None = None,
) -> list[tuple[int, ...]]:
    """The "no optimizer" path: one single step over every operand, in order."""
    return [tuple(range(len(inputs)))]


_NO_PATH = _no_path

_BUILTIN_ATTRS["none"] = ()

#: Names that are **only** ever an implicit default, never a writable spelling:
#: ``_api.contract``/``contract_path`` normalise ``True``/``None`` to ``"auto"``,
#: so the planner must still plan it. It is deliberately kept out of
#: :data:`_BUILTIN_ATTRS` (hence out of :data:`PATH_OPTIONS`) so that
#: ``get_path_fn("auto")`` raises like every other removed name — ``_api._find_path``
#: decides "planner or registry" with :func:`is_builtin_path_name`, which is the
#: one place that has to know about this entry.
_IMPLICIT_DEFAULT_NAMES: frozenset[str] = frozenset({"auto"})

#: The **advertised** names — the ones a caller should write. Hyphen spellings are
#: aliases, not part of this list; they resolve to the same objects as their
#: underscore counterparts.
PATH_OPTIONS_MAIN: frozenset[str] = frozenset({
    "greedy",
    "optimal",
    "dp",
    "branch_1",
    "branch_2",
    "branch_all",
    "none",
})

#: Every name :func:`get_path_fn` answers: the seven leaves plus the spellings kept
#: for existing callers — the hyphen aliases ``branch-1``/``branch-2``/``branch-all``
#: and ``dp``'s ``dynamic-programming``. Nothing else is in here: the degenerate
#: aliases (``eager``, ``opportunistic``, ``random-greedy*``), ``branch`` and the
#: default-only ``auto``/``auto-hq`` all raise ``KeyError``.
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
    """Whether ``name`` is planned by :mod:`pyqcd.einsum._plan_py`.

    True for the registered builtins **and** for :data:`_IMPLICIT_DEFAULT_NAMES`
    (``auto``): the planner does serve the default value, it just is not a name a
    caller may write. Registered names are not builtin — ``register_path_fn`` must
    keep overriding a name that the planner could otherwise also answer.
    """
    key = name.lower()
    if key in _REGISTERED:
        return False
    return key in _BUILTIN_ATTRS or key in _IMPLICIT_DEFAULT_NAMES


def register_path_fn(name: str, fn: Callable[..., Any]) -> None:
    """Add path finding function ``fn`` as an option with ``name``."""
    key = name.lower()
    if key in _REGISTERED or key in _BUILTIN_ATTRS:
        raise KeyError(f"Path optimizer '{name}' already exists.")

    _REGISTERED[key] = fn
    PATH_OPTIONS.add(key)


def get_path_fn(name: str) -> Any:
    """Get the path finding function registered as ``name``.

    Every name in :data:`PATH_OPTIONS` resolves: the seven leaves, the hyphen
    aliases (``branch-1`` ≡ ``branch_1``: the **same** object, not a second
    implementation), ``dynamic-programming`` ≡ ``dp``, and ``none``, which
    resolves to :data:`_NO_PATH` — the "no optimizer" sentinel — instead of
    raising ``KeyError``.

    ``auto`` is **not** a name: it is ``optimize``'s default value
    (:data:`_IMPLICIT_DEFAULT_NAMES`), so it raises like any other removed
    spelling. Every removed name raises ``KeyError`` listing the legal options.
    """
    key = name.lower()

    try:
        return _REGISTERED[key]
    except KeyError:
        pass

    if key == "none":
        return _NO_PATH

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
