"""Primary contraction routines: :func:`contract_path` and :func:`contract`."""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any

from . import paths
from ._pathinfo import PathInfo
from ._sharing import einsum_cache_wrap
from . import (_autotune, _cache, _equivalence, _kernel_force, _profile_cache,
               _shape_paths, _specialize, _threads)
from .backends import dispatch as backends
from .backends.dispatch import get_func, parse_backend

__all__ = ["contract", "contract_path", "last_kernel", "last_profile",
           "cache_clear", "cache_stats"]

_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EINSUM_PUNCTUATION = ",->."
_BASE_TYPES = (bool, int, float, complex, str, bytes)

_MODULES: dict[str, Any] = {}
_NATIVE_PLAN_CACHE: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
_NATIVE_PLAN_CACHE_LIMIT = 128
_CONTRACT_PROGRAM_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_CONTRACT_PROGRAM_CACHE_LIMIT = 64
# 规划结果缓存：重复的 (方程, 形状, dtype, 后端, 优化器) 不必重新规划。规划在
# 小/中型通用方程上占总调用时间的 31–84%（8 算子链实测 0.76 ms / 0.86 ms），
# 而解析只占 0.002 ms，所以这里只缓存 contraction_list。
_CONTRACT_PLAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_CONTRACT_PLAN_CACHE_LIMIT = 128

# 规划器（``contract_path`` 的 ``_find_path``）结果缓存：只对可哈希的 ``optimize``
# 生效，键含输入集合、输出集合、size_dict、memory_limit、eq 与形状——少一维就会串味。
_PLAN_PATH_CACHE = _cache.LRUCache("api.find_path", maxsize=512)
_FUSED_ENV_OWNED = False
# 最近一次成功调用的 (签名, 程序)，**必须单次赋值**。
#
# 原先是 ``_LAST_CONTRACT_SIGNATURE`` 与 ``_LAST_CONTRACT_PROGRAM`` 两个独立
# 全局，靠「先写签名、再写程序」维持配对。但快路径是「先读签名比较、后读程序」，
# 两步之间若被别的线程换掉程序，就会把 **A 的签名配上 B 的程序**直接执行——
# 程序只校验操作数个数/形状/dtype，**不校验方程**，于是静默算错方程且无异常
# （2026-11-15 巡查实证）。合成一个元组后，读与写各是一次原子引用操作，
# 撕裂窗口消失。
_LAST_CONTRACT: tuple[tuple[Any, ...], Any] | None = None
_NATIVE_DISABLED: set[tuple[Any, ...]] = set()
_NATIVE_WARNED_REASONS: set[str] = set()

# Compiled native contexts reused by plain ``contract()`` calls. Without this
# cache every non-expression call rebuilt the backend context from the plan bytes
# (CUDA: a full ``cudaMalloc`` workspace per call, KUPL: a plan re-parse).
_NATIVE_CONTEXT_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_NATIVE_CONTEXT_CACHE_LIMIT = 16
_NATIVE_CONTEXT_CACHE_ENV = "PYQCD_EINSUM_CONTEXT_CACHE_LIMIT"
_NATIVE_CONTEXT_LIMIT_RESOLVED = False
_NATIVE_CONTEXT_CACHE_LOCK = threading.RLock()
# Keys whose plan could not be compiled into a reusable context. They keep the
# historical per-call native execution instead of being disabled outright.
_NATIVE_CONTEXT_UNAVAILABLE: set[tuple[Any, ...]] = set()


def _sibling(name: str) -> Any:
    """Import (and cache) a sibling module of this package."""
    try:
        return _MODULES[name]
    except KeyError:
        module = _MODULES[name] = importlib.import_module(f".{name}", __package__)
        return module


def _parser() -> Any:
    return _sibling("parser")


def _helpers() -> Any:
    return _sibling("helpers")


def _blas() -> Any:
    return _sibling("blas")


def _planner() -> Any:
    return _sibling("_plan_py")


def _executor() -> Any:
    return _sibling("_execute")


def _native_executor() -> Any:
    return _sibling("_native")


def __getattr__(name: str) -> Any:
    """Expose ``ArrayShaped``/``shape_only`` from :mod:`._expr` without a cycle."""
    if name in ("ArrayShaped", "shape_only"):
        return getattr(_sibling("_expr"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


##### helpers


def _choose_memory_arg(memory_limit: Any, size_list: Sequence[int]) -> int | None:
    """Turn the public ``memory_limit`` value into a number of elements."""
    if memory_limit == "max_input":
        return max(size_list)

    if isinstance(memory_limit, str):
        raise ValueError("memory_limit must be None, int, or the string Literal['max_input'].")

    if memory_limit is None:
        return None

    if memory_limit < 1:
        if memory_limit == -1:
            return None
        raise ValueError("Memory limit must be larger than 0, or -1")

    return int(memory_limit)


def _normalize_optimize(optimize: Any) -> Any:
    """Public ``optimize`` spelling → the value the planner understands.

    ``"none"`` (case insensitive, surrounding blanks ignored) means *no optimizer*,
    which is exactly ``False``: one n-ary einsum, no pairwise planning. Both entry
    points (:func:`contract_path` and :func:`contract`) run through here so the two
    cannot drift apart.

    Every other value — other strings, ``False``/``True``/``None``, callables and
    explicit path lists — is returned **unchanged**; unknown optimizer names must
    keep raising in :func:`_find_path` instead of being silently swallowed here.

    ``"none"`` is normalised **without** knowing the kernel, so it stays usable
    for the topology kernel too: that one is handed its order by the kernel and
    never consults ``optimize`` for planning. See :func:`_resolve_optimize`.
    """
    if isinstance(optimize, str) and optimize.strip().lower() == "none":
        return False
    return optimize


def _resolve_optimize(optimize: Any, forced: tuple[str, ...]) -> Any:
    """``optimize`` after accounting for the kernel that will actually run.

    ``optimize="none"`` has a second meaning for the **topology** kernel, which
    is the one kernel that cannot take an external order: its step order comes
    from the kernel itself. For that kernel ``"none"`` is the only legal
    ``optimize`` value ("the kernel decides the order"), so it is accepted as
    written. Everywhere else — ``generic`` / ``backend`` / ``dedicated`` — the
    same spelling is simply "no optimizer", i.e. :func:`_normalize_optimize`'s
    ``False``: the equation's own operand order, no planning.

    ``forced`` is :func:`pyqcd.einsum._kernel_force.forced_labels`' result.
    """
    if isinstance(optimize, str) and optimize.strip().lower() == "none":
        if forced == ("topology",):
            return "none"            # 顺序由内核决定：这是唯一合法的 optimize 输入
        return False                 # 其余对象：等同于不使用 optimize
    return _normalize_optimize(optimize)


#: 拓扑内核在**生成计划**时用的收缩序（内部常量，**不是**用户参数）。
#: v3 的收缩序固化在编译期步表 ``topology_steps.h`` 里，这个值只影响 pack 流量建模
#: 按哪一份计划记账。历史上它曾以 ``optimize_topology_v2`` 的名字对外暴露，那个参数
#: 已在 0.5.9 随 v2 通用执行器删除、并于 0.5.12 把内部残留一并清掉。
_TOPOLOGY_PLAN_ORDER = "greedy"


def _normalize_as_equivalence(value: Any) -> bool | None:
    """``as_equivalence`` 的三态归一：显式布尔、可识别字符串，或 ``None``（跟随环境）。

    存在的理由是一个**反转开关**陷阱：``bool("false")`` 是 ``True``，所以直接把字符串
    交给布尔判定会让"关掉改写"变成"强制改写"。这里显式映射 ``true/false/1/0``
    （大小写与首尾空白不敏感），``force`` 与 ``true`` 同义（CLI 的第四态，
    见 :func:`pyqcd.einsum.test.parse_as_equivalence`）；其余字符串一律 ``ValueError``
    ——既不是开也不是关的取值必须由调用方改写，不许猜。
    """
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "force"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(
        f"as_equivalence={value!r} 不可识别：用 True/False（或 ``true``/``false``/"
        "``1``/``0``），``None`` 表示跟随 PYQCD_EINSUM_AS_EQUIVALENCE")


def _filter_einsum_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter out default contract kwargs to pass to various backends."""
    kwargs = kwargs.copy()
    ret: dict[str, Any] = {}

    order = kwargs.pop("order", "K")
    if order != "K":
        ret["order"] = order

    casting = kwargs.pop("casting", "safe")
    if casting != "safe":
        ret["casting"] = casting

    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        ret["dtype"] = dtype

    out = kwargs.pop("out", None)
    if out is not None:
        ret["out"] = out

    ret.update(kwargs)
    return ret


def _get_shape(x: Any) -> tuple[int, ...]:
    """Shape of an array-like object; fallback for ``parser.get_shape``."""
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if isinstance(x, _BASE_TYPES):
        return ()
    if isinstance(x, Sequence):
        shape = []
        while isinstance(x, Sequence) and not isinstance(x, _BASE_TYPES):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)
    raise ValueError(
        f"Cannot determine the shape of {x}, can only determine the shape of array-like objects."
    )


def _has_valid_einsum_chars_only(einsum_str: str) -> bool:
    return all((c in _EINSUM_SYMBOLS) or (c in _EINSUM_PUNCTUATION) for c in einsum_str)


def _find_output_str(subscripts: str) -> str:
    tmp_subscripts = subscripts.replace(",", "")
    return "".join(s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1)


def _convert_to_valid_einsum_chars(einsum_str: str) -> str:
    get_symbol = _parser().get_symbol
    symbols = sorted(set(einsum_str) - set(",->"))
    replacer = {x: get_symbol(i) for i, x in enumerate(symbols)}
    return "".join(replacer.get(x, x) for x in einsum_str)


##### path and contraction-list bookkeeping


_PARAM_CACHE: dict[int, tuple[frozenset[str], bool]] = {}


def _signature_of(fn: Any) -> tuple[frozenset[str], bool]:
    """Accepted parameter names of ``fn`` plus whether it takes ``**kwargs``."""
    key = id(fn)
    cached = _PARAM_CACHE.get(key)
    if cached is None:
        try:
            parameters = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):  # pragma: no cover - builtins
            cached = (frozenset(), True)
        else:
            names = frozenset(
                p.name for p in parameters if p.kind is not inspect.Parameter.VAR_KEYWORD
            )
            has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
            cached = (names, has_kwargs)
        _PARAM_CACHE[key] = cached
    return cached


def _accepts_param(fn: Any, name: str) -> bool:
    names, has_kwargs = _signature_of(fn)
    return has_kwargs or name in names


def _accepts_var_kwargs(fn: Any) -> bool:
    return _signature_of(fn)[1]


def _normalize_path(path: Any) -> list[tuple[int, ...]]:
    return [tuple(int(i) for i in step) for step in path]


def _plan_steps(
    eq: str,
    input_shapes: Sequence[Sequence[int]],
    optimize: str,
    memory_limit: int | None,
    use_blas: bool,
) -> list[tuple[Any, ...]]:
    """Ask the planner for the contraction steps of ``eq``."""
    shapes = [tuple(int(d) for d in shape) for shape in input_shapes]
    plan = _planner()
    if _accepts_param(plan.optimize, "use_blas"):
        steps = plan.optimize(eq, shapes, optimize, memory_limit, use_blas=use_blas)
    else:  # pragma: no cover - older planner without the ``use_blas`` knob
        steps = plan.optimize(eq, shapes, optimize, memory_limit)

    if steps is None:
        raise RuntimeError(f"planner returned no contraction steps for '{eq}'")
    return [tuple(step) for step in steps]


def _find_path(
    optimize: Any,
    num_ops: int,
    input_sets: Sequence[frozenset[str]],
    output_set: frozenset[str],
    size_dict: dict[str, int],
    memory_limit: int | None,
    eq: str,
    input_shapes: Sequence[Sequence[int]],
    use_blas: bool,
) -> tuple[Any, Any]:
    """Resolve ``optimize`` into a path, or into planner supplied steps."""
    single_step = _normalize_path([tuple(range(num_ops))])

    if optimize is False:
        return single_step, None

    if isinstance(optimize, str):
        if num_ops <= 2:
            return single_step, None
        if paths.is_builtin_path_name(optimize):
            try:
                steps = _plan_steps(eq, input_shapes, optimize, memory_limit, use_blas)
            except KeyError:
                # planner does not implement this name: fall back to the registry
                pass
            else:
                path = [tuple(sorted(int(i) for i in step[0])) for step in steps]
                return path, steps

        path_fn = paths.get_path_fn(optimize)
        return _normalize_path(path_fn(input_sets, output_set, size_dict, memory_limit)), None

    if callable(optimize):
        if num_ops <= 2:
            return single_step, None
        return (
            _normalize_path(optimize(input_sets, output_set, size_dict, memory_limit)),
            None,
        )

    return _normalize_path(optimize), None


def _find_path_cached(optimize: Any, num_ops: int,
                      input_sets: Sequence[frozenset[str]],
                      output_set: frozenset[str], size_dict: dict[str, int],
                      memory_limit: int | None, eq: str,
                      input_shapes: Sequence[Sequence[int]], use_blas: bool,
                      *, use: bool | None = None) -> tuple[Any, Any]:
    """:func:`_find_path` 的缓存外壳。

    规划是纯函数：结果只依赖这九个入参，因此可以安全复用。两种情形**不缓存**——
    显式路径（``list``/``tuple``）是调用方的数据，可调用优化器自带状态。
    """
    if not isinstance(optimize, (str, bool)) and optimize is not None:
        return _find_path(optimize, num_ops, input_sets, output_set, size_dict,
                          memory_limit, eq, input_shapes, use_blas)
    key = (optimize, int(num_ops), tuple(input_sets), output_set,
           tuple(sorted(size_dict.items())), memory_limit, eq,
           tuple(tuple(int(dim) for dim in shape) for shape in input_shapes),
           bool(use_blas))
    cached = _PLAN_PATH_CACHE.get(key, None, use=use)
    if cached is not None:
        return cached                    # 只读复用：调用方不得原地修改 path/steps
    result = _find_path(optimize, num_ops, input_sets, output_set, size_dict,
                        memory_limit, eq, input_shapes, use_blas)
    _PLAN_PATH_CACHE.put(key, result, use=use)
    return result


def cache_stats() -> dict[str, Any]:
    """所有 einsum 缓存的命中统计：``{"enabled": bool, "caches": {名字: info}}``。"""
    return _cache.stats()


def cache_clear() -> None:
    """清空所有 einsum 缓存并归零统计。"""
    _cache.clear()


def _build_contraction_list(
    path: Sequence[Sequence[int]],
    input_list: Sequence[str],
    input_shapes: Sequence[Sequence[int]],
    output_set: frozenset[str],
    output_subscript: str,
    size_dict: dict[str, int],
    use_blas: bool,
) -> tuple[list[tuple[Any, ...]], list[int], list[int], list[int]]:
    """Build the contraction list for an explicitly supplied ``path``."""
    helpers, blas = _helpers(), _blas()
    find_output_shape = getattr(_parser(), "find_output_shape", None)

    input_list = list(input_list)
    input_shapes = [tuple(shape) for shape in input_shapes]
    input_sets = [frozenset(term) for term in input_list]

    contraction_list: list[tuple[Any, ...]] = []
    cost_list: list[int] = []
    scale_list: list[int] = []
    size_list: list[int] = []
    total = len(path)

    for cnum, contract_inds in enumerate(path):
        # make sure we remove inds from right to left
        contract_inds = tuple(sorted((int(i) for i in contract_inds), reverse=True))

        contraction = helpers.find_contraction(contract_inds, input_sets, output_set)
        if len(contraction) == 3:
            out_inds, input_sets, idx_removed = contraction
            idx_contract = frozenset(out_inds) | frozenset(idx_removed)
        else:  # pragma: no cover - opt_einsum style 4-tuple
            out_inds, input_sets, idx_removed, idx_contract = contraction

        cost_list.append(
            helpers.flop_count(idx_contract, bool(idx_removed), len(contract_inds), size_dict)
        )
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shapes.pop(x) for x in contract_inds]

        if use_blas:
            do_blas = blas.can_blas(tmp_inputs, "".join(out_inds), idx_removed, tmp_shapes)
        else:
            do_blas = False

        if cnum == total - 1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        if find_output_shape is not None:
            result_shape = tuple(find_output_shape(tmp_inputs, tmp_shapes, idx_result))
        else:
            result_shape = tuple(size_dict[char] for char in idx_result)

        input_list.append(idx_result)
        input_shapes.append(result_shape)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # saving the remaining terms at each step is expensive for large expressions
        remaining: tuple[str, ...] | None = tuple(input_list) if len(input_list) <= 20 else None

        contraction_list.append((contract_inds, idx_removed, einsum_str, remaining, do_blas))

    return contraction_list, cost_list, scale_list, size_list


def _measure_steps(
    steps: Sequence[tuple[Any, ...]],
    size_dict: dict[str, int],
    use_blas: bool,
) -> tuple[list[tuple[Any, ...]], list[int], list[int], list[int]]:
    """Derive the costs and intermediate sizes of planner supplied steps."""
    helpers = _helpers()

    contraction_list: list[tuple[Any, ...]] = []
    cost_list: list[int] = []
    scale_list: list[int] = []
    size_list: list[int] = []

    for step in steps:
        if len(step) != 5:
            raise RuntimeError(f"expected a 5-field contraction step, got {step!r}")
        inds, idx_removed, einsum_str, remaining, do_blas = step

        left, separator, right = einsum_str.partition("->")
        if not separator:
            raise RuntimeError(f"contraction step without an output: {einsum_str!r}")

        idx_contract = frozenset(left.replace(",", ""))
        cost_list.append(helpers.flop_count(idx_contract, bool(idx_removed), len(inds), size_dict))
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(frozenset(right), size_dict))

        contraction_list.append(
            (tuple(inds), frozenset(idx_removed), einsum_str, remaining, do_blas if use_blas else False)
        )

    return contraction_list, cost_list, scale_list, size_list


##### public API


def contract_path(
    subscripts: Any,
    *operands: Any,
    use_blas: bool = True,
    optimize: Any = "auto",
    memory_limit: Any = None,
    shapes: bool = False,
    cache: bool | None = None,
    **kwargs: Any,
) -> tuple[list[tuple[int, ...]], PathInfo]:
    """Find a contraction order ``path``, without performing the contraction.

    Parameters:
        subscripts: Specifies the subscripts for summation.
        *operands: The arrays (or, with ``shapes=True``, the shapes) to contract.
        use_blas: Whether pairwise contractions may be dispatched to BLAS.
        optimize: ``False`` (or the string ``"none"``, case insensitive) for a
            single n-ary einsum, an explicit path, the name of a registered path
            optimizer, a ``PathOptimizer`` instance, or ``True``/``None``/``"auto"``
            to let the planner decide.
        memory_limit: Upper bound on the number of elements of any intermediate.
            ``None``/``-1`` means unlimited and ``"max_input"`` means the largest
            input tensor.
        shapes: Whether shapes rather than arrays have been supplied.
        cache: ``None`` follows ``PYQCD_EINSUM_CACHE`` (on by default); ``False``
            disables the planner cache for this call (neither read nor write).

    Returns:
        ``(path, PathInfo)``: the contraction order and a printable summary.
    """
    optimize = _normalize_optimize(optimize)
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    einsum_call_arg = kwargs.pop("einsum_call", False)
    if len(kwargs):
        raise TypeError(f"Did not understand the following kwargs: {kwargs.keys()}")

    parser, helpers = _parser(), _helpers()

    # Python side parsing
    operands_ = [subscripts] + list(operands)
    input_subscripts, output_subscript, operands_prepped = parser.parse_einsum_input(
        operands_, shapes=shapes
    )

    input_list = input_subscripts.split(",")
    input_sets = [frozenset(term) for term in input_list]
    if shapes:
        input_shapes = [tuple(shape) for shape in operands_prepped]
    else:
        get_shape = getattr(parser, "get_shape", _get_shape)
        input_shapes = [tuple(get_shape(operand)) for operand in operands_prepped]

    output_set = frozenset(output_subscript)
    indices = frozenset(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict: dict[str, int] = {}
    for tnum, term in enumerate(input_list):
        shape = input_shapes[tnum]

        if len(shape) != len(term):
            raise ValueError(
                f"Einstein sum subscript '{input_list[tnum]}' does not contain the "
                f"correct number of indices for operand {tnum}."
            )
        for cnum, char in enumerate(term):
            dim = int(shape[cnum])

            if char in size_dict:
                # for broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        f"Size of label '{char}' for operand {tnum} ({size_dict[char]}) does not match previous "
                        f"terms ({dim})."
                    )
            else:
                size_dict[char] = dim

    # size of each input array plus the output array
    input_size_list = [
        helpers.compute_size_by_dict(term, size_dict) for term in input_list + [output_subscript]
    ]
    memory_arg = _choose_memory_arg(memory_limit, input_size_list)

    num_ops = len(input_list)
    inner_product = (sum(len(term) for term in input_sets) - len(indices)) > 0
    naive_cost = helpers.flop_count(indices, inner_product, num_ops, size_dict)

    eq = f"{input_subscripts}->{output_subscript}"
    path, steps = _find_path_cached(
        optimize, num_ops, input_sets, output_set, size_dict, memory_arg, eq,
        input_shapes, use_blas, use=cache
    )

    if steps is None:
        contraction_list, cost_list, scale_list, size_list = _build_contraction_list(
            path, input_list, input_shapes, output_set, output_subscript, size_dict, use_blas
        )
    else:
        contraction_list, cost_list, scale_list, size_list = _measure_steps(
            steps, size_dict, use_blas
        )

    opt_cost = sum(cost_list)

    if einsum_call_arg:
        return operands_prepped, contraction_list  # type: ignore[return-value]

    path_info = PathInfo(
        contraction_list,
        input_subscripts,
        output_subscript,
        indices,
        path,
        scale_list,
        naive_cost,
        opt_cost,
        size_list,
        size_dict,
    )

    return path, path_info


@einsum_cache_wrap
def _einsum(*operands: Any, **kwargs: Any) -> Any:
    """Base einsum, but with pre-parse for valid characters if a string is given."""
    fn = get_func("einsum", kwargs.pop("backend", "numpy"))

    if not isinstance(operands[0], str):
        return fn(*operands, **kwargs)

    einsum_str, operands = operands[0], operands[1:]

    # do we need to temporarily map indices into [a-z,A-Z] range?
    if not _has_valid_einsum_chars_only(einsum_str):
        if "->" not in einsum_str:
            einsum_str += "->" + _find_output_str(einsum_str)
        einsum_str = _convert_to_valid_einsum_chars(einsum_str)

    kwargs = _filter_einsum_defaults(kwargs)
    return fn(einsum_str, *operands, **kwargs)


#: 最近一次专用内核尝试的记录，由 :func:`last_kernel` 读出。
#: ``None`` 表示本进程还没有任何调用走进专用分支。
_LAST_KERNEL: dict[str, Any] | None = None


def last_kernel() -> dict[str, Any] | None:
    """最近一次调用的选路记录：``requested`` / ``ran`` / ``attempted`` / ``reasons``。

    ``requested`` 是本次调用要求的**逻辑名**：``None`` = 调用方没给 ``kernel_force``
    （按缺省 ``generic`` 执行）；``()`` = 明确要求 ``generic``（不试专用内核）；
    ``("dedicated",)`` / ``("topology",)`` / ``("backend",)`` = 强制该条路线。

    **``None`` 只可能来自"调用方没给"**：``_kernel_force.forced_labels`` 永不返回
    ``None``，缺省同样解析成 ``()``。所以这两个字段是同一条事实的两种拼写，报告里
    要判断"用户是否真的点名"只能看这里记录的那个值照抄了哪个输入，不能用它反推
    显式性（见 ``_kernel_force.forced_labels`` 的 docstring）。

    ``ran`` 是实际跑成的标签（``None`` = 一个都没跑成；``backend`` 表示直接走了后端
    自己的 einsum），``attempted`` 是真正下发过的具体标签（``kupl_dedicated`` …），
    ``reasons`` 是逐标签的失败原因（与 ``KernelUnavailableError.reasons`` 同源）。

    记录在每次 ``contract()`` 走到选路处**整体覆盖**，因此读到的一定是最近一次调用的
    结果，不会残留上一次的命中——"到底跑了哪个内核"不必再去翻 ``LAST_FAILURE`` 或负
    缓存的内部状态。``requested`` 只报**调用方**的要求（参数或
    ``PYQCD_EINSUM_KERNEL_FORCE``）；配置画像给出的建议不是调用方的要求，它的命中由
    自己的一行提示报告。
    """
    return None if _LAST_KERNEL is None else dict(_LAST_KERNEL)


_SPECIALIZED_KERNEL_WARNED: set[tuple[str, str]] = set()


def _specialized_kernel_attempts(
    operands: Sequence[Any],
) -> tuple[tuple[str, Any, Any], ...]:
    """Return only the native entry points for this operand backend."""
    if _cupy_operands(operands):
        from . import _kernel_cuda

        return (
            ("cuda_dedicated", _kernel_cuda, _kernel_cuda.try_dedicated),
            ("cuda_topology", _kernel_cuda, _kernel_cuda.try_topology_apply),
        )
    from . import _kernel_kupl

    return (
        ("kupl_dedicated", _kernel_kupl, _kernel_kupl.try_dedicated),
        ("kupl_topology", _kernel_kupl, _kernel_kupl.try_topology_apply),
    )


def _resolve_forced_label(
    name: str,
    operands: Sequence[Any],
) -> str | None:
    """逻辑名 → 本次操作数后端的 attempt 标签。

    ``dedicated`` → ``kupl_dedicated``（numpy 操作数）/ ``cuda_dedicated``（cupy），
    ``topology`` 同理。角色从标签后缀取（``kupl_dedicated`` → ``dedicated``），所以
    映射跟着 ``_specialized_kernel_attempts`` 的当前后端走，不写死第二份后端表。
    ``name`` 不在尝试表里时返回 ``None``——调用方按"未尝试"报告，不猜一个可能跑不了
    的入口。
    """
    for label, _module, _attempt in _specialized_kernel_attempts(operands):
        if label.rpartition("_")[2] == name:
            return label
    return None


def _attempt_kernel(label: str, module: Any, attempt: Any, subscripts: Any,
                    operands: Sequence[Any]) -> tuple[Any, str]:
    """旁路负缓存与预算门槛后**真试一次**；返回 ``(result or None, 失败原因)``。

    "强制即真试一次"：强制模式下不许被上一次的失败记录挡住，所以每个 attempt 前套上
    内核模块自己的 ``bypass_negative_cache()``（模块没提供就照旧）。异常收成该标签的
    原因向上报告，绝不静默跳过。

    **同时旁路预算门槛**（``bypass_plan_gate()``）：arena / 单步 GEMM / pack 流量那几道
    门是**选路策略**（"自动分派时这一格不划算"），不是能力上限。用户点名
    ``--kernel-force dedicated`` 时他要的就是"这一个内核在这条方程上到底多快"——
    在选路层就把他挡回去，等于永远不给那个数字（2026-11-19 用户实测：
    ``pack/flops = 3.950 >= 1.0`` 直接判死，连一次下发都没有）。门槛仍会照算并写进
    ``LAST_INFO['budget_gate']``，所以"跑了但超预算"这条事实不会丢。
    """
    # ``*_topology`` 入口照旧收 ``optimize``（它只影响 pack 流量建模，不参与 v3 的
    # 收缩序），值来自内部常量 ``_TOPOLOGY_PLAN_ORDER``——不再是透传参数。
    extra = ({"optimize": _TOPOLOGY_PLAN_ORDER}
             if label.rpartition("_")[2] == "topology" else {})
    bypass = getattr(module, "bypass_negative_cache", None)
    gate = getattr(module, "bypass_plan_gate", None)
    gates = ExitStack()
    try:
        if gate is not None:
            gates.enter_context(gate())
        if bypass is None:
            result = attempt(subscripts, operands, **extra)
        else:
            with bypass():
                result = attempt(subscripts, operands, **extra)
    except Exception as exc:  # noqa: BLE001 - 收成原因，由调用方统一报告
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        gates.close()
    if result is not None:
        return result, ""
    # 失败后才读原因：早读会拿到上一次调用残留的值
    return None, str(module.LAST_FAILURE.get("reason", ""))


def _run_forced_kernel(
    subscripts: Any,
    operands: Sequence[Any],
    requested: str,
    record: dict[str, Any],
) -> Any:  # noqa: D401 - 见 docstring
    """强制内核的唯一执行点：真试一次，跑不成即抛。

    ``requested`` 是**单个逻辑名**（``dedicated`` / ``topology``）：先按操作数后端解析
    成具体标签再下发。失败时把"逻辑名 → 具体标签"的对照一并交给异常，消息里两种名字
    都在（用户对得上自己写的取值，也看得见哪个入口真的被试过）。

    没有"依次尝试多个"的分支：逗号链式已取消（见 :mod:`._kernel_force`），一个调用
    只有一条被要求的路线，跑不成就是跑不成。

    拓扑内核的 ``optimize`` 取自内部常量 :data:`_TOPOLOGY_PLAN_ORDER`（见
    :func:`_attempt_kernel`）——调用方不再有可传的"计划次序"参数。
    """
    concrete = _resolve_forced_label(requested, operands)
    if concrete is not None:
        by_label = {entry[0]: entry for entry in _specialized_kernel_attempts(operands)}
        _label, module, attempt = by_label[concrete]
        result, reason = _attempt_kernel(
            concrete, module, attempt, subscripts, operands)
        record["attempted"] = (concrete,)
        if result is not None:
            record["ran"] = concrete
            record["reasons"] = {}
            return result
        reasons = {concrete: reason}
        resolved = {requested: (concrete,)}
    else:
        # 该后端没有这条路线（例如只有 ``kupl_*`` 的尝试表里查 ``cuda_*``）：
        # 一次都没下发，如实报"未尝试"，不猜一个跑不了的入口。
        reasons = {}
        resolved = {}
        record["attempted"] = ()
    record["ran"] = None
    record["reasons"] = dict(reasons)
    raise _kernel_force.KernelUnavailableError(
        requested, subscripts, reasons, record["attempted"], resolved)


_KERNEL_SKIP_WARNED: set[tuple[str, str]] = set()


def _kernel_labels_for(entry: dict[str, Any]) -> tuple[str, ...]:
    """Which kernel labels can serve this registered shape."""
    declared = entry.get("kernels")
    if declared is not None:
        return tuple(str(label) for label in declared)
    # 字面条目只带 name/kind/note：按形态关键词解析，规则由 _specialize 单点声明。
    return _specialize.kernel_labels(str(entry.get("name", "")))


def _no_kernel_reason(entry: dict[str, Any]) -> str:
    """内核没留下原因时的兜底说明。

    专用 kernel 只接受注册表里的字面拼写，对拓扑等价的其他写法会静默返回
    ``None``（不改 ``LAST_FAILURE``）；这种命中必须说清"为什么没跑"。
    """
    subscripts = str(entry.get("subscripts", ""))
    if subscripts:
        return ("the equation is topologically equivalent to "
                f"{subscripts}, but the specialised kernel only accepts that "
                "exact spelling")
    return "unknown reason"


def _warn_kernel_skipped_once(entry: dict[str, Any], reasons: list[str]) -> None:
    """该形态注册了专用 kernel，但这次没有跑成 —— 立刻告诉调用方原因。

    静默回落是数值契约（结果仍然正确），但它会伪装成"PyQCD 很慢"，所以命中
    kernel 形态却没跑 kernel 时必须报警；同一(形态, 原因)只报一次，避免刷屏。
    """
    detail = "; ".join(reason for reason in reasons if reason) or _no_kernel_reason(entry)
    key = (str(entry.get("name", "?")), detail)
    if key in _KERNEL_SKIP_WARNED:
        return
    _KERNEL_SKIP_WARNED.add(key)
    warnings.warn(
        f"pyqcd.einsum: specialised kernel did not run for "
        f"{entry.get('name', '?')} ({detail}). The contraction falls back to the "
        "generic path — results are still correct, but slower. Check "
        "`python -m pyqcd --gpu-probe`, PYQCD_EINSUM_KERNEL_* switches and "
        "PYQCD_CUDA_ARCHITECTURES.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_specialized_kernel_once(exc: BaseException) -> None:
    """专用内核抛异常时告警，**每个不同的失败各报一次**。

    与 :func:`_warn_kernel_skipped_once` 同一粒度（那里用 ``(形态, 原因)`` 作键）。
    先前这里是进程级布尔：先来一个无关异常，后面真正的失败就再也报不出来——用户会
    照着那条无关的提示去排查。
    """
    key = (type(exc).__name__, str(exc)[:120])
    if key in _SPECIALIZED_KERNEL_WARNED:
        return
    _SPECIALIZED_KERNEL_WARNED.add(key)
    warnings.warn(
        "pyqcd.einsum specialized kernel failed; using the generic path: "
        f"{type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )


def _original_einsum(backend_name: str, subscripts: str, operands: Sequence[Any],
                     out: Any = None, optimize: Any = None):
    """交给 ``opt_einsum.contract``：外部实现的 einsum，收缩序由 ``optimize`` 决定。

    这是 ``kernel_force="original"`` 的唯一执行点，与 :func:`_backend_einsum` 同层
    ——两者都**不进**专用内核分派，区别只在"谁来规划与执行"：
    ``backend`` 是后端自己的 ``einsum``，``original`` 是 ``opt_einsum``。

    ``optimize`` 在本库是**下划线叶子**（``branch_1`` / ``branch_all`` / ``none``），
    而 ``opt_einsum`` 只认连字符与 ``False``（实测 ``KeyError: Path optimizer
    'branch_1' not found``；字符串 ``'none'`` / ``'false'`` 同样 KeyError，只有
    ``False`` 表示"不规划"）。转型单点在
    :func:`pyqcd.einsum._kernel_force.opt_einsum_path_name`，这里不再写第二份映射。

    ``backend_name`` 由**操作数**推断（``_opt_einsum_backend_name``），不读命令行：
    ``opt_einsum`` 的 ``backend`` 缺省 ``'auto'`` 按数组模块选，只传 numpy 副本
    等于在另一台机器上做参照（2026-11-19 用户实测过这个坑）。

    失败一律外抛：``original`` 是**强制**标签，跑不成就该报错，不静默回落。
    """
    try:
        import opt_einsum as oe
    except ImportError as exc:
        # ``opt_einsum`` 是**可选依赖**（``pip install "pyqcd[einsum]"``）。缺它时
        # 必须抛一个**说清装什么**的错误：裸 ``ModuleNotFoundError`` 会让用户以为
        # 是 pyqcd 自己的 bug，而 ``original`` 的语义是"强制即真试一次、跑不成即抛"，
        # 与其它内核的 ``KernelUnavailableError`` 一致才对。
        raise _kernel_force.KernelUnavailableError(
            _kernel_force.ORIGINAL_LABEL, subscripts,
            {_kernel_force.ORIGINAL_LABEL: (
                "opt_einsum is not installed; install it with "
                '`pip install "pyqcd[einsum]"` (or `pip install opt_einsum`)')},
            attempted=(_kernel_force.ORIGINAL_LABEL,)) from exc

    from ._kernel_force import opt_einsum_path_name

    if backend_name is None:
        raise ValueError(
            "opt_einsum does not have a backend for these operands: "
            f"{type(operands[0]).__module__ if operands else 'no operand'}")
    path = opt_einsum_path_name(optimize)
    kwargs: dict[str, Any] = {"optimize": path, "backend": backend_name}
    if out is not None:
        kwargs["out"] = out
    try:
        return oe.contract(subscripts, *operands, **kwargs)
    except TypeError:
        # ``out=`` 在某些后端（实测 torch）没有实现：去掉它再试一次，但**不吞掉**
        # 第一次的真实原因——重试也失败时把两个异常都写进消息。
        if "out" not in kwargs:
            raise
        kwargs.pop("out")
        try:
            return oe.contract(subscripts, *operands, **kwargs)
        except Exception as exc:                # noqa: BLE001 - 供上层如实报告
            raise RuntimeError(
                f"opt_einsum.contract(out=...) is not supported by backend "
                f"{backend_name!r}, and the retry without out= failed too: "
                f"{type(exc).__name__}: {exc}") from exc


def _opt_einsum_backend_name(operands: Sequence[Any]) -> str | None:
    """``opt_einsum.contract(backend=...)`` 该填什么；它不支持该后端时给 ``None``。

    判据用 ``oe.backends.has_backend``（公开接口），失败才看内部表——后者只是兜底，
    不让"公开接口不存在"变成"这个后端不可用"。
    """
    name = "numpy"
    for operand in operands:
        module = type(operand).__module__.split(".", 1)[0]
        if module:
            name = module
            break
    if name == "numpy":
        return "numpy"
    try:
        import opt_einsum as oe
    except Exception:                            # noqa: BLE001 - 没装就说不支持
        return None
    probe = getattr(oe.backends, "has_backend", None)
    if callable(probe):
        try:
            return name if probe(name) else None
        except Exception:                        # noqa: BLE001 - 探测失败按不支持
            return None
    return name if name in getattr(oe.backends, "_FUNCS_BACKENDS", {}) else None


def _backend_einsum(backend_name: str, subscripts: str, operands: Sequence[Any],
                     out: Any = None):
    """Call the backend's own einsum, preferring its optimised path."""
    einsum_call = backends.get_func("einsum", backend_name)
    kwargs: dict[str, Any] = {}
    if len(operands) > 1:
        kwargs["optimize"] = True      # numpy/cupy/torch 内部会用 tensordot/cuBLAS
    if out is not None:
        kwargs["out"] = out
    try:
        return einsum_call(subscripts, *operands, **kwargs)
    except TypeError:
        kwargs.pop("optimize", None)
        return einsum_call(subscripts, *operands, **kwargs)


def _cupy_operands(operands: Sequence[Any]) -> bool:
    """True when every operand is a cupy array (the GPU delegation case)."""
    if not operands:
        return False
    for operand in operands:
        if not type(operand).__module__.startswith("cupy"):
            return False
    return True


def _is_elementwise_equation(subscripts: str) -> bool:
    """True when the equation sums over nothing (element-wise / permutation / diagonal).

    Measured on this machine, ``ij,ij->ij`` at 4000² costs 89 ms through the step
    executor versus 29 ms for one native ``einsum`` call, and ``ii->i`` costs 0.53 ms
    versus 0.01 ms. Those equations have no BLAS structure at all, so the backend's
    own kernel is strictly better — the planner adds only bookkeeping.
    """
    text = subscripts.replace(" ", "")
    if "->" not in text:
        return False                      # implicit output: leave to the normal path
    lhs, rhs = text.split("->", 1)
    terms = [term for term in lhs.split(",") if term]
    if not terms:
        return False
    if len(terms) == 1:
        return True                       # transpose / diagonal / trace / scale
    if "..." in text:
        return False                      # ellipsis needs the planner's broadcasting
    summed = set("".join(terms)) - set(rhs)
    return not summed


def _axis_permuted(operand: Any, perm: Sequence[int]) -> Any:
    """按 ``numpy.transpose(perm)`` 语义重排轴（torch 要 ``permute``）。"""
    if type(operand).__module__.split(".")[0] == "torch":
        return operand.permute(*perm)
    return operand.transpose(tuple(perm))


def _contiguous_copy(operand: Any) -> Any:
    """落成连续内存：``transpose`` 得到的是视图，会被原生特化的连续门拒掉。"""
    contiguous = getattr(operand, "contiguous", None)
    if callable(contiguous):
        return contiguous()
    module = type(operand).__module__.split(".")[0]
    if module == "cupy":                                 # pragma: no cover - 需 GPU
        import cupy

        return cupy.ascontiguousarray(operand)
    try:
        import numpy

        return numpy.ascontiguousarray(operand)
    except Exception:                                    # noqa: BLE001
        return operand


def _expected_output_shape(subscripts: str,
                           operands: Sequence[Any]) -> tuple[int, ...] | None:
    """原方程要求的输出形状；算不出（省略号/隐式输出/形状对不上）时返回 ``None``。

    等价改写只在内部发生，调用方拿到的必须是**原方程**的答案，所以这里按原方程
    独立算一遍输出形状，用来卡住"映射层万一给出错字段"的情形。
    """
    try:
        text = "".join(str(subscripts).split())
        if "..." in text or "->" not in text:
            return None
        lhs, rhs = text.split("->")
        terms = lhs.split(",")
        if len(terms) != len(operands):
            return None
        sizes: dict[str, int] = {}
        for term, operand in zip(terms, operands):
            shape = _get_shape(operand)
            if len(shape) != len(term):
                return None
            for label, size in zip(term, shape):
                known = sizes.setdefault(label, size)
                if known != size:
                    return None
        return tuple(sizes[label] for label in rhs)
    except Exception:                                    # noqa: BLE001
        return None


##### 等价改写：表里的可执行载荷

#: 已经做过**数值自检**的载荷（token 见 :func:`_equivalence_token`）。首次套用一条
#: 载荷时要与"重新匹配一次"的结果对拍，通过后同一份载荷不再付这份代价。
_EQUIVALENCE_VERIFIED: set[tuple[Any, ...]] = set()
#: 自检失败的载荷：本进程内**不再套用**（每次都退回重新匹配）。表是**只读**的，所以
#: 这里只影响本进程，不写回——"这条载荷不可信"由下次 tune 重新测量。
_EQUIVALENCE_REJECTED: set[tuple[Any, ...]] = set()
_EQUIVALENCE_LOCK = threading.Lock()
#: 自检的容差：两份结果来自同一条收缩的两种写法，只该差舍入。
_EQUIVALENCE_SELF_CHECK_RTOL = 1e-12


def _equivalence_payload(row: dict[str, Any] | None) -> Any | None:
    """配置表的一行 → **可直接施加**的 :class:`~._equivalence.Equivalence`；不可用时 ``None``。

    载荷四列由 :func:`._profile_cache.equivalence_payload` 解码（JSON → 元组），这里只把
    解出来的字段装成与 :func:`._equivalence.normalize_equation` 同形的对象，于是"照表
    施加"与"查完再施加"共用同一段骨架（:func:`_apply_equivalence`）。

    不是等价行（没有 ``target_equation`` / 没有可动字段）、坏 JSON 一律返回 ``None``：
    调用方据此**退回重新匹配**，绝不抛异常（表只影响选路，坏了只该慢一点）。
    """
    if not isinstance(row, dict):
        return None
    try:
        payload = _profile_cache.equivalence_payload(row)
    except Exception:                                    # noqa: BLE001 - 载荷绝不外抛
        return None
    if not isinstance(payload, dict):
        return None
    equation = str(payload.get("target_equation") or "")
    if not equation:
        return None
    return _equivalence.Equivalence(
        equation=equation,
        operand_order=payload.get("operand_order"),
        operand_permutations=payload.get("operand_permutations") or (),
        output_permutation=payload.get("output_permutation"),
        description="配置表的等价行（跳过匹配，直接施加）",
        kind="",
    )


def _apply_equivalence(subscripts: str, operands: Sequence[Any], mapping: Any, *,
                       use_blas: bool, optimize: Any, memory_limit: Any,
                       backend: str, specialize: bool, kernel_force: str | None,
                       profile: bool = False) -> Any | None:
    """把一条改写**照做**：重排 → 按需 transpose + 连续副本 → 递归 → 按需转回。

    ``mapping`` 只需要有 ``equation`` / ``operand_order`` / ``operand_permutations``
    / ``output_permutation`` / ``kind`` 五个字段——既可以是
    :func:`._equivalence.normalize_equation` 给的 :class:`~._equivalence.Equivalence`，
    也可以是由配置表的等价行装出来的同形对象。这里**不查映射、不读表**，任何一步不
    适用（含"轴序对不上张量秩"这类不可信字段）都返回 ``None``——绝不降级成"半个改写"。

    ``profile`` 是内层递归调用的画像开关：走**等价行载荷**时打开（载荷里的 equation
    就是语料写法，它的配置行才值得读），走"查完再施加"时保持原来的关闭。

    计时不在这里做：``wall_ms`` 的口径是**整个** ``contract()`` 的墙钟时间
    （含本函数的重排与收尾），由外层那一次调用负责，所以这里没有"只计内部收缩"
    的分段计时，也没有可以与之相加的第二列。
    """
    if mapping.kind == "identity":
        return None
    if not (mapping.operand_order or mapping.operand_permutations
            or mapping.output_permutation):
        return None

    order = mapping.operand_order
    if order is None:
        ordered = list(operands)
    else:
        if len(order) != len(operands) or sorted(order) != list(range(len(operands))):
            return None                  # 不是整个操作数集合的排列：载荷不可信
        ordered = [operands[index] for index in order]

    perms = mapping.operand_permutations or ()
    adjusted: list[Any] = []
    for index, operand in enumerate(ordered):
        perm = perms[index] if index < len(perms) else None
        if perm is None:
            adjusted.append(operand)
            continue
        rank = len(_get_shape(operand))
        if len(perm) != rank or sorted(perm) != list(range(rank)):
            return None                  # 轴置换对不上张量秩：载荷不可信
        adjusted.append(_contiguous_copy(_axis_permuted(operand, perm)))

    result = contract(mapping.equation, *adjusted, out=None, use_blas=use_blas,
                      optimize=optimize, memory_limit=memory_limit,
                      backend=backend, specialize=specialize,
                      kernel_force=kernel_force, as_equivalence=False,
                      profile=profile)
    if mapping.output_permutation is not None:
        result = result.transpose(tuple(mapping.output_permutation))

    expected = _expected_output_shape(subscripts, operands)
    if expected is not None and tuple(result.shape) != expected:
        raise RuntimeError(
            "等价映射改变了输出形状：原方程要求 "
            f"{expected}，按改写 {mapping.equation!r} 得到 {tuple(result.shape)}；"
            "这是映射层的缺陷（结果不予返回）。请把该方程报告给维护者，"
            "并可用 as_equivalence=False 绕过。")
    return result


def _equivalence_token(subscripts: str, mapping: Any) -> tuple[Any, ...]:
    """一份载荷的**内容**指纹：同一份内容只需自检一次（进程内）。

    含原方程与改写的全部字段——同一份载荷换一条原方程就是另一件事，不能共用结论。
    """
    order = getattr(mapping, "operand_order", None)
    output = getattr(mapping, "output_permutation", None)
    perms = tuple(None if perm is None else tuple(perm)
                  for perm in (getattr(mapping, "operand_permutations", ()) or ()))
    return ("".join(str(subscripts).split()),
            str(getattr(mapping, "equation", "")),
            None if order is None else tuple(order),
            perms,
            None if output is None else tuple(output))


def _results_agree(left: Any, right: Any,
                   tolerance: float = _EQUIVALENCE_SELF_CHECK_RTOL) -> bool:
    """两份结果是否在 ``tolerance`` 相对误差内一致；算不出来就返回 ``False``。

    自检只用来决定"要不要相信表里的载荷"，所以**判不了就当不一致**：那会退回重新
    匹配一次，代价是一次多余的查询，而不是一个未经验证的改写。
    """
    try:
        worst = float(abs(left - right).max())
        scale = float(abs(right).max())
    except Exception:                                    # noqa: BLE001 - 判不了即不一致
        return False
    return worst <= tolerance * (scale if scale > 0.0 else 1.0)


def _payload_result(subscripts: str, operands: Sequence[Any], mapping: Any,
                    applied: Any, token: tuple[Any, ...], *,
                    cache: bool | None, call: dict[str, Any]) -> Any | None:
    """首次套用一条表里的载荷时的**数值自检**：与"重新匹配一次"的结果对拍。

    表里的载荷可能是别的机器/别的语料版本写下的，中间也许换过映射层。第一次真拿它
    跑出来的结果，必须和"现在重新匹配一次再施加"的结果一致（rel <= 1e-12）才继续
    相信它；不一致、或者现在根本匹配不到，就**降级为重新匹配的结果**（本进程内不再
    套用这条载荷）。匹配抛异常同样按失败处理：宁可多查一次，也不返回未经验证的改写。

    表是**只读**的（§1 唯一写者是 tune），所以这里刷新不了那一行：不一致只在**本
    进程**里生效，磁盘上的旧载荷留给下次 tune 重新测量。
    """
    call = dict(call)
    try:
        fresh = _equivalence.normalize_equation(subscripts, use=cache)
    except Exception:                                    # noqa: BLE001 - 不许影响数值
        fresh = None
    fresh_result = (None if fresh is None
                    else _apply_equivalence(subscripts, operands, fresh, **call))
    if fresh_result is not None and _results_agree(applied, fresh_result):
        with _EQUIVALENCE_LOCK:
            _EQUIVALENCE_VERIFIED.add(token)
        return applied
    with _EQUIVALENCE_LOCK:
        _EQUIVALENCE_REJECTED.add(token)
    return fresh_result


def _contract_via_equivalence(subscripts: str, operands: Sequence[Any], *,
                              use_blas: bool, optimize: Any, memory_limit: Any,
                              backend: str, specialize: bool,
                              kernel_force: str | None,
                              cache: bool | None = None,
                              payload: Any = None) -> Any | None:
    """用等价写法跑一次并把结果转回原输出轴序；不适用时返回 ``None``。

    只在真正的通用收缩路径上被调用（逐元素与 cupy 委派早已返回）。任何一步不满足
    都返回 ``None`` 让调用方走原路径——这里**绝不**降级成"半个改写"。

    ``specialize`` 是调用方**已求值**的开关（``_specialize.enabled()`` 只算一次），
    因此这里是纯布尔判断，不再重复读环境变量。

    ``payload`` 是配置表等价行给出的**可执行改写**（:func:`_equivalence_payload` 装出）：
    给了它就直接套用、**不调用** :func:`._equivalence.normalize_equation`——首次套用会做
    一次数值自检（见 :func:`_payload_result`），之后同一份载荷连自检都省掉。载荷不可用
    （缺字段 / 与张量秩对不上）时静默退回下面那条"真的查一次"的路径。

    表只读：这里**不写**任何结论，所以一次查询的结论不会跨进程留下来——下次同类调用
    照常重查，直到有人为这条键 tune。
    """
    if not _equivalence.enabled():
        return None
    if callable(optimize) or isinstance(optimize, (list, tuple)):
        return None          # 显式路径/可调用优化器绑定的是原操作数次序，不能重排
    if not specialize:
        return None          # 映射的目的是命中特化形态，关掉就没意义

    call = dict(use_blas=use_blas, optimize=optimize, memory_limit=memory_limit,
                backend=backend, specialize=specialize, kernel_force=kernel_force)

    if payload is not None:
        token = _equivalence_token(subscripts, payload)
        applied = None
        if token not in _EQUIVALENCE_REJECTED:
            try:
                applied = _apply_equivalence(subscripts, operands, payload,
                                             profile=True, **call)
            except Exception:                            # noqa: BLE001 - 见下
                # 表里的载荷可能是**别的进程**写下的：字段自洽不等于真的对得上这批
                # 张量（输出形状守卫抛错就是这种情形）。任何异常都按"这条载荷不可信"
                # 处理——记进本进程的拒绝表、退回下面真的查一次。查一次若也失败，
                # 那个异常会照常抛给调用方，所以这里不会把**真实缺陷**吞掉。
                with _EQUIVALENCE_LOCK:
                    _EQUIVALENCE_REJECTED.add(token)
                applied = None
        if applied is not None:
            if token in _EQUIVALENCE_VERIFIED:
                return applied           # 已自检过：纯施加，一次匹配都不做
            return _payload_result(subscripts, operands, payload, applied, token,
                                   cache=cache, call=call)
        # 载荷不可用（缺字段 / 轴序对不上 / 之前自检失败）：退回真的查一次。

    try:
        mapping = _equivalence.normalize_equation(subscripts, use=cache)
    except Exception:                                    # noqa: BLE001 - 不许影响数值
        return None
    if mapping is None:
        return None
    return _apply_equivalence(subscripts, operands, mapping, **call)


##### 配置画像（profile）接线

#: 等价改写跑成时用的标签：**不可重放**（原方程上强制具体内核必然"未尝试"），
#: 只用于 :func:`last_profile` 的诊断输出。
_EQUIVALENCE_LABEL = "equivalence"

# 命中提示的去重表：每 (方程, 形状, 配置) 每进程只打一行（VERBOSE 打开了就每次必打）。
_PROFILE_NOTICED: set[tuple[Any, ...]] = set()
#: "配置表文件不存在"的提示：**每进程只打一次**。缺的是同一张表，同一句话按每个键
#: 重复一遍没有新信息，只把日志冲掉。
_PROFILE_MISSING_NOTICED = False
_PROFILE_NOTICE_LOCK = threading.Lock()

#: 最近一次画像调用的观测（:func:`last_profile` 读出；含整段 ``contract()`` 的 ``wall_ms``）。
_LAST_PROFILE: dict[str, Any] | None = None


def last_profile() -> dict[str, Any] | None:
    """最近一次**画像**调用的观测副本；本进程还没有画像调用时 ``None``。

    字段：``key`` / ``kernel`` / ``optimize`` / ``requested_optimize`` / ``hit`` /
    ``force`` / ``equivalence`` / ``note`` / ``fallback`` / ``wall_ms``。

    ``wall_ms`` 是**整个** :func:`contract` 调用的墙钟：含等价改写（若有）、读表
    （若有）与未命中时的规划-匹配——就是配置表 ``wall_ms`` 那一列的口径（§4），
    tune 据此选最优。``profile=False`` 的调用不更新它。
    """
    return None if _LAST_PROFILE is None else dict(_LAST_PROFILE)


def _profile_devices(operands: Sequence[Any], backend_name: str) -> tuple[str, str]:
    """``(device, user_device)``：具体计算设备与用户视角的设备口径。

    ``device`` 是**建键**用的那一列（cupy 的设备对象只有编号、torch 的 device 是
    ``cpu`` / ``cuda:0``）；``user_device`` 只给调用方（CLI 复现同一把键）参考。
    两者都取不到就给空串——画像缓存不该因为一个怪后端抛错。
    """
    raw = getattr(operands[0], "device", None)
    if raw is None:
        text = "cpu" if backend_name in ("numpy", "cpu") else ""
    elif hasattr(raw, "id"):                 # cupy 的设备对象只有编号
        text = f"{backend_name}:{int(raw.id)}"
    else:                                    # torch 的 device：``cpu`` / ``cuda:0``
        text = str(raw)
    user = "cpu" if backend_name in ("numpy", "cpu") else (text or backend_name)
    return text or "", user


def _profile_dtype(operands: Sequence[Any]) -> str:
    """dtype 列：同一个 dtype 就给名字，混合 dtype 去重后用 ``+`` 连接。"""
    names: list[str] = []
    for operand in operands:
        text = str(operand.dtype)
        if text not in names:
            names.append(text)
    return "+".join(names)


#: 配置表 ``optimize`` 列里"不用优化器"的写法：可读、可重放（读回即 ``False``）。
_OPTIMIZE_NONE_LABEL = "none"


def _profile_optimize(optimize: Any) -> str:
    """``optimize`` 列的文本：字符串原样，``False`` → ``none``，callable 只记类型名。

    ``False`` 写成 ``none`` 而不是 ``str(False)``：这一列是给人看、也要能重放的，
    而 ``none`` 正是 :func:`_normalize_optimize` 认得的输入写法（读回见
    :func:`_profile_optimize_value`），于是"写进去的"和"能填回来的"是同一种拼写。
    """
    normalized = _normalize_optimize(optimize)
    if normalized is False:
        return _OPTIMIZE_NONE_LABEL
    if isinstance(normalized, (str, bool)) or normalized is None:
        return str(normalized)
    return type(normalized).__name__


def _profile_optimize_is_default(optimize: Any) -> bool:
    """调用方是否**没给** ``optimize``（``None``/``True``/``"auto"``/空串）。

    只有"没给"时才允许表里的 ``optimize`` 生效；显式给了就别拿缓存覆盖用户。
    """
    if optimize is None or optimize is True:
        return True
    return isinstance(optimize, str) and optimize.strip().lower() in ("", "auto")


def _profile_optimize_value(value: Any) -> Any:
    """表里的 ``optimize`` → 可施加的值；缺省/不可用返回 ``None``。

    ``auto`` 是**缺省值**（不是"表给出的选择"），``none``（表里 ``False`` 的写法，
    见 :func:`_profile_optimize`）读回成 ``False``，其余取值必须在
    :data:`pyqcd.einsum.paths.PATH_OPTIONS` 里才施加——表里写了个怪名字时宁可退回
    调用方的值，也绝不因为一行缓存把调用打崩。
    """
    text = _profile_optimize(value).strip().lower()
    if text == _OPTIMIZE_NONE_LABEL:
        return False                     # 双向：读回的表值与调用方写法同一个意思
    # ``auto`` 是 ``optimize`` 的**缺省值**（不是表给出的选择），读回时不可施加；
    # 它与内核标签的 ``auto`` 无关——那一档已被删除，这里保留的是合法取值本身。
    if not text or text in ("auto", "true", "false"):
        return None
    return text if text in paths.PATH_OPTIONS else None


def _profile_force_label(kernel: str) -> str | None:
    """画像里记下的 kernel 标签 → 本次可交给 ``kernel_force=`` 的逻辑名。

    具体专用内核（``kupl_dedicated`` / ``cuda_topology`` …）按角色归一成
    ``dedicated`` / ``topology``；``backend`` **也会被返回**——"后端 einsum 最快"
    是画像的合法结论，缓存存在的意义就是复用它。``generic`` **也会被返回**：它同样
    可重放（``kernel_force="generic"`` 合法），而且实测常是最快的一条（10 操作数方程
    上 generic 385 ms vs topology 2735 ms）。代价是它不再进入专用分派、因此不会为
    这家方程打"内核没跑成"的诊断——按缓存选最快时必须接受这个取舍。
    """
    label = kernel.strip().lower()
    if not label:
        return None
    if label == _kernel_force.BACKEND_LABEL:
        # ``backend`` 是"记录里这条最快"的合法结论——大规模生产里 backend.einsum
        # 常常真的比专用内核快（小算子上尤其是）。既然缓存存在的意义就是复用它，
        # 这里必须允许把 backend 作为建议交出去；调用方显式给了 kernel_force 或
        # 有 out= 时仍会被上层挡住（见 _profile_begin 的 caller_force/out 判定）。
        return _kernel_force.BACKEND_LABEL
    if label == _kernel_force.ORIGINAL_LABEL:
        # ``original`` 与 ``backend`` 同地位：tune 会在同一张网格里测它，它当然也可能
        # 是最快的那一格。同样允许复用它，同样会被上层的 caller_force/out 挡住。
        return _kernel_force.ORIGINAL_LABEL
    if label == "generic":
        # ``generic`` 同样可重放（``kernel_force="generic"`` 是合法取值），而且实测
        # 常是最快的一条：10 操作数方程上 generic 385 ms vs topology 2735 ms。
        # 代价是它不再进入专用分派、因此不会为这家方程打"内核没跑成"的诊断——
        # 这是"按缓存选最快"必须接受的取舍，且用户可在需要诊断时显式设 auto。
        return "generic"
    role = label.rpartition("_")[2]          # ``kupl_dedicated`` → ``dedicated``
    for name in (role, label):
        if name in _kernel_force.KERNEL_LABELS:
            return name
    return None


def _profile_verbose() -> bool:
    """**完整详解**是否开启：``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1``，或 CLI 的
    ``--explain true`` / ``--verbose true``（经 ``_specialize.set_detail``）。

    判定单点在 ``_specialize.verbose_enabled()``：画像提示、``announce`` 与
    ``announce_generic`` 共用同一个档，否则会出现"命中行静默了、特化提示还照打"
    的半静音状态。
    """
    return _specialize.verbose_enabled()


def _profile_ms_text(row: Mapping[str, Any]) -> str:
    """行里的耗时原文（``wall_ms``）；没有就空串。

    只用于提示行与报告——这一列的口径永远是"整次 ``contract()`` 的墙钟"（§4）。
    """
    return str(row.get("wall_ms") or "")


def _to_float(value: Any) -> float | None:
    """字符串/数字 → ``float``；空、``None``、坏值一律 ``None``（不猜 0）。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_notice(info: dict[str, Any], *, row: dict[str, Any] | None = None,
                    applied: bool = False) -> None:
    """一行提示：**施加**了的命中每形态每进程打一次；其余只在 VERBOSE 时打。

    ``row is None`` = 未命中；``row`` 非空但 ``applied`` 为假 = 命中但没施加
    （``out=`` / 调用方已强制内核等）。后两种默认静默，免得把"有一条记录"说成
    "已经按它执行"。

    **已施加**时只打一行，且把记录钉到 ``<pyqcd.einsum.profile.csv 绝对路径>:<数据行号>``
    （行号由 :func:`._profile_cache.read_winner` 与行一起返回）：同一个 ``kernel=`` 可能
    出现在许多行里，没有行号用户翻不到"到底是哪一条在生效"（2026-11-17 用户实测）。
    这一行之后本次调用**不再**打 ``announce_generic`` / ``announce`` / 等价解释，
    除非 ``--explain true`` / ``--verbose true``（见 :func:`_profile_verbose`）。
    """
    if not _specialize.notice_enabled():
        return
    verbose = _profile_verbose()
    if not applied and not verbose:
        return
    token = (info["key"]["equation"], info["key"]["shape"],
             str((row or {}).get("kernel", "")),
             str((row or {}).get("optimize", "")))
    with _PROFILE_NOTICE_LOCK:
        if not verbose and token in _PROFILE_NOTICED:
            return
        _PROFILE_NOTICED.add(token)
    if applied and row is not None:
        source = _profile_cache.profile_path()
        line = str(row.get("line") or "").strip()
        where = f"{source}:{line}" if line else source
        if _specialize.notice_language() == "zh":
            print(
                f"[pyqcd.einsum] profile 命中：{where}"
                f" kernel={row.get('kernel', '')}"
                f" optimize={row.get('optimize', '')}"
                f" wall_ms={_profile_ms_text(row)}（已套用）",
                flush=True,
            )
        else:
            print(
                f"[pyqcd.einsum] profile hit: {where}"
                f" kernel={row.get('kernel', '')}"
                f" optimize={row.get('optimize', '')}"
                f" wall_ms={_profile_ms_text(row)} (applied)",
                flush=True,
            )
    elif row is not None:
        print(
            f"[pyqcd.einsum] profile 命中未施加：backend={info['key']['backend']}"
            f" equation={info['key']['equation']} shape={info['key']['shape']}"
            f" kernel={row.get('kernel', '')}（本次的显式开关优先）",
            flush=True,
        )
    else:
        print(
            f"[pyqcd.einsum] profile 未命中：backend={info['key']['backend']}"
            f" equation={info['key']['equation']} shape={info['key']['shape']}"
            "（按现有逻辑执行；这条键要进表得先 tune）",
            flush=True,
        )



def _profile_read(key: dict[str, Any]) -> dict[str, Any] | None:
    """读该键**唯一**的一行；三种结局按 §2 处理。

    * 命中 1 行 → 返回它；
    * 同键多行 → 把**全部**候选行打到 stderr，然后让
      :class:`~._profile_cache.DuplicateProfileEntry` 向上冒泡（不猜、不选最优）；
    * 文件不存在 → 一行警告"需要先 tune" + 该配置的 tune 命令模板，返回 ``None``
      让调用方**继续**按无命中执行；其他读失败也按无命中（表坏了只该慢一点）。
    """
    try:
        return _profile_cache.read_winner(key)
    except _profile_cache.DuplicateProfileEntry as exc:
        print(f"[pyqcd.einsum] {exc}", file=sys.stderr, flush=True)
        raise
    except _profile_cache.ProfileMissing as exc:
        _warn_profile_missing(exc)
        return None
    except Exception:                                    # noqa: BLE001 - 表坏了只该慢一点
        return None


def _warn_profile_missing(exc: BaseException) -> None:
    """配置表不存在：**每进程一次**警告 + 现成的 tune 命令，然后继续无命中执行。"""
    global _PROFILE_MISSING_NOTICED
    with _PROFILE_NOTICE_LOCK:
        if _PROFILE_MISSING_NOTICED:
            return
        _PROFILE_MISSING_NOTICED = True
    print(f"[pyqcd.einsum] 警告：{exc}\n  本次按未命中执行（方程、dtype、形状与数值都不受影响）。",
          file=sys.stderr, flush=True)


def _profile_conflict_message(key: Mapping[str, Any], row: Mapping[str, Any],
                              field: str, recorded: str, requested: str) -> str:
    """冲突的报错正文：键、表里的值、用户给的值、那一行在哪、怎么解决。"""
    source = _profile_cache.profile_path()
    line = str(row.get("line") or "").strip()
    where = f"{source}:{line}" if line else source
    return (
        "einsum 配置表命中行与你的显式配置冲突（不猜、不覆盖）：\n"
        f"  键: shape={key.get('shape', '')} dtype={key.get('dtype', '')}"
        f" device={key.get('device', '')} equation={key.get('equation', '')}"
        f" backend={key.get('backend', '')}\n"
        f"  冲突: {field}  表中={recorded!r}  显式={requested!r}  （{where}）\n"
        "  处理：去掉该显式参数，或删掉/改正那一行；也可以 profile=False 让本次不读表。")


def _profile_check_conflict(key: dict[str, Any], row: dict[str, Any], *,
                            caller_force: tuple[str, ...] | None, optimize: Any,
                            as_equivalence: bool | str | None,
                            has_equivalence: bool,
                            caller_force_given: bool | None = None) -> None:
    """用户**显式**配置与命中行冲突 → 抛 :class:`~._profile_cache.ProfileConflict`。

    只比"用户真的给了的"三个维度（``kernel_force`` / ``optimize`` /
    ``as_equivalence``）：显式要求既不能被静默忽略，也不能被表悄悄改掉。``out=`` /
    ``specialize=False`` 只是"不施加"，不是冲突；``optimize="auto"`` 是缺省值，不算
    显式。

    ``caller_force`` 是**解析出的**强制集，``caller_force_given`` 是"调用方真的给过"的
    显式位（``None`` = 由 ``caller_force is not None`` 推，兼容只有解析结果的调用方）。
    两者必须分开：缺省取值同样解析成 ``generic``（``()``），只看取值就会让每一次默认
    调用都撞上表里的具体标签。
    """
    if caller_force_given is None:
        caller_force_given = caller_force is not None
    recorded_kernel = _profile_force_label(str(row.get("kernel") or ""))
    if caller_force_given and caller_force is not None and recorded_kernel is not None:
        # ``caller_force == ()`` 是一个**显式**要求（"只走通用路径"），不是"没给"。
        # 过滤掉它会让"显式 generic 撞上表中 topology"静默通过，正好违反 §2.4——
        # 所以这里要把它还原成 ``{"generic"}`` 再比。
        allowed = ({_kernel_force.GENERIC_LABEL} if caller_force == ()
                   else set(caller_force))
        if allowed and recorded_kernel not in allowed:
            requested = ",".join(caller_force) or _kernel_force.GENERIC_LABEL
            raise _profile_cache.ProfileConflict(
                _profile_conflict_message(key, row, "kernel_force",
                                          recorded_kernel, requested),
                field="kernel_force", recorded=recorded_kernel,
                requested=requested, key=key)

    recorded_optimize = _profile_optimize_value(row.get("optimize"))
    if recorded_optimize is not None and not _profile_optimize_is_default(optimize):
        # 两侧都先归到同一拼写再比：表里记 ``none``、调用方写 ``"none"`` 是同一个要求，
        # 不能因为一边是 ``False``、另一边是字符串就报冲突。
        requested_optimize = _profile_optimize(optimize).strip().lower()
        recorded_text = _profile_optimize(recorded_optimize).strip().lower()
        if requested_optimize != recorded_text:
            raise _profile_cache.ProfileConflict(
                _profile_conflict_message(key, row, "optimize",
                                          recorded_text, requested_optimize),
                field="optimize", recorded=recorded_text,
                requested=requested_optimize, key=key)

    if (has_equivalence and as_equivalence is not None
            and not _equivalence.as_equivalence_enabled(as_equivalence)):
        raise _profile_cache.ProfileConflict(
            _profile_conflict_message(
                key, row, "as_equivalence",
                "该键的等价行（走等价转换后更快）", str(as_equivalence)),
            field="as_equivalence", recorded="equivalence-row",
            requested=str(as_equivalence), key=key)


def _profile_begin(profile: bool | None, profile_dir: str | None, subscripts: Any,
                   operands: Sequence[Any], *, backend: str, out: Any,
                   specialize_on: bool, caller_force: tuple[str, ...] | None,
                   caller_force_given: bool, optimize: Any,
                   as_equivalence: bool | str | None = None) -> dict[str, Any] | None:
    """画像入口：返回本次调用的记录槽，``None`` 表示本次不画像（零副作用）。

    开关与目录的解析**完全**交给 :mod:`._profile_cache`（显式参数 >
    ``PYQCD_EINSUM_PROFILE[_DIR]`` > 默认 ``./.cache``）。``profile=False`` 在第一个
    判断就返回：不 ``configure``、不读表、不建目录——**零 IO**。

    读表只有三种结局（见 :func:`_profile_read`）：命中 1 行、同键多行（报错并输出
    全部候选）、文件不存在（警告 + tune 命令模板后继续）。命中只填 ``force`` /
    ``optimize`` / ``equivalence`` 三个槽位——方程、dtype、形状与数值一概不动。

    用户显式给了 ``kernel_force`` / ``optimize`` / ``as_equivalence`` 且与命中行冲突时
    抛 :class:`~._profile_cache.ProfileConflict`（向上冒泡到 CLI）。``out=`` /
    ``specialize=False`` 只让内核建议"不施加"，不是冲突。

    ``caller_force`` 是解析出的强制集（``generic`` → ``()``），``caller_force_given``
    是"调用方真的给过"的显式位。两者必须分开：缺省取值同样解析成 ``()``，拿它当
    "显式要求"会让每一次默认调用都走进"未施加"分支。
    """
    if not _profile_cache.enabled(profile):
        return None
    if not isinstance(subscripts, str) or not operands:
        return None                     # 交错/字典形式没有可作键的方程文本
    if any(not hasattr(o, "shape") or not hasattr(o, "dtype") for o in operands):
        return None
    if profile_dir:
        _profile_cache.configure(profile_dir)   # 显式目录优先；切换是幂等的
    else:
        _profile_cache.ensure_configured()
    backend_name = parse_backend(operands, backend)
    device, _user_device = _profile_devices(operands, backend_name)
    key = {
        "shape": _profile_cache.format_shape([o.shape for o in operands]),
        "dtype": _profile_dtype(operands),
        "device": device,
        "equation": "".join(subscripts.split()),
        "backend": backend_name,
    }
    row = _profile_read(key)
    if row is None:
        # 「默认第一次调用先 tune 再真跑」（用户裁定 2026-11-19）：缺行时跑一整轮
        # ``--tune``，它自己负责把胜出行写进表；本函数随后**重读一次**——重读是必须
        # 的，因为 tune 写的是"实测出来的最优格"，不重读就等于测完不用。
        #
        # 三道边界都在 ``_autotune`` 里：开关环境变量、同键每进程只试一次、
        # 以及"tune 自己调用 contract 时不回来"的互斥。这里只负责"重读"这一件事，
        # 不判断"值不值得 tune"——那是用户的缺省行为，不是本层的策略。
        if _autotune.run_for(key=key, subscripts=subscripts, operands=operands,
                             backend=backend_name, device=device,
                             dtype=str(key.get("dtype") or "")):
            row = _profile_read(key)
    info: dict[str, Any] = {
        "key": key,
        "requested_optimize": _profile_optimize(optimize),
        "optimize": None,               # 命中行给出的 optimize（能施加时）
        "wall_ms": 0.0,
        "kernel": "generic",
        "note": "",
        "hit": row,
        "force": None,
        "equivalence": None,
        "fallback": False,
    }
    if row is None:
        _profile_notice(info)
        return info

    payload = _equivalence_payload(row)
    info["equivalence"] = payload
    _profile_check_conflict(key, row, caller_force=caller_force, optimize=optimize,
                            as_equivalence=as_equivalence,
                            has_equivalence=payload is not None,
                            caller_force_given=caller_force_given)

    label = str(row.get("kernel") or "")
    force = _profile_force_label(label)
    recorded_optimize = _profile_optimize_value(row.get("optimize"))
    if recorded_optimize is not None and _profile_optimize_is_default(optimize):
        info["optimize"] = recorded_optimize
    if force is None and recorded_optimize is None:
        # 等价行（或没有可用性能载荷的行）：只提供改写，不提内核、不施加开关。
        info["note"] = ("profile 等价行：按表改写" if payload is not None
                        else "profile 命中但没有可用配置")
        if payload is None:
            _profile_notice(info, row=row)
        return info
    if force is None or caller_force_given or out is not None or not specialize_on:
        info["note"] = f"profile 命中未施加（kernel={label or '?'}）"
        _profile_notice(info, row=row)
        return info
    info["force"] = force
    info["note"] = f"profile 命中：kernel={label or '?'}"
    _profile_notice(info, row=row, applied=True)
    return info


def _profile_end(info: dict[str, Any] | None, result: Any) -> Any:
    """收尾：把**整个** ``contract()`` 的墙钟记进 ``wall_ms``，原样返回结果。

    §4 的口径：``wall_ms`` 就是这一次 ``contract()`` 的墙钟——含等价转换（若有）、
    读表（若有）与未命中时的规划-匹配。没有第二列计时可以与它相加或混比，也没有
    "只算收缩执行"的拆分（旧的拆分计时已删除）。起点由 :func:`contract` 在**函数
    第一行**取，所以这里量到的必然是整段调用。

    这里**不写表**（配置表只有 tune 能写），观测只留在 :func:`last_profile`。
    """
    global _LAST_PROFILE
    if info is None:
        return result
    started = info.get("started")
    if started is not None:
        info["wall_ms"] = (time.perf_counter() - started) * 1000.0
    _LAST_PROFILE = info
    return result


def _profile_retry(info: dict[str, Any], subscripts: Any, operands: Sequence[Any],
                   **call: Any) -> Any:
    """画像建议跑不成：退回**原**分派一次，绝不把异常抛给调用方。

    只在 ``KernelUnavailableError`` 由画像强制引起时调用（用户自己强制的标签
    仍然照旧抛出）。重试带 ``profile=False``：本次观测由外层统一收尾，不重复。
    """
    info["fallback"] = True
    info["force"] = None
    info["note"] = "profile 建议不可用，已回退自动分派"
    result = contract(subscripts, *operands, profile=False, **call)
    return _profile_end(info, result)


def contract(
    subscripts: Any,
    *operands: Any,
    out: Any = None,
    use_blas: bool = True,
    optimize: Any = "auto",
    memory_limit: Any = None,
    backend: str = "auto",
    specialize: bool | None = None,
    kernel_force: str | None = None,
    as_equivalence: bool | str | None = None,
    cache: bool | None = None,
    profile: bool | None = None,
    profile_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    """Evaluate an einsum expression with an optimized contraction order.

    Drop-in replacement for :func:`numpy.einsum` that contracts the operands
    pairwise in a cost minimizing order.

    Parameters
    ----------
    subscripts : str
        Einstein summation subscripts, e.g. ``"ij,jk->ik"``. The ``->`` part is
        optional (implicit output) and ``...`` broadcasting is supported.
    *operands : array_like
        Two or more tensors. All operands must share one dtype; results keep it.
    out : ndarray, optional
        Preallocated buffer written **in place**. Its shape and dtype must match
        the contraction output, otherwise a ``ValueError`` is raised. Passing
        ``out`` disables the specialized-kernel path.
    use_blas : bool
        Allow pairwise contractions to be dispatched to BLAS (default ``True``).
    optimize : bool | str | list | PathOptimizer
        ``False`` (or the string ``"none"``, case insensitive, surrounding blanks
        ignored) for a single n-ary einsum, an explicit path, the name of a
        registered optimizer (``"greedy"``/``"optimal"``/``"dp"``), or
        ``True``/``None``/``"auto"`` (default) to let the planner decide.
    （``optimize_topology_v2`` **已彻底删除**：0.5.9 从签名移除、0.5.12 清掉内部残留。
    传它会得到 ``TypeError``，因为它从来就只有一个接收方，而那个接收方在 0.5.4 就
    随 v2 通用执行器一起删除了。要换收缩序请用 ``optimize=``。）
    memory_limit : int | str | None
        Upper bound on any intermediate's element count. ``None``/``-1`` means
        unlimited, ``"max_input"`` means the largest input tensor.
    backend : str
        ``"auto"`` (default), ``"numpy"``, ``"cupy"``, ``"torch"`` or a
        registered backend name.
    specialize : bool | None
        ``None`` (default) follows ``PYQCD_EINSUM_SPECIALIZE``; ``False`` forces
        the generic path for this call; ``True`` forces the specialized path.
    kernel_force : str | None
        Which specialised kernel **must** run. All four values are forced ones —
        there is no ``"auto"`` (automatic routing belongs to ``--tune``'s measured
        winner, see :mod:`._kernel_force`): ``None`` (default) resolves to
        ``"generic"``. ``"generic"`` skips the specialised kernels entirely (no
        warning). ``"dedicated"``/``"topology"`` allow exactly that kernel **of the
        operand backend** (``kupl_*`` for numpy, ``cuda_*`` for cupy); **one value
        only**, a comma-separated chain raises ``ValueError``. Forcing bypasses the
        kernels' negative cache, and a kernel that does not run raises
        ``KernelUnavailableError`` instead of falling back silently. ``"backend"``
        leaves the specialised dispatch altogether and calls the backend's own
        ``einsum``; it cannot be combined with the other labels. An explicit
        ``kernel_force`` wins over ``PYQCD_EINSUM_KERNEL_FORCE``; both are resolved
        on every call. Forcing any value contradicts ``out=`` and
        ``specialize=False``, which raise ``ValueError``.
        ``"generic"`` skips the shape lookup (``_specialize.match``) and the
        equivalence rewrite as well: both exist only to land on a specialised form,
        so on an explicitly generic call they are pure overhead and their notices
        would contradict what was asked for.
    as_equivalence : bool | str | None
        ``None`` (default) follows ``PYQCD_EINSUM_AS_EQUIVALENCE`` (on by default);
        ``False`` disables the automatic rewrite for this call; ``"force"`` is
        accepted as ``True``. When on and the equation does not already run on a
        specialised form, the equivalence layer rewrites it into a corpus spelling,
        permutes the operands accordingly and transposes the result back — **the
        returned array is the answer to the original equation**, with the original
        shape and dtype, so callers never see the rewrite. Skipped for ``out=``,
        extra ``**kwargs``, element-wise and cupy-delegated equations,
        ``specialize=False``, ``kernel_force="generic"`` (the rewrite exists to
        reach a specialised form, which that call refuses) and explicit optimizer
        paths (those bind the original operand order).

        With the profile on, a **configuration table row for this key** can carry an
        executable payload (``target_equation`` + the three permutation columns);
        a hit is then applied directly, skipping the equivalence search. The table
        is **read-only for ordinary calls** — nothing is written back, so a lookup
        is never turned into a stored conclusion; only ``--tune`` writes rows. An
        explicit ``as_equivalence`` that disagrees with such a row raises
        :class:`~._profile_cache.ProfileConflict` (see ``profile``).
    cache : bool | None
        ``None`` (default) follows ``PYQCD_EINSUM_CACHE`` (on by default); ``False``
        disables every einsum cache for this call — matchers
        (``_specialize.match``, ``_equivalence.normalize_equation``) and the planner
        (``_find_path``) then neither read nor write, so nothing is warmed or
        poisoned by a diagnostic call. Per-cache hit/miss counters are available via
        :func:`pyqcd.einsum.cache_stats`.
    profile : bool | None
        ``None`` (default) follows ``PYQCD_EINSUM_PROFILE`` (on by default); ``False``
        turns the configuration profile off for this call with **no side effect at
        all** — no directory, no read, no write: **zero IO**. When on, the
        configuration table ``pyqcd.einsum.profile.csv`` is read by **exact key**
        ``(shape, dtype, device, equation, backend)`` and the single row found
        decides **only** which specialised kernel is forced, which planner
        ``optimize`` is used and whether a stored equivalence payload is applied —
        the equation, dtype, shape and numbers are untouched.

        * **One row per key.** Two or more rows for the same key raise
          :class:`~._profile_cache.DuplicateProfileEntry` and print **every**
          candidate row; the winner is never guessed.
        * **Missing table.** One warning ("tune first") plus a ready-to-run tune
          command template, then the call proceeds exactly as if there were no hit.
        * **Explicit configuration wins — or fails loudly.** A caller/env supplied
          ``kernel_force``, ``optimize`` or ``as_equivalence`` that **contradicts**
          the row raises :class:`~._profile_cache.ProfileConflict` (never silently
          ignored, never silently overridden).

        A suggestion that does not run falls back to the original dispatch once
        (never raising ``KernelUnavailableError`` at the caller); an adopted hit
        prints **one** line naming the applied row as
        ``<absolute pyqcd.einsum.profile.csv>:<data-row number>``, and that line
        replaces the incidental notices of this call (``announce_generic`` /
        ``announce`` / equivalence). Raising the detail level
        (``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1``, or the CLI's ``--explain true`` /
        ``--verbose true`` through ``_specialize.set_detail``) restores them. A
        recorded *specialised* kernel (``kupl_dedicated`` / ``cuda_topology`` …) is
        replayed as ``dedicated`` / ``topology``; a recorded ``generic`` is replayed
        too (``kernel_force="generic"`` is legal and is often the fastest recorded
        configuration), at the cost of the auto-dispatch diagnostics for that
        equation.

        **Timing.** The table has exactly one clock: ``wall_ms`` = the wall clock
        of the **whole** ``contract()`` call — equivalence rewrite, table read and
        planning/matching included. It is *not* "the contraction only", there is no
        second column to add to it, and tune is the only writer that fills it. The
        observation of the current call is available from :func:`last_profile`.
    profile_dir : str | None
        Directory of the configuration table. Resolution order: this parameter,
        ``PYQCD_EINSUM_PROFILE_DIR``, then ``./.cache`` (relative to the current
        working directory). Ignored when the profile is off. Passing a **different**
        directory switches to that table for the rest of the process; **omitting**
        it keeps whatever directory is already pinned (``None``/empty means "not
        given", never "go back to the environment/default"). Only a first,
        never-configured call resolves the default.
    **kwargs
        Forwarded to the underlying backend.

    Returns
    -------
    ndarray
        The contraction result, in the operands' dtype. When ``out`` is given,
        the returned object is that same buffer.

    Raises
    ------
    ValueError
        Mismatched operand counts, conflicting label sizes, unknown or
        duplicated output labels, an ``out`` buffer with the wrong shape, an
        unknown ``kernel`` label, or ``kernel_force=<label>`` combined with ``out=`` /
        ``specialize=False``.
    TypeError
        Non-string subscripts or unsupported operand types.
    KernelUnavailableError
        A specific ``kernel_force`` label was forced but did not run. Carries the
        requested/attempted labels and each label's failure reason.
    pyqcd.einsum._profile_cache.DuplicateProfileEntry
        The configuration table holds more than one row for this key. All candidate
        rows are printed and the exception propagates — the winner is never guessed.
    pyqcd.einsum._profile_cache.ProfileConflict
        An explicit ``kernel_force`` / ``optimize`` / ``as_equivalence`` contradicts
        the configuration row matched for this key. Delete the explicit argument,
        fix/remove that row, or pass ``profile=False`` for this call.

    Notes
    -----
    Specialized kernels are selected by **topology** (label-independent), so
    ``"ij,jk->ik"`` and ``"ij,jb->ib"`` take the same path. A failed native
    kernel falls back to the generic path with one ``RuntimeWarning``; results
    stay correct. Set ``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`` to log every hit.
    :func:`last_kernel` reports which kernel the most recent call actually ran.
    """
    # §4：``wall_ms`` 量的是**整段** ``contract()``（含等价转换、读表、规划-匹配），
    # 所以起点就在函数的第一个语句取。
    entry_started = time.perf_counter()

    # ``optimize`` 的归一化要等**内核定型**：``"none"`` 对 topology 是"顺序由内核
    # 决定"（该内核唯一合法的 optimize 输入），对其余内核是"不使用 optimize"。
    # 因此这里只记下原值，解析放在 ``forced`` 求值之后（见 ``_resolve_optimize``）。
    requested_optimize = optimize

    # ``optimize_topology_v2`` 已在 0.5.12 **彻底删除**（0.5.9 删公开签名、0.5.12 清
    # 内部残留）：拓扑内核的 v2 通用执行器在 0.5.4 就删了，这个参数从来没有第二个
    # 接收方。用户仍传旧名时由下面这道守卫明确报错——它不能落进 ``**kwargs`` 被当成
    # 后端关键字静默吃掉（那会让人以为"换了次序"，实际什么也没变）。
    if "optimize_topology_v2" in kwargs:
        raise TypeError(
            "optimize_topology_v2 已在 0.5.12 彻底删除（0.5.9 起就已不在签名里）："
            "拓扑内核的 v2 通用执行器在 0.5.4 一并删除后，该参数没有任何接收方。"
            "v3 的收缩序固化在编译期步表 topology_steps.h 里，不随调用变化；"
            "要换收缩序请用 optimize=。"
        )

    # ``as_equivalence`` 先归一成三态：``bool("false")`` 是 ``True``，直接把字符串交给
    # 布尔判定会**反转**这个开关（本文件所有判定都只认布尔/None）。
    as_equivalence = _normalize_as_equivalence(as_equivalence)

    global _FUSED_ENV_OWNED, _LAST_KERNEL

    # 首次调用时把线程配置补到已加载的 OpenMP/BLAS 运行时上（幂等、极廉价）。
    _threads.ensure_runtime()

    # 专用路径开关只求值**一次**：专用门、等价门与画像门共用它，
    # 避免每次调用重复读环境变量。
    specialize_on = _specialize.enabled(specialize)

    # 强制内核：显式 ``kernel_force=`` 优先于 ``PYQCD_EINSUM_KERNEL_FORCE``。取值**单值**：
    # ``generic``（缺省）→ ``()``；``dedicated`` / ``topology`` / ``backend`` → 长度 1 的
    # 元组。没有"未指定 → 自动分派"的档（自动选路交给 tune 的实测结论）。
    #
    # ``caller_force_given`` 是"**用户真的给过**"的显式位，在解析与任何默认填充之前取：
    # 缺省取值同样解析成 ``generic``，拿解析结果反过来判断显式性，会把每一次默认调用都
    # 当成"用户点名了 generic"（画像命中因此永远不施加，那唯一一行 ``<file>:<行号>``
    # 提示也就永远走不到）。环境变量那一路不需要这个位：它与表冲突时照样由
    # ``_profile_check_conflict`` 抛 ``ProfileConflict``（见 ``caller_force == ()`` 分支）。
    caller_force_given = kernel_force is not None
    forced = _kernel_force.forced_labels(kernel_force)
    optimize = _resolve_optimize(requested_optimize, forced)
    if (optimize is True) or (optimize is None):
        optimize = "auto"
    if forced:
        # "强制即真试一次"：凡是会让专用内核根本跑不到的开关，一律早失败，
        # 不做"半强制"的静默回落。
        if not isinstance(subscripts, str):
            raise ValueError(
                "kernel_force= 只能用于字符串方程；交错/字典形式的输入没有专用内核可强制")
        if out is not None:
            raise ValueError(
                "out= 会禁用专用内核，不能与强制指定内核同时使用：去掉 out=，"
                f"或把 kernel_force 设为 {_kernel_force.GENERIC_LABEL!r}")

    # 配置画像：命中只决定内核目标、收缩序与等价改写，方程/dtype/形状/数值一概不动
    # （冲突报错见 _profile_check_conflict，建议跑不成的回退见 _profile_retry）。
    # 这一层**只读**：配置表只有 tune 能写。
    info = _profile_begin(profile, profile_dir, subscripts, operands,
                          backend=backend, out=out, specialize_on=specialize_on,
                          caller_force=forced,
                          caller_force_given=caller_force_given, optimize=optimize,
                          as_equivalence=as_equivalence)
    requested_backend, requested_kernel_force = backend, kernel_force
    requested_forced = forced
    forced_by_profile = False
    if info is not None:
        info["started"] = entry_started        # wall_ms 的起点：整个 contract()
        if info["force"] is not None:
            kernel_force = info["force"]
            forced = _kernel_force.forced_labels(kernel_force)
            forced_by_profile = True
        if info.get("optimize") is not None:
            optimize = info["optimize"]


    if forced == (_kernel_force.BACKEND_LABEL,):
        # ``backend`` = "别走专用内核，直接调后端自己的 einsum"。它既不进专用
        # 分派，也不做等价改写（改写的目的正是命中专用形态），所以在分派之前返回。
        _LAST_KERNEL = {
            "requested": requested_forced,
            "ran": _kernel_force.BACKEND_LABEL,
            "attempted": (),
            "reasons": {},
        }
        backend_result = _backend_einsum(
            parse_backend(operands, backend), subscripts, operands)
        if info is not None:
            info["kernel"] = _kernel_force.BACKEND_LABEL
        return _profile_end(info, backend_result)

    if forced == (_kernel_force.ORIGINAL_LABEL,):
        # ``original`` = "交给 opt_einsum"，与 ``backend`` 同一地位（都绕过专用内核
        # 与等价改写），区别只在谁来规划与执行。``optimize`` **照收**：这正是它与
        # ``backend`` 的实质差异——``backend.einsum`` 不接受我们的叶子名，而
        # ``opt_einsum`` 接受（名字转型见 ``_kernel_force.opt_einsum_path_name``）。
        # 失败不回落：强制标签的语义就是"跑不成即抛"。
        backend_name = _opt_einsum_backend_name(operands)
        if backend_name is None:
            raise _kernel_force.KernelUnavailableError(
                _kernel_force.ORIGINAL_LABEL, subscripts,
                {_kernel_force.ORIGINAL_LABEL: (
                    "opt_einsum has no backend for these operands "
                    f"({type(operands[0]).__module__ if operands else 'no operand'})")},
                attempted=(_kernel_force.ORIGINAL_LABEL,))
        _LAST_KERNEL = {
            "requested": requested_forced,
            "ran": _kernel_force.ORIGINAL_LABEL,
            "attempted": (),
            "reasons": {},
        }
        original_result = _original_einsum(
            backend_name, subscripts, operands, optimize=optimize)
        if info is not None:
            info["kernel"] = _kernel_force.ORIGINAL_LABEL
        return _profile_end(info, original_result)

    # ``kernel_force="generic"`` 解析成 ``()``：调用方**明确要求**不走专用内核。匹配与
    # 等价改写的唯一目的都是命中特化形态，这条路上两者都不需要——跳过它们只是省时间
    # （一次签名/摘要/语料查询，外加一次改写查询），并且不再打"通用路径 / 专用路径"
    # 两条提示：已经明确要求通用，再提示一遍等于自说自话（2026-11-17 用户实测：
    # ``--kernel-force generic`` 的输出里同时出现了 generic 与 specialized 两条）。
    #
    # **只认调用方的显式要求**（参数或 ``PYQCD_EINSUM_KERNEL_FORCE``）。画像给出的
    # ``generic`` 建议是**计时结论**，不是语义指令：它不该顺手关掉 ``as_equivalence``
    # 那条独立开关（``_einsum_as_equivalence_cache_contract`` 要求 env=1 时同一进程内
    # 连续调用仍必须真的查一次映射）。画像那条路上执行路线逐字不变，只是提示被下面的
    # ``quiet_profile`` 静音。
    generic_only = forced == () and not forced_by_profile

    # 画像命中**已施加**时只留那唯一一行命中提示：解释性的 announce_generic /
    # announce / 等价提示一律静音（``--explain true`` / ``--verbose true`` 恢复）。
    # 这是**纯输出**开关，不改变方程、dtype、形状或任何数值。
    quiet_profile = bool(
        info is not None and info.get("force") is not None
        and not _profile_verbose())

    # 方程形态专用路径：命中时打印提示；specialize=False 关闭本次调用的专用执行。
    # 只回收本模块自己设置的开关，不覆盖调用方通过环境变量显式给出的设置。
    specialized = None
    if isinstance(subscripts, str):
        if specialize_on:
            if _FUSED_ENV_OWNED:
                os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
                _FUSED_ENV_OWNED = False
            if generic_only:
                # 强制通用：**不匹配、不改写、不提示**，直接进下面的通用收缩。
                # 线程档仍按"通用路径"设；last_kernel() 如实记录"要求了 generic、
                # 一次都没试"，报告里的 kernel 行因此不会说谎。
                _threads.apply_profile("generic")
                _LAST_KERNEL = {
                    "requested": requested_forced, "ran": None, "attempted": (),
                    "reasons": {}}
            else:
                specialized = _specialize.match(subscripts, use=cache)
                # 内核路径与 BLAS 路径想要相反的线程配置：命中特化形态用 kernel 档，
                # 其余用 generic 档。切换只在档位真正变化时发生。
                if specialized is not None:
                    _threads.apply_profile("kernel")
                else:
                    _threads.apply_profile("generic")
                    # 通用内核首次命中提醒（每个进程一次，PYQCD_EINSUM_NOTICE=0 可关）。
                    # 只在确实没有特化形态时打，避免和 announce() 重复；画像命中已施加
                    # 时静音（那唯一一行命中提示已经说明了配置来源）。
                    if not quiet_profile:
                        _specialize.announce_generic(subscripts)
                # 本次调用的内核记录：默认"没尝试任何专用内核"，真跑内核的分支填上结果。
                # 整体覆盖写，所以 last_kernel() 绝不会读到上一次调用残留的命中。
                record: dict[str, Any] = {
                    "requested": requested_forced, "ran": None, "attempted": (),
                    "reasons": {}}
                _LAST_KERNEL = record
                if specialized is not None and out is None:
                    if forced == ():
                        # 走到这里说明 ``generic`` 是**画像建议**（调用方显式要求时已在
                        # 上面 early-return）：照旧一个专用内核都不跑，只提示命中的形态；
                        # 画像命中已施加时这一行也静音。
                        if not quiet_profile:
                            _specialize.announce(specialized)
                    elif forced:
                        # 强制具体标签：按用户给的次序真试一次，跑不成即抛。
                        try:
                            kernel_result = _run_forced_kernel(
                                subscripts, operands, forced[0], record)
                        except _kernel_force.KernelUnavailableError:
                            if not forced_by_profile:
                                raise
                            return _profile_retry(
                                info, subscripts, operands, out=out, use_blas=use_blas,
                                optimize=optimize, memory_limit=memory_limit,
                                backend=requested_backend, specialize=specialize,
                                kernel_force=requested_kernel_force,
                                as_equivalence=as_equivalence, cache=cache,
                                **kwargs)
                        if info is not None:
                            info["kernel"] = record["ran"] or "generic"
                        if not quiet_profile:
                            _specialize.announce(specialized, record["ran"] or "")
                        return _profile_end(info, kernel_result)
                    else:
                        # 专用 kernel 失败必须回退通用路径（契约：原生失败只 warning
                        # 一次，不把异常抛给调用方）。
                        announce_kernel = ""
                        kernel_reasons: dict[str, str] = {}
                        try:
                            kernel_result, announce_kernel, kernel_reasons = _run_specialized_kernel(
                                subscripts, operands)
                        except Exception as exc:  # noqa: BLE001 - 见上
                            kernel_result = None
                            _warn_specialized_kernel_once(exc)
                        if info is not None:
                            info["kernel"] = announce_kernel or "generic"
                        record["ran"] = announce_kernel or None
                        record["attempted"] = tuple(kernel_reasons) + (
                            (announce_kernel,) if announce_kernel else ())
                        record["reasons"] = dict(kernel_reasons)
                        if not quiet_profile:
                            _specialize.announce(specialized, announce_kernel)
                        if kernel_result is not None:
                            return _profile_end(info, kernel_result)
                        if str(specialized.get("kind", "")).startswith("kernel"):
                            relevant = _kernel_labels_for(specialized)
                            reasons = [kernel_reasons.get(label, "")
                                       for label in relevant] or list(kernel_reasons.values())
                            _warn_kernel_skipped_once(specialized, reasons)
                elif forced:
                    # 语料没命中却强制了具体标签：旧行为是静默走通用路径，这里必须真试
                    # 一次，并明确报出"未尝试"或具体原因。
                    #
                    # **先做等价改写**（``as_equivalence`` 开启时）：改写的唯一目的就是
                    # 命中特化形态，而"原方程不在语料里"正是改写能帮上忙的场景。此前
                    # 直接拿原方程去强制内核，必然 ``KernelUnavailableError``，改写
                    # 根本没机会生效（2026-11-18 用户实测：
                    # `kernel_force=topology` + `as_equivalence=true` 必定失败，而同一条
                    # 改写后的方程单独跑 ``try_topology_apply`` 却成功）。
                    if (isinstance(subscripts, str) and out is None and not kwargs
                            and not callable(optimize)
                            and not isinstance(optimize, (list, tuple))
                            and _equivalence.as_equivalence_enabled(as_equivalence)
                            and specialize_on):
                        rewritten = _contract_via_equivalence(
                            subscripts, operands, use_blas=use_blas, optimize=optimize,
                            memory_limit=memory_limit, backend=backend,
                            specialize=specialize_on, kernel_force=kernel_force,
                            cache=cache,
                            payload=info.get("equivalence") if info is not None else None)
                        if rewritten is not None:
                            if info is not None:
                                info["kernel"] = ("" if kernel_force is None
                                                  else str(kernel_force).strip().lower()
                                                  ) or _EQUIVALENCE_LABEL
                            return _profile_end(info, rewritten)
                    try:
                        kernel_result = _run_forced_kernel(
                            subscripts, operands, forced[0], record)
                    except _kernel_force.KernelUnavailableError:
                        if not forced_by_profile:
                            raise
                        return _profile_retry(
                            info, subscripts, operands, out=out, use_blas=use_blas,
                            optimize=optimize, memory_limit=memory_limit,
                            backend=requested_backend, specialize=specialize,
                            kernel_force=requested_kernel_force,
                            as_equivalence=as_equivalence, cache=cache,
                            **kwargs)
                    if info is not None:
                        info["kernel"] = record["ran"] or "generic"
                    return _profile_end(info, kernel_result)
        else:
            if forced:
                raise ValueError(
                    "specialize=False / PYQCD_EINSUM_SPECIALIZE=0 关掉了专用内核路径，"
                    "不能与强制指定内核同时使用；把 specialize 恢复为 None 或 True 再强制")
            os.environ[_specialize.FUSED_DISABLE_ENV] = "1"
            _FUSED_ENV_OWNED = True

    # GPU 上的通用（非融合）方程：cupy 自己的 einsum 直接走 cuBLAS，
    # 实测在 16×1024² complex128 上比我们的原生批量 GEMM 快 5 倍。
    # 在规划之前就委派，连契约解析的开销也省掉。
    if (isinstance(subscripts, str) and out is None and not kwargs
            and _cupy_operands(operands) and not _cuda_fused_shape(subscripts)):
        cupy_result = _backend_einsum("cupy", subscripts, operands)
        if info is not None:
            info["kernel"] = _kernel_force.BACKEND_LABEL
        return _profile_end(info, cupy_result)

    # 逐元素/纯置换类方程没有 BLAS 结构：直接交给 backend 的原生 einsum，
    # 比逐步执行 + 计划缓存更快（实测 Hadamard 4000² 89 ms -> 29 ms）。
    if (isinstance(subscripts, str) and out is None and not kwargs
            and _is_elementwise_equation(subscripts)):
        elementwise_backend = parse_backend(operands, backend)
        elementwise_result = _backend_einsum(
            elementwise_backend, subscripts, operands)
        if info is not None:
            info["kernel"] = _kernel_force.BACKEND_LABEL
        return _profile_end(info, elementwise_result)

    # 形状特化路径：少量结构特殊的方程（语料之外）用一次性推导出的固定算法，
    # 不规划、不建步表。只接管无 out=/无 kwargs 的调用；前置条件不满足即回落。
    if (isinstance(subscripts, str) and out is None and not kwargs
            and specialize is not False):
        shape_result = _shape_paths.try_execute(subscripts, operands)
        if shape_result is not None:
            return _profile_end(info, shape_result)

    # 等价映射自动施加（``as_equivalence``，缺省跟随 PYQCD_EINSUM_AS_EQUIVALENCE，
    # 默认开启）：方程不是语料写法时，改写成等价写法跑一次，再把结果转回原输出轴序。
    # 放在这里的理由：逐元素与 cupy 委派这两条热路径已在上面返回，不受影响；走到这里
    # 的都是真正的通用收缩——正是映射能帮上忙的场景。任何不适用都退回原路径。
    # 语料命中的方程**不必**再改写（改写的目的正是命中语料形态），所以只在未命中时施加；
    # ``kernel_force="generic"`` 下也**整段跳过**：改写唯一的目的就是命中特化形态，而
    # 调用方已经明确要求通用（改写只会在内层再打一条 "specialized path" 自相矛盾）。
    if (isinstance(subscripts, str) and out is None and not kwargs
            and not callable(optimize)
            and not isinstance(optimize, (list, tuple))
            and _equivalence.as_equivalence_enabled(as_equivalence)
            and specialized is None
            and specialize_on
            and not generic_only):
        # 唯一可能省下匹配的输入是**配置表的等价行**（tune 判出"走等价转换后更快"时
        # 写下的那一行）：它带着可执行的改写载荷，命中它就照表施加、连
        # ``normalize_equation`` 都不必调（载荷首次套用时做一次数值自检，见
        # ``_payload_result``）。载荷缺失/坏 JSON/对不上张量一律退回真的查一次。
        #
        # 表是**只读**的：这里不写任何"查过没有"的标记，所以一次查询的结论不会跨进程
        # 留下来——要留下就得为这条键 tune（§1 唯一写者）。画像关闭时 ``info`` 为
        # ``None``，本段连表都不碰（零 IO）。
        payload = info.get("equivalence") if info is not None else None
        equivalence_result = _contract_via_equivalence(
            subscripts, operands, use_blas=use_blas, optimize=optimize,
            memory_limit=memory_limit, backend=backend, specialize=specialize_on,
            kernel_force=kernel_force, cache=cache, payload=payload)
        if equivalence_result is not None:
            # 本次是**等价改写**跑成的：原方程上强制具体内核会失败（它本来就不在
            # 语料里），所以这个标签不可重放；但必须如实标注，别沿用初始值
            # ``generic``——那会把"改写+topology"谎报成"通用路径"（实跑复现过）。
            if info is not None:
                # 记录**可重放的那个选择**：调用方（或画像）给了内核时，那个值本身
                # 就是合法且可复现的建议。只有"没给、纯靠改写跑成"才记
                # `equivalence`——它在原方程上强制内核必然"未尝试"，不可重放。
                _text = "" if kernel_force is None else str(kernel_force).strip().lower()
                info["kernel"] = _text or _EQUIVALENCE_LABEL
            return _profile_end(info, equivalence_result)

    global _LAST_CONTRACT

    operands_list = [subscripts] + list(operands)

    cached_key = None
    cacheable_operands = all(
        hasattr(operand, "shape") and hasattr(operand, "dtype")
        for operand in operands
    )
    optimize_key = _optimize_cache_key(optimize)
    cache_enabled = (
        isinstance(subscripts, str)
        and cacheable_operands
        and optimize_key is not None
        and not kwargs
        and optimize is not False
        # 语料外的方程不走编译程序缓存：那条门会**直接执行**原生程序，绕过
        # `_core_contract` 里"无 batch 维就别用原生执行器"的判据（实测纯 2D 链
        # 慢 6–10%）。计划复用不受这里影响，它对所有方程都生效。
        and specialized is not None
        and specialize is not False
    )
    if cache_enabled:
        resolved_backend = parse_backend(operands, backend)
        last_signature = _contract_signature(
            subscripts, resolved_backend, optimize_key, use_blas, memory_limit, operands
        )
        memo = _LAST_CONTRACT
        if memo is not None and last_signature == memo[0]:
            try:
                memo_result = memo[1](list(operands), out=out)
            except Exception:
                # Shapes, dtypes or devices changed under the same objects: drop
                # the memo and fall through to the shape keyed lookup.
                _LAST_CONTRACT = None
            else:
                return _profile_end(info, memo_result)
        cached_key = _contract_program_key(
            subscripts, operands, resolved_backend, optimize_key, memory_limit, use_blas
        )
        cached_program = _cached_contract_program(cached_key)
        if cached_program is not None:
            try:
                program_result = cached_program(list(operands), out=out)
            except Exception:
                if cached_key is not None:
                    _CONTRACT_PROGRAM_CACHE.pop(cached_key, None)
            else:
                _LAST_CONTRACT = (last_signature, cached_program)
                return _profile_end(info, program_result)

    # if no optimization, run pure einsum
    if optimize is False:
        input_subscripts, output_subscript, arrays = _parser().parse_einsum_input(operands_list)
        selected_backend = parse_backend(arrays, backend)
        plain_result = _einsum(
            f"{input_subscripts}->{output_subscript}",
            *arrays,
            backend=selected_backend,
            out=out,
            **kwargs,
        )
        return _profile_end(info, plain_result)

    # grab non-einsum kwargs
    gen_expression = kwargs.pop("_gen_expression", False)
    constants_dict = kwargs.pop("_constants_dict", {})
    full_str = operands_list[0] if gen_expression else None

    plan_key = None
    # 2 算子的规划也要重建整条收缩表（步表 + BLAS 判定），建键比它便宜得多，
    # 因此 2 算子以上的图都启用计划缓存。
    if isinstance(subscripts, str) and not kwargs and len(operands) >= 2:
        plan_key = _contract_program_key(
            subscripts, operands, parse_backend(operands, backend),
            _plan_cache_optimize_key(optimize, optimize_key), memory_limit, use_blas,
        )
    cached_plan = _cached_contract_plan(plan_key)
    if cached_plan is not None:
        try:
            _input_subscripts, _output_subscript, operands_prepped = (
                _parser().parse_einsum_input(operands_list))
        except Exception:                 # 形状/内容变了：丢弃缓存重新规划
            _CONTRACT_PLAN_CACHE.pop(plan_key, None)
            cached_plan = None
        else:
            contraction_list = cached_plan
    if cached_plan is None:
        operands_prepped, contraction_list = contract_path(  # type: ignore[misc]
            *operands_list,
            optimize=optimize,
            memory_limit=memory_limit,
            einsum_call=True,
            use_blas=use_blas,
            # ``cache=False`` 的公开语义是"本次调用所有 einsum 缓存既不读也不写"，
            # 规划缓存（``api.find_path``）也在其中；这里不把开关传下去，那半句话就
            # 只剩匹配器成立（诊断用的 cache=False 调用仍会热身/污染规划缓存）。
            cache=cache,
        )
        _cache_contract_plan(plan_key, contraction_list)

    if gen_expression:
        from ._expr import ContractExpression

        return _profile_end(
            info, ContractExpression(full_str, contraction_list, constants_dict, **kwargs))

    resolved_backend = parse_backend(operands_prepped, backend)
    if cache_enabled:
        if cached_key is None:
            cached_key = _contract_program_key(
                subscripts, operands_prepped, resolved_backend, optimize_key,
                memory_limit, use_blas,
            )
        program = _compile_native_expression(
            contraction_list, operands_prepped, resolved_backend
        )
        if program is not None:
            _cache_contract_program(cached_key, program)
            try:
                compiled_result = program(list(operands_prepped), out=out)
            except Exception:
                _CONTRACT_PROGRAM_CACHE.pop(cached_key, None)
            else:
                _LAST_CONTRACT = (_contract_signature(
                    subscripts, resolved_backend, optimize_key, use_blas,
                    memory_limit, operands_prepped,
                ), program)
                return _profile_end(info, compiled_result)

    core_result = _core_contract(operands_prepped, contraction_list, backend=backend,
                                 out=out,
                                 batch_like=_has_batch_index(subscripts),
                                 equation=subscripts if isinstance(subscripts, str) else None,
                                 **kwargs)
    return _profile_end(info, core_result)


def _run_step(step: Any, operands: list[Any], backend: str, out: Any, kwargs: dict[str, Any]) -> Any:
    """Run one contraction step and return the produced array.

    ``_execute.execute_step`` executes a single step in isolation, so it is
    handed exactly the operands that step consumes (renumbered positions keep
    the step's pop order, hence the einsum term order, unchanged).
    """
    inds = tuple(step[0])
    order = sorted(inds)
    window = [operands[p] for p in order]
    rank = {position: index for index, position in enumerate(order)}
    local_step = (tuple(rank[x] for x in inds),) + tuple(step[1:])

    execute_step = _executor().execute_step
    if kwargs and _accepts_var_kwargs(execute_step):
        return execute_step(local_step, window, backend, out, **kwargs)
    return execute_step(local_step, window, backend, out)


def _native_backend(backend: str) -> str | None:
    """Map an array backend to the matching native execution backend."""
    if backend in ("numpy", "cpu"):
        return "kupl"
    if backend in ("cupy", "gpu"):
        return "cuda"
    return None


def _native_compatible(operands: Sequence[Any], backend: str, kwargs: dict[str, Any]) -> bool:
    """Reject cases where a native call would copy, reinterpret or lose semantics."""
    if kwargs or _native_backend(backend) is None:
        return False
    for operand in operands:
        if operand is None or not hasattr(operand, "shape") or not hasattr(operand, "dtype"):
            return False
        if any(int(extent) == 0 for extent in operand.shape):
            # The native kernels require positive packed extents. The Python
            # execution path implements zero-length reductions as exact zeros.
            return False
        flags = getattr(operand, "flags", None)
        if flags is not None and not bool(getattr(flags, "c_contiguous", True)):
            return False
        if flags is None and hasattr(operand, "is_contiguous") and not operand.is_contiguous():
            return False
    return True


def _fused_disabled() -> bool:
    """当前调用是否禁用原生融合路径（specialize=False 时由 env 标记）。

    它现在**只进缓存键**（``_native_plan_key`` 与 ``_try_native_contract`` /
    ``_compile_native_expression`` 各自的 key），不再决定"跑不跑融合核"——CUDA
    融合核已删除。**不要**因为"名字还在、语义没人读"就删掉它：删掉会让两种
    ``specialize`` 状态共用同一份原生计划/编译缓存。
    """
    return os.environ.get(_specialize.FUSED_DISABLE_ENV) is not None


def _native_plan_key(
    operands: Sequence[Any], contraction_list: Sequence[tuple[Any, ...]], backend: str
) -> tuple[Any, ...]:
    return (
        backend,
        _fused_disabled(),
        tuple(tuple(int(value) for value in operand.shape) for operand in operands),
        tuple(str(operand.dtype) for operand in operands),
        tuple(
            int(operand.device.id)
            if hasattr(operand, "device") and hasattr(operand.device, "id")
            else -1
            for operand in operands
        ),
        tuple(
            (
                tuple(int(value) for value in step[0]),
                tuple(sorted(step[1])),
                str(step[2]),
                None if step[3] is None else tuple(str(value) for value in step[3]),
                str(step[4]),
            )
            for step in contraction_list
        ),
    )


def _fallback_signature(message: str, reason: str | None) -> str:
    """回落告警的去重签名：**同一个根因只报一次**。

    不能直接拿整条文案当键——同一个 ``NativeUnavailable``（例如库缺失）会在"编译"
    与"执行"两个层各自拼出不同措辞，于是同一根因报两次；反过来也不能只用一个
    进程级布尔，那样第二个**不同**故障会被吞掉。折中：优先用根因里稳定的那段
    （``NativeUnavailable: …`` 到分号/换行），没有就退回整条文案。
    """
    if reason:
        return reason
    # 同一根因在不同层会有不同措辞（"compile failed with status 7" vs
    # "execution failed with status 7"），所以优先归一到状态码。
    status = re.search(r"status\s*[= ]\s*(\d+)", message)
    if status is not None:
        return f"native-status-{status.group(1)}"
    match = re.search(r"NativeUnavailable:[^;\n]+", message)
    if match is not None:
        return match.group(0).strip()
    return message


def _warn_native_fallback_once(message: str, reason: str | None = None) -> None:
    """Emit a native fallback warning, **once per distinct root cause**.

    早期实现是一个进程级布尔：第二个**不同原因**的失败会被彻底吞掉，于是长进程里
    用户只看到第一条原因、照着它去修却永远修不好。现在按 :func:`_fallback_signature`
    去重——同一根因只报一次，不同根因各报一次。
    """
    key = _fallback_signature(message, reason)
    if key in _NATIVE_WARNED_REASONS:
        return
    _NATIVE_WARNED_REASONS.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=4)


# 原生编译失败的一次性诊断：库缺失、库无法 dlopen、库没有可复用编译入口是三种不同的
# 修法，所以文案要点名是哪一种，并给出归档绝对路径、build_info() 的后端与可执行修复步骤。
_NATIVE_ARCHIVE_LABEL = {
    "kupl": "pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so",
    "cuda": "pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so",
}
_NATIVE_REBUILD_COMMAND = {"kupl": "bash install.sh", "cuda": "bash install.sh cuda"}
_NATIVE_REPAIR_HINT = (
    "reinstall pyqcd from a build that ships {archive}, or rebuild it in a source "
    "checkout ({build}) after checking the C/C++ compiler and the build log; then "
    "confirm with `python -m pyqcd` and `python -m pyqcd --gpu-probe`"
)


def _native_library_report(native_backend: str) -> dict[str, Any]:
    """Path, presence and failure class of a native archive (never raises)."""
    try:
        if native_backend == "kupl":
            from .kupl import einsum as kupl_einsum

            return kupl_einsum.library_report()
        from ._native import library_path as backend_library_path

        path = os.path.abspath(str(backend_library_path(native_backend)))
        present = os.path.isfile(path)
        return {
            "path": path,
            "present": present,
            "status": "compiler context unavailable" if present else "library missing",
        }
    except Exception:  # noqa: BLE001 - 诊断失败绝不能改变调用方的行为
        return {"path": "unknown", "present": False, "status": "unknown"}


def _native_compile_warning(
    native_backend: str, exc: BaseException, *, per_call: bool
) -> str:
    """One-line, actionable warning for a native compiler that cannot be reused."""
    report = _native_library_report(native_backend)
    try:
        info = _native_executor().build_info()
        build_backend = str(info.get("backend", "unknown"))
        array_backend = str(info.get("array_backend", "unknown"))
    except Exception:  # noqa: BLE001 - 同上
        build_backend = array_backend = "unknown"
    hint = _NATIVE_REPAIR_HINT.format(
        archive=_NATIVE_ARCHIVE_LABEL.get(native_backend, "the native archive"),
        build=_NATIVE_REBUILD_COMMAND.get(native_backend, "bash install.sh"),
    )
    execution = "per-call native execution" if per_call else "Python execution"
    resilience = ""
    if report.get("load_retry"):
        resilience += f"; load_retry={report['load_retry']}"
    return (
        f"pyqcd.einsum native {native_backend} compiler unavailable "
        f"({report['status']}); using {execution}, results stay correct: "
        f"{type(exc).__name__}: {exc}; library={report['path']} "
        f"(present={report['present']}){resilience}; backend={build_backend} "
        f"array_backend={array_backend}; fix: {hint}"
    )


def _native_context_cache_limit() -> int:
    """Bound of the compiled-context cache (``0`` disables caching).

    ``_NATIVE_CONTEXT_CACHE_LIMIT`` may be assigned directly; the environment
    variable ``PYQCD_EINSUM_CONTEXT_CACHE_LIMIT`` is read once on first use.
    """
    global _NATIVE_CONTEXT_CACHE_LIMIT, _NATIVE_CONTEXT_LIMIT_RESOLVED
    if not _NATIVE_CONTEXT_LIMIT_RESOLVED:
        _NATIVE_CONTEXT_LIMIT_RESOLVED = True
        raw = os.environ.get(_NATIVE_CONTEXT_CACHE_ENV, "").strip()
        if raw:
            try:
                configured = int(raw)
            except ValueError:
                configured = -1
            if configured >= 0:
                _NATIVE_CONTEXT_CACHE_LIMIT = configured
    return _NATIVE_CONTEXT_CACHE_LIMIT


def _cached_native_context(key: tuple[Any, ...]) -> Any | None:
    """Return the compiled context cached for ``key`` (LRU touch), or ``None``."""
    with _NATIVE_CONTEXT_CACHE_LOCK:
        program = _NATIVE_CONTEXT_CACHE.get(key)
        if program is not None:
            _NATIVE_CONTEXT_CACHE.move_to_end(key)
        return program


def _store_native_context(key: tuple[Any, ...], program: Any | None) -> None:
    """Insert ``program`` into the bounded LRU, evicting the least recently used."""
    limit = _native_context_cache_limit()
    with _NATIVE_CONTEXT_CACHE_LOCK:
        if limit <= 0 or program is None:
            # Caching disabled: drop what is there so no context stays pinned.
            _NATIVE_CONTEXT_CACHE.clear()
            return
        _NATIVE_CONTEXT_CACHE[key] = program
        _NATIVE_CONTEXT_CACHE.move_to_end(key)
        while len(_NATIVE_CONTEXT_CACHE) > limit:
            # Evicted programs lose their last reference here; their destructor
            # releases the native context (``applyDestroyQcd`` and friends).
            _NATIVE_CONTEXT_CACHE.popitem(last=False)


def _drop_native_context(key: tuple[Any, ...]) -> None:
    """Forget the compiled context of ``key`` (used after a failed call)."""
    with _NATIVE_CONTEXT_CACHE_LOCK:
        _NATIVE_CONTEXT_CACHE.pop(key, None)


def _native_plan_bytes(
    key: tuple[Any, ...],
    contraction_list: Sequence[tuple[Any, ...]],
    operands: Sequence[Any],
    native_cache: dict[tuple[Any, ...], Any] | None,
) -> bytes | None:
    """Return the encoded plan for ``key``, encoding and caching it once."""
    plan = native_cache.get(key) if native_cache is not None else _NATIVE_PLAN_CACHE.get(key)
    if plan is not None:
        if native_cache is None:
            _NATIVE_PLAN_CACHE.move_to_end(key)
        return plan

    try:
        from ._native_plan import encode_execution_plan

        plan = encode_execution_plan(contraction_list, operands)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        _NATIVE_DISABLED.add(key)
        _warn_native_fallback_once(
            f"pyqcd.einsum native plan unavailable; using Python execution: {exc}"
        )
        return None

    if native_cache is None:
        _NATIVE_PLAN_CACHE[key] = plan
        if len(_NATIVE_PLAN_CACHE) > _NATIVE_PLAN_CACHE_LIMIT:
            _NATIVE_PLAN_CACHE.popitem(last=False)
    return plan


def _call_native_program(
    program: Any,
    key: tuple[Any, ...],
    operands: Sequence[Any],
    out: Any,
    native_backend: str,
) -> tuple[bool, Any]:
    """Run ``program`` with *this* call's operands, disabling the key on failure."""
    try:
        return True, program(list(operands), out=out)
    except Exception as exc:
        # Fail closed: never retry a program that produced an error for this key.
        _NATIVE_DISABLED.add(key)
        _drop_native_context(key)
        _warn_native_fallback_once(
            f"pyqcd.einsum native {native_backend} execution unavailable; "
            f"using Python execution: {type(exc).__name__}: {exc}"
        )
        return False, None


def _execute_native_plan(
    plan: bytes,
    key: tuple[Any, ...],
    operands: Sequence[Any],
    out: Any,
    native_backend: str,
) -> tuple[bool, Any]:
    """Execute ``plan`` without a reusable context (historical behaviour)."""
    try:
        result = _native_executor().execute(
            plan, list(operands), out=out, backend=native_backend
        )
    except Exception as exc:  # Native failures must fall back without changing semantics.
        _NATIVE_DISABLED.add(key)
        _warn_native_fallback_once(
            f"pyqcd.einsum native {native_backend} execution unavailable; "
            f"using Python execution: {type(exc).__name__}: {exc}"
        )
        return False, None
    return True, result


_CUDA_FUSED_EQUATIONS = ("ab,...bce,cd->...dae",)


def _cuda_fused_shape(equation: str | None) -> bool:
    """True when the CUDA executor has a fused kernel for this equation.

    这个名单是**CUDA 执行器的准入证**，不是"融合核存在性"的记录：名单里的方程会被
    放行进原生执行器，名单外的直接回落 ``cupy.einsum``（实测在 16×1024² 的
    complex128 上快约 5 倍）。逐项改动它等于改选路，不是改注释——加方程前先确认
    原生侧确实有对应的步表。
    """
    if not equation:
        return True                       # 未知时保持原生（保守）
    normalised = equation.replace(" ", "")
    return normalised in _CUDA_FUSED_EQUATIONS


def _has_batch_index(subscripts: str | None) -> bool:
    """True when a label is shared by two operands **and** kept in the output.

    That is exactly the shape family where the native plan executor wins big
    (measured 6–11x): numpy would loop per batch, while the native plan does one
    batched GEMM. Pure ``ndim>=3`` is the wrong test — a double contraction such as
    ``abc,bcd->ad`` has three-dimensional operands but no batch index, and there the
    native executor is 1.5x *slower* than numpy's tensordot ladder.
    """
    if not subscripts:
        return False
    text = subscripts.replace(" ", "")
    if "..." in text:
        return True                      # ellipsis always means batched axes
    if "->" not in text:
        return False
    lhs, rhs = text.split("->", 1)
    terms = [term for term in lhs.split(",") if term]
    if len(terms) < 2:
        return False
    output = set(rhs)
    seen: set[str] = set()
    for term in terms:
        labels = set(term)
        if labels & seen & output:
            return True
        seen |= labels
    return False


def _batched_operands(operands: Sequence[Any]) -> bool:
    """Fallback when the equation text is unavailable (e.g. interleaved form)."""
    for operand in operands:
        shape = getattr(operand, "shape", None)
        if shape is not None and len(shape) >= 3:
            return True
    return False


def _kupl_plan_enabled() -> bool:
    """Whether the KUPL generic plan executor may run (default: off)."""
    return os.environ.get("PYQCD_EINSUM_KUPL_PLAN", "").strip().lower() in {
        "1", "true", "on", "yes"}


def _try_native_contract(
    operands: Sequence[Any],
    contraction_list: Sequence[tuple[Any, ...]],
    backend: str,
    out: Any,
    kwargs: dict[str, Any],
    native_cache: dict[tuple[Any, ...], Any] | None = None,
    batch_like: bool | None = None,
    equation: str | None = None,
) -> tuple[bool, Any]:
    """Return ``(handled, result)`` for one native execution attempt."""
    if not _native_compatible(operands, backend, kwargs):
        return False, None
    native_backend = _native_backend(backend)
    assert native_backend is not None
    if native_backend == "cuda" and not _cuda_fused_shape(equation):
        # cupy 的 einsum 直接走 cuBLAS（含批量 GEMM），实测在 16×1024² 的
        # complex128 上比我们的原生批量 GEMM 快约 5 倍；只有存在融合 kernel 的
        # 方程（γ 夹心）才值得走原生执行器。
        return False, None
    wants_batch_path = (_batched_operands(operands) if batch_like is None
                        else batch_like)
    if (native_backend == "kupl" and not _kupl_plan_enabled()
            and not wants_batch_path):
        # CPU 计划执行器分两类形态（交错实测）：
        #   · 纯 2D 链/矩阵乘：比 numpy 分步慢约 6–10%（打包/置换没有额外收益）；
        #   · 带 batch 维（ndim≥3）或 ellipsis：比 numpy 的逐 batch 循环快 5–10 倍。
        # 所以只在高维形态上启用；CUDA 侧不受影响（融合 kernel 在其执行器内部）。
        return False, None
    if native_cache is None:
        key = _native_plan_key(operands, contraction_list, native_backend)
    else:
        key = (
            native_backend,
            _fused_disabled(),
            tuple(str(operand.dtype) for operand in operands),
            tuple(tuple(int(value) for value in operand.shape) for operand in operands),
        )
    if key in _NATIVE_DISABLED:
        return False, None

    if native_cache is not None:
        cached = native_cache.get(key)
        if cached is not None and hasattr(cached, "__call__"):
            return _call_native_program(cached, key, operands, out, native_backend)
    else:
        # Plain ``contract()``: reuse the compiled context of this exact plan.
        cached = _cached_native_context(key)
        if cached is not None:
            return _call_native_program(cached, key, operands, out, native_backend)

    plan = _native_plan_bytes(key, contraction_list, operands, native_cache)
    if plan is None:
        return False, None

    if native_cache is None and key in _NATIVE_CONTEXT_UNAVAILABLE:
        return _execute_native_plan(plan, key, operands, out, native_backend)

    if native_backend == "cuda":
        # CUDA 没有可复用的编译上下文：compiled-context ABI 已随融合 kernel 一起删除。
        # 逐次下发同一份计划（``contract_plan``），数值与形状不变。
        return _execute_native_plan(plan, key, operands, out, native_backend)

    try:
        program = _native_executor().compile(plan, backend=native_backend)
    except Exception as exc:
        if native_cache is None:
            # No reusable context for this plan: keep executing the plan per call
            # instead of dropping the whole native path.
            _NATIVE_CONTEXT_UNAVAILABLE.add(key)
            _warn_native_fallback_once(
                _native_compile_warning(native_backend, exc, per_call=True)
            )
            return _execute_native_plan(plan, key, operands, out, native_backend)
        _NATIVE_DISABLED.add(key)
        _warn_native_fallback_once(
            _native_compile_warning(native_backend, exc, per_call=False)
        )
        return False, None

    if native_cache is not None:
        native_cache[key] = program
    else:
        _store_native_context(key, program)
    return _call_native_program(program, key, operands, out, native_backend)


def _compile_native_expression(
    contraction_list: Sequence[tuple[Any, ...]],
    operands: Sequence[Any],
    backend: str,
    kwargs: dict[str, Any] | None = None,
) -> Any | None:
    """Compile a reusable native program for a fixed expression, or return ``None``."""
    if not _native_compatible(operands, backend, kwargs or {}):
        return None
    native_backend = _native_backend(backend)
    if native_backend is None:
        return None
    if native_backend == "cuda":
        # CUDA 没有可复用的编译上下文（该 ABI 已随融合 kernel 删除）：每次调用都由
        # ``_try_native_contract`` 下发同一份计划，因此这里没有"预编译程序"可返回。
        return None
    try:
        from ._native_plan import encode_execution_plan

        plan = encode_execution_plan(contraction_list, operands)
        return _native_executor().compile(plan, backend=native_backend)
    except Exception as exc:                             # noqa: BLE001
        # 编译失败是可回落的，但**必须留痕**：否则调用方只会看到"表达式建好了，
        # 但每次执行都慢一个量级"，而没有任何线索指向这里。
        _warn_native_fallback_once(
            "pyqcd.einsum: native expression compilation failed; using the "
            f"generic path: {type(exc).__name__}: {exc}",
            reason=f"expression-compile:{type(exc).__name__}:{exc}")
        return None


def _optimize_cache_key(optimize: Any) -> Any | None:
    """Return a hashable, mutation proof cache token for ``optimize``.

    ``None`` means "not cacheable". Strings, booleans and tuples are already
    immutable, so they double as their own token; an explicit ``path`` given as
    a list of step tuples is frozen into a tuple so that later edits of the list
    cannot hit a stale program (opt_einsum accepts both spellings). Callables and
    ``PathOptimizer`` instances get ``None`` here (no mutation proof exists, so
    the program cache stays off); the plan cache keys them by identity instead,
    see :func:`_plan_cache_optimize_key`.
    """
    if optimize is None or isinstance(optimize, (str, bool)):
        return optimize
    if isinstance(optimize, tuple):
        try:
            hash(optimize)
        except TypeError:
            return None
        return optimize
    if isinstance(optimize, list):
        try:
            return tuple(tuple(int(index) for index in step) for step in optimize)
        except (TypeError, ValueError):
            return None
    return None


class _OptimizeIdentity:
    """Identity token for a ``callable`` / ``PathOptimizer`` ``optimize``.

    Two different optimizers must never share a cached plan, and the plain
    ``id()`` of an object is not a valid cache key on its own: once the object
    is collected its id can be handed out again to a different optimizer, which
    would make the stale plan look like a hit. The token therefore compares by
    *object identity* (so even a recycled id only produces a harmless hash
    collision) and keeps a strong reference to the optimizer (so an entry stays
    meaningful, and comparable, for as long as it is cached). Repeated calls with
    the same object build a new token that is equal and hashes alike, hence they
    hit. The retained reference is bounded by the cache limits and is released
    together with the entry.
    """

    __slots__ = ("optimizer", "token")

    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.token = id(optimizer)

    def __hash__(self) -> int:
        return self.token

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, _OptimizeIdentity)
                and other.optimizer is self.optimizer)

    def __repr__(self) -> str:
        return f"<optimize {type(self.optimizer).__name__} at {self.token:#x}>"


def _plan_cache_optimize_key(optimize: Any, optimize_key: Any) -> Any | None:
    """Cache token for ``optimize`` inside the plan cache key.

    ``optimize_key`` is reused as-is when it exists (immutable argument). A
    callable/``PathOptimizer`` has no mutation proof token, but its identity is
    enough to keep different optimizers apart while still caching the plan of a
    repeated call, so it is replaced by an :class:`_OptimizeIdentity`. Everything
    else stays uncacheable (``None``), matching ``cache_enabled``: an unknown
    ``optimize`` argument disables caching.
    """
    if optimize_key is not None:
        return optimize_key
    return _OptimizeIdentity(optimize) if callable(optimize) else None


def _contract_signature(
    subscripts: Any,
    backend: str,
    optimize_key: Any,
    use_blas: bool,
    memory_limit: Any,
    operands: Sequence[Any],
) -> tuple[Any, ...]:
    """Identity memo key: the same program can be replayed for these operands.

    Shapes, dtypes and device ids are deliberately left to the native program,
    which validates them on every call and raises (``fail closed``) on a change.
    """
    return (
        subscripts,
        backend,
        optimize_key,
        bool(use_blas),
        memory_limit,
        _fused_disabled(),
        tuple(id(operand) for operand in operands),
    )


def _contract_program_key(
    subscripts: Any,
    operands: Sequence[Any],
    backend: str,
    optimize_key: Any,
    memory_limit: Any,
    use_blas: bool,
) -> tuple[Any, ...] | None:
    if not isinstance(subscripts, str) or _native_backend(backend) is None:
        return None
    try:
        return (
            subscripts,
            _native_backend(backend),
            tuple(tuple(int(value) for value in operand.shape) for operand in operands),
            tuple(str(operand.dtype) for operand in operands),
            tuple(
                int(operand.device.id)
                if hasattr(operand, "device") and hasattr(operand.device, "id")
                else -1
                for operand in operands
            ),
            optimize_key,
            memory_limit,
            bool(use_blas),
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _cached_contract_plan(key: tuple[Any, ...] | None) -> Any | None:
    if key is None:
        return None
    plan = _CONTRACT_PLAN_CACHE.get(key)
    if plan is not None:
        _CONTRACT_PLAN_CACHE.move_to_end(key)
    return plan


def _cache_contract_plan(key: tuple[Any, ...] | None, plan: Any) -> None:
    if key is None:
        return
    _CONTRACT_PLAN_CACHE[key] = plan
    _CONTRACT_PLAN_CACHE.move_to_end(key)
    while len(_CONTRACT_PLAN_CACHE) > _CONTRACT_PLAN_CACHE_LIMIT:
        _CONTRACT_PLAN_CACHE.popitem(last=False)


def _cached_contract_program(key: tuple[Any, ...] | None) -> Any | None:
    if key is None:
        return None
    program = _CONTRACT_PROGRAM_CACHE.get(key)
    if program is not None:
        _CONTRACT_PROGRAM_CACHE.move_to_end(key)
    return program


def _cache_contract_program(key: tuple[Any, ...] | None, program: Any | None) -> None:
    if key is None or program is None:
        return
    _CONTRACT_PROGRAM_CACHE[key] = program
    _CONTRACT_PROGRAM_CACHE.move_to_end(key)
    while len(_CONTRACT_PROGRAM_CACHE) > _CONTRACT_PROGRAM_CACHE_LIMIT:
        _CONTRACT_PROGRAM_CACHE.popitem(last=False)


def _core_contract(
    operands_: Sequence[Any],
    contraction_list: Sequence[tuple[Any, ...]],
    backend: str | None = "auto",
    batch_like: bool | None = None,
    equation: str | None = None,
    evaluate_constants: bool = False,
    out: Any = None,
    native_cache: dict[tuple[Any, ...], Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``contraction_list`` step by step on ``operands_``."""
    specified_out = out is not None

    operands = list(operands_)
    backend = parse_backend(operands, backend)

    # `shared_intermediates()` 期间必须走（被缓存的）Python 路径：原生执行会绕过
    # 共享缓存，使缓存恒为空、失去共享语义。
    sharing = False
    if not evaluate_constants:
        try:
            from . import _sharing

            sharing = _sharing.currently_sharing()
        except Exception:  # noqa: BLE001 - 探测失败按未共享处理
            sharing = False

    if not evaluate_constants and not sharing:
        # 回落是硬不变量：原生路径里的任何异常（包括规划/编译阶段，而不只是执行阶段）
        # 都必须降级到 Python 路径，最多警告一次，绝不把异常抛给调用方。
        try:
            handled, native_result = _try_native_contract(
                operands, contraction_list, backend, out, kwargs,
                native_cache=native_cache, batch_like=batch_like,
                equation=equation)
        except Exception as exc:  # noqa: BLE001 - 见上
            handled, native_result = False, None
            _warn_native_fallback_once(
                "pyqcd.einsum native path failed; using Python execution: "
                f"{type(exc).__name__}: {exc}")
        if handled:
            return native_result

    last = len(contraction_list) - 1
    for num, step in enumerate(contraction_list):
        inds = tuple(step[0])

        # break out upon finding the first non-constant operand when preparing
        # the constant pre-pass of an expression
        if evaluate_constants and any(operands[x] is None for x in inds):
            return operands, contraction_list[num:]

        handle_out = out if (specified_out and (num == last)) else None
        new_view = _run_step(step, operands, backend, handle_out, kwargs)

        # ``execute_step`` works on a copy of the operand list, so the
        # bookkeeping is ours to keep in sync with the contraction list
        for position in sorted(inds, reverse=True):
            operands.pop(position)
        operands.append(new_view)

    if specified_out:
        return out
    return operands[0]
