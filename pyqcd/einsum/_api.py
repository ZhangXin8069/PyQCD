"""Primary contraction routines: :func:`contract_path` and :func:`contract`."""

from __future__ import annotations

import importlib
import inspect
import os
import re
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from . import paths
from ._pathinfo import PathInfo
from ._sharing import einsum_cache_wrap
from . import (_cache, _equivalence, _kernel_force, _profile_cache, _shape_paths,
               _specialize, _threads)
from .backends import dispatch as backends
from .backends.dispatch import get_func, parse_backend

__all__ = ["contract", "contract_path", "last_kernel", "cache_clear", "cache_stats"]

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
        optimize: ``False`` for a single n-ary einsum, an explicit path, the name
            of a registered path optimizer, a ``PathOptimizer`` instance, or
            ``True``/``None``/``"auto"`` to let the planner decide.
        memory_limit: Upper bound on the number of elements of any intermediate.
            ``None``/``-1`` means unlimited and ``"max_input"`` means the largest
            input tensor.
        shapes: Whether shapes rather than arrays have been supplied.
        cache: ``None`` follows ``PYQCD_EINSUM_CACHE`` (on by default); ``False``
            disables the planner cache for this call (neither read nor write).

    Returns:
        ``(path, PathInfo)``: the contraction order and a printable summary.
    """
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
    """最近一次专用内核尝试的记录：``requested`` / ``ran`` / ``attempted`` / ``reasons``。

    ``requested`` 是本次调用要求的**逻辑名**（``None`` = 自动分派，``()`` = 只走通用
    路径，``dedicated`` / ``topology`` / ``backend`` = 强制），``ran`` 是实际跑成的
    标签（``None`` = 一个都没跑成；``backend`` 表示直接走了后端自己的 einsum），
    ``attempted`` 是按尝试次序排列的具体标签（``kupl_dedicated`` …），``reasons`` 是
    逐标签的失败原因（与 ``KernelUnavailableError.reasons`` 同源）。记录在每次
    ``contract()`` 走进专用分支时**整体覆盖**，因此读到的一定是最近一次调用的结果，
    不会残留上一次的命中——"到底跑了哪个内核"不必再去翻 ``LAST_FAILURE`` 或负缓存的
    内部状态。``requested`` 只报**调用方**的要求（参数或 ``PYQCD_EINSUM_KERNEL_FORCE``）；
    配置画像给出的建议不是调用方的要求，它的命中由自己的一行提示报告。
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


def _run_specialized_kernel(
    subscripts: Any,
    operands: Sequence[Any],
    allowed: tuple[str, ...] | None = None,
) -> tuple[Any, str, dict[str, str]]:
    """依次尝试各专用 kernel。

    ``allowed`` 决定允许走哪些标签：

    * ``None``（缺省）→ 旧分派逐字不变：``*_dedicated`` → ``*_topology``，第一个成功者胜。
    * ``()`` → 一次也不尝试，立即返回 ``(None, "", {})``。
    * ``(label, ...)`` → **只**尝试这些标签，次序按 ``allowed`` 的书写次序（强制内核时
      "用户先写的先试"）；attempt 表里没有的标签直接跳过，由调用方报告"未尝试"。
      强制模式下每个 attempt 前套上内核模块自己的 ``bypass_negative_cache()``
      （"强制即真试一次"；模块没提供就照旧），并把 attempt 抛出的异常收成该标签的
      失败原因继续试下一个——强制模式不接受静默跳过。

    Returns ``(result or None, hit label or "", {label: failure reason})`` — the
    per-label reasons let the caller report the *relevant* cause instead of
    whichever attempt happened to run last.
    """
    reasons: dict[str, str] = {}
    if allowed is not None and not allowed:
        return None, "", {}
    attempts = _specialized_kernel_attempts(operands)
    if allowed is None:
        ordered = attempts
    else:
        by_label = {entry[0]: entry for entry in attempts}
        ordered = tuple(by_label[label] for label in allowed if label in by_label)
    for label, module, attempt in ordered:
        if allowed is None:
            # 自动分派：异常照旧向上抛，由 contract 的 warning + 回落兜住。
            result = attempt(subscripts, operands)
        else:
            bypass = getattr(module, "bypass_negative_cache", None)
            try:
                if bypass is None:
                    result = attempt(subscripts, operands)
                else:
                    with bypass():
                        result = attempt(subscripts, operands)
            except Exception as exc:  # noqa: BLE001 - 收成原因，最后统一抛
                reasons[label] = f"{type(exc).__name__}: {exc}"
                continue
        if result is not None:
            return result, label, reasons
        # 失败后才读原因：早读会拿到上一次调用残留的值
        reasons[label] = str(module.LAST_FAILURE.get("reason", ""))
    return None, "", reasons


def _resolve_forced_labels(
    forced: Sequence[str],
    operands: Sequence[Any],
) -> dict[str, tuple[str, ...]]:
    """把逻辑名解析成本次操作数后端的 attempt 标签。

    ``dedicated`` → ``kupl_dedicated``（numpy 操作数）/ ``cuda_dedicated``（cupy），
    ``topology`` 同理。角色从标签后缀取（``kupl_dedicated`` → ``dedicated``），所以
    映射跟着 ``_specialized_kernel_attempts`` 的当前后端走，不写死第二份后端表。
    ``forced`` 里出现尝试表没有的角色时**不给映射**——调用方按"未尝试"报告，
    不猜一个可能跑不了的入口。
    """
    by_role: dict[str, str] = {}
    for label, _module, _attempt in _specialized_kernel_attempts(operands):
        by_role[label.rpartition("_")[2]] = label
    resolved: dict[str, tuple[str, ...]] = {}
    for name in forced:
        concrete = by_role.get(name)
        if concrete is not None:
            resolved[name] = (concrete,)
    return resolved


def _run_forced_kernel(
    subscripts: Any,
    operands: Sequence[Any],
    requested: tuple[str, ...],
    record: dict[str, Any],
) -> Any:
    """强制内核的唯一执行点：按 ``requested`` 次序真试一次，跑不成即抛。

    ``requested`` 是**逻辑名**（``dedicated`` / ``topology``），先按操作数后端解析成
    具体标签再下发；失败时把"逻辑名 → 具体标签"的对照一并交给异常，消息里两种名字
    都在（用户对得上自己写的取值，也看得见哪个入口真的被试过）。

    与 auto 分派只有两点不同：①每个 attempt 前旁路负缓存（"强制即真试一次"，
    不再被上一次的失败记录挡住）；②一个都没跑成时抛
    :class:`~._kernel_force.KernelUnavailableError`，而不是 warning + 回落。
    跑成时的结果与 auto 命中完全同源（同一个 ``attempt`` 入口），不改任何数值。
    """
    resolved = _resolve_forced_labels(requested, operands)
    allowed = tuple(label for labels in resolved.values() for label in labels)
    result, ran, reasons = _run_specialized_kernel(
        subscripts, operands, allowed=allowed)
    record["ran"] = ran or None
    record["attempted"] = tuple(reasons) + ((ran,) if ran else ())
    record["reasons"] = dict(reasons)
    if result is None:
        raise _kernel_force.KernelUnavailableError(
            requested, subscripts, reasons, record["attempted"], resolved)
    return result


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


##### 等价改写的跨进程负标记

#: ``as_equivalence=`` 的第四个取值：**忽略**负标记，真的再查一次等价映射，并在查完
#: 后刷新标记。API 与 CLI 都可能直接把这个字符串传进来。
_AS_EQ_FORCE = "force"


def _equivalence_marked_none(key: dict[str, Any] | None) -> bool:
    """``key`` 是否被标记成"查过、无法等价改写"。

    标记只是**省时间**的提示：查询/写入都是可选能力（``_profile_cache`` 不提供这两个
    函数时用 ``getattr`` 降级），任何异常、任何非字典返回值都按"未标记"处理——读不到
    就必须照常往下查，绝不因为一个可选缓存把等价步骤整段关掉。
    """
    if key is None:
        return False
    lookup = getattr(_profile_cache, "equivalence_marker", None)
    if lookup is None:
        return False
    try:
        marker = lookup(key)
    except Exception:                                    # noqa: BLE001 - 标记绝不外抛
        return False
    if not isinstance(marker, dict):
        return False
    return str(marker.get("state", "")).strip().lower() == "none"


def _record_equivalence(key: dict[str, Any] | None, mapping: Any) -> None:
    """把一次查询的结论落成标记：``mapping is None`` → ``"none"``，否则 ``"ok"``。

    ``as_equivalence="force"`` 走的是同一条路径，因此旧标记会被新结论覆盖（刷新）。
    写失败、未实现、``key is None``（画像关闭）都静默跳过。
    """
    if key is None:
        return
    record = getattr(_profile_cache, "record_equivalence", None)
    if record is None:
        return
    state = "none" if mapping is None else "ok"
    kind = "" if mapping is None else str(getattr(mapping, "kind", ""))
    try:
        record(key, state, kind=kind)
    except Exception:                                    # noqa: BLE001 - 标记绝不外抛
        return


def _note_equivalence_skipped() -> None:
    """VERBOSE 下说明"这一步是按缓存标记跳过的"，免得把跳过看成"查过但没找到"。"""
    if _profile_verbose():
        print("[pyqcd.einsum] equivalence 已跳过（缓存标记：此 equation 无法等价改写）",
              flush=True)


def _contract_via_equivalence(subscripts: str, operands: Sequence[Any], *,
                              use_blas: bool, optimize: Any, memory_limit: Any,
                              backend: str, specialize: bool,
                              kernel_force: str | None,
                              cache: bool | None = None,
                              marker_key: dict[str, Any] | None = None) -> Any | None:
    """用等价写法跑一次并把结果转回原输出轴序；不适用时返回 ``None``。

    只在真正的通用收缩路径上被调用（逐元素与 cupy 委派早已返回）。任何一步不满足
    都返回 ``None`` 让调用方走原路径——这里**绝不**降级成"半个改写"。

    ``specialize`` 是调用方**已求值**的开关（``_specialize.enabled()`` 只算一次），
    因此这里是纯布尔判断，不再重复读环境变量。

    ``marker_key`` 是画像键（``None`` = 画像关闭、不碰标记）：真的查过之后把结论
    记成跨进程标记，下一次同类调用就可能在
    :func:`_equivalence_marked_none` 那道廉价门上直接跳过。标记只影响"要不要花时间
    查映射"，命中/跳过都不改方程、dtype、形状或结果。
    """
    if not _equivalence.enabled():
        return None
    if callable(optimize) or isinstance(optimize, (list, tuple)):
        return None          # 显式路径/可调用优化器绑定的是原操作数次序，不能重排
    if not specialize:
        return None          # 映射的目的是命中特化形态，关掉就没意义
    try:
        mapping = _equivalence.normalize_equation(subscripts, use=cache)
    except Exception:                                    # noqa: BLE001 - 不许影响数值
        # 查崩了不算结论：不落标记，下一次照常重查（标记宁缺勿错）。
        return None
    # 查完就落标记（force 也走这里 → 旧标记被新结论覆盖）：下次同类调用可在廉价门上
    # 直接跳过。写入失败/未实现都静默，绝不影响本次结果。
    _record_equivalence(marker_key, mapping)
    if mapping is None or mapping.kind == "identity":
        return None
    if not (mapping.operand_order or mapping.operand_permutations
            or mapping.output_permutation):
        return None

    order = mapping.operand_order
    if order is None:
        ordered = list(operands)
    else:
        if len(order) != len(operands):
            return None
        ordered = [operands[index] for index in order]

    perms = mapping.operand_permutations or ()
    adjusted: list[Any] = []
    for index, operand in enumerate(ordered):
        perm = perms[index] if index < len(perms) else None
        if perm is None:
            adjusted.append(operand)
            continue
        if len(perm) != len(_get_shape(operand)):
            return None
        adjusted.append(_contiguous_copy(_axis_permuted(operand, perm)))

    result = contract(mapping.equation, *adjusted, out=None, use_blas=use_blas,
                      optimize=optimize, memory_limit=memory_limit,
                      backend=backend, specialize=specialize,
                      kernel_force=kernel_force, as_equivalence=False,
                      profile=False)
    if mapping.output_permutation is not None:
        result = result.transpose(mapping.output_permutation)

    expected = _expected_output_shape(subscripts, operands)
    if expected is not None and tuple(result.shape) != expected:
        raise RuntimeError(
            "等价映射改变了输出形状：原方程要求 "
            f"{expected}，按改写 {mapping.equation!r} 得到 {tuple(result.shape)}；"
            "这是映射层的缺陷（结果不予返回）。请把该方程报告给维护者，"
            "并可用 as_equivalence=False 绕过。")
    return result


##### 配置画像（profile）接线

# 本进程**已 load() 的目录**（绝对路径）。每个目录只读一次；目录换了就重读，
# 目录不变就纯内存查表：`_profile_cache.best` / `record` 在 `_loaded` 与目录就位后
# 都不再碰磁盘。
_PROFILE_READY: str | None = None    # 已 load() 的**绝对目录**；None = 尚未加载
#: 等价改写跑成时记录的标签：**不可重放**（原方程上强制具体内核必然"未尝试"），
#: 但必须如实标注——沿用 `generic` 会把"改写+topology"谎报成"通用路径"。
_EQUIVALENCE_LABEL = "equivalence"
#: ``--tune`` 期间为真：记录时绕过 5% 稀疏门，留下整张地图。
_PROFILE_TUNE = False
_PROFILE_LOCK = threading.Lock()

# 命中提示的去重表：每 (方程, 形状, 配置) 每进程只打一行（VERBOSE 打开了就每次必打）。
_PROFILE_NOTICED: set[tuple[Any, ...]] = set()
_PROFILE_NOTICE_LOCK = threading.Lock()


def _profile_devices(operands: Sequence[Any], backend_name: str) -> tuple[str, str]:
    """``(device, user_device)``：具体计算设备与用户视角的设备口径。

    两者都只用来建键，取不到就给空串——画像缓存不该因为一个怪后端抛错。
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


def _profile_optimize(optimize: Any) -> str:
    """optimize 列：字符串/布尔原样，callable 与 PathOptimizer 只记类型名。"""
    if isinstance(optimize, (str, bool)) or optimize is None:
        return str(optimize)
    return type(optimize).__name__


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
    """``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1``：连未命中也要打一行。"""
    return os.environ.get(_specialize.VERBOSE_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


def _profile_notice(info: dict[str, Any], *, row: dict[str, Any] | None = None,
                    applied: bool = False) -> None:
    """一行提示：**施加**了的命中每形态每进程打一次；其余只在 VERBOSE 时打。

    ``row is None`` = 未命中；``row`` 非空但 ``applied`` 为假 = 命中但没施加
    （``out=`` / 调用方已强制内核等）。后两种默认静默，免得把"有一条记录"说成
    "已经按它执行"。
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
        source = _profile_cache.summary().get("file", _profile_cache.PROFILE_NAME)
        print(
            f"[pyqcd.einsum] profile 命中：backend={row.get('backend', '')}"
            f" kernel={row.get('kernel', '')}"
            f" compute_ms={row.get('compute_ms', '')}（来自 {source}）",
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
            "（按现有逻辑规划，结果已记入内存）",
            flush=True,
        )


def _profile_begin(profile: bool | None, profile_dir: str | None, subscripts: Any,
                   operands: Sequence[Any], *, backend: str, out: Any,
                   specialize_on: bool, caller_force: tuple[str, ...] | None,
                   optimize: Any) -> dict[str, Any] | None:
    """画像入口：返回本次调用的记录槽，``None`` 表示本次不画像（零副作用）。

    开关与目录的分辨率**完全**交给 :mod:`._profile_cache`（显式参数 >
    ``PYQCD_EINSUM_PROFILE[_DIR]`` > 默认 ``./.cache``）。``profile=False`` 在第一个
    判断就返回：不 ``configure``、不 ``load``、不 ``record``，也不建目录。

    进程内只做**一次** IO（``configure`` + ``load``）。命中只填 ``force`` /
    ``backend`` 两个槽位——方程、dtype、形状与数值一概不动。``caller_force`` 非
    ``None``（调用方参数或环境变量已经指定了内核）、``out=``、``specialize=False``
    这些情形下命中只被记下而不施加：显式要求永远压过画像建议。
    """
    global _PROFILE_READY
    if not _profile_cache.enabled(profile):
        return None
    if not isinstance(subscripts, str) or not operands:
        return None                     # 交错/字典形式没有可作键的方程文本
    if any(not hasattr(o, "shape") or not hasattr(o, "dtype") for o in operands):
        return None
    # 门闸必须按**已配置的目录**判定，不能用一个单向布尔：同一进程里
    # ``profile_dir`` 从 A 换到 B 时，布尔会让 B 被完全忽略（继续用 A 的索引）。
    # ``configure`` 自己幂等（路径不变就只做一次字符串比较，变了就冲队列+清索引），
    # 所以这里每调用一次它，代价是几个字符串操作；``load`` 只在目录真的变了时跑。
    resolved_dir = (_profile_cache.configure(profile_dir) if profile_dir
                    else _profile_cache.ensure_configured())
    if _PROFILE_READY != resolved_dir:
        with _PROFILE_LOCK:
            if _PROFILE_READY != resolved_dir:
                _profile_cache.load(force=True)
                _PROFILE_READY = resolved_dir
    backend_name = parse_backend(operands, backend)
    device, user_device = _profile_devices(operands, backend_name)
    key = {
        "user_device": user_device,
        "backend": backend_name,
        "dtype": _profile_dtype(operands),
        "equation": "".join(subscripts.split()),
        "shape": _profile_cache.format_shape([o.shape for o in operands]),
        "device": device,
    }
    row = _profile_cache.best(key)
    info: dict[str, Any] = {
        "key": key,
        "optimize": _profile_optimize(optimize),
        "init_ms": 0.0,
        "compute_ms": 0.0,
        "kernel": "generic",
        "note": "",
        "hit": row,
        "force": None,
        "backend": None,
        "fallback": False,
    }
    if row is None:
        _profile_notice(info)
        return info
    label = str(row.get("kernel") or "")
    force = _profile_force_label(label)
    if force is None or caller_force is not None or out is not None \
            or not specialize_on:
        info["note"] = f"profile 命中未施加（kernel={label or '?'}）"
        _profile_notice(info, row=row)
        return info
    info["force"] = force
    if str(backend or "").strip().lower() in ("", "auto"):
        recorded = str(row.get("backend") or "")
        if recorded and recorded != _kernel_force.AUTO_LABEL:
            info["backend"] = recorded
    info["note"] = f"profile 命中：kernel={label or '?'}"
    _profile_notice(info, row=row, applied=True)
    return info


def _profile_clock(info: dict[str, Any] | None) -> float | None:
    """计算段的起点；未画像时返回 ``None``，调用点据此整段跳过。"""
    return None if info is None else time.perf_counter()


def _profile_spend(info: dict[str, Any] | None, started: float | None) -> None:
    """把一段"规划/绑定"耗时累进 ``init_ms``（命中缓存的那一段不进来）。"""
    if info is not None and started is not None:
        info["init_ms"] += (time.perf_counter() - started) * 1000.0


def _profile_record(info: dict[str, Any]) -> None:
    """把本次观测写进内存（不落盘：落不落盘由 :mod:`._profile_cache` 的稀疏口径定）。

    已经施加了命中的调用没有新信息，不重复记录；命中被丢弃/回退的调用照记，
    让"实际跑成的那条配置"有机会在下一次胜出。
    """
    applied_hit = (info.get("hit") is not None
                   and (info.get("force") is not None
                        or info.get("backend") is not None))
    if applied_hit and not info.get("fallback"):
        return
    _profile_cache.record({
        **info["key"],
        "kernel": info.get("kernel") or "generic",
        "optimize": info["optimize"],
        "init_ms": info["init_ms"],
        "compute_ms": info["compute_ms"],
        "note": info.get("note", ""),
    }, force_write=_PROFILE_TUNE)


def _profile_end(info: dict[str, Any] | None, result: Any,
                 started: float | None = None) -> Any:
    """记录本次观测并原样返回结果；``info is None`` 时就是一次普通返回。

    ``compute_ms`` 只覆盖 ``started`` 之后的那一段——解析、键构造、规划与绑定都
    不在计时区里（规划/绑定另记 ``init_ms``）。
    """
    if info is None:
        return result
    if started is not None:
        info["compute_ms"] = (time.perf_counter() - started) * 1000.0
    _profile_record(info)
    return result


def _profile_retry(info: dict[str, Any], subscripts: Any, operands: Sequence[Any],
                   **call: Any) -> Any:
    """画像建议跑不成：退回**原**分派一次，绝不把异常抛给调用方。

    只在 ``KernelUnavailableError`` 由画像强制引起时调用（用户自己强制的标签
    仍然照旧抛出）。重试带 ``profile=False``：本次观测由外层统一记录，不重复。
    """
    info["fallback"] = True
    info["force"] = None
    info["note"] = "profile 建议不可用，已回退自动分派"
    started = time.perf_counter()
    result = contract(subscripts, *operands, profile=False, **call)
    return _profile_end(info, result, started)


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
        ``False`` for a single n-ary einsum, an explicit path, the name of a
        registered optimizer (``"greedy"``/``"optimal"``/``"dp"``), or
        ``True``/``None``/``"auto"`` (default) to let the planner decide.
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
        Which specialised kernel may run. ``None``/``"auto"`` (default) keeps the
        original dispatch: try ``*_dedicated`` then ``*_topology``, and fall back
        to the generic path with one ``RuntimeWarning`` when none of them runs.
        ``"generic"`` skips the specialised kernels entirely (no warning).
        ``"dedicated"``/``"topology"`` allow exactly that kernel **of the operand
        backend** (``kupl_*`` for numpy, ``cuda_*`` for cupy), comma separated to
        allow both, tried in the order written; forcing bypasses the kernels'
        negative cache, and a kernel that does not run raises
        ``KernelUnavailableError`` instead of falling back silently. ``"backend"``
        leaves the specialised dispatch altogether and calls the backend's own
        ``einsum``; it cannot be combined with the other labels. An explicit
        ``kernel_force`` wins over ``PYQCD_EINSUM_KERNEL_FORCE``; both are resolved
        on every call. Forcing any value contradicts ``out=`` and
        ``specialize=False``, which raise ``ValueError``.
    as_equivalence : bool | str | None
        ``None`` (default) follows ``PYQCD_EINSUM_AS_EQUIVALENCE`` (on by default);
        ``False`` disables the automatic rewrite for this call; ``"force"`` enables
        it and **ignores the cross-process negative marker** (see below), really
        querying the equivalence table again. When on and the
        equation does not already run on a specialised form, the equivalence layer
        rewrites it into a corpus spelling, permutes the operands accordingly and
        transposes the result back — **the returned array is the answer to the
        original equation**, with the original shape and dtype, so callers never
        see the rewrite. Skipped for ``out=``, extra ``**kwargs``, element-wise and
        cupy-delegated equations, ``specialize=False``, and explicit optimizer
        paths (those bind the original operand order). Every real lookup records a
        cross-process marker (``none`` / ``ok``) in the profile cache; with the
        profile on, a ``none`` marker lets later calls of the same
        ``(device, backend, dtype, equation, shape)`` skip the lookup entirely —
        skipping changes no equation, dtype, shape or number, only the time spent
        looking. ``profile=False`` neither reads nor writes the marker.
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
        all** — no ``configure``, no ``load``, no ``record``, not even a directory.
        When on, the profile is read **once per process**
        (``_profile_cache.configure`` + ``load``) and the best recorded
        ``(host, device, backend, dtype, equation, shape)`` row decides **only** which
        specialised kernel is forced (``kernel_force``) and which backend is chosen —
        the equation, dtype, shape and numbers are untouched. A suggestion that does
        not run falls back to the original dispatch once (never raising
        ``KernelUnavailableError`` at the caller); a hit prints one line. A recorded
        *specialised* kernel (``kupl_dedicated`` / ``cuda_topology`` …) is replayed as
        ``dedicated`` / ``topology``; a recorded **``backend``** is also replayed — it
        means "the backend own ``einsum`` was fastest here", which is a legitimate
        conclusion on small shapes and is the whole point of keeping a profile. A
        recorded ``generic`` is *recorded and reported but not replayed* (it would mute
        the auto-dispatch warning). A caller/env supplied ``kernel_force`` always wins
        over the profile.
    profile_dir : str | None
        Directory of the profile cache. Resolution order: this parameter,
        ``PYQCD_EINSUM_PROFILE_DIR``, then ``./.cache`` (relative to the current
        working directory). Ignored when the profile is off. Passing a **different**
        directory switches to that table for the rest of the process (the index and
        the pending queue are flushed/reset accordingly); **omitting** it keeps
        whatever directory is already pinned (``None``/empty means "not given", never
        "go back to the environment/default"). Only a first, never-configured call
        resolves the default.
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

    Notes
    -----
    Specialized kernels are selected by **topology** (label-independent), so
    ``"ij,jk->ik"`` and ``"ij,jb->ib"`` take the same path. A failed native
    kernel falls back to the generic path with one ``RuntimeWarning``; results
    stay correct. Set ``PYQCD_EINSUM_SPECIALIZE_VERBOSE=1`` to log every hit.
    :func:`last_kernel` reports which kernel the most recent call actually ran.
    """
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    global _FUSED_ENV_OWNED, _LAST_KERNEL

    # 首次调用时把线程配置补到已加载的 OpenMP/BLAS 运行时上（幂等、极廉价）。
    _threads.ensure_runtime()

    # 专用路径开关只求值**一次**：专用门、等价门与画像门共用它，
    # 避免每次调用重复读环境变量。
    specialize_on = _specialize.enabled(specialize)

    # 强制内核：显式 ``kernel_force=`` 优先于 ``PYQCD_EINSUM_KERNEL_FORCE``。三态语义：
    # ``None`` = 自动分派（旧行为逐字不变）；``()`` = 完全跳过专用内核；
    # 非空元组 = 只允许这些**逻辑名**（按书写次序，运行时按后端解析成具体标签），
    # 跑不成就抛 KernelUnavailableError。
    forced = _kernel_force.forced_labels(kernel_force)
    if forced:
        # "强制即真试一次"：凡是会让专用内核根本跑不到的开关，一律早失败，
        # 不做"半强制"的静默回落。
        if not isinstance(subscripts, str):
            raise ValueError(
                "kernel_force= 只能用于字符串方程；交错/字典形式的输入没有专用内核可强制")
        if out is not None:
            raise ValueError(
                "out= 会禁用专用内核，不能与强制指定内核同时使用："
                "去掉 out=，或把 kernel_force 设回 'auto'")

    # 配置画像：本进程首次进入且画像开启时读一次；命中只决定内核目标与后端选择，
    # 方程/dtype/形状/数值一概不动（失败回退见 _profile_retry）。调用方/环境已经
    # 指定了内核的调用不受画像影响——那才是真正的"显式优先"。
    info = _profile_begin(profile, profile_dir, subscripts, operands,
                          backend=backend, out=out, specialize_on=specialize_on,
                          caller_force=forced, optimize=optimize)
    requested_backend, requested_kernel_force = backend, kernel_force
    requested_forced = forced
    forced_by_profile = False
    if info is not None:
        if info["force"] is not None:
            kernel_force = info["force"]
            forced = _kernel_force.forced_labels(kernel_force)
            forced_by_profile = True
        if info["backend"] is not None:
            backend = info["backend"]

    if forced == (_kernel_force.BACKEND_LABEL,):
        # ``backend`` = "别走专用内核，直接调后端自己的 einsum"。它既不进专用
        # 分派，也不做等价改写（改写的目的正是命中专用形态），所以在分派之前返回。
        _LAST_KERNEL = {
            "requested": requested_forced,
            "ran": _kernel_force.BACKEND_LABEL,
            "attempted": (),
            "reasons": {},
        }
        started = _profile_clock(info)
        backend_result = _backend_einsum(
            parse_backend(operands, backend), subscripts, operands)
        if info is not None:
            info["kernel"] = _kernel_force.BACKEND_LABEL
        return _profile_end(info, backend_result, started)

    # 方程形态专用路径：命中时打印提示；specialize=False 关闭本次调用的专用执行。
    # 只回收本模块自己设置的开关，不覆盖调用方通过环境变量显式给出的设置。
    specialized = None
    if isinstance(subscripts, str):
        if specialize_on:
            if _FUSED_ENV_OWNED:
                os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
                _FUSED_ENV_OWNED = False
            specialized = _specialize.match(subscripts, use=cache)
            # 内核路径与 BLAS 路径想要相反的线程配置：命中特化形态用 kernel 档，
            # 其余用 generic 档。切换只在档位真正变化时发生。
            if specialized is not None:
                _threads.apply_profile("kernel")
            else:
                _threads.apply_profile("generic")
                # 通用内核首次命中提醒（每个进程一次，PYQCD_EINSUM_NOTICE=0 可关）。
                # 只在确实没有特化形态时打，避免和 announce() 重复。
                _specialize.announce_generic(subscripts)
            # 本次调用的内核记录：默认"没尝试任何专用内核"，真跑内核的分支填上结果。
            # 整体覆盖写，所以 last_kernel() 绝不会读到上一次调用残留的命中。
            record: dict[str, Any] = {
                "requested": requested_forced, "ran": None, "attempted": (),
                "reasons": {}}
            _LAST_KERNEL = record
            if specialized is not None and out is None:
                if forced == ():
                    # generic：完全跳过专用内核（不发 warning，也不碰负缓存）。
                    _specialize.announce(specialized)
                elif forced:
                    # 强制具体标签：按用户给的次序真试一次，跑不成即抛。
                    started = _profile_clock(info)
                    try:
                        kernel_result = _run_forced_kernel(
                            subscripts, operands, forced, record)
                    except _kernel_force.KernelUnavailableError:
                        if not forced_by_profile:
                            raise
                        return _profile_retry(
                            info, subscripts, operands, out=out, use_blas=use_blas,
                            optimize=optimize, memory_limit=memory_limit,
                            backend=requested_backend, specialize=specialize,
                            kernel_force=requested_kernel_force,
                            as_equivalence=as_equivalence, cache=cache, **kwargs)
                    if info is not None:
                        info["kernel"] = record["ran"] or "generic"
                    _specialize.announce(specialized, record["ran"] or "")
                    return _profile_end(info, kernel_result, started)
                else:
                    # 专用 kernel 失败必须回退通用路径（契约：原生失败只 warning
                    # 一次，不把异常抛给调用方）。
                    announce_kernel = ""
                    kernel_reasons: dict[str, str] = {}
                    started = _profile_clock(info)
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
                    _specialize.announce(specialized, announce_kernel)
                    if kernel_result is not None:
                        return _profile_end(info, kernel_result, started)
                    if str(specialized.get("kind", "")).startswith("kernel"):
                        relevant = _kernel_labels_for(specialized)
                        reasons = [kernel_reasons.get(label, "")
                                   for label in relevant] or list(kernel_reasons.values())
                        _warn_kernel_skipped_once(specialized, reasons)
            elif forced:
                # 语料没命中却强制了具体标签：旧行为是静默走通用路径，这里必须真试
                # 一次，并明确报出"未尝试"或具体原因。
                started = _profile_clock(info)
                try:
                    kernel_result = _run_forced_kernel(
                        subscripts, operands, forced, record)
                except _kernel_force.KernelUnavailableError:
                    if not forced_by_profile:
                        raise
                    return _profile_retry(
                        info, subscripts, operands, out=out, use_blas=use_blas,
                        optimize=optimize, memory_limit=memory_limit,
                        backend=requested_backend, specialize=specialize,
                        kernel_force=requested_kernel_force,
                        as_equivalence=as_equivalence, cache=cache, **kwargs)
                if info is not None:
                    info["kernel"] = record["ran"] or "generic"
                return _profile_end(info, kernel_result, started)
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
        started = _profile_clock(info)
        cupy_result = _backend_einsum("cupy", subscripts, operands)
        if info is not None:
            info["kernel"] = _kernel_force.BACKEND_LABEL
        return _profile_end(info, cupy_result, started)

    # 逐元素/纯置换类方程没有 BLAS 结构：直接交给 backend 的原生 einsum，
    # 比逐步执行 + 计划缓存更快（实测 Hadamard 4000² 89 ms -> 29 ms）。
    if (isinstance(subscripts, str) and out is None and not kwargs
            and _is_elementwise_equation(subscripts)):
        elementwise_backend = parse_backend(operands, backend)
        started = _profile_clock(info)
        elementwise_result = _backend_einsum(
            elementwise_backend, subscripts, operands)
        if info is not None:
            info["kernel"] = _kernel_force.BACKEND_LABEL
        return _profile_end(info, elementwise_result, started)

    # 形状特化路径：少量结构特殊的方程（语料之外）用一次性推导出的固定算法，
    # 不规划、不建步表。只接管无 out=/无 kwargs 的调用；前置条件不满足即回落。
    if (isinstance(subscripts, str) and out is None and not kwargs
            and specialize is not False):
        started = _profile_clock(info)
        shape_result = _shape_paths.try_execute(subscripts, operands)
        if shape_result is not None:
            return _profile_end(info, shape_result, started)

    # 等价映射自动施加（``as_equivalence``，缺省跟随 PYQCD_EINSUM_AS_EQUIVALENCE，
    # 默认开启）：方程不是语料写法时，改写成等价写法跑一次，再把结果转回原输出轴序。
    # 放在这里的理由：逐元素与 cupy 委派这两条热路径已在上面返回，不受影响；走到这里
    # 的都是真正的通用收缩——正是映射能帮上忙的场景。任何不适用都退回原路径。
    # 语料命中的方程**不必**再改写（改写的目的正是命中语料形态），所以只在未命中时施加。
    if (isinstance(subscripts, str) and out is None and not kwargs
            and not callable(optimize)
            and not isinstance(optimize, (list, tuple))
            and _equivalence.as_equivalence_enabled(as_equivalence)
            and specialized is None
            and specialize_on):
        # 廉价门（在 ``_split`` / ``_Shape`` / 语料索引之前）：长跑里同一条语料外方程
        # 会被反复问"有没有等价写法"，答案却不变。跨进程标记说"这条查过、没有"时直接
        # 放弃等价步骤——**只省时间**：标记不参与方程、dtype、形状或结果。
        # ``as_equivalence="force"`` = 不信标记、再查一遍（查完照常刷新标记）；
        # 画像关闭时 ``info`` 为 ``None``，此处既不查也不写标记（零 IO）。
        marker_key = info.get("key") if info is not None else None
        if (as_equivalence != _AS_EQ_FORCE
                and _equivalence_marked_none(marker_key)):
            _note_equivalence_skipped()
        else:
            started = _profile_clock(info)
            equivalence_result = _contract_via_equivalence(
                subscripts, operands, use_blas=use_blas, optimize=optimize,
                memory_limit=memory_limit, backend=backend, specialize=specialize_on,
                kernel_force=kernel_force, cache=cache, marker_key=marker_key)
            if equivalence_result is not None:
                # 本次是**等价改写**跑成的：原方程上强制具体内核会失败（它本来就不在
                # 语料里），所以这个标签不可重放；但必须如实标注，别沿用初始值
                # ``generic``——那会把"改写+topology"谎报成"通用路径"（实跑复现过）。
                if info is not None:
                    # 记录**可重放的那个选择**：调用方显式强制了内核时，强制值本身
                    # 就是合法且可复现的建议（`generic` 实测比 topology 快 7 倍，丢掉
                    # 它等于让缓存永远学不到这条）。只有"没强制、纯靠改写跑成"才记
                    # `equivalence`——它在原方程上强制内核必然"未尝试"，不可重放。
                    _text = "" if kernel_force is None else str(kernel_force).strip().lower()
                    info["kernel"] = (_text if _text and _text != _kernel_force.AUTO_LABEL
                                      else _EQUIVALENCE_LABEL)
                return _profile_end(info, equivalence_result, started)

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
                started = _profile_clock(info)
                memo_result = memo[1](list(operands), out=out)
            except Exception:
                # Shapes, dtypes or devices changed under the same objects: drop
                # the memo and fall through to the shape keyed lookup.
                _LAST_CONTRACT = None
            else:
                return _profile_end(info, memo_result, started)
        cached_key = _contract_program_key(
            subscripts, operands, resolved_backend, optimize_key, memory_limit, use_blas
        )
        cached_program = _cached_contract_program(cached_key)
        if cached_program is not None:
            try:
                started = _profile_clock(info)
                program_result = cached_program(list(operands), out=out)
            except Exception:
                if cached_key is not None:
                    _CONTRACT_PROGRAM_CACHE.pop(cached_key, None)
            else:
                _LAST_CONTRACT = (last_signature, cached_program)
                return _profile_end(info, program_result, started)

    # if no optimization, run pure einsum
    if optimize is False:
        input_subscripts, output_subscript, arrays = _parser().parse_einsum_input(operands_list)
        selected_backend = parse_backend(arrays, backend)
        started = _profile_clock(info)
        plain_result = _einsum(
            f"{input_subscripts}->{output_subscript}",
            *arrays,
            backend=selected_backend,
            out=out,
            **kwargs,
        )
        return _profile_end(info, plain_result, started)

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
    planning = _profile_clock(info)
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
        )
        _cache_contract_plan(plan_key, contraction_list)
        # 命中计划缓存的这一次不记 init_ms（没有任何规划/绑定发生）。
        _profile_spend(info, planning)

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
        compiling = _profile_clock(info)
        program = _compile_native_expression(
            contraction_list, operands_prepped, resolved_backend
        )
        _profile_spend(info, compiling)
        if program is not None:
            _cache_contract_program(cached_key, program)
            try:
                started = _profile_clock(info)
                compiled_result = program(list(operands_prepped), out=out)
            except Exception:
                _CONTRACT_PROGRAM_CACHE.pop(cached_key, None)
            else:
                _LAST_CONTRACT = (_contract_signature(
                    subscripts, resolved_backend, optimize_key, use_blas,
                    memory_limit, operands_prepped,
                ), program)
                return _profile_end(info, compiled_result, started)

    started = _profile_clock(info)
    core_result = _core_contract(operands_prepped, contraction_list, backend=backend,
                                 out=out,
                                 batch_like=_has_batch_index(subscripts),
                                 equation=subscripts if isinstance(subscripts, str) else None,
                                 **kwargs)
    return _profile_end(info, core_result, started)


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
    """当前调用是否禁用原生融合路径（specialize=False 时由 env 标记）。"""
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
    """True when the CUDA executor has a fused kernel for this equation."""
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
