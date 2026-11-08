"""Primary contraction routines: :func:`contract_path` and :func:`contract`."""

from __future__ import annotations

import importlib
import inspect
import os
import threading
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from . import paths
from ._pathinfo import PathInfo
from ._sharing import einsum_cache_wrap
from . import _specialize, _threads
from .backends.dispatch import get_func, parse_backend

__all__ = ["contract", "contract_path"]

_EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EINSUM_PUNCTUATION = ",->."
_BASE_TYPES = (bool, int, float, complex, str, bytes)

_MODULES: dict[str, Any] = {}
_NATIVE_PLAN_CACHE: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
_NATIVE_PLAN_CACHE_LIMIT = 128
_CONTRACT_PROGRAM_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_CONTRACT_PROGRAM_CACHE_LIMIT = 64
_FUSED_ENV_OWNED = False
_LAST_CONTRACT_SIGNATURE: tuple[Any, ...] | None = None
_LAST_CONTRACT_PROGRAM: Any | None = None
_NATIVE_DISABLED: set[tuple[Any, ...]] = set()
_NATIVE_WARNING_EMITTED = False

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
    path, steps = _find_path(
        optimize, num_ops, input_sets, output_set, size_dict, memory_arg, eq, input_shapes, use_blas
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


_SPECIALIZED_KERNEL_WARNED = False


def _run_specialized_kernel(
    subscripts: Any, operands: Sequence[Any]
) -> tuple[Any, str] | tuple[None, str]:
    """依次尝试各专用 kernel；返回 ``(结果或 None, 命中的 kernel 名或 '')``。"""
    from . import _kernel_cuda, _kernel_kupl

    for label, attempt in (
        ("kupl_nucleon_2pt", lambda: _kernel_kupl.try_nucleon_2pt(subscripts, operands)),
        ("kupl_3pt", lambda: _kernel_kupl.try_three_pt(subscripts, operands)),
        ("cuda_nucleon_2pt", lambda: _kernel_cuda.try_nucleon_2pt(subscripts, operands)),
        ("cuda_3pt", lambda: _kernel_cuda.try_three_pt(subscripts, operands)),
    ):
        result = attempt()
        if result is not None:
            return result, label
    return None, ""


def _warn_specialized_kernel_once(exc: BaseException) -> None:
    global _SPECIALIZED_KERNEL_WARNED

    if _SPECIALIZED_KERNEL_WARNED:
        return
    _SPECIALIZED_KERNEL_WARNED = True
    warnings.warn(
        "pyqcd.einsum specialized kernel failed; using the generic path: "
        f"{type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )


def contract(
    subscripts: Any,
    *operands: Any,
    out: Any = None,
    use_blas: bool = True,
    optimize: Any = "auto",
    memory_limit: Any = None,
    backend: str = "auto",
    specialize: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Evaluate an einsum expression with an optimized contraction order.

    Drop-in replacement for :func:`numpy.einsum` that contracts the operands
    pairwise in a cost minimizing order.
    """
    if (optimize is True) or (optimize is None):
        optimize = "auto"

    # 首次调用时把线程配置补到已加载的 OpenMP/BLAS 运行时上（幂等、极廉价）。
    _threads.ensure_runtime()

    # 方程形态专用路径：命中时打印提示；specialize=False 关闭本次调用的专用执行。
    # 只回收本模块自己设置的开关，不覆盖调用方通过环境变量显式给出的设置。
    global _FUSED_ENV_OWNED
    specialized = None
    if isinstance(subscripts, str):
        if _specialize.enabled(specialize):
            if _FUSED_ENV_OWNED:
                os.environ.pop(_specialize.FUSED_DISABLE_ENV, None)
                _FUSED_ENV_OWNED = False
            specialized = _specialize.match(subscripts)
            # 内核路径与 BLAS 路径想要相反的线程配置：命中特化形态用 kernel 档，
            # 其余用 generic 档。切换只在档位真正变化时发生。
            if specialized is not None:
                _threads.apply_profile("kernel")
            else:
                _threads.apply_profile("generic")
            if specialized is not None and out is None:
                announce_kernel = ""
                if out is None:
                    # 专用 kernel 失败必须回退通用路径（契约：原生失败只 warning
                    # 一次，不把异常抛给调用方）。
                    try:
                        kernel_result, announce_kernel = _run_specialized_kernel(
                            subscripts, operands)
                    except Exception as exc:  # noqa: BLE001 - 见上
                        kernel_result = None
                        _warn_specialized_kernel_once(exc)
                    _specialize.announce(specialized, announce_kernel)
                    if kernel_result is not None:
                        return kernel_result
                else:
                    _specialize.announce(specialized)
        else:
            os.environ[_specialize.FUSED_DISABLE_ENV] = "1"
            _FUSED_ENV_OWNED = True

    global _LAST_CONTRACT_SIGNATURE, _LAST_CONTRACT_PROGRAM

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
        and specialized is not None
        and specialize is not False
    )
    if cache_enabled:
        resolved_backend = parse_backend(operands, backend)
        last_signature = _contract_signature(
            subscripts, resolved_backend, optimize_key, use_blas, memory_limit, operands
        )
        if last_signature == _LAST_CONTRACT_SIGNATURE and _LAST_CONTRACT_PROGRAM is not None:
            try:
                return _LAST_CONTRACT_PROGRAM(list(operands), out=out)
            except Exception:
                # Shapes, dtypes or devices changed under the same objects: drop
                # the memo and fall through to the shape keyed lookup.
                _LAST_CONTRACT_SIGNATURE = None
                _LAST_CONTRACT_PROGRAM = None
        cached_key = _contract_program_key(
            subscripts, operands, resolved_backend, optimize_key, memory_limit, use_blas
        )
        cached_program = _cached_contract_program(cached_key)
        if cached_program is not None:
            try:
                result = cached_program(list(operands), out=out)
                _LAST_CONTRACT_SIGNATURE = last_signature
                _LAST_CONTRACT_PROGRAM = cached_program
                return result
            except Exception:
                if cached_key is not None:
                    _CONTRACT_PROGRAM_CACHE.pop(cached_key, None)

    # if no optimization, run pure einsum
    if optimize is False:
        input_subscripts, output_subscript, arrays = _parser().parse_einsum_input(operands_list)
        selected_backend = parse_backend(arrays, backend)
        return _einsum(
            f"{input_subscripts}->{output_subscript}",
            *arrays,
            backend=selected_backend,
            out=out,
            **kwargs,
        )

    # grab non-einsum kwargs
    gen_expression = kwargs.pop("_gen_expression", False)
    constants_dict = kwargs.pop("_constants_dict", {})
    full_str = operands_list[0] if gen_expression else None

    operands_prepped, contraction_list = contract_path(  # type: ignore[misc]
        *operands_list,
        optimize=optimize,
        memory_limit=memory_limit,
        einsum_call=True,
        use_blas=use_blas,
    )

    if gen_expression:
        from ._expr import ContractExpression

        return ContractExpression(full_str, contraction_list, constants_dict, **kwargs)

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
                result = program(list(operands_prepped), out=out)
                _LAST_CONTRACT_SIGNATURE = _contract_signature(
                    subscripts, resolved_backend, optimize_key, use_blas,
                    memory_limit, operands_prepped,
                )
                _LAST_CONTRACT_PROGRAM = program
                return result
            except Exception:
                _CONTRACT_PROGRAM_CACHE.pop(cached_key, None)

    return _core_contract(operands_prepped, contraction_list, backend=backend, out=out, **kwargs)


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


def _warn_native_fallback_once(message: str) -> None:
    """Emit the one-time native fallback warning."""
    global _NATIVE_WARNING_EMITTED
    if _NATIVE_WARNING_EMITTED:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=4)
    _NATIVE_WARNING_EMITTED = True


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


def _try_native_contract(
    operands: Sequence[Any],
    contraction_list: Sequence[tuple[Any, ...]],
    backend: str,
    out: Any,
    kwargs: dict[str, Any],
    native_cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[bool, Any]:
    """Return ``(handled, result)`` for one native execution attempt."""
    if not _native_compatible(operands, backend, kwargs):
        return False, None
    native_backend = _native_backend(backend)
    assert native_backend is not None
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
                f"pyqcd.einsum native {native_backend} compiler unavailable; "
                f"using per-call native execution: {type(exc).__name__}: {exc}"
            )
            return _execute_native_plan(plan, key, operands, out, native_backend)
        _NATIVE_DISABLED.add(key)
        _warn_native_fallback_once(
            f"pyqcd.einsum native {native_backend} compiler unavailable; "
            f"using Python execution: {type(exc).__name__}: {exc}"
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
    except Exception:
        return None


def _optimize_cache_key(optimize: Any) -> Any | None:
    """Return a hashable, mutation proof cache token for ``optimize``.

    ``None`` means "not cacheable". Strings, booleans and tuples are already
    immutable, so they double as their own token; an explicit ``path`` given as
    a list of step tuples is frozen into a tuple so that later edits of the list
    cannot hit a stale program (opt_einsum accepts both spellings).
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
                native_cache=native_cache)
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
