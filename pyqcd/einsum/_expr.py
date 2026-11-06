"""Reusable contraction expressions for fixed shapes."""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Collection, Iterable, Sequence

from . import backends
from ._api import _compile_native_expression, _core_contract, _parser, contract_path

__all__ = [
    "ArrayShaped",
    "ContractExpression",
    "contract_expression",
    "format_const_einsum_str",
    "shape_only",
]

ArrayShaped = namedtuple("ArrayShaped", ["shape"])


def shape_only(shape: Sequence[int]) -> ArrayShaped:
    """Dummy array-like object which has a shape only."""
    return ArrayShaped(tuple(shape))


def format_const_einsum_str(einsum_str: str, constants: Iterable[int]) -> str:
    """Add brackets to the constant terms in ``einsum_str``.

    For example ``'ab,bc,cd->ad'`` with constants ``[0, 2]`` becomes
    ``'bc,[ab,cd]->ad'``. No-op if there are no constants.
    """
    constants = set(constants)
    if not constants:
        return einsum_str

    if "->" in einsum_str:
        lhs, rhs = einsum_str.split("->")
        arrow = "->"
    else:
        lhs, rhs, arrow = einsum_str, "", ""

    wrapped_terms = [f"[{t}]" if i in constants else t for i, t in enumerate(lhs.split(","))]

    formatted_einsum_str = "{}{}{}".format(",".join(wrapped_terms), arrow, rhs)

    # merge adjacent constants
    return formatted_einsum_str.replace("],[", ",")


def _convert_interleaved_input(operands: Sequence[Any]) -> tuple[str, tuple[Any, ...]]:
    """Convert ``operand, indices, operand, indices, ...`` input to subscripts."""
    convert = getattr(_parser(), "convert_interleaved_input", None)
    if convert is not None:
        return convert(operands)
    return _local_convert_interleaved_input(operands)


def _local_convert_interleaved_input(operands: Sequence[Any]) -> tuple[str, tuple[Any, ...]]:
    import itertools

    get_symbol = _parser().get_symbol

    tmp_operands = list(operands)
    operand_list: list[Any] = []
    subscript_list: list[Any] = []
    for _ in range(len(operands) // 2):
        operand_list.append(tmp_operands.pop(0))
        subscript_list.append(tmp_operands.pop(0))

    output_list = tmp_operands[-1] if len(tmp_operands) else None

    try:
        symbol_set = set(itertools.chain.from_iterable(subscript_list))
        symbol_set.discard(Ellipsis)
        symbol_map = {symbol: get_symbol(idx) for idx, symbol in enumerate(sorted(symbol_set))}
    except TypeError:
        raise TypeError(
            "For this input type lists must contain either Ellipsis or hashable and comparable object (e.g. int, str)."
        ) from None

    def convert_subscripts(old_sub: Sequence[Any]) -> str:
        new_sub = ""
        for symbol in old_sub:
            new_sub += "..." if symbol is Ellipsis else symbol_map[symbol]
        return new_sub

    subscripts = ",".join(convert_subscripts(sub) for sub in subscript_list)
    if output_list is not None:
        subscripts += "->" + convert_subscripts(output_list)

    return subscripts, tuple(operand_list)


class ContractExpression:
    """A stored contraction list that can be evaluated with plain arrays."""

    def __init__(
        self,
        contraction: str,
        contraction_list: Sequence[tuple[Any, ...]],
        constants_dict: dict[int, Any],
        **kwargs: Any,
    ) -> None:
        self.contraction = format_const_einsum_str(contraction, constants_dict.keys())
        self.contraction_list = contraction_list
        self.kwargs = kwargs

        # need to know _full_num_args to parse constants with, and num_args to call with
        self._full_num_args = contraction.count(",") + 1
        self.num_args = self._full_num_args - len(constants_dict)

        # likewise need to know full contraction list
        self._full_contraction_list = contraction_list

        self._constants_dict = constants_dict
        self._evaluated_constants: dict[str, Any] = {}
        self._backend_expressions: dict[str, Any] = {}
        self._native_cache: dict[tuple[Any, ...], Any] = {}
        self._native_expression_programs: dict[tuple[str, str, tuple[int, ...]], Any] = {}

    def evaluate_constants(self, backend: str | None = "auto") -> None:
        """Contract the constant operands and store the remaining expression."""
        tmp_const_ops = [self._constants_dict.get(i, None) for i in range(self._full_num_args)]
        backend = backends.parse_backend(tmp_const_ops, backend)

        try:
            new_ops, new_contraction_list = backends.evaluate_constants(
                backend, tmp_const_ops, self
            )
        except KeyError:
            new_ops, new_contraction_list = self(
                *tmp_const_ops, backend=backend, evaluate_constants=True
            )

        self._evaluated_constants[backend] = new_ops
        self.contraction_list = new_contraction_list

    def _get_evaluated_constants(self, backend: str) -> Any:
        try:
            return self._evaluated_constants[backend]
        except KeyError:
            self.evaluate_constants(backend)
            return self._evaluated_constants[backend]

    def _get_backend_expression(self, arrays: Sequence[Any], backend: str) -> Any:
        try:
            return self._backend_expressions[backend]
        except KeyError:
            fn = backends.build_expression(backend, arrays, self)
            self._backend_expressions[backend] = fn
            return fn

    def _contract(
        self,
        arrays: Sequence[Any],
        out: Any = None,
        backend: str | None = "auto",
        evaluate_constants: bool = False,
    ) -> Any:
        """The normal, core contraction."""
        contraction_list = (
            self._full_contraction_list if evaluate_constants else self.contraction_list
        )

        native_backend = {"numpy": "numpy", "cupy": "cupy"}.get(backend)
        native_inputs = all(
            hasattr(array, "shape") and hasattr(array, "dtype") for array in arrays
        )
        if (
            not evaluate_constants
            and native_inputs
            and native_backend is not None
            and not self.kwargs
        ):
            dtype_key = str(arrays[0].dtype) if arrays else ""
            device_key = tuple(
                int(array.device.id)
                if hasattr(array, "device") and hasattr(array.device, "id")
                else -1
                for array in arrays
            )
            program_key = (native_backend, dtype_key, device_key)
            if program_key not in self._native_expression_programs:
                program = _compile_native_expression(
                    contraction_list, arrays, native_backend
                )
                self._native_expression_programs[program_key] = program or False
            program = self._native_expression_programs[program_key]
            if program is not False:
                try:
                    return program(list(arrays), out=out)
                except Exception:
                    self._native_expression_programs[program_key] = False

        return _core_contract(
            list(arrays),
            contraction_list,
            out=out,
            backend=backend,
            evaluate_constants=evaluate_constants,
            native_cache=self._native_cache,
            **self.kwargs,
        )

    def _contract_with_conversion(
        self,
        arrays: Sequence[Any],
        out: Any,
        backend: str,
        evaluate_constants: bool = False,
    ) -> Any:
        """Contract with a different backend, converting to and from numpy."""
        if evaluate_constants:
            return backends.evaluate_constants(backend, arrays, self)

        result = self._get_backend_expression(arrays, backend)(*arrays)

        if out is not None:
            out[()] = result
            return out

        return result

    def __call__(
        self,
        *arrays: Any,
        out: Any = None,
        backend: str = "auto",
        evaluate_constants: bool = False,
    ) -> Any:
        """Evaluate this expression with a set of arrays."""
        backend = backends.parse_backend(arrays, backend)

        correct_num_args = self._full_num_args if evaluate_constants else self.num_args

        if len(arrays) != correct_num_args:
            raise ValueError(
                f"This `ContractExpression` takes exactly {self.num_args} array arguments but received {len(arrays)}."
            )

        if self._constants_dict and not evaluate_constants:
            # fill in the missing non-constant terms with newly supplied arrays
            ops_var, ops_const = iter(arrays), self._get_evaluated_constants(backend)
            ops: Sequence[Any] = [next(ops_var) if op is None else op for op in ops_const]
        else:
            ops = arrays

        try:
            # check if the backend requires special preparation / calling
            #   but also ignore non-numpy arrays -> assume user wants same type back
            if backends.has_backend(backend) and all(
                backends.infer_backend(x) == "numpy" for x in arrays
            ):
                return self._contract_with_conversion(
                    ops, out, backend, evaluate_constants=evaluate_constants
                )

            return self._contract(ops, out=out, backend=backend, evaluate_constants=evaluate_constants)

        except ValueError as err:
            original_msg = str(err.args) if err.args else ""
            msg = (
                "Internal error while evaluating `ContractExpression`. Note that few checks are performed"
                " - the number and rank of the array arguments must match the original expression. "
                f"The internal error was: '{original_msg}'",
            )
            err.args = msg
            raise

    def __repr__(self) -> str:
        if self._constants_dict:
            constants_repr = f", constants={sorted(self._constants_dict)}"
        else:
            constants_repr = ""
        return f"<ContractExpression('{self.contraction}'{constants_repr})>"

    def __str__(self) -> str:
        s = [self.__repr__()]
        for i, c in enumerate(self.contraction_list):
            s.append(f"\n  {i + 1}.  ")
            s.append(f"'{c[2]}'" + (f" [{c[-1]}]" if c[-1] else ""))
            s.append(f"\neinsum_kwargs={self.kwargs}")
        return "".join(s)


def contract_expression(
    subscripts: Any,
    *shapes: Any,
    constants: Collection[int] | None = None,
    use_blas: bool = True,
    optimize: Any = "auto",
    memory_limit: Any = None,
    **kwargs: Any,
) -> ContractExpression:
    """Generate a reusable expression for a contraction with fixed shapes.

    Parameters:
        subscripts: Specifies the subscripts for summation.
        *shapes: Shapes (or, for constants, the arrays themselves) to optimize for.
        constants: Indices of ``shapes`` holding constant arguments; the
            contraction between them is computed once per backend and reused.
        use_blas: Whether pairwise contractions may be dispatched to BLAS.
        optimize: Path optimizer to use, as in :func:`contract_path`.
        memory_limit: Upper bound on the number of elements of any intermediate.

    Returns:
        A :class:`ContractExpression` callable as ``expr(*arrays, out=None, backend="auto")``.
    """
    if not optimize:
        raise ValueError("Can only generate expressions for optimized contractions.")

    for arg in ("out", "backend"):
        if kwargs.get(arg, None) is not None:
            raise ValueError(
                f"'{arg}' should only be specified when calling a `ContractExpression`, not when building it."
            )

    if not isinstance(subscripts, str):
        subscripts, shapes = _convert_interleaved_input((subscripts,) + shapes)

    constants = constants or ()
    constants_dict = {i: shapes[i] for i in constants}

    # apart from constant arguments, make dummy arrays
    dummy_arrays = [s if i in constants else shape_only(s) for i, s in enumerate(shapes)]

    _, path_info = contract_path(
        subscripts,
        *dummy_arrays,
        use_blas=use_blas,
        optimize=optimize,
        memory_limit=memory_limit,
    )

    return ContractExpression(subscripts, path_info.contraction_list, constants_dict, **kwargs)
