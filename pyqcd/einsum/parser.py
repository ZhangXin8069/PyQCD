"""NumPy compatible parsing of einsum subscripts, without any array dependency."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

__all__ = [
    "alpha_canonicalize",
    "find_output_shape",
    "find_output_str",
    "get_shape",
    "get_symbol",
    "parse_einsum_input",
]

_einsum_symbols_base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def is_valid_einsum_char(x: str) -> bool:
    """Whether ``x`` is a plain einsum character."""
    return (x in _einsum_symbols_base) or (x in ",->.")


def has_valid_einsum_chars_only(einsum_str: str) -> bool:
    """Whether every character of ``einsum_str`` is a plain einsum character."""
    return all(map(is_valid_einsum_char, einsum_str))


def get_symbol(i: int) -> str:
    """Symbol for integer ``i``: the 52 letters first, then unicode skipping surrogates."""
    if i < len(_einsum_symbols_base):
        return _einsum_symbols_base[i]
    codepoint = i + 140
    if codepoint >= 0xD800:
        codepoint += 0x800
    return chr(codepoint)


def gen_unused_symbols(used: str, n: int) -> Iterator[str]:
    """Yield ``n`` symbols not contained in ``used``."""
    i = cnt = 0
    while cnt < n:
        s = get_symbol(i)
        i += 1
        if s in used:
            continue
        yield s
        cnt += 1


def convert_to_valid_einsum_chars(einsum_str: str) -> str:
    """Rewrite ``einsum_str`` onto the plain einsum alphabet."""
    symbols = sorted(set(einsum_str) - set(",->"))
    replacer = {x: get_symbol(i) for i, x in enumerate(symbols)}
    return "".join(replacer.get(x, x) for x in einsum_str)


def alpha_canonicalize(equation: str) -> str:
    """Rename the indices of ``equation`` in order of first appearance."""
    rename: dict[str, str] = {}
    for name in equation:
        if name in ".,->":
            continue
        if name not in rename:
            rename[name] = get_symbol(len(rename))
    return "".join(rename.get(x, x) for x in equation)


def find_output_str(subscripts: str) -> str:
    """Implicit output string: indices appearing exactly once, sorted."""
    tmp_subscripts = subscripts.replace(",", "")
    return "".join(s for s in sorted(set(tmp_subscripts)) if tmp_subscripts.count(s) == 1)


def find_output_shape(inputs: Sequence[str], shapes: Sequence[Sequence[int]], output: str) -> tuple:
    """Output shape for ``inputs``/``shapes``, broadcasting size-1 dimensions away."""
    output_shape = []
    for c in output:
        dimensions = [shape[term.find(c)] for term, shape in zip(inputs, shapes) if c in term]
        output_shape.append(next((dimension for dimension in dimensions if dimension != 1), 1))
    return tuple(output_shape)


_BaseTypes = (bool, int, float, complex, str, bytes)


def get_shape(x: Any) -> tuple:
    """Shape of an array-like object, a base scalar or a (nested) sequence."""
    if hasattr(x, "shape"):
        return x.shape
    elif isinstance(x, _BaseTypes):
        return ()
    elif isinstance(x, Sequence):
        shape = []
        while isinstance(x, Sequence) and not isinstance(x, _BaseTypes):
            shape.append(len(x))
            x = x[0]
        return tuple(shape)
    else:
        raise ValueError(f"Cannot determine the shape of {x}, can only determine the shape of array-like objects.")


def convert_subscripts(old_sub: Sequence[Any], symbol_map: dict) -> str:
    """Translate an interleaved subscript list using ``symbol_map``."""
    new_sub = ""
    for s in old_sub:
        if s is Ellipsis:
            new_sub += "..."
        else:
            new_sub += symbol_map[s]
    return new_sub


def convert_interleaved_input(operands: Sequence[Any]) -> tuple[str, tuple]:
    """Convert ``(op, sub, op, sub, ..., [out])`` input to a plain subscript string."""
    tmp_operands = list(operands)
    operand_list = []
    subscript_list = []
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
        )

    subscripts = ",".join(convert_subscripts(sub, symbol_map) for sub in subscript_list)
    if output_list is not None:
        subscripts += "->"
        subscripts += convert_subscripts(output_list, symbol_map)

    return subscripts, tuple(operand_list)


def parse_einsum_input(operands: Any, shapes: bool = False) -> tuple[str, str, list]:
    """Parse einsum operands into ``(input_subscripts, output_subscript, operands)``.

    With ``shapes=True`` the operands are shapes rather than arrays, and broadcasting
    and ellipsis widths are inferred from them.
    """
    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        if shapes:
            if any(hasattr(o, "shape") for o in operands[1:]):
                raise ValueError(
                    "shapes is set to True but given at least one operand looks like an array"
                    " (at least one operand has a shape attribute). "
                )
        operands = operands[1:]
    else:
        subscripts, operands = convert_interleaved_input(operands)

    if shapes:
        operand_shapes = operands
    else:
        operand_shapes = [get_shape(o) for o in operands]

    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        ellipse_inds = "".join(gen_unused_symbols(used, max(len(x) for x in operand_shapes)))
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                if operand_shapes[num] == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(len(operand_shapes[num]), 1) - (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace("...", "")
                else:
                    split_subscripts[num] = sub.replace("...", ellipse_inds[-ellipse_count:])

        subscripts = ",".join(split_subscripts)

        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            output_subscript = find_output_str(subscripts)
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts, output_subscript = subscripts, find_output_str(subscripts)

    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError(f"Output character '{char}' appeared more than once in the output.")
        if char not in input_subscripts:
            raise ValueError(f"Output character '{char}' did not appear in the input")

    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError(
            f"Number of einsum subscripts, {len(input_subscripts.split(','))}, must be equal to the "
            f"number of operands, {len(operands)}."
        )

    return input_subscripts, output_subscript, operands
