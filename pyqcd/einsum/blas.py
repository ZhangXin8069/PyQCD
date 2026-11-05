"""Decide whether a pairwise contraction maps onto a BLAS call."""

from collections.abc import Sequence

__all__ = ["can_blas"]


def can_blas(
    inputs: list[str],
    result: str,
    idx_removed,
    shapes: Sequence[Sequence[int]] | None = None,
) -> str | bool:
    """BLAS flavour usable for ``inputs -> result``, or ``False`` if none.

    Values are ``"DOT"``, ``"DOT/EINSUM"``, ``"GEMM"``, ``"GEMV/EINSUM"``,
    ``"OUTER/EINSUM"``, ``"TDOT"``; the ``/EINSUM`` suffix marks cases where a
    plain einsum is preferable but a tensordot based fallback is still valid.
    ``shapes`` additionally rejects broadcast contraction dimensions.
    """
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        if nl + nr - 1 == int(c in result):
            return False

    if shapes is not None:
        for c in idx_removed:
            if shapes[0][input_left.find(c)] != shapes[1][input_right.find(c)]:
                return False

    if len(idx_removed) == 0:
        return "OUTER/EINSUM"

    sets = [set(x) for x in inputs]
    keep_left = sets[0] - idx_removed
    keep_right = sets[1] - idx_removed
    rs = len(idx_removed)

    if inputs[0] == inputs[1]:
        return "DOT"

    elif sets[0] == sets[1]:
        return "DOT/EINSUM"

    if input_left[-rs:] == input_right[:rs]:
        return "GEMM"

    elif input_left[:rs] == input_right[-rs:]:
        return "GEMM"

    elif input_left[-rs:] == input_right[-rs:]:
        return "GEMM"

    elif input_left[:rs] == input_right[:rs]:
        return "GEMM"

    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        return "GEMV/EINSUM"

    else:
        return "TDOT"
