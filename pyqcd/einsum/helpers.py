"""Index bookkeeping and cost helpers for contraction planning."""

from collections.abc import Collection, Iterable, Sequence

__all__ = ["compute_size_by_dict", "find_contraction", "flop_count"]


def compute_size_by_dict(indices: Iterable, size_dict: dict) -> int:
    """Product of the sizes of ``indices`` looked up in ``size_dict``."""
    ret = 1
    for i in indices:
        ret *= size_dict[i]
    return ret


def find_contraction(
    positions: Collection[int],
    input_sets: Sequence,
    output_set,
) -> tuple[frozenset, list, frozenset]:
    """Contract ``positions`` of ``input_sets`` against the overall ``output_set``.

    Returns ``(new_result, remaining, idx_removed)``; the contracted index set is
    recoverable as ``new_result | idx_removed``.
    """
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idx_contract = frozenset.union(*inputs)
    idx_remain = output_set.union(*remaining)

    new_result = idx_remain & idx_contract
    idx_removed = idx_contract - new_result
    remaining.append(new_result)

    return new_result, remaining, idx_removed


def flop_count(idx_contraction: Collection, inner: bool, num_terms: int, size_dict: dict) -> int:
    """Number of scalar multiplications of a contraction over ``idx_contraction``."""
    overall_size = compute_size_by_dict(idx_contraction, size_dict)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor
