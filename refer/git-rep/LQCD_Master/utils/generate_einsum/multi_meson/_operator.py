"""_operator (internal) — Multi-meson operator construction.

Builds product operators like (π·π)(x) or (D·D·D_s)(x)
by multiplying individual meson_operator() results.

Uses the existing Operator.__mul__ — no changes to hadron_operator.py needed.
"""

from pathlib import Path

try:
    from ..hadron_operator import meson_operator, Operator
except ImportError:
    import sys
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from hadron_operator import meson_operator, Operator


def multi_meson_operator(*meson_specs) -> Operator:
    """Build a product of meson operators at the same spacetime point.

    Parameters
    ----------
    *meson_specs : tuples of (anti_flavor, quark_flavor, gamma)
        Each tuple defines one meson via meson_operator().
        Order matters for Wick contraction labeling.

    Returns
    -------
    Operator
        Product Operator with all tensors from all mesons.

    Examples
    --------
    >>> O = multi_meson_operator(
    ...     ("c", "u", "g5"),   # D⁰ = c̄·γ₅·u
    ...     ("c", "d", "g5"),   # D⁰ = c̄·γ₅·d
    ...     ("c", "s", "g5"),   # D_s⁺ = c̄·γ₅·s
    ... )
    """
    if not meson_specs:
        raise ValueError("At least one meson required")

    result_op = meson_operator(*meson_specs[0])[0]
    for spec in meson_specs[1:]:
        next_op = meson_operator(*spec)[0]
        result_op = result_op * next_op

    return result_op
