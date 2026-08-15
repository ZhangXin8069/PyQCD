"""codegen — PyQUDA einsum format for multi-meson 2-point functions.

Converts Wick contraction output into PyQUDA contract() calls.
Handles arbitrary numbers of propagators and gamma matrices.

Key differences from single-meson codegen:
  - No special meson 2pt optimization (gamma simplification)
  - Generic operant → contract() formatting
  - dag→conj conversion for backward propagators
"""

from pathlib import Path
from typing import List

try:
    from .._wick_translate import _simplify_gammas, _rename_op
except ImportError:
    import sys
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from _wick_translate import _simplify_gammas, _rename_op

# Gamma names recognized in PyQUDA
_GAMMAS = frozenset(("G5", "g1", "g2", "g3", "g4", "gtg5", "Cg5", "I4"))


def _is_gamma(op: str) -> bool:
    return op in _GAMMAS


def _dag_to_conj(op_str: str) -> str:
    """Convert 'G5 @ prop_l.dag() @ G5' → 'prop_l.conj()'.

    This handles the backward-propagator convention, converting
    wicklib's γ₅-hermiticity form to PyQUDA's conj() form.
    If the string is not a backward propagator, return unchanged.
    """
    if op_str.startswith("G5 @ ") and " @ G5" in op_str:
        inner = op_str[5:-5].strip()
        if inner and inner.endswith(".dag()"):
            return f"{inner[:-6]}.conj()"
    return op_str


def pyquda_format_contract(term) -> str:
    """Format a contraction Term → contract() string.

    Generic formatter: adds wtzyx prefix to propagators,
    leaves gamma and epsilon operands as-is, and handles
    the dag→conj conversion for backward propagators.
    """
    if not term.einsum_subs:
        return ""

    parts = [p.strip() for p in term.einsum_subs.split(",") if p.strip()]
    ops = list(term.operands)

    new_subs = []
    new_ops = []
    for part, op in zip(parts, ops):
        if _is_gamma(op) or "epsilon" in op:
            new_subs.append(part)
        elif op.startswith("G5 @ ") and " @ G5" in op:
            # Backward propagator: dag → conj, add wtzyx
            new_subs.append(f"wtzyx{part}")
            new_ops.append(_dag_to_conj(op))
        else:
            # Forward or same-point propagator
            new_subs.append(f"wtzyx{part}")
            new_ops.append(op)

    subs_final = ", ".join(new_subs) + " -> t"
    args = ", ".join(new_ops)

    c = getattr(term, "coefficient", 1)
    if c == 0:
        return ""
    if c == 1:
        return f"contract('{subs_final}', {args})"
    if c == -1:
        return f"-contract('{subs_final}', {args})"
    return f"{c} * contract('{subs_final}', {args})"
