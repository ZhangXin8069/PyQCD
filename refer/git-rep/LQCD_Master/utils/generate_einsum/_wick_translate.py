"""_wick_translate — Shared wicklib ↔ Tensor ↔ PyQUDA naming conversion layer.

Provides:
  _to_wicklib()      — Tensor[] → wicklib Operator (Block construction)
  _rename_op()       — wicklib operand string → PyQUDA variable name
  _simplify_gammas() — gamma product simplification (gamma² = I)

Owned by generate_einsum/; imported by two_pt.contract, two_pt.codegen and three_pt.
"""

from pathlib import Path

try:
    from .hadron_operator import Tensor, Operator
    from .wicklib.quark import QuarkField
    from .wicklib.gamma import Gamma, GAMMA_0, GAMMA_5, C
    from .wicklib.operator import Quark, Diquark, AntiQuark, QuarkBilinear
    from .wicklib.index import ColorIndex, SpinIndex
except ImportError:
    import sys
    _p = str(Path(__file__).resolve().parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from hadron_operator import Tensor, Operator
    from wicklib.quark import QuarkField
    from wicklib.gamma import Gamma, GAMMA_0, GAMMA_5, C
    from wicklib.operator import Quark, Diquark, AntiQuark, QuarkBilinear
    from wicklib.index import ColorIndex, SpinIndex


# ======================================================================
# Gamma lookup (Tensor gamma name → wicklib Gamma object)
# ======================================================================

_GAMMA_MAP = {
    "g1": Gamma(1),
    "g2": Gamma(2),
    "g3": Gamma(4),
    "g4": Gamma(8),
    "g5": GAMMA_5,
    "gtg5": Gamma(7),
    "Cg5": C @ GAMMA_5,
    "Cg1": C @ Gamma(1),
}


# ======================================================================
# Tensor[] → wicklib Operator (explicit Block construction) 
#                    meson and baryon operator
# ======================================================================
# 
def _to_wicklib(op: Operator, location: str = "x"):
    """Convert Tensor[] → wicklib Operator using explicit Block construction."""
    has_eps = any(t.type == "epsilon" for t in op.tensors)

    if has_eps:
        # Baryon B = epsilon(q_a^T . Cg5/g1 . q_b) . q_c
        qs = [t for t in op.tensors if t.type == "quark"]
        gs = [t for t in op.tensors if t.type == "gamma"]
        dg = _GAMMA_MAP.get(gs[0].name, GAMMA_5) if gs else GAMMA_5
        ci = ColorIndex.new()
        si = SpinIndex.new()
        diq = Diquark(QuarkField(qs[0].flavor), QuarkField(qs[1].flavor), dg)
        spec = Quark(QuarkField(qs[2].flavor), Gamma(0))
        return diq.at(location, ci) * spec.at(location, si, ci), si

    # M = q_bar(barred) . Gamma . f(unbarred)
    anti = [t for t in op.tensors if t.type == "antiquark"]
    quark = [t for t in op.tensors if t.type == "quark"]
    gs = [t for t in op.tensors if t.type == "gamma"]
    gamma = _GAMMA_MAP.get(gs[0].name, GAMMA_5) if gs else GAMMA_5
    return QuarkBilinear(
        barred=QuarkField(anti[0].flavor),
        unbarred=QuarkField(quark[0].flavor),
        gamma=gamma,
    ).at(location), None


# ======================================================================
# wicklib output → name substitution tables
# ======================================================================

_NAME_GAMMA = {
    0: "I4", 1: "g1", 2: "g2", 4: "g3", 8: "g4",
    15: "G5", 10: "Cmat", 5: "Cg5", 7: "gtg5", 11: "Cg1",
}
_FLAV_PROP = {
    "u": "prop_l", "d": "prop_l", "s": "prop_s",
    "c": "prop_c", "b": "prop_b",
}


def _rename_op(op_str: str) -> str:
    """Convert wicklib operand strings → our naming convention.

    In propag_{flav}_{sink}_{source}:
      sink = quark position, source = antiquark position
      so sink→source = forward, source→sink = backward.

    Backward propagator: S_bwd = S_fwd.conj() (element-wise).
    The gamma5-hermiticity wrapping (G5 @ ... @ G5) is absorbed into
    the gamma-insertion operands in the output format.
    """
    if op_str == "epsilon":
        return "epsilon"
    if op_str == "projector":
        return "Tmat"
    if op_str.startswith("gamma("):
        idx = int(op_str.replace("gamma(", "").replace(")", ""))
        return _NAME_GAMMA.get(idx, f"gamma{idx}")
    if op_str.startswith("propag_"):
        raw = op_str.replace("propag_", "")
        parts = raw.split("_")
        flav = parts[0] if parts else "l"
        snk = parts[1] if len(parts) > 1 else "x"  # quark position
        src = parts[2] if len(parts) > 2 else "y"  # antiquark position
        pn = _FLAV_PROP.get(flav, f"prop_{flav}")
        if snk == src:
            return pn  # same-point
        # backward: S_bwd = G5 @ S.dag() @ G5 (gamma5-hermiticity, dag = conj+transpose)
        return f"G5 @ {pn}.dag() @ G5" if snk > src else pn
    return op_str


def _simplify_gammas(expr: str) -> str:
    """Simplify gamma products: cancel adjacent identical gammas (gamma² = I).

    Example: "G5 @ g1 @ g1 @ G5" → "I4"
    """
    _GAMMA_NAMES = frozenset(("G5", "g1", "g2", "g3", "g4", "gtg5", "Cg5", "I4"))
    tokens = expr.split(" @ ")
    changed = True
    while changed:
        changed = False
        i = 0
        new_tokens = []
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] in _GAMMA_NAMES
                and tokens[i] == tokens[i + 1]
            ):
                i += 2  # skip both: gamma·gamma = I
                changed = True
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    if not tokens:
        return "I4"
    return " @ ".join(tokens)
