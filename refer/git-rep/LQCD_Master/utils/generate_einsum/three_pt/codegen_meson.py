"""codegen_meson_backup — Meson 3pt PyQUDA code generation.

Baryon-style interface + meson-style code output.

Interface (matches codegen_baryon.gen_code):
  gen_pyquda_code(sink, source, current, projector,
           t_sep,)

Input:
  sink, source, current: tuples (Operator, is_baryon) from hadron_operator
    e.g. (meson_operator("u","s","g5"), False)

Output:
  Complete PyQUDA main.py string for meson 3pt.

Validation:
  - Builds wicklib Correlator and checks it produces >=1 term.
  - Extracts all flavors and gamma names from operator tensors directly.
"""

import sys
from pathlib import Path

try:
    from .._wick_translate import _to_wicklib
    from ..wicklib.correlator import Correlator
    from ..hadron_operator import meson_operator, current_operator
except ImportError:
    _parent = str(Path(__file__).resolve().parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from _wick_translate import _to_wicklib
    from wicklib.correlator import Correlator
    from hadron_operator import meson_operator, current_operator

_LIGHT = frozenset({"u", "d"})

# Gamma tensor name → PyQUDA gamma() call
_GAMMA_TENSOR_TO_CALL = {
    "G5": "gamma.gamma(15)",
    "g1": "gamma.gamma(1)",
    "gx": "gamma.gamma(1)",
    "gamma_x": "gamma.gamma(1)",    
    "g2": "gamma.gamma(2)",
    "g3": "gamma.gamma(4)",
    "g4": "gamma.gamma(8)",
    "gtg5": "gamma.gamma(7)",
    "I4": "gamma.gamma(0)",
}

# Gamma tensor name → short display name
_GAMMA_DISPLAY = {
    "G5": "G5",
    "g1": "g1",
    "g2": "g2",
    "g3": "g3",
    "g4": "g4",
    "gx": "g1",
    "gy": "g2",
    "gz": "g3",
    "gt": "g4",
    "gtg5": "gtg5",
    "I4": "I4",
}


def var(f):
    return "prop_l" if f in _LIGHT else f"prop_{f}"


def _gamma_call(name):
    """Gamma tensor name → PyQUDA gamma() call string."""
    if name in _GAMMA_TENSOR_TO_CALL:
        return _GAMMA_TENSOR_TO_CALL[name]
    # Composite gamma: parse digits, e.g. g1g5 → gamma.gamma(1) @ gamma.gamma(15)
    parts = []
    i = 0
    while i < len(name):
        if name[i] == "g" and i + 1 < len(name) and name[i + 1].isdigit():
            parts.append(f"gamma.gamma({name[i+1]})")
            i += 2
        elif name[i] == "G" and i + 1 < len(name) and name[i + 1] == "5":
            parts.append("gamma.gamma(15)")
            i += 2
        else:
            i += 1
    if len(parts) >= 1:
        return " @ ".join(parts)
    return name


def _gamma_display(name):
    return _GAMMA_DISPLAY.get(name, name)


def _get_tensors(op):
    """Extract tensor list from an operator (handles tuple wrapper)."""
    obj = op[0] if isinstance(op, tuple) else op
    return list(obj.tensors)


def gen_meson_3pt_code(
    snk_op,
    src_op,
    cur_op,
    t_sep="t_sep"
):
    # ═══════════════════════════════════════════════════════════════════
    #  0. Validate via wicklib Correlator (check non-empty)
    # ═══════════════════════════════════════════════════════════════════
    #_op = snk_op[0] if isinstance(snk_op, tuple) else snk_op
    #w_snk, _ = _to_wicklib(_op, "x")
    #_op2 = src_op[0] if isinstance(src_op, tuple) else src_op
    #w_src, _ = _to_wicklib(_op2, "y")
    #_op3 = cur_op[0] if isinstance(cur_op, tuple) else cur_op
    #w_cur, _ = _to_wicklib(_op3, "z")
    snk_op = snk_op[0] if isinstance(snk_op, tuple) else snk_op
    src_op = src_op[0] if isinstance(src_op, tuple) else src_op
    cur_op = cur_op[0] if isinstance(cur_op, tuple) else cur_op
    w_snk, _ = _to_wicklib(snk_op, "x")
    w_src, _ = _to_wicklib(src_op, "y")
    w_cur, _ = _to_wicklib(cur_op, "z")

    corr = Correlator(w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)

    if not corr.terms:
        return "/* No valid terms from Wick contraction */"

    # ═══════════════════════════════════════════════════════════════════
    #  1. Extract flavors and gamma names from operator tensors
    # ═══════════════════════════════════════════════════════════════════
    snk_tensors = list(snk_op.tensors)
    src_tensors = list(src_op.tensors)
    cur_tensors = list(cur_op.tensors)
    spectator_flavor = next(t.flavor for t in snk_tensors if t.type == "antiquark")
    forward_flavor   = next(t.flavor for t in cur_tensors if t.type == "quark")
    seq_flavor       = next(t.flavor for t in cur_tensors if t.type == "antiquark")

    snk_gamma_name = next(t.name for t in snk_tensors if t.type == "gamma")
    src_gamma_name = next(t.name for t in src_tensors if t.type == "gamma")
    cur_gamma_name = next(t.name for t in cur_tensors if t.type == "gamma")

    prop_spec = var(spectator_flavor)
    prop_fwd = var(forward_flavor)
#    prop_seq = f"prop_{seq_flavor}" if seq_flavor not in _LIGHT else f"prop_{seq_flavor}"

    # ═══════════════════════════════════════════════════════════════════
    #  2. Generate PyQUDA code
    # ═══════════════════════════════════════════════════════════════════
    lines = [
        "# ═══════════════════════════════════════════════════════",
        f"# Meson 3pt:",
        f"#   spectator  = {spectator_flavor}  ({prop_spec})",
        f"#   forward    = {forward_flavor}  ({prop_fwd})",
        f"#   sequential = {seq_flavor} (prop_seq)",
        f"#   sink Gamma = {_gamma_display(snk_gamma_name)}",
        f"#   src Gamma  = {_gamma_display(src_gamma_name)}",
        f"#   cur Gamma  = {_gamma_display(cur_gamma_name)}",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# ------------------------------------------------------------------",
        "#  Gamma matrices",
        "# ------------------------------------------------------------------",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_snk = cp.asarray({_gamma_call(snk_gamma_name)},",
        " dtype=cp.complex128)",
        f"Gamma_src = cp.asarray({_gamma_call(src_gamma_name)},",
        " dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray({_gamma_call(cur_gamma_name)},",
        " dtype=cp.complex128)",
        "",
        "# ------------------------------------------------------------------",
        "#  G5-conjugate gammas:  Γ̄ = γ₅ . Γ . γ₅",
        "#  These appear in the sink block.",
        "# ------------------------------------------------------------------",
        "Gamma_snk_bar = G5 @ Gamma_snk @ G5",
        "Gamma_src_bar = G5 @ Gamma_src @ G5",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 1 -- Sink block",
        "#  B(x) = Γ̄_snk . S_spectator . Γ̄_src",
        "#  No Wick summation needed: meson 3pt has a single topology.",
        "# ------------------------------------------------------------------",
        f"# spectator = {spectator_flavor}  ({prop_spec})",
        "",
        "B = core.LatticePropagator(latt_info)",
        "B.data = contract(",
        "    'AB, wtzyxBCab, CD -> wtzyxADab',",
        f"    Gamma_snk_bar, {prop_spec}.data, Gamma_src_bar)",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 2 -- Sequential source + solve",
        "# ------------------------------------------------------------------",
        "# Sequential source at t_sink",
        f"src_seq = source.sequential12(B, {t_sep})",
        "",
        "# Sequential inversion",
        f"dirac_{seq_flavor}.loadGauge(gauge_stout)",
        f"prop_seq = core.invertPropagator(dirac_{seq_flavor}, src_seq)",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 3 -- Final contraction",
        "#  C_3pt(t) = Tr[ γ₅·G†_seq·γ₅ · Γ_cur · S_fwd ]",
        "# ------------------------------------------------------------------",
        "# 3a. Outer G5-dagger: tmp = γ₅ · S_seq† · γ₅ = G(0,x)",
        "tmp_prop = core.LatticePropagator(latt_info)",
        "tmp_prop.data = contract(",
        "    'AB, wtzyxCBji, CD -> wtzyxADij',",
        "    G5, prop_seq.data.conj(), G5)",
        "",
        f"# 3b. Trace contraction: Tr[ G(0,x) · Γ_cur · S_fwd ]",
        f"# forward = {forward_flavor}  ({prop_fwd})",
        "three_pt_local = contract(",
        "    'wtzyxijba, jk, wtzyxkiab -> t',",
        f"    tmp_prop.data, Gamma_cur, {prop_fwd}.data)",
        "",
        "# 3c. Sum over spatial sites",
        "",
        "# 3d. MPI gather",
        "C3_t = core.gatherLattice(",
        "     array.arrayAsNumpy(three_pt_local), [0, -1, -1, -1])",
        "",
    ]

    return "\n".join(lines)
