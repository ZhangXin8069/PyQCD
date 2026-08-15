"""codegen — PyQUDA sequential source and contraction code for 3pt functions.

Meson and baryon 3pt codegen are separated into dedicated functions.
Meson 3pt codegen takes operators directly (no intermediate contraction step).
"""

from pathlib import Path
from typing import List

try:
    from .._wick_translate import _rename_op
except ImportError:
    import sys
    _pp = str(Path(__file__).resolve().parent.parent)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)
    from _wick_translate import _rename_op


# ======================================================================
# Shared helpers
# ======================================================================

_PYQUDA_GAMMA_IDX = {
    "g1": "1", "g2": "2", "g3": "4", "g4": "8",
    "g5": "15", "G5": "15", "gtg5": "7", "I4": "0", "I": "0",
}


def _gamma_g_call(gamma_name: str) -> str:
    name = gamma_name.strip().lower()
    if name.startswith("gamma"):
        idx = name[5:]
        name = f"g{idx}" if idx.isdigit() else name
    return f"gamma.gamma({_PYQUDA_GAMMA_IDX.get(name, '0')})"


# ═══════════════════════════════════════════════════════════════════════
# Meson 3pt helpers
# ═══════════════════════════════════════════════════════════════════════



_LIGHT = frozenset({"u", "d"})


def _extract_flavors(op) -> tuple:
    """Extract (antiquark_flavor, quark_flavor, gamma_name) from an operator."""
    tensors = op[0].tensors if isinstance(op, tuple) else op.tensors
    return (
        next(t.flavor for t in tensors if t.type == "antiquark"),
        next(t.flavor for t in tensors if t.type == "quark"),
        next(t.name for t in tensors if t.type == "gamma"),
    )


def _var(flavor: str) -> str:
    return "prop_l" if flavor in _LIGHT else f"prop_{flavor}"


# ======================================================================
# MESON 3pt — unified codegen (one function, no intermediate contraction)
# ======================================================================

def gen_meson_3pt_code(
    src,
    snk,
    current,
    src_name: str = "source",
    snk_name: str = "sink",
    out: str = "out_path",
) -> str:
    """Generate the complete PyQUDA code for a meson 3pt measurement.

    Unified entry point — covers sink block, sequential source setup,
    inversion placeholder, final contraction, MPI gather, and save.
    """
    src_anti, src_quark, src_gamma = _extract_flavors(src)
    snk_anti, snk_quark, snk_gamma = _extract_flavors(snk)
    cur_anti, cur_quark, cur_gamma = _extract_flavors(current)

    assert src_anti == snk_anti, (
        f"Flavor mismatch: src anti {src_anti} vs snk anti {snk_anti}")
    assert src_quark == cur_quark, (
        f"Flavor mismatch: src quark {src_quark} vs cur quark {cur_quark}")
    assert snk_quark == cur_anti, (
        f"Flavor mismatch: snk quark {snk_quark} vs cur anti {cur_anti}")

    spectator_flavor = src_anti
    forward_flavor = cur_quark
    seq_flavor = cur_anti
    prop_spec = _var(spectator_flavor)
    prop_fwd = _var(forward_flavor)
   
    lines = [
        "# ═══════════════════════════════════════════════════════",
        f"# Meson 3pt: {src_name} -> {snk_name}",
        f"#   spectator  = {spectator_flavor}  ({prop_spec})",
        f"#   forward    = {forward_flavor}  ({prop_fwd})",
        f"#   sequential = {seq_flavor} (prop_seq)", 
        f"#   sink Gamma = {snk_gamma}",
        f"#   src Gamma  = {src_gamma}",
        f"#   cur Gamma  = {cur_gamma}",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# ------------------------------------------------------------------",
        "#  Gamma matrices",
        "# ------------------------------------------------------------------",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_snk = cp.asarray({_gamma_g_call(snk_gamma)},",
        " dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray({_gamma_g_call(cur_gamma)},",
        " dtype=cp.complex128)",
        f"Gamma_src = cp.asarray({_gamma_g_call(src_gamma)},",
        " dtype=cp.complex128)",
        "",
        "# ------------------------------------------------------------------",
        "#  G5-conjugate gammas:  Γ̄ = γ₅ . Γ . γ₅  (self-adjoint → Γ†=Γ)",
        "#  These appear in the sink block; see PDF Eq 6.",
        "# ------------------------------------------------------------------",
        "Gamma_snk_bar = G5 @ Gamma_snk @ G5",
        "Gamma_src_bar = G5 @ Gamma_src @ G5",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 1 -- Sink block  (PDF Eq 6)",
        "#  B(x) = Γ̄_snk . S_spectator . Γ̄_src",
        "#  No G5-dagger or conj needed — this is a pure γ₅ sandwich.",
        "# ------------------------------------------------------------------",
        f"# spectator = {spectator_flavor}  ({prop_spec})",
        "B = core.LatticePropagator(latt_info)",
        "B.data = contract(",
        "    'αj, wtzyxjiba, iβ -> wtzyxαβba',",
        f"    Gamma_snk_bar, {prop_spec}.data, Gamma_src_bar)",
        "",
        "# ------------------------------------------------------------------",
        "#  Step 2 -- Sequential source + solve",
        "# ------------------------------------------------------------------",
        "src_seq = source.sequential12(B, t_sink)",
        "",
        "# Sequential inversion: insert your CG / BiCGstab solver here",
        "# After inversion, prop_seq = G_seq(x,0). THe following is an example:",
        "dirac_{seq_flavor}.loadGauge(gauge_stout)",
        "prop_seq = core.invertPropagator(dirac_{seq_flavor}, src_seq) ",
        "# ------------------------------------------------------------------",
        "#  Step 3 -- Final contraction  (PDF Eq 7)",
        "#  C_3pt(t) = Tr[ γ₅·G†_seq·γ₅ · Γ_cur · S_fwd ]",
        "#  The outer G5-dagger (γ₅·G†_seq·γ₅) gives back G(0,x).",
        "# ------------------------------------------------------------------",
        "",
        "# 3a. Outer G5-dagger: tmp = γ₅ · S_seq† · γ₅ = G(0,x)",
        "tmp_prop = core.LatticePropagator(latt_info)",
        "tmp_prop.data = contract(",
        "    'AB, wtzyxCBji, CD -> wtzyxADij',",
        "    G5, prop_seq.data.conj(), G5)",
        "",
        "# 3b. Trace contraction: Tr[ G(0,x) · Γ_cur · S_fwd ]",
        f"# forward = {forward_flavor}  ({prop_fwd})",
        "three_pt_site = contract(",
        "    'wtzyxijba, jk, wtzyxkiab -> wtzyx',",
        f"    tmp_prop.data, Gamma_cur, {prop_fwd}.data)",
        "",
        "# 3c. Trace spatial volume -> time-slice correlator",
        "three_pt_local = contract('wtzyx -> t', three_pt_site)",
        "",
        "# 3d. MPI gather",
        "C3_t = core.gatherLattice(",
        "    cp.array(three_pt_local), [0, -1, -1, -1])",
        "",
        "# 3e. Save",
        "if core.getMPIRank() == 0:",
        f"    np.save({out}, cp.asnumpy(C3_t))",
    ]
    return "\n".join(lines)


# ======================================================================
# BARYON 3pt
# ======================================================================

# Structural operand name -> einsum index set lookup
_STRUCTURAL_MAP = {
    frozenset("ijk"): "epsilon",
    frozenset("lmn"): "epsilon",
    frozenset("AB"): "Cg5",
    frozenset("GH"): "Cg5",
    frozenset("ID"): "projector",
}


def _derive_structural_operand(indices: str) -> str:
    idx_set = frozenset(indices)
    if idx_set in _STRUCTURAL_MAP:
        return _STRUCTURAL_MAP[idx_set]
    return indices


def _get_baryon_propagators(sb: dict) -> List[str]:
    props = []
    if "var_a" in sb:
        props.append(sb["var_a"])
    if "var_b" in sb:
        props.append(sb["var_b"])
    for k, v in sb.items():
        if k.startswith("var_") and k not in ("var_a", "var_b"):
            if isinstance(v, str) and v not in props:
                props.append(v)
    return props


def gen_seq_source_baryon(
    src_name: str,
    snk_name: str,
    result: dict,
    phase: str = "phase_sink",
) -> str:
    """Sequential source code for a baryon 3pt.

    Sink block: B = Sum sign . contract(wtzxy, ijk, lmn, AB, GH, ID, prop1, prop2, ...)
    Summed over all Wick topologies.
    Then: G5-dagger -> sequential12 -> invert.
    """
    cg = result["current_gamma"]
    terms = result["sink_terms"]

    lines = [
        "# ═══════════════════════════════════════════════════════",
        f"# Sequential source: {src_name} -> {snk_name} (baryon 3pt)",
        "# ═══════════════════════════════════════════════════════",
        "",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_snk = cp.asarray({_gamma_g_call(cg)}, dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray({_gamma_g_call(cg)}, dtype=cp.complex128)",
        "",
        "# Epsilon tensor (3D color anti-symmetric)",
        "epsilon = cp.zeros((3, 3, 3), dtype=cp.float64)",
        "epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1.0",
        "epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1.0",
        "",
        "# Gamma matrices for baryon spin/pariy structure",
        "Cmat = cp.asarray("
        "gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)",
        "Cg5 = Cmat @ G5",
        "Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)"
        "  # P_plus",
        "",
    ]

    lines.append(f"# {len(terms)} topology(ies)")
    for i, sb in enumerate(terms):
        lines.append(
            f"#   topo {i}: sign={sb['sign']:+d}"
            f"  {sb.get('description', '')}"
        )
    lines.append("")

    lines.append("# Sink block = sum over all Wick topologies")
    lines.append("B = core.LatticePropagator(latt_info)")
    lines.append("B.data = (")

    indent = "    "
    for i, sb in enumerate(terms):
        sgn = "+" if sb["sign"] >= 0 else "-"
        desc = sb.get("description", "")
        einsum = sb["einsum"]
        props = _get_baryon_propagators(sb)

        ops = ["phase_sink", "epsilon", "epsilon", "Cg5", "Cg5", "Tmat"]
        ops.extend(f"{pv}.data" for pv in props)

        lines.append(f"{indent}{sgn} contract('{einsum}',")
        for j in range(0, len(ops), 3):
            chunk = ops[j:j + 3]
            chunk_str = ", ".join(chunk) + ","
            lines.append(f"{indent*2}{chunk_str}")
        lines.append(f"{indent}),  # {desc}")

    lines.append(")")
    lines.append("")
    lines.append("# G5-dagger for sequential source")
    lines.append("B.data = contract(")
    lines.append("    'AB, wtzyxCBji, CD -> wtzyxADij',")
    lines.append("    G5, B.data.conj(), G5)")
    lines.append("")
    lines.append("# Sequential source at t_sink")
    lines.append("src_seq = source.sequential12(B, t_sink)")
    lines.append("")
    lines.append("# Sequential solve (insert your inversion code here)")

    return "\n".join(lines)


def gen_final_contract_baryon(result: dict, out: str = "out_path") -> str:
    """Final contraction for a baryon 3pt.

    3-operand einsum: Tr[ G~_seq @ Gamma_cur @ S_fwd ]
    """
    fwd = result["fwd_var"]

    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# Final contraction: Tr[ G~_seq @ Gamma_cur @ S_fwd ]",
        f"# Forward: {result['fwd_flavor']} ({fwd})",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# G5-dagger on sequential propagator",
        "tmp_prop = core.LatticePropagator(latt_info)",
        "tmp_prop.data = contract(",
        "    'AB, wtzyxCBji, CD -> wtzyxADij',",
        "    G5, prop_seq.data.conj(), G5)",
        "",
        "three_pt_site = contract(",
        "    'wtzyxijba, jk, wtzyxkiab -> wtzyx',",
        f"    tmp_prop.data, Gamma_cur, {fwd}.data)",
        "",
        "three_pt_local = contract('wtzyx -> t', three_pt_site)",
        "C3_t = core.gatherLattice(",
        "    cp.array(three_pt_local), [0, -1, -1, -1])",
        "",
        "if core.getMPIRank() == 0:",
        f"    np.save({out}, cp.asnumpy(C3_t))",
    ]
    return "\n".join(lines)
