"""codegen — PyQUDA sequential source and contraction code for 3pt functions.

Operands are derived from the einsum string by parsing index groups.
No hardcoded operand layouts — the codegen adapts to any contraction structure.

Reference: two_pt.codegen pattern — coordinate-based determination via _rename_op().
"""

from pathlib import Path
from typing import List

try:
    from .._wick_translate import _rename_op
except ImportError:
    import sys
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from _wick_translate import _rename_op

# Gamma index mapping for gamma.gamma() calls
_PYQUDA_GAMMA_IDX = {
    "g1": "1",
    "g2": "2",
    "g3": "4",
    "g4": "8",
    "g5": "15",
    "G5": "15",
    "gtg5": "7",
    "I4": "0",
    "I": "0",
}


def _gamma_g_call(gamma_name: str) -> str:
    name = gamma_name.strip().lower()
    if name.startswith("gamma"):
        idx = name[5:]
        name = f"g{idx}" if idx.isdigit() else name
    return f"gamma.gamma({_PYQUDA_GAMMA_IDX.get(name, '0')})"


def _parse_einsum(einsum: str) -> dict:
    """Parse an einsum string into index groups with metadata."""
    parts = einsum.split("->")
    output = parts[1].strip()
    raw_groups = [g.strip() for g in parts[0].split(",")]

    groups = []
    n_props = 0
    n_structural = 0
    for g in raw_groups:
        if g == "wtzyx":
            n_structural += 1
            groups.append({"raw": g, "type": "phase", "indices": g})
        elif g.startswith("wtzyx"):
            n_props += 1
            groups.append({"raw": g, "type": "prop", "indices": g[5:]})
        else:
            n_structural += 1
            groups.append({"raw": g, "type": "struct", "indices": g})

    return {
        "groups": groups,
        "output": output,
        "n_props": n_props,
        "n_structural": n_structural,
        "is_baryon": n_structural > 1,
    }


# ======================================================================
# Sequential source code generation
# ======================================================================


def gen_seq_source_code(
    src_name: str,
    snk_name: str,
    result: dict,
    phase: str = "phase_sink",
) -> str:
    """Generate sequential source code block.

    Operand structure is derived by parsing the einsum string,
    then resolved through _rename_op() for consistent naming.
    """
    cg = result["current_gamma"]
    parsed = _parse_einsum(result["sink_terms"][0]["einsum"])
    is_baryon = parsed["is_baryon"]
    has_snk_gamma = "sink_gamma" in result

    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# Sequential source:"
        f" {src_name} -> {snk_name}"
        f" ({'baryon' if is_baryon else 'meson'} 3pt)",
        "# ═══════════════════════════════════════════════════════",
        "",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_snk = cp.asarray({_gamma_g_call(cg)}, dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray({_gamma_g_call(cg)}, dtype=cp.complex128)",
        "",
    ]

    # Override Gamma_snk with the actual sink gamma if available
    if has_snk_gamma and result["sink_gamma"] != cg:
        sg = result["sink_gamma"]
        lines[-3] = (
            f"Gamma_snk = cp.asarray({_gamma_g_call(sg)},"
            " dtype=cp.complex128)"
        )

    if is_baryon:
        lines.append("# Epsilon tensor (GPU: cp.zeros, NOT np.zeros)")
        lines.append("epsilon = cp.zeros((3, 3, 3), dtype=cp.float64)")
        lines.append("epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1.0")
        lines.append(
            "epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1.0"
        )
        lines.append("")
        lines.append("# Gamma matrices (GPU)")
        lines.append(
            "Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8),"
            " dtype=cp.complex128)"
        )
        lines.append("Cg5 = Cmat @ G5")
        lines.append(
            "Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5,"
            " dtype=cp.complex128)  # P_plus"
        )
        lines.append("")

    # Topology summary
    lines.append(f"# {len(result['sink_terms'])} topology(ies)")
    for i, sb in enumerate(result["sink_terms"]):
        lines.append(
            f"#   topo {i}: sign={sb['sign']:+d}"
            f"  {sb.get('description', '')}"
        )
    lines.append("")

    # ── Sink block — build operands from parsed einsum ──
    lines.append("# Sink block B(x) = sum over all Wick topologies")
    if has_snk_gamma:
        lines.append(
            "# B carries the sink operator gamma:"
            " B = S_spectator @ Gamma_snk"
        )
    lines.append("B = core.LatticePropagator(latt_info)")
    lines.append("B.data = (")

    indent = "    "
    for i, sb in enumerate(result["sink_terms"]):
        sgn = "+" if sb["sign"] >= 0 else "-"
        indent2 = indent + "    "
        desc = sb.get("description", "")

        # Parse this term's einsum
        term_parsed = _parse_einsum(sb["einsum"])
        groups = term_parsed["groups"]

        # Build operands from index groups (generic, works for baryon + meson)
        raw_ops = []
        prop_idx = 0
        for g in groups:
            if g["type"] == "phase":
                raw_ops.append(phase)
            elif g["type"] == "prop":
                var_key = f"var_{chr(ord('a') + prop_idx)}"
                var_name = sb.get(var_key, f"prop_{prop_idx}")
                raw_ops.append(f"{var_name}.data")
                prop_idx += 1
            else:
                # Structural operand
                # For meson: use Gamma_snk from result dict
                snk_gamma = sb.get("sink_gamma")
                if snk_gamma:
                    raw_ops.append("Gamma_snk")
                else:
                    # Baryon: epsilon, Cg5, projector via mapping
                    op_name = _derive_structural_operand(g["indices"])
                    raw_ops.append(op_name)

        # Resolve through _rename_op
        ops = [_rename_op(o) for o in raw_ops]

        lines.append(f"{indent}{sgn} contract('{sb['einsum']}',")
        for j in range(0, len(ops), 3):
            chunk = ops[j:j + 3]
            chunk_str = ", ".join(chunk) + ","
            lines.append(f"{indent2}{chunk_str}")
        lines.append(f"{indent}),  # {desc}")

    lines.append(")")
    lines.append("")

    if has_snk_gamma and not is_baryon:
        # ── Meson 3pt (PDF Eq 6): replace sink block with γ₅ sandwich ──
        # The old sink block (B = S_spect · Γ_snk) + G5-dagger is WRONG.
        # Correct: B = Γ̄_snk · S_spect · Γ̄_src, no G5-dagger.
        spectator_var = _rename_op(
            result["sink_terms"][0].get("var_a", "prop_l"))
        lines[-1] = (
            "# Meson: B = Γ̄_snk · S_spectator · Γ̄_src  (no G5-dagger)")
        lines.append("")  # keep blank line before G5-dagger
        src_gamma = result["src_gamma"]
        lines.append(
            f"Gamma_src = cp.asarray("
            f"{_gamma_g_call(src_gamma)}, dtype=cp.complex128)")
        lines.append("")
        lines.append("# Γ̄ = γ₅ · Γ · γ₅  (PDF Eq 6)")
        lines.append("Gamma_snk_bar = G5 @ Gamma_snk @ G5")
        lines.append("Gamma_src_bar = G5 @ Gamma_src @ G5")
        lines.append("")
        lines.append("B = core.LatticePropagator(latt_info)")
        lines.append("B.data = contract(")
        lines.append(
            "    'αj, wtzyxjiba, iβ -> wtzyxαβba',")
        lines.append(
            f"    Gamma_snk_bar, {spectator_var}.data,"
            " Gamma_src_bar)")
        lines.append(")")
        lines.append("")
        # Skip G5-dagger for meson
    else:
        # ── Baryon: keep old G5-dagger ──
        lines.append("# G5-dagger for sequential source")
        lines.append(
            "B.data = contract("
            "'AB, wtzyxCBji, CD -> wtzyxADij',"
            " G5, B.data.conj(), G5)")
        lines.append("")

    # Sequential source
    lines.append("# Sequential source at t_sink")
    lines.append("src_seq = source.sequential12(B, t_sink)")
    lines.append("")
    lines.append("# Sequential solve (insert your inversion code here)")

    return "\n".join(lines)


# ======================================================================
# Final contraction code generation
# ======================================================================


def gen_final_contract_code(result: dict, out: str = "out_path") -> str:
    """Generate final contraction code block.

    Always 3-operand contraction: Tr[ tmp · Gamma_cur · S_fwd ].
    (PDF Eq 7 — no extra source gamma factor.)
    """
    fwd = result["fwd_var"]

    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# Final contraction: Tr[ G̃_seq @ Gamma_cur @ S_fwd ]",
        f"# Forward: {result['fwd_flavor']} ({result['fwd_var']})",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# G5-dagger on sequential propagator",
        "tmp_prop = core.LatticePropagator(latt_info)",
        "tmp_prop.data = contract("
        "'AB, wtzyxCBji, CD -> wtzyxADij',"
        " G5, prop_seq.data.conj(), G5)",
        "",
    ]

    # Always 3-operand contraction (baryon + meson, PDF Eq 7)
    # Tr[ tmp_prop · Gamma_cur · S_fwd ] — no extra Gamma_src
    lines.append("three_pt_site = contract(")
    lines.append(
        "    'wtzyxijba, jk, wtzyxkiab -> wtzyx',"
    )
    lines.append(
        f"    tmp_prop.data, Gamma_cur, {fwd}.data)"
    )

    lines.append("")
    lines.append("three_pt_local = contract('wtzyx -> t', three_pt_site)")
    lines.append("C3_t = core.gatherLattice(")
    lines.append("    cp.array(three_pt_local), [0, -1, -1, -1])")
    lines.append("")
    lines.append("if core.getMPIRank() == 0:")
    lines.append(f"    np.save({out}, cp.asnumpy(C3_t))")
    return "\n".join(lines)


# ======================================================================
# Structural operand mapping
# ======================================================================

_STRUCTURAL_MAP = {
    frozenset("ijk"): "epsilon",
    frozenset("lmn"): "epsilon",
    frozenset("AB"): "Cg5",
    frozenset("GH"): "Cg5",
    frozenset("ID"): "projector",
}


def _derive_structural_operand(indices: str) -> str:
    """Derive operand name from a structural index group."""
    idx_set = frozenset(indices)
    if idx_set in _STRUCTURAL_MAP:
        return _STRUCTURAL_MAP[idx_set]
    return indices
