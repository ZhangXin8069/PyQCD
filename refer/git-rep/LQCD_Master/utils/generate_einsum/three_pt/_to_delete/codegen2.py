"""codegen2 — PyQUDA sequential source and contraction code for 3pt functions.

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
    """Parse an einsum string into index groups with metadata.

    Distinguishes:
      - 'wtzyx' ALONE          → phase field (structural index group)
      - 'wtzyx' + extra chars  → propagator (e.g. 'wtzyxAGil')
      - no 'wtzyx' prefix      → structural (epsilon, Cg5, Tmat, etc.)
    """
    parts = einsum.split("->")
    output = parts[1].strip()
    raw_groups = [g.strip() for g in parts[0].split(",")]

    groups = []
    n_props = 0
    n_structural = 0
    for g in raw_groups:
        if g == "wtzyx":
            # Bare wtzyx = spacetime phase field (structural)
            n_structural += 1
            groups.append({"raw": g, "type": "phase", "indices": g})
        elif g.startswith("wtzyx"):
            # wtzyx + extra chars = propagator (spin-color indices)
            n_props += 1
            groups.append({"raw": g, "type": "prop", "indices": g[5:]})
        else:
            # Structural group: "ijk", "AB", "ID", etc.
            n_structural += 1
            groups.append({"raw": g, "type": "struct", "indices": g})

    return {
        "groups": groups,
        "output": output,
        "n_props": n_props,
        "n_structural": n_structural,
        "is_baryon": n_structural > 1,  # meson: 1 prop group only → n_structural=0
        # is_baryon implies n_structural >= 4 (phase + 2 eps + Cg5 + Cg5 + projector)
    }


def _derive_structural_operand(indices: str, is_baryon: bool) -> str:
    """Derive operand name from a structural index group.

    For baryon: indices like 'ijk' → 'epsilon', 'AB' → 'Cg5', 'ID' → 'projector'
    For phase: indices == 'wtzyx' → None (handled separately)
    """
    idx_set = frozenset(indices)

    # Structural operand mapping (baryon only)
    _STRUCTURAL_MAP = {
        frozenset("ijk"): "epsilon",
        frozenset("lmn"): "epsilon",
        frozenset("AB"): "Cg5",
        frozenset("GH"): "Cg5",
        frozenset("ID"): "projector",
    }

    if idx_set in _STRUCTURAL_MAP:
        return _STRUCTURAL_MAP[idx_set]
    return indices


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

        if is_baryon:
            # Build operands from index groups
            raw_ops = []
            prop_idx = 0
            for g in groups:
                if g["type"] == "phase":
                    raw_ops.append(phase)
                elif g["type"] == "prop":
                    # Map to var_a, var_b, ...
                    var_key = f"var_{chr(ord('a') + prop_idx)}"
                    var_name = sb.get(var_key, f"prop_{prop_idx}")
                    raw_ops.append(f"{var_name}.data")
                    prop_idx += 1
                else:
                    # Structural: epsilon, Cg5, projector
                    op_name = _derive_structural_operand(g["indices"], True)
                    raw_ops.append(op_name)

            # Resolve through _rename_op
            ops = [_rename_op(o) for o in raw_ops]

            lines.append(f"{indent}{sgn} contract('{sb['einsum']}',")
            # Group in lines of 3, last line has remaining
            for j in range(0, len(ops), 3):
                chunk = ops[j:j + 3]
                chunk_str = ", ".join(chunk) + ","
                lines.append(f"{indent2}{chunk_str}")
            lines.append(f"{indent}),  # {desc}")

        else:
            # Meson: single propagator, phase is scalar-multiplied
            # Convention: conj for the spectator anti-quark propagator
            var_a = sb.get("var_a", "prop_0")
            ops = [f"{var_a}.data * {phase}"]
            lines.append(f"{indent}{sgn} contract('{sb['einsum']}',")
            lines.append(f"{indent2}{ops[0]},")
            lines.append(f"{indent2}  # spectator: {sb.get('flavor_a', '')}")
            lines.append(f"{indent}),  # {sb.get('description', '')}")

    lines.append(")")
    lines.append("")

    # G5-dagger
    lines.append("# G5-dagger for sequential source")
    lines.append(
        "B.data = contract("
        "'AB, wtzyxCBji, CD -> wtzyxADij',"
        " G5, B.data.conj(), G5)"
    )
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
    """Generate final contraction code block."""
    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# Final contraction: Tr[ G_seq_dag @ Gamma_cur @ S_fwd ]",
        f"# Forward: {result['fwd_flavor']} ({result['fwd_var']})",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# G5-dagger on sequential propagator",
        "tmp_prop = core.LatticePropagator(latt_info)",
        "tmp_prop.data = contract("
        "'AB, wtzyxCBji, CD -> wtzyxADij',"
        " G5, prop_seq.data.conj(), G5)",
        "",
        "three_pt_site = contract(",
        "    'wtzyxijba, jk, wtzyxkiab -> wtzyx',",
        f"    tmp_prop.data, Gamma_cur, {result['fwd_var']}.data)",
        "",
        "three_pt_local = contract('wtzyx -> t', three_pt_site)",
        "C3_t = core.gatherLattice(",
        "    cp.array(three_pt_local), [0, -1, -1, -1])",
        "",
        "if core.getMPIRank() == 0:",
        f"    np.save({out}, cp.asnumpy(C3_t))",
    ]
    return "\n".join(lines)
