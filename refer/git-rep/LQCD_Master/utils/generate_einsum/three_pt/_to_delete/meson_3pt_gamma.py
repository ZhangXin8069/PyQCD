"""meson_3pt_gamma — Corrected meson 3pt contraction + code generation.

Fixes both the sink gamma and source gamma factors that were missing
in the original codegen.

The old code uses a trace-only sink block (wtzyxjiba -> wtzyx) that drops
BOTH the sink and source gamma matrices. For pseudoscalar mesons (γ₅) this
works partially by accident (G5-dagger masks the sink γ₅), but non-γ₅
gammas give incorrect results.

This file wraps the existing contract module and explicitly includes:
  1. Γ_snk in the sink block  →  B = S_spectator · Γ_snk
  2. Γ_src† in the final contraction  →  Tr[G̃_seq · Γ_cur · S_fwd · Γ_src†]
"""

from pathlib import Path

try:
    from ..hadron_operator import meson_operator
    from ..wicklib.gamma import Gamma, GAMMA_5, C
    from .contract import contract_meson_3pt as _raw_contract_meson_3pt
    from .._wick_translate import _rename_op
    from .codegen2 import gen_seq_source_code, gen_final_contract_code
except ImportError:
    import sys
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from hadron_operator import meson_operator
    from wicklib.gamma import Gamma, GAMMA_5, C
    from three_pt.contract import contract_meson_3pt as _raw_contract_meson_3pt
    from _wick_translate import _rename_op
    from three_pt.codegen2 import gen_seq_source_code, gen_final_contract_code


_GAMMA_IDX = {
    "I4": 0, "g1": 1, "g2": 2, "g3": 4, "g4": 8,
    "G5": 15, "gtg5": 7,
    "Cmat": 10, "Cg5": 5, "Cg1": 11,
}


def _gamma_to_idx(name: str) -> int:
    """Convert gamma name (like 'G5', 'g1') to wicklib index."""
    n = name.strip()
    if n.startswith("gamma"):
        n = f"g{n[5:]}" if n[5:].isdigit() else n
    return _GAMMA_IDX.get(n, 0)


def _pyquda_idx(name: str) -> str:
    """Convert gamma name to PyQUDA gamma.gamma() argument."""
    return str(_gamma_to_idx(name))


def _extract_gamma(op_input) -> str:
    """Extract the gamma name from a meson operator."""
    tensors = op_input[0].tensors
    for t in tensors:
        if t.type == "gamma":
            return t.name
    return "I4"


def _dirac_adjoint_factor(gamma_name: str) -> int:
    """Compute the factor from Dirac adjoint: γ₄ · Γ† · γ₄.

    Returns the factor applied:  Γ̄ = factor × Γ
    (e.g., G5̄ = -1 × G5,  g1̄ = +1 × g1).
    """
    idx = _gamma_to_idx(gamma_name)
    g = Gamma(idx)
    dag = g.D
    # dag.factor tells us how much of the original gamma remains
    return dag.factor


def _build_meson_sink_einsum() -> str:
    """Meson sink block einsum with sink gamma insertion.

    wtzyxjiba, jk -> wtzyxkiba

    The gamma operates on the spin_out index of the spectator propagator,
    producing a LatticePropagator output compatible with G5-dagger.
    """
    return "wtzyxjiba, jk -> wtzyxkiba"


# ======================================================================
# Fixed meson 3pt contract
# ======================================================================

def contract_meson_3pt_gamma(sink, source, current):
    """Meson 3pt contract with corrected sink and source gamma handling.

    Parameters are the same as contract_meson_3pt().
    Returns a patched result dict with additional gamma metadata.
    """
    raw = _raw_contract_meson_3pt(sink, source, current)

    sink_gamma = _extract_gamma(sink)
    src_gamma = _extract_gamma(source)
    sb_einsum = _build_meson_sink_einsum()

    fixed_terms = []
    for st in raw["sink_terms"]:
        new_st = dict(st)
        new_st["einsum"] = sb_einsum
        new_st["sink_gamma"] = sink_gamma
        fixed_terms.append(new_st)

    return {
        "sink_terms": fixed_terms,
        "fwd_var": raw["fwd_var"],
        "fwd_flavor": raw["fwd_flavor"],
        "current_gamma": raw["current_gamma"],
        "sink_gamma": sink_gamma,
        "src_gamma": src_gamma,
        "src_gamma_dag_factor": _dirac_adjoint_factor(src_gamma),
        "n_topologies": raw["n_topologies"],
    }


# ======================================================================
# Sequential source code generation
# ======================================================================

def gen_seq_source_code_meson(
    src_name, snk_name, result, phase="phase_sink"
):
    """Generate sequential source code.

    Includes both the sink gamma (in B) and the source gamma (in the
    final contraction, via gen_final_contract_code_meson).
    """
    cg = result.get("current_gamma", "g1")
    sg = result.get("sink_gamma", "G5")

    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# Sequential source:"
        f" {src_name} -> {snk_name} (meson 3pt)",
        "# ═══════════════════════════════════════════════════════",
        "",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_snk = cp.asarray(gamma.gamma({_pyquda_idx(sg)}),"
        " dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray(gamma.gamma({_pyquda_idx(cg)}),"
        " dtype=cp.complex128)",
        "",
    ]

    # Topology summary
    lines.append(f"# {len(result['sink_terms'])} topology(ies)")
    for sb in result["sink_terms"]:
        lines.append(
            f"#   topo: sign={sb['sign']:+d}"
            f"  {sb.get('description', '')}"
        )
    lines.append("")

    # Sink block with Gamma_snk
    lines.append("# Sink block B(x) = sum over all Wick topologies")
    lines.append("# B = S_spectator @ Gamma_snk")
    lines.append("B = core.LatticePropagator(latt_info)")
    lines.append("B.data = (")

    indent = "    "
    for sb in result["sink_terms"]:
        sgn = "+" if sb["sign"] >= 0 else "-"
        indent2 = indent + "    "
        desc = sb.get("description", "")
        var_a = sb.get("var_a", "prop_0")
        ops = [f"{var_a}.data", "Gamma_snk"]

        lines.append(f"{indent}{sgn} contract('{sb['einsum']}',")
        lines.append(f"{indent2}{ops[0]},")
        lines.append(f"{indent2}{ops[1]},")
        lines.append(f"{indent2}  # spectator: {sb.get('flavor_a', '')}")
        lines.append(f"{indent}),  # {desc}")

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

def gen_final_contract_code_meson(result, out="out_path"):
    """Generate final contraction code with source gamma.

    Full contraction:
      Tr[ γ₅ · G_seq† · γ₅  ·  Γ_cur  ·  S_fwd  ·  Γ_src† ]
         └── G̃_seq ──┘

    where Γ_src† = factor × Γ_src (Dirac adjoint of source gamma).
    """
    src_gamma = result.get("src_gamma", "I4")
    dag_factor = result.get("src_gamma_dag_factor", 1)

    src_var = f"Gamma_src"
    dag_expr = f"cp.asarray(gamma.gamma({_pyquda_idx(src_gamma)}), dtype=cp.complex128)"
    if dag_factor == -1:
        dag_expr = f"-{dag_expr}"

    lines = [
        "# ═══════════════════════════════════════════════════════",
        "# Final contraction: Tr[ G̃_seq @ Gamma_cur @ S_fwd @ Gamma_src† ]",
        f"# Forward: {result['fwd_flavor']} ({result['fwd_var']})",
        f"# Source gamma: {src_gamma} (Dirac conj factor={dag_factor:+d})",
        "# ═══════════════════════════════════════════════════════",
        "",
        "# G5-dagger on sequential propagator",
        "tmp_prop = core.LatticePropagator(latt_info)",
        "tmp_prop.data = contract("
        "'AB, wtzyxCBji, CD -> wtzyxADij',"
        " G5, prop_seq.data.conj(), G5)",
        "",
        f"{src_var} = {dag_expr}",
        "",
        "three_pt_site = contract(",
        "    'wtzyxijba, jk, wtzyxklba, lm -> wtzyx',",
        f"    tmp_prop.data, Gamma_cur,"
        f" {result['fwd_var']}.data, {src_var})",
        "",
        "three_pt_local = contract('wtzyx -> t', three_pt_site)",
        "C3_t = core.gatherLattice(",
        "    cp.array(three_pt_local), [0, -1, -1, -1])",
        "",
        "if core.getMPIRank() == 0:",
        f"    np.save({out}, cp.asnumpy(C3_t))",
    ]
    return "\n".join(lines)
