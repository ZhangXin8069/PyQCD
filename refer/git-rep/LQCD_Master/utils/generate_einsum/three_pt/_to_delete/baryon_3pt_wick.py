"""baryon_3pt_wick — Baryon 3pt codegen using wicklib contraction results.

Generates complete PyQUDA code (sink block + sequential source + final contraction)
directly from wicklib topology enumeration.

Fixes the label bug in contract.sink_block_einsum(): the sink block must
only include the TWO spectator propagators, NOT the current wire. The output
free indices correspond to the current wire slot.

Usage:
    from three_pt.baryon_3pt_wick import gen_baryon_3pt_code

    code = gen_baryon_3pt_code(sink_op, source_op, cur_op, src_name, snk_name)
    print(code)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from .._wick_translate import _to_wicklib
    from ..wicklib.correlator import Correlator as _WickCorr
    from ..wicklib.operator import SpinProjector
except ImportError:
    import sys
    from pathlib import Path
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from _wick_translate import _to_wicklib
    from wicklib.correlator import Correlator as _WickCorr
    from wicklib.operator import SpinProjector

# ── Baryon slot label conventions ──
# Source side: a(A,i), b(B,j), c(D,k)
# Sink   side: a(G,l), b(H,m), c(I,n)

SRC_SPIN = {"a": "A", "b": "B", "c": "D"}
SNK_SPIN = {"a": "G", "b": "H", "c": "I"}
SRC_COL  = {"a": "i", "b": "j", "c": "k"}
SNK_COL  = {"a": "l", "b": "m", "c": "n"}

_LIGHT = frozenset({"u", "d"})

# ── Wicklib matrix label translation ──
_ROW_SNK = {1: "c", 2: "b", 3: "a"}   # sink rows → slot
_COL_SRC = {0: "a", 1: "b", 2: "c"}   # source cols → slot


# ═══════════════════════════════════════════════════════════════════════
# Label helpers
# ═══════════════════════════════════════════════════════════════════════

def _var(f: str) -> str:
    """PyQUDA variable name for a flavour."""
    return "prop_l" if f in _LIGHT else f"prop_{f}"


def _prop_label(src_slot: str, snk_slot: str) -> str:
    """Forward-propagator einsum label: wtzyx{src_spin}{snk_spin}{src_col}{snk_col}.

    Exception: (c,c) reverses spin order to match PyQUDA propagator convention.
    """
    if src_slot == "c" and snk_slot == "c":
        spin = f"{SNK_SPIN[snk_slot]}{SRC_SPIN[src_slot]}"  # ID
    else:
        spin = f"{SRC_SPIN[src_slot]}{SNK_SPIN[snk_slot]}"
    return f"wtzyx{spin}{SRC_COL[src_slot]}{SNK_COL[snk_slot]}"


def _free_label(cur_src: str, cur_snk: str) -> str:
    """Free-index label for the current wire: {snk_spin}{src_spin}{snk_col}{src_col}.

    This is the "output" label — the sink block leaves these indices open
    for the sequential source to attach.
    """
    return (
        f"{SNK_SPIN[cur_snk]}{SRC_SPIN[cur_src]}"
        f"{SNK_COL[cur_snk]}{SRC_COL[cur_src]}"
    )


def _gamma_g_call(gamma_name: str) -> str:
    """Gamma name → gamma.gamma(N) call."""
    idx_map = {
        "g1": "1", "g2": "2", "g3": "4", "g4": "8",
        "G5": "15", "gtg5": "7", "I4": "0", "g5": "15",
    }
    return f"gamma.gamma({idx_map.get(gamma_name, '0')})"


# ═══════════════════════════════════════════════════════════════════════
# Topology extraction from wicklib
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Topology:
    """One baryon 3pt Wick topology."""
    cur_src: str          # Source slot carrying the current (a/b/c)
    cur_snk: str          # Sink slot carrying the current
    spectators: Dict[str, str]  # {src_slot: snk_slot} for spectators (2 entries)
    sign: int             # Fermion sign (-1 or +1), includes D−A negation


def _extract_flavors(op_input) -> Dict[str, str]:
    """Extract {slot: flavor} from baryon_operator() output."""
    op = op_input[0] if isinstance(op_input, tuple) else op_input
    tensors = op.tensors
    return {"a": tensors[1].flavor, "b": tensors[3].flavor, "c": tensors[4].flavor}


def _extract_current_info(current) -> Tuple[str, str, str]:
    """Extract (flavor_in, flavor_out, gamma) from current_operator()."""
    if not isinstance(current, tuple) and hasattr(current, "flavor_in"):
        return current.flavor_in, current.flavor_out, getattr(current, "gamma", "")
    op = current[0] if isinstance(current, tuple) else current
    tensors = op.tensors
    return (
        next(t.flavor for t in tensors if t.type == "antiquark"),  # flavor_out
        next(t.flavor for t in tensors if t.type == "quark"),      # flavor_in
        next(t.name for t in tensors if t.type == "gamma"),
    )


def _enumerate_topologies(sink, source, current) -> List[Topology]:
    """Use wicklib to enumerate all baryon 3pt Wick topologies.

    Each topology contains:
      - cur_src, cur_snk: which source/sink slots carry the current
      - spectators: {src_slot: snk_slot} for the other two quarks
      - sign: overall fermion sign (negated for D−A convention)
    """
    src_flavors = _extract_flavors(source)
    snk_flavors = _extract_flavors(sink)
    cur_out, cur_in, _ = _extract_current_info(current)

    # Build wicklib operators
    sr = sink[0] if isinstance(sink, tuple) else sink
    sr2 = source[0] if isinstance(source, tuple) else source
    cr = current[0] if isinstance(current, tuple) else current

    w_snk, si_snk = _to_wicklib(sr, "x")
    w_src, si_src = _to_wicklib(sr2, "y")
    w_cur = _to_wicklib(cr, "z")

    # Correlator with spin projector
    T = SpinProjector.P_plus(si_src, si_snk)
    corr = _WickCorr(T * w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)

    topologies = []
    for term in corr.terms:
        # Walk the sparse 3×3 adjacency matrix
        cur_src = cur_snk = None
        conn = {}    # {src_slot: (snk_slot, is_current)}
        us = set()   # source slots seen
        us2 = set()  # sink slots seen

        for ri in range(len(term.matrix)):
            for ci in range(len(term.matrix[ri])):
                e = term.matrix[ri][ci]
                if e.flavor is None:
                    continue
                is_cur = "z" in (e.sink, e.source)
                if ri == 0:
                    # Row 0 is always the current row
                    cur_src = _COL_SRC.get(ci)
                    if cur_src is None:
                        for s in ["a", "b", "c"]:
                            if src_flavors[s] == e.flavor:
                                cur_src = s
                                break
                    us.add(cur_src)
                elif is_cur:
                    # Sink-side current slot
                    cur_snk = _ROW_SNK[ri]
                    us2.add(cur_snk)
                else:
                    ss = _ROW_SNK[ri]
                    sc = _COL_SRC[ci]
                    conn[sc] = (ss, False)
                    us.add(sc)
                    us2.add(ss)

        if cur_snk is None:
            continue

        # Record the current connection
        conn[cur_src] = (cur_snk, True)

        # Fill missing spectator identities
        for s in ["a", "b", "c"]:
            if s not in us:
                for t in ["a", "b", "c"]:
                    if t not in us2:
                        conn[s] = (t, False)
                        us.add(s)
                        us2.add(t)
                        break

        if len(conn) != 3:
            continue

        # Extract spectator-only mapping
        spectators = {s: t for s, (t, thru) in conn.items() if not thru}
        sign = -round(term.factor.real)  # negated for D−A convention

        topologies.append(Topology(
            cur_src=cur_src,
            cur_snk=cur_snk,
            spectators=spectators,
            sign=sign,
        ))

    return topologies


# ═══════════════════════════════════════════════════════════════════════
# Sink block einsum generation (CORRECT version)
# ═══════════════════════════════════════════════════════════════════════

def _sink_einsum(topo: Topology) -> str:
    """Generate the 8-operand sink block einsum for this topology.

    The three propagator lines are:
      1. Forward (current) line: from src.cur_src to snk.cur_snk
         → NOT in the sink block; it becomes the forward propagator
      2. Spectator 1, 2: the two non-current source→sink mappings
         → These are the TWO propagator operands in the sink block

    Output: free indices for the current wire's sink-side slot.
    """
    spec_items = sorted(topo.spectators.items())
    p1 = _prop_label(spec_items[0][0], spec_items[0][1])
    p2 = _prop_label(spec_items[1][0], spec_items[1][1])
    output = _free_label(topo.cur_src, topo.cur_snk)

    return (
        f"wtzyx, ijk, lmn, AB, GH, ID, {p1}, {p2} -> wtzyx{output}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Complete code generation
# ═══════════════════════════════════════════════════════════════════════

def gen_baryon_3pt_code(
    sink, source, current,
    src_name: str = "source",
    snk_name: str = "sink",
    phase: str = "phase_sink",
) -> str:
    """Generate complete baryon 3pt PyQUDA code from wicklib topologies.

    Produces a single code block with:
      1. Gamma / epsilon setup
      2. Sink block B(x) from all topologies
      3. G5-dagger transformation
      4. Sequential source solve
      5. G5-dagger transformation on sequential propagator
      6. Final contraction: Tr[ ψ̃_seq · Γ_cur · S_fwd ]
      7. MPI gather + save

    Parameters
    ----------
    sink : tuple(Operator, ...)
        Sink baryon operator (e.g. from baryon_operator()).
    source : tuple(Operator, ...)
        Source baryon operator.
    current : tuple(Operator, ...)
        Current operator (e.g. from current_operator()).
    src_name, snk_name : str
        Names for comments.
    phase : str
        Phase variable name used in einsum (default: "phase_sink").

    Returns
    -------
    str
        Complete PyQUDA code block.
    """
    # ── Enumerate topologies ──
    topologies = _enumerate_topologies(sink, source, current)
    cur_out, cur_in, cur_gamma = _extract_current_info(current)
    n_topo = len(topologies)

    # Gather flavors
    src_flavors = _extract_flavors(source)
    snk_flavors = _extract_flavors(sink)

    # ── Build code ──
    lines = [
        "# ═══════════════════════════════════════════════════════",
        f"# Baryon 3pt: {src_name} -> {snk_name}",
        f"#   Current: {cur_gamma}  "
        f"({cur_in} -> {cur_out})",
        f"#   Topologies: {n_topo}",
        "# ═══════════════════════════════════════════════════════",
        "",
        "I4 = cp.eye(4, dtype=cp.complex128)",
        "G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)",
        f"Gamma_cur = cp.asarray({_gamma_g_call(cur_gamma)},"
        " dtype=cp.complex128)",
        "",
        "# Epsilon tensor (GPU)",
        "epsilon = cp.zeros((3, 3, 3), dtype=cp.float64)",
        "epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1.0",
        "epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1.0",
        "",
        "# Dirac gamma matrices (GPU)",
        "Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8),"
        " dtype=cp.complex128)",
        "Cg5 = Cmat @ G5",
        "Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5,"
        " dtype=cp.complex128)  # P_plus",
        "",
    ]

    # ── Topology summary ──
    lines.append(f"# {n_topo} topology(ies)")
    for i, topo in enumerate(topologies):
        # Build description: all three connections
        desc_parts = []
        for s in ["a", "b", "c"]:
            if s == topo.cur_src:
                t = topo.cur_snk
                desc_parts.append(f"{src_flavors[s]}(src.{s})->[J]->{snk_flavors[t]}(snk.{t})")
            elif s in topo.spectators:
                t = topo.spectators[s]
                desc_parts.append(f"{src_flavors[s]}(src.{s})->{snk_flavors[t]}(snk.{t})")
        lines.append(f"#   [{i}] sign={topo.sign:+d}"
                      f"  cur={topo.cur_src}->{topo.cur_snk}"
                      f"  {', '.join(desc_parts)}")
    lines.append("")

    # ── Sink block B(x) ──
    lines.append("# Sink block B(x): spectator contraction")
    lines.append("#   Two spectator propagators per topology")
    lines.append("B = core.LatticePropagator(latt_info)")
    lines.append("B.data = (")

    indent = "    "
    indent2 = indent + "    "

    for i, topo in enumerate(topologies):
        einsum = _sink_einsum(topo)
        sign_op = "+" if topo.sign >= 0 else "-"

        # Spectator flavors → variable names
        spec_items = sorted(topo.spectators.items())
        spec_flavors = [src_flavors[s] for s, _ in spec_items]
        var_a, var_b = _var(spec_flavors[0]), _var(spec_flavors[1])

        lines.append(f"{indent}{sign_op}"
                      f" contract('{einsum}',")
        lines.append(f"{indent2}{phase}, epsilon, epsilon,")
        lines.append(f"{indent2}Cg5, Cg5, Tmat,")
        lines.append(f"{indent2}{var_a}.data, {var_b}.data,")
        lines.append(f"{indent}),  # topo {i}: cur"
                      f" {topo.cur_src}->{topo.cur_snk}")

    lines.append(")")
    lines.append("")

    # ── G5-dagger on B ──
    lines.append("# G5-dagger: B̃ = γ₅ · B† · γ₅")
    lines.append("B.data = contract(")
    lines.append(
        "    'AB, wtzyxCBji, CD -> wtzyxADij',"
    )
    lines.append("    G5, B.data.conj(), G5)")
    lines.append("")

    # ── Sequential source ──
    lines.append("# Sequential source at t_sink")
    lines.append("src_seq = source.sequential12(B, t_sink)")
    lines.append("")
    lines.append("# Sequential solve (insert your inversion code here)")
    lines.append("")

    # ── Final contraction ──
    fwd_var = _var(cur_in)
    lines.append(
        "# ═══════════════════════════════════════════════════════")
    lines.append(
        "# Final contraction: Tr[ G̃_seq @ Gamma_cur @ S_fwd ]")
    lines.append(
        f"# Forward propagator: {cur_in} ({fwd_var})")
    lines.append(
        "# ═══════════════════════════════════════════════════════")
    lines.append("")
    lines.append("# G5-dagger on sequential propagator")
    lines.append("tmp_prop = core.LatticePropagator(latt_info)")
    lines.append("tmp_prop.data = contract(")
    lines.append(
        "    'AB, wtzyxCBji, CD -> wtzyxADij',")
    lines.append("    G5, prop_seq.data.conj(), G5)")
    lines.append("")
    lines.append("# Tr[ tmp_prop · Gamma_cur · S_fwd ]")
    lines.append("three_pt_site = contract(")
    lines.append(
        "    'wtzyxijba, jk, wtzyxkiab -> wtzyx',")
    lines.append(
        f"    tmp_prop.data, Gamma_cur, {fwd_var}.data)")
    lines.append("")
    lines.append("three_pt_local = contract('wtzyx -> t',"
                  " three_pt_site)")
    lines.append("C3_t = core.gatherLattice(")
    lines.append("    cp.array(three_pt_local), [0, -1, -1, -1])")
    lines.append("")
    lines.append("if core.getMPIRank() == 0:")
    lines.append(
        "    np.save(out_path, cp.asnumpy(C3_t))")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    _parent = str(Path(__file__).resolve().parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from hadron_operator import baryon_operator, current_operator

    # p -> p (V, u->u)
    cur = current_operator("u", "u", "g1")
    snk, _ = baryon_operator("u", "d", "u")
    src, _ = baryon_operator("u", "d", "u")
    print("=== p -> p (V, u->u) ===")
    print(gen_baryon_3pt_code(snk, src, cur, "p", "p"))
    print()

    # Λ -> p (V, s->u)
    cur2 = current_operator("u", "s", "g1")
    snk2, _ = baryon_operator("u", "d", "u")
    src2, _ = baryon_operator("u", "d", "s")
    print("=== Λ -> p (V, s->u) ===")
    print(gen_baryon_3pt_code(snk2, src2, cur2, "L", "p"))
    print()

    # Λ -> Λ (V, s->s)
    cur3 = current_operator("s", "s", "g1")
    snk3, _ = baryon_operator("u", "d", "s")
    src3, _ = baryon_operator("u", "d", "s")
    print("=== Λ -> Λ (V, s->s) ===")
    print(gen_baryon_3pt_code(snk3, src3, cur3, "L", "L"))
