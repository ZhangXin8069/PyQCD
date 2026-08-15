"""contract — Baryon and meson 3pt contraction + topology enumeration + einsum generation.

Meson 3pt is corrected to include both sink and source gamma matrices.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from utils.generate_einsum._wick_translate import _to_wicklib
    from utils.generate_einsum.wicklib.correlator import Correlator as _WickCorr
    from utils.generate_einsum.wicklib.operator import SpinProjector
    from utils.generate_einsum.wicklib.gamma import Gamma
except ImportError:
    from _wick_translate import _to_wicklib
    from wicklib.correlator import Correlator as _WickCorr
    from wicklib.operator import SpinProjector
    from wicklib.gamma import Gamma


# ── Internal data structures ──

@dataclass
class _CurrentInfo:
    flavor_in: str
    flavor_out: str
    gamma: str


@dataclass
class ContractionTopology:
    name: str
    fermion_sign: int
    connections: Dict[str, Tuple[str, bool]]
    description: str = ""


def from_current_operator(current_input) -> _CurrentInfo:
    """Extract flavor/gamma from current_operator() output or old-style object."""
    if not isinstance(current_input, tuple) and hasattr(current_input, "flavor_in"):
        return _CurrentInfo(
            current_input.flavor_in,
            current_input.flavor_out,
            getattr(current_input, "gamma", ""),
        )
    op = current_input[0] if isinstance(current_input, tuple) else current_input
    tensors = op.tensors
    return _CurrentInfo(
        flavor_in=tensors[2].flavor,
        flavor_out=tensors[0].flavor,
        gamma=tensors[1].name,
    )


# ═══════════════════════════════════════════════════════════════════════
# Label conventions (baryon)
# ═══════════════════════════════════════════════════════════════════════
SRC_EPS = {"a": "i", "b": "j", "c": "k"}
SNK_EPS = {"a": "l", "b": "m", "c": "n"}
SRC_SPIN = {"a": "A", "b": "B", "c": "D"}
SNK_SPIN = {"a": "G", "b": "H", "c": "I"}
SRC_COL = {"a": "i", "b": "j", "c": "k"}
SNK_COL = {"a": "l", "b": "m", "c": "n"}

_LIGHT = frozenset({"u", "d"})


def _extract_baryon_flavors(op_input):
    """Extract slot->flavor dict from baryon_operator() output."""
    tensors = op_input[0].tensors if isinstance(op_input, tuple) else op_input.tensors
    return {"a": tensors[1].flavor, "b": tensors[3].flavor, "c": tensors[4].flavor}


def _proj_idx(p: str = "Pplus") -> str:
    return "ID" if p in ("Pplus", "P+", "Pminus", "P-") else p


def _prop_label(src_slot: str, snk_slot: str) -> str:
    spin = (
        f"{SNK_SPIN[snk_slot]}{SRC_SPIN[src_slot]}"
        if (src_slot == "c" and snk_slot == "c")
        else f"{SRC_SPIN[src_slot]}{SNK_SPIN[snk_slot]}"
    )
    return f"wtzyx{spin}{SRC_COL[src_slot]}{SNK_COL[snk_slot]}"


def _output_label(src_slot: str, snk_slot: str) -> str:
    return (
        f"{SNK_SPIN[snk_slot]}{SRC_SPIN[src_slot]}"
        f"{SNK_COL[snk_slot]}{SRC_COL[src_slot]}"
    )


def _var(f: str) -> str:
    return "prop_l" if f in _LIGHT else f"prop_{f}"


# ═══════════════════════════════════════════════════════════════════════
# Sink block einsum generation (baryon)
# ═══════════════════════════════════════════════════════════════════════

def sink_block_einsum(src, snk, topo: ContractionTopology) -> dict:
    r"""Generate the 8-operand ``einsum`` string for one baryon sink block.

    Given a specific Wick topology (from :func:`contract_baryon_3pt`), this
    function writes the concrete PyQUDA ``einsum`` that computes:

    .. math::
        B_{x;\alpha\beta}^{ij} =
        \sum_{\text{internal}} \bigl[
            \varepsilon_{abc}
            \varepsilon_{lmn}
            (C\gamma_5)_{\alpha\beta}
            (\Gamma_{\text{proj}})_{AB}
            S_1 \otimes S_2
        \bigr]

    where the two spectator propagators are either ``wtzyxGHij`` etc.
    and the output carries the correct spin/colour index ordering for
    the subsequent sequential-source contraction.

    The einsum always has 8 operands (in order):

    #. ``wtzyx`` — coordinate delta (MPI gather placeholder)
    #. ``ijk`` — source epsilon
    #. ``lmn`` — sink epsilon
    #. ``AB`` — source (Cγ₅)⁻¹
    #. ``GH`` — sink Cγ₅
    #. ``{projector}`` — spin projector (e.g. ``PP``, ``ID``, ``PM``)
    #. *spectator-1* — full propagator einsum string
    #. *spectator-2* — full propagator einsum string

    The output index string is ``wtzyx{snk_spin}{src_spin}{snk_col}{src_col}``,
    e.g. ``wtzyxGIli`` or ``wtzyxGGlj`` for the ``sb`` (heavy-strange anchor
    attached to the current) case.

    Parameters
    ----------
    src : object
        Source baryon operator.
    snk : object
        Sink baryon operator.
    topo : ContractionTopology
        Wick topology specifying:

        - ``connections``: ``{src_slot: (snk_slot, is_current)}`` mapping
        - ``fermion_sign``: permutation parity (integer) from wicklib
        - ``description``: human-readable label

    Returns
    -------
    dict
        ``einsum`` : str
            The 8-operand einsum string ready for PyQUDA.
        ``sign`` : int
            Fermion sign, already negated (D−A convention).
        ``current`` : (str, str)
            Source and sink slots carrying the current, e.g. ``("c", "c")``.
        ``n_props`` : int
            Number of spectator propagators (always 2).
        ``var_a``, ``var_b`` : str
            PyQUDA variable names for the two spectator propagators.
        ``flavor_a``, ``flavor_b`` : str
            Flavour tags of the two spectators.
        ``fwd_var``, ``fwd_flavor`` : str
            Forward (sequential-source) propagator info.
        ``description`` : str
            Verbose topology description for debugging.

    Notes
    -----
    **sb case** (``cur_src == "c" and cur_snk == "c"``):
    The heavy/strange quark carries the current.  Both spectators are
    identity-connected (``a->a``, ``b->b``).  Output index order is
    ``GADB`` → ``GI…`` because slot ``c`` has reversed spin ordering
    with ``SNK_SPIN["c"]="I"`` and ``SRC_SPIN["c"]="D"``.

    **Non-sb case**:
    The current is on a light slot (usually ``a``).  One spectator
    propagator is identity-connected (e.g. ``b->b``), the other carries
    an off-diagonal connection.  The code special-cases the ``b->b``
    identity for the ordering used by the sequential-source downstream
    code.

    **Sign convention**:
    The raw Wicklib factor is negated (``-topo.fermion_sign``) to
    align with the standard D−A sequential-source formula.
    """
    src_flavors = _extract_baryon_flavors(src)
    snk_flavors = _extract_baryon_flavors(snk)
    conn = topo.connections

    # Locate the current-carrying slot
    cur_src = cur_snk = None
    for s in ["a", "b", "c"]:
        if s in conn and conn[s][1]:
            cur_src, cur_snk = s, conn[s][0]
            break
    if cur_src is None:
        raise ValueError("No current slot")

    # Determine propagator labels and output index order
    sb = (cur_src == "c" and cur_snk == "c")
    if sb:
        # Current on heavy/strange: both spectators are identity (a->a, b->b)
        p1, p2 = _prop_label("a", "a"), _prop_label("b", "b")
        output = _output_label("c", "c")
    else:
        # Current on a light slot: one spectator identity, one off-diagonal
        p1 = _prop_label(cur_src, cur_snk)
        ps = [s for s in ["a", "b", "c"] if s in conn and not conn[s][1]]
        if "b" in ps and conn["b"][0] == "b":
            p2 = _prop_label("b", "b")
            r = [s for s in ps if s != "b"][0]
            output = _output_label(r, conn[r][0])
        elif ps:
            s0 = ps[0]
            p2 = _prop_label(s0, conn[s0][0])
            r = [s for s in ps if s != s0][0]
            output = _output_label(r, conn[r][0])
        else:
            p2 = _prop_label("b", "b")
            output = _output_label("a", "c")
    einsum = (
        f"wtzyx, ijk, lmn, AB, GH,"
        f" {_proj_idx()}, {p1}, {p2} -> wtzyx{output}"
    )
    cur_in = src_flavors[cur_src]
    cur_out = snk_flavors[cur_snk]
    src_f = [src_flavors[s] for s in ["a", "b", "c"]]
    src_f.remove(cur_in)
    fa, fb = src_f[0], src_f[1]
    return {
        "einsum": einsum,
        "sign": -topo.fermion_sign,
        "current": (cur_src, cur_snk),
        "n_props": 2,
        "var_a": _var(fa),
        "var_b": _var(fb),
        "flavor_a": fa,
        "flavor_b": fb,
        "fwd_var": _var(cur_in),
        "fwd_flavor": cur_in,
        "description": topo.description,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baryon 3pt entry point
# ═══════════════════════════════════════════════════════════════════════

_ROW_SNK = {1: "c", 2: "b", 3: "a"}
_COL_SRC = {0: "a", 1: "b", 2: "c"}


def contract_baryon_3pt(sink, source, current, projector="P_plus") -> dict:
    r"""Enumerate and format all baryon 3-point contraction topologies.

    This is the single entry point for baryon 3pt code generation.  It works by:

    1. Building Wick-contraction operators for the sink, source, current,
       and the spin-parity projector :math:`T = \frac12(1+\gamma_4)`.
    2. Wick-contracting everything via :class:`wicklib.correlator.Correlator`,
       which enumerates all Wick pairings and evaluates the associated flavour
       and gamma-matrix traces.
    3. Mapping each Wicklib contraction term's adjacency matrix back to the
       ``{a,b,c}`` slot notation used in PyQUDA code generation.
    4. For each topology, calling :func:`sink_block_einsum` to produce the
       concrete 8-operand ``einsum`` string.

    Parameters
    ----------
    sink : object
        Output (sink) baryon operator, typically from :func:`baryon_operator`.
        Internally unwrapped via ``sink[0]`` (tuple support) or used directly.
    source : object
        Input (source) baryon operator.
    current : object
        Current insertion operator, typically from :func:`current_operator`.
    projector : str, optional
        Spin projector type (default ``"P_plus"``).  Passed to
        :func:`SpinProjector.P_plus` as the source-to-sink projector.

    Returns
    -------
    dict
        ``sink_terms`` : list of dict
            One entry per distinct Wick topology.  Each entry has the
            keys produced by :func:`sink_block_einsum`:

            - ``einsum``: 8-operand einsum string (sink block)
            - ``sign``: overall fermion sign (already negated for the D−A convention)
            - ``current``: ``(src_slot, snk_slot)`` — which slots carry the current
            - ``n_props``: number of spectator propagators (always 2 for baryon)
            - ``var_a``, ``var_b``: spectator propagator variable names
              (e.g. ``"prop_l"`` for light quarks, ``"prop_s"`` for strange)
            - ``flavor_a``, ``flavor_b``: corresponding flavour tags
            - ``fwd_var``, ``fwd_flavor``: forward (sequential) propagator
            - ``description``: human-readable connection summary

        ``fwd_var`` : str
            Variable name for the forward (non-spectator, sequential-source)
            propagator.
        ``fwd_flavor`` : str
            Flavour of the forward propagator.
        ``current_gamma`` : str
            Gamma-matrix name of the inserted current.
        ``n_topologies`` : int
            Total number of distinct Wick topologies found.

    Notes
    -----
    **Slots**: the three baryon quark lines are labelled ``a``, ``b``, ``c``.
    ``c`` is conventionally the heavy/strange anchor; ``a`` and ``b`` are the
    two light (or mixed-light) quarks.

    **Current identification**: the Wicklib contraction tags the current vertex
    with coordinate ``"z"``.  The row in the adjacency matrix whose sink
    flavour matches ``cur_in`` carries the current.  That row's column index
    is mapped back to the source slot (``cur_src``) and its row index to the
    sink slot (``cur_snk``).

    **Missing-link repair**: Wicklib only reports non-identity contractions.
    If a spectator quark has no contraction (i.e. the row/col is missing from
    the sparse matrix), the code fills it in as a slot-to-slot identity
    (``src.s -> snk.s``) so that all three quarks are accounted for.

    **Sign convention**: the overall fermion sign is computed by Wicklib
    (from permutation parity of Wick pairings) and negated here to match the
    D−A convention used in the sequential-source literature.
    """
    src_flavors = _extract_baryon_flavors(source)
    snk_flavors = _extract_baryon_flavors(sink)
    cur_info = from_current_operator(current)
    cur_in = cur_info.flavor_in
    sr = sink[0] if isinstance(sink, tuple) else sink
    sr2 = source[0] if isinstance(source, tuple) else source
    cr = current[0] if isinstance(current, tuple) else current
    w_snk, si_snk = _to_wicklib(sr, "x")
    w_src, si_src = _to_wicklib(sr2, "y")
    w_cur = _to_wicklib(cr, "z")
    T = SpinProjector.P_plus(si_src, si_snk)
    corr = _WickCorr(T * w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)
    terms = []
    for term in corr.terms:
        # ── Phase 1: walk the Wicklib adjacency matrix ──
        # term.matrix is a 3×3 sparse matrix indexed by
        #   rows (sink side):    _ROW_SNK = {1: "c", 2: "b", 3: "a"}
        #   columns (source side): _COL_SRC = {0: "a", 1: "b", 2: "c"}
        # Each non-null entry e = (flavour, sink_coord, source_coord)
        # connects a source slot to a sink slot.
        cur_src = cur_snk = None
        conn = {}
        us = set()   # source slots already seen
        us2 = set()  # sink slots already seen
        for ri in range(len(term.matrix)):
            for ci in range(len(term.matrix[ri])):
                e = term.matrix[ri][ci]
                if e.flavor is None:
                    continue
                is_cur = "z" in (e.sink, e.source)
                if ri == 0:
                    # Row 0 is always the current row.
                    # Map its column to a source slot.
                    cur_src = _COL_SRC.get(ci)
                    if cur_src is None:
                        for s in ["a", "b", "c"]:
                            if src_flavors[s] == e.flavor:
                                cur_src = s
                                break
                    us.add(cur_src)
                elif is_cur:
                    # Row ri > 0 with coords containing "z":
                    # this is the sink-side current slot.
                    cur_snk = _ROW_SNK[ri]
                    us2.add(cur_snk)
                else:
                    # Spectator contraction: source slot -> sink slot.
                    ss = _ROW_SNK[ri]
                    sc = _COL_SRC[ci]
                    conn[sc] = (ss, False)
                    us.add(sc)
                    us2.add(ss)
        if cur_snk is None:
            continue
        # Record the current-vertex connection as "thru" = True.
        conn[cur_src] = (cur_snk, True)
        if len(conn) != 3:
            continue

        # ── Phase 2: fill in any missing spectator identity ---
        # Wicklib omits diagonal entries (src.s -> snk.s) when the
        # quark is uncontracted.  Patch them in so conn always has
        # exactly 3 entries.
        for s in ["a", "b", "c"]:
            if s not in us:
                for t in ["a", "b", "c"]:
                    if t not in us2:
                        conn[s] = (t, False)
                        us.add(s)
                        us2.add(t)
                        break

        # ── Phase 3: build topology + generate einsum ──
        desc = ", ".join(
            f"{src_flavors[s]}(src.{s})->[J]->{snk_flavors[t]}(snk.{t})"
            if thru
            else f"{src_flavors[s]}(src.{s})->{snk_flavors[t]}(snk.{t})"
            for s, (t, thru) in sorted(conn.items())
        )
        topo = ContractionTopology(
            f"topo_{len(terms)}", round(term.factor.real), conn, desc
        )
        sb = sink_block_einsum(source, sink, topo)
        terms.append(sb)
    return {
        "sink_terms": terms,
        "fwd_var": _var(cur_in),
        "fwd_flavor": cur_in,
        "current_gamma": cur_info.gamma,
        "n_topologies": len(terms),
    }


# ═══════════════════════════════════════════════════════════════════════
# Meson 3pt — corrected with sink AND source gamma handling
# ═══════════════════════════════════════════════════════════════════════

_GAMMA_IDX = {
    "I4": 0, "g1": 1, "g2": 2, "g3": 4, "g4": 8,
    "G5": 15, "gtg5": 7,
}


def _gamma_to_idx(name: str) -> int:
    n = name.strip()
    if n.startswith("gamma"):
        n = f"g{n[5:]}" if n[5:].isdigit() else n
    return _GAMMA_IDX.get(n, 0)


def _dirac_adjoint_factor(gamma_name: str) -> int:
    """Γ̄ = factor × Γ, computed via wicklib Gamma.D."""
    idx = _gamma_to_idx(gamma_name)
    return Gamma(idx).D.factor


def _extract_gamma(op_input) -> str:
    """Extract gamma name from a meson operator."""
    tensors = op_input[0].tensors if isinstance(op_input, tuple) else op_input.tensors
    for t in tensors:
        if t.type == "gamma":
            return t.name
    return "I4"


def _meson_sink_block_einsum() -> str:
    """Meson sink block einsum WITH sink gamma insertion.

    Uses: wtzyxjiba, jk -> wtzyxkiba
    Output is a LatticePropagator (compatible with G5-dagger).
    """
    return "wtzyxjiba, jk -> wtzyxkiba"


def _parse_op(op_tuple):
    """Extract (antiquark_flavor, quark_flavor) from a meson operator."""
    op = op_tuple[0] if isinstance(op_tuple, tuple) else op_tuple
    tensors = op.tensors
    return (
        next(t.flavor for t in tensors if t.type == "antiquark"),
        next(t.flavor for t in tensors if t.type == "quark"),
    )

### In fact there are two ways: One is to use _WickCorr, and the other is to hard-code
##
def contract_meson_3pt(sink, source, current) -> dict:
    """Meson 3pt contract with corrected gamma handling.

    Both sink and source gamma matrices are included in the result.
    Sink gamma → sink block: B = S_spectator · Γ_snk
    Source gamma → final contraction: Tr[G̃_seq · Γ_cur · S_fwd · Γ_src†]
    """
    snk_anti, snk_quark = _parse_op(sink)
    src_anti, src_quark = _parse_op(source)
    cur_anti, cur_quark = _parse_op(current)

    # Validate via wicklib
    sr = sink[0] if isinstance(sink, tuple) else sink
    sr2 = source[0] if isinstance(source, tuple) else source
    cr = current[0] if isinstance(current, tuple) else current
    w_snk = _to_wicklib(sr, "x")
    w_src = _to_wicklib(sr2, "y")
    w_cur = _to_wicklib(cr, "z")
    corr = _WickCorr(w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)

    # Gamma info
    sink_gamma = _extract_gamma(sink)
    src_gamma = _extract_gamma(source)

    # Roles: psi(src_anti) × psibar(snk_anti) = spectator
    spectator = src_anti
    forward = cur_quark
    sb = _meson_sink_block_einsum()

    return {
        "sink_terms": [
            {
                "sign": 1,
                "einsum": sb,
                "n_props": 1,
                "var_a": _var(spectator),
                "var_b": None,
                "flavor_a": spectator,
                "flavor_b": None,
                "sink_gamma": sink_gamma,
                "description": f"spectator {spectator}",
            }
        ],
        "fwd_var": _var(forward),
        "fwd_flavor": forward,
        "current_gamma": next(
            t.name for t in cr.tensors if t.type == "gamma"
        ),
        "sink_gamma": sink_gamma,
        "src_gamma": src_gamma,
        "src_gamma_dag_factor": _dirac_adjoint_factor(src_gamma),
        "n_topologies": len(corr.terms),
    }
