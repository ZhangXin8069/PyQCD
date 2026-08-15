"""sink_block — Wicklib-driven baryon 3pt sink block einsum generation.

Builds the sink block einsum string and propagator variable assignments
directly from the wicklib adjacency matrix — no hardcoded slot-level
if-else.
"""

from typing import Dict, List, Tuple

# ── Slot-level spin/color index mapping ──────────────────────────────
SRC_SPIN = {"a": "A", "b": "B", "c": "D"}
SNK_SPIN = {"a": "G", "b": "H", "c": "I"}
SRC_COL = {"a": "i", "b": "j", "c": "k"}
SNK_COL = {"a": "l", "b": "m", "c": "n"}


def _prop_einsum(src_slot: str, snk_slot: str) -> str:
    spin = f"{SRC_SPIN[src_slot]}{SNK_SPIN[snk_slot]}"
    return f"wtzyx{spin}{SRC_COL[src_slot]}{SNK_COL[snk_slot]}"


def _output_einsum(curr_src: str, curr_snk: str) -> str:
    return (f"{SNK_SPIN[curr_snk]}{SRC_SPIN[curr_src]}"
            f"{SNK_COL[curr_snk]}{SRC_COL[curr_src]}")


def _var(flavor: str) -> str:
    _MASS = {"u": "prop_l", "d": "prop_l", "s": "prop_s",
             "c": "prop_c", "b": "prop_b"}
    return _MASS.get(flavor, f"prop_{flavor}")


class SinkBlockTopology:
    """One baryon 3pt sink-block contraction topology."""

    def __init__(self, *, einsum: str, sign: int,
                 var_a: str, var_b: str,
                 flavor_a: str, flavor_b: str,
                 fwd_var: str, fwd_flavor: str,
                 curr_src: str, curr_snk: str):
        self.einsum = einsum
        self.sign = sign
        self.var_a = var_a
        self.var_b = var_b
        self.flavor_a = flavor_a
        self.flavor_b = flavor_b
        self.fwd_var = fwd_var
        self.fwd_flavor = fwd_flavor
        self.curr_src = curr_src
        self.curr_snk = curr_snk


# ── Matrix parsing ───────────────────────────────────────────────────

_ROW_SNK = {1: "c", 2: "b", 3: "a"}
_COL_SRC = {0: "a", 1: "b", 2: "c"}


def _parse_adjacency(term, src_flavors):
    """Parse wicklib 4x4 adjacency matrix into conn + current slots.

    Returns (conn, curr_src, curr_snk) or None.

    Row 0 = current vertex (coordinate z).
    Rows 1-3 = three quark lines (sink side), mapped via _ROW_SNK.
    Cols 0-2 = three source slots, mapped via _COL_SRC.
    Col 3 = current vertex (coordinate z) on sink side.
    """
    conn: Dict[str, Tuple[str, bool]] = {}
    seen_src: set = set()
    seen_snk: set = set()
    curr_src = curr_snk = None

    for ri in range(len(term.matrix)):
        for ci in range(len(term.matrix[ri])):
            e = term.matrix[ri][ci]
            if e.flavor is None:
                continue
            is_cur = "z" in (e.sink, e.source)

            # ── Row 0: current vertex (source side) ──
            if ri == 0:
                if not is_cur:
                    continue
                src_slot = _COL_SRC.get(ci)
                if src_slot is None:
                    for s in ["a", "b", "c"]:
                        if s not in seen_src and src_flavors.get(s) == e.flavor:
                            src_slot = s
                            break
                if src_slot is not None:
                    curr_src = src_slot
                    seen_src.add(src_slot)
                continue

            # ── Rows 1-3: quark lines ──
            snk_slot = _ROW_SNK[ri]

            if is_cur:
                # Sink-side current — only record curr_snk, do NOT
                # create a conn entry (the source-side current row
                # already identified curr_src).
                curr_snk = snk_slot
                seen_snk.add(snk_slot)
                continue

            # Spectator connection
            src_slot = _COL_SRC.get(ci)
            if src_slot is None:
                for s in ["a", "b", "c"]:
                    if s not in seen_src and src_flavors.get(s) == e.flavor:
                        src_slot = s
                        break
            if src_slot is None:
                continue

            conn[src_slot] = (snk_slot, False)
            seen_src.add(src_slot)
            seen_snk.add(snk_slot)

    if curr_snk is None or curr_src is None:
        return None

    # Add the current-carrying line as a conn entry
    conn[curr_src] = (curr_snk, True)
    seen_src.add(curr_src)
    seen_snk.add(curr_snk)

    # ── Fill missing identity connections ──
    for s in ["a", "b", "c"]:
        if s not in seen_src:
            for t in ["a", "b", "c"]:
                if t not in seen_snk:
                    conn[s] = (t, False)
                    seen_src.add(s)
                    seen_snk.add(t)
                    break

    return conn if len(conn) == 3 else None, curr_src, curr_snk


# ── Einsum builder ───────────────────────────────────────────────────

def _topology_from_conn(conn, curr_src, curr_snk,
                        src_flavors, snk_flavors,
                        proj_idx: str, raw_sign: int):
    """Build one SinkBlockTopology."""
    spectators: List[Tuple[str, str]] = []

    for src_slot, (snk_slot, is_cur) in conn.items():
        if is_cur:
            continue  # skip — it's the forward line
        spectators.append((src_slot, snk_slot))

    if len(spectators) != 2:
        return None

    # Build the 8-operand einsum
    op1_einsum = _prop_einsum(spectators[0][0], spectators[0][1])
    op2_einsum = _prop_einsum(spectators[1][0], spectators[1][1])
    out_einsum = _output_einsum(curr_src, curr_snk)

    einsum_str = (
        f"wtzyx, ijk, lmn, AB, GH, {proj_idx},"
        f" {op1_einsum}, {op2_einsum} -> wtzyx{out_einsum}"
    )

    # Propagator variable names (alphabetically sorted by flavor for determinism)
    fa = src_flavors[spectators[0][0]]
    fb = src_flavors[spectators[1][0]]
    fwd_flavor = src_flavors[curr_src]

    if fa > fb or (fa == fb and spectators[0][0] > spectators[1][0]):
        spectators[0], spectators[1] = spectators[1], spectators[0]
        fa, fb = fb, fa

    return SinkBlockTopology(
        einsum=einsum_str,
        sign=-raw_sign,
        var_a=_var(fa), var_b=_var(fb),
        flavor_a=fa, flavor_b=fb,
        fwd_var=_var(fwd_flavor), fwd_flavor=fwd_flavor,
        curr_src=curr_src, curr_snk=curr_snk,
    )


# ── Main entry point ─────────────────────────────────────────────────

def build_sink_block_topologies(
    corr_terms: list,
    src_flavors: Dict[str, str],
    snk_flavors: Dict[str, str],
    projector: str = "Pplus",
) -> List[SinkBlockTopology]:
    """Generate one SinkBlockTopology per wicklib term.

    Parameters
    ----------
    corr_terms : list
        ``corr.terms`` from ``wicklib.correlator.Correlator``.
    src_flavors, snk_flavors : dict
        ``{slot: flavor}`` maps from :func:`_extract_baryon_flavors`.
    projector : str
        Short name (default ``Pplus``).

    Returns
    -------
    list of SinkBlockTopology
    """
    proj_short = {"Pplus": "ID", "P+": "ID", "Pminus": "ID", "P-": "ID"}
    proj_idx = proj_short.get(projector, projector)

    topologies: List[SinkBlockTopology] = []

    for term in corr_terms:
        parsed = _parse_adjacency(term, src_flavors)
        if parsed is None:
            continue
        conn, curr_src, curr_snk = parsed
        topo = _topology_from_conn(conn, curr_src, curr_snk,
                                   src_flavors, snk_flavors,
                                   proj_idx, round(term.factor.real))
        if topo is not None:
            topologies.append(topo)

    return topologies
