#!/usr/bin/env python3
"""dump_wick.py — Print raw wicklib contraction results for all baryon 3pt cases.

Usage:
    cd QCD_Master/utils/generate_einsum
    python3 three_pt/dump_wick.py
"""
import sys
from pathlib import Path
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from hadron_operator import baryon_operator, current_operator
from three_pt.contract import _to_wicklib, _extract_baryon_flavors, _ROW_SNK, _COL_SRC
from wicklib.correlator import Correlator
from wicklib.operator import SpinProjector


def dump(label, src_f, snk_f, cur_out, cur_in, gamma="g1"):
    """Build wicklib correlator and print all contraction terms."""
    cur = current_operator(cur_out, cur_in, gamma)
    snk, _ = baryon_operator(*snk_f)
    src, _ = baryon_operator(*src_f)

    src_flav = _extract_baryon_flavors(src)
    snk_flav = _extract_baryon_flavors(snk)

    sr = snk[0] if isinstance(snk, tuple) else snk
    sr2 = src[0] if isinstance(src, tuple) else src
    cr = cur[0] if isinstance(cur, tuple) else cur

    w_snk, si_snk = _to_wicklib(sr, "x")
    w_src, si_src = _to_wicklib(sr2, "y")
    w_cur = _to_wicklib(cr, "z")

    T = SpinProjector.P_plus(si_src, si_snk)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"  Source flavors: {src_flav}")
    print(f"  Sink flavors:   {snk_flav}")
    print(f"  Current: {cur_in} -> {cur_out}  gamma={gamma}")
    print(f"  Row -> slot: {_ROW_SNK}")
    print(f"  Col -> slot: {_COL_SRC}")

    try:
        corr = Correlator(T * w_snk * w_cur * w_src.adjoint())
        corr.simplify(degenerate=False)
    except ValueError as e:
        if "unmatched quark" in str(e):
            print(f"  Wicklib: No valid Wick contractions (unmatched quark)")
        else:
            print(f"  Wicklib error: {e}")
        return

    print(f"  Wicklib terms: {len(corr.terms)}")
    print(sep)

    for ti, term in enumerate(corr.terms):
        fac = term.factor
        print(f"\n  Term {ti}: factor = {fac}  (real={round(fac.real):+d})")
        print(f"    Adjacency matrix ({len(term.matrix)}x{len(term.matrix[0])}):")
        print(f"                   col 0(a)  col 1(b)  col 2(c)")
        for ri in range(len(term.matrix)):
            rl = _ROW_SNK.get(ri, f"r{ri}")
            entries = []
            for ci in range(len(term.matrix[ri])):
                e = term.matrix[ri][ci]
                if e.flavor is None:
                    entries.append("   --   ")
                else:
                    z = "z" if "z" in (e.sink, e.source) else " "
                    entries.append(f"  {e.flavor}{z}  ")
            print(f"    row {ri}({rl}):  {''.join(entries)}")

        # Build connection topology
        cur_src = None
        cur_snk = None
        conn = {}
        us = set()
        us2 = set()

        for ri in range(len(term.matrix)):
            for ci in range(len(term.matrix[ri])):
                e = term.matrix[ri][ci]
                if e.flavor is None:
                    continue
                is_cur = "z" in (e.sink, e.source)
                if ri == 0:
                    # Row 0: current source side (or source for non-current entries)
                    cur_src = _COL_SRC.get(ci)
                    if cur_src is None:
                        for s in ["a","b","c"]:
                            if src_flav[s] == e.flavor:
                                cur_src = s
                                break
                    us.add(cur_src)
                elif is_cur:
                    cur_snk = _ROW_SNK[ri]
                    us2.add(cur_snk)
                else:
                    ss = _ROW_SNK[ri]
                    sc = _COL_SRC[ci]
                    conn[sc] = (ss, False)
                    us.add(sc)
                    us2.add(ss)

        if cur_snk is None:
            print(f"    -> Filtered: current sink not found in rows 1-3")
            continue

        conn[cur_src] = (cur_snk, True)

        # Fill missing identities
        for s in ["a","b","c"]:
            if s not in us:
                for t in ["a","b","c"]:
                    if t not in us2:
                        conn[s] = (t, False)
                        us.add(s)
                        us2.add(t)
                        break

        if len(conn) != 3:
            print(f"    -> Filtered: incomplete connections (got {len(conn)})")
            continue

        # Print connections
        desc_parts = []
        for s in ["a","b","c"]:
            t, thru = conn[s]
            f = src_flav[s]
            sf = snk_flav[t]
            arrow = "[J]" if thru else "---"
            desc_parts.append(f"src.{s}({f})--{arrow}->snk.{t}({sf})")
        print(f"    Connections: {', '.join(desc_parts)}")


if __name__ == "__main__":
    CASES = [
        ("p -> p  (u->u)",         ("u","d","u"), ("u","d","u"), "u", "u"),
        ("Lambda -> Lambda  (s->s)", ("u","d","s"), ("u","d","s"), "s", "s"),
        ("Lambda -> p  (s->u)",    ("u","d","s"), ("u","d","u"), "u", "s"),
        ("Xi -> Lambda  (s->u)",    ("u","d","s"), ("d","s","s"), "u", "s"),
    ]
    for label, sf, snkf, co, ci in CASES:
        dump(label, sf, snkf, co, ci)
