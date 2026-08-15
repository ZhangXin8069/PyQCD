#!/usr/bin/env python3
"""audit_3pt.py — Comprehensive audit of the OLD sink_block_einsum() logic.

Compares old `contract_baryon_3pt()` output against direct wicklib analysis.
"""
import sys, itertools
from pathlib import Path

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from hadron_operator import baryon_operator, current_operator
from three_pt import contract_baryon_3pt as old_contract
from three_pt.contract import (
    sink_block_einsum, _extract_baryon_flavors, _ROW_SNK, _COL_SRC,
    _to_wicklib, from_current_operator, ContractionTopology,
)
from wicklib.correlator import Correlator as _WickCorr
from wicklib.operator import SpinProjector

SRC_SPIN = {"a": "A", "b": "B", "c": "D"}
SNK_SPIN = {"a": "G", "b": "H", "c": "I"}
SRC_COL = {"a": "i", "b": "j", "c": "k"}
SNK_COL = {"a": "l", "b": "m", "c": "n"}
_LIGHT = frozenset({"u", "d"})


def var(f):
    return "prop_l" if f in _LIGHT else f"prop_{f}"


def prop_label(s, t):
    if s == "c" and t == "c":
        spin = f"{SNK_SPIN[t]}{SRC_SPIN[s]}"
    else:
        spin = f"{SRC_SPIN[s]}{SNK_SPIN[t]}"
    return f"wtzyx{spin}{SRC_COL[s]}{SNK_COL[t]}"


def free_label(s, t):
    return f"{SNK_SPIN[t]}{SRC_SPIN[s]}{SNK_COL[t]}{SRC_COL[s]}"


# ═══════════════════════════════════════════════════════════════════════
# Audit: use wicklib to enumerate topologies, then check old code
# ═══════════════════════════════════════════════════════════════════════

def audit_case(label, src_f, snk_f, cur_out, cur_in, gamma="g1"):
    cur = current_operator(cur_out, cur_in, gamma)
    snk, _ = baryon_operator(*snk_f)
    src, _ = baryon_operator(*src_f)

    src_flav = _extract_baryon_flavors(src)
    snk_flav = _extract_baryon_flavors(snk)
    cur_info = from_current_operator(cur)

    # ── Build wicklib operators ──
    sr = snk[0] if isinstance(snk, tuple) else snk
    sr2 = source_op = src[0] if isinstance(src, tuple) else src
    cr = cur[0] if isinstance(cur, tuple) else cur

    w_snk, si_snk = _to_wicklib(sr, "x")
    w_src, si_src = _to_wicklib(sr2, "y")
    w_cur = _to_wicklib(cr, "z")

    T = SpinProjector.P_plus(si_src, si_snk)
    corr = _WickCorr(T * w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)

    # ── Old code topologies ──
    old_r = old_contract(snk, src, cur)
    old_terms = old_r["sink_terms"]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Src flavors: {src_flav}  Snk flavors: {snk_flav}")
    print(f"  Current: {cur_info.flavor_in} -> {cur_info.flavor_out}")
    print(f"{'='*70}")

    if len(corr.terms) != len(old_terms):
        print(f"  ⚠  TOPO COUNT MISMATCH: wicklib={len(corr.terms)} old={len(old_terms)}")
        return

    # ── For each wicklib term, reconstruct what the old code should produce ──
    for ti, (term, ot) in enumerate(zip(corr.terms, old_terms)):
        # Reconstruct the connection topology from wicklib adj matrix
        cur_src = cur_snk = None
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
                    cur_src = _COL_SRC.get(ci)
                    if cur_src is None:
                        for s in ["a","b","c"]:
                            if src_flav[s] == e.flavor:
                                cur_src = s; break
                    us.add(cur_src)
                elif is_cur:
                    cur_snk = _ROW_SNK[ri]
                    us2.add(cur_snk)
                else:
                    ss = _ROW_SNK[ri]
                    sc = _COL_SRC[ci]
                    conn[sc] = (ss, False)
                    us.add(sc); us2.add(ss)
        if cur_snk is None:
            continue
        conn[cur_src] = (cur_snk, True)
        # fill missing identities
        for s in ["a","b","c"]:
            if s not in us:
                for t in ["a","b","c"]:
                    if t not in us2:
                        conn[s] = (t, False)
                        us.add(s); us2.add(t); break
        if len(conn) != 3:
            continue

        # Re-run the old sink_block_einsum with this topology
        desc = ", ".join(
            f"{src_flav[s]}(src.{s})->[J]->{snk_flav[t]}(snk.{t})"
            if thru else
            f"{src_flav[s]}(src.{s})->{snk_flav[t]}(snk.{t})"
            for s,(t,thru) in sorted(conn.items())
        )
        topo = ContractionTopology(f"topo_{ti}", round(term.factor.real), conn, desc)
        expected = sink_block_einsum(src, snk, topo)  # what old code SHOULD produce

        # Compare
        e_match = (expected["einsum"] == ot["einsum"])
        s_match = (expected["sign"] == ot["sign"])
        v_match = (expected["var_a"] == ot["var_a"]
                   and expected["var_b"] == ot["var_b"])
        f_match = (expected["flavor_a"] == ot["flavor_a"]
                   and expected["flavor_b"] == ot["flavor_b"])

        status = "✅" if (e_match and s_match and v_match) else "❌"
        print(f"\n  {status} Topo {ti}: cur ({cur_src}->{cur_snk})  "
              f"perm={round(term.factor.real):+d}")
        print(f"    {desc}")

        if not e_match:
            print(f"    EINSUM MISMATCH:")
            print(f"      produced: {ot['einsum']}")
            print(f"      expected: {expected['einsum']}")
        if not s_match:
            print(f"    SIGN MISMATCH: old={ot['sign']:+d} expected={expected['sign']:+d}")
        if not v_match:
            print(f"    VAR MISMATCH:  old_va={ot['var_a']} exp={expected['var_a']}"
                  f"  old_vb={ot['var_b']} exp={expected['var_b']}")
        if not f_match:
            print(f"    FLAVOR MISMATCH: old_fa={ot['flavor_a']} exp={expected['flavor_a']}"
                  f"  old_fb={ot['flavor_b']} exp={expected['flavor_b']}")


if __name__ == "__main__":
    CASES = [
        ("p → p  (u→u)", ("u","d","u"), ("u","d","u"), "u", "u"),
        ("p → p  (d→d)", ("u","d","u"), ("u","d","u"), "d", "d"),
        ("Λ → Λ  (s→s)", ("u","d","s"), ("u","d","s"), "s", "s"),
        ("Λ → p   (s→u)", ("u","d","s"), ("u","d","u"), "u", "s"),
        ("Λc→Λ   (c→s)", ("u","d","c"), ("u","d","s"), "s", "c"),
        ("Λc→p   (c→u)", ("u","d","c"), ("u","d","u"), "u", "c"),
        ("Ξc→Ξ   (c→s)", ("d","s","c"), ("d","s","s"), "s", "c"),
        ("Λb→Λc  (b→c)", ("u","d","b"), ("u","d","c"), "c", "b"),
        ("Λb→Λ   (b→s)", ("u","d","b"), ("u","d","s"), "s", "b"),
        ("Λb→p   (b→u)", ("u","d","b"), ("u","d","u"), "u", "b"),
        ("Ξb→Ξ   (b→s)", ("d","s","b"), ("d","s","s"), "s", "b"),
        ("Σ+→Σ+ (u→u)",  ("u","u","s"), ("u","u","s"), "u", "u"),
        ("Σ+→Λ   (d→u)", ("u","u","s"), ("u","d","s"), "d", "u"),
        ("Ξ0→Ξ0  (s→s)", ("u","s","s"), ("u","s","s"), "s", "s"),
        ("Ξ0→Λ   (d→s)", ("u","s","s"), ("u","d","s"), "d", "s"),
        ("Ω−→Ω−  (s→s)", ("s","s","s"), ("s","s","s"), "s", "s"),
        ("Ω−→Ξ0  (u→s)", ("s","s","s"), ("u","s","s"), "u", "s"),
        ("Ωcc→Ωc (s→c)", ("s","c","c"), ("s","c","s"), "s", "c"),
        ("Ωccc→Ωcc(c→s)", ("c","c","c"), ("c","c","s"), "s", "c"),
    ]
    for label, sf, snkf, co, ci in CASES:
        audit_case(label, sf, snkf, co, ci)
