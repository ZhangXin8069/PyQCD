#!/usr/bin/env python3
"""demo_3pt.py — Comprehensive 3pt demo (baryon + meson) with codegen output."""
import sys
from pathlib import Path
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path: sys.path.insert(0, _parent)

from hadron_operator import baryon_operator, meson_operator, current_operator
from three_pt import contract_baryon_3pt, contract_meson_3pt
from three_pt.codegen_baryon import gen_seq_source_baryon, gen_final_contract_baryon
from three_pt.codegen_meson import gen_meson_3pt_code


def baryon(label, src_name, snk_name, src_f, snk_f, f_out, f_in, gamma):
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    cur = current_operator(f_out, f_in, gamma)
    snk, _ = baryon_operator(*snk_f)
    src, _ = baryon_operator(*src_f)
    try:
        r = contract_baryon_3pt(snk, src, cur)
        print(f"  Topologies: {r['n_topologies']}")
        for i, s in enumerate(r['sink_terms']):
            print(f"    [{i}] sign={s['sign']:+d}  {s.get('description','')}")
            print(f"        props: {s['var_a']}({s['flavor_a']}), {s['var_b']}({s['flavor_b']})")
            print(f"        einsum: {s['einsum']}")

        print(f"\n-- gen_seq_source_baryon('{src_name}', '{snk_name}') --")
        print(_indent(gen_seq_source_baryon(src_name, snk_name, r), 2))

        print(f"\n-- gen_final_contract_baryon() --")
        print(_indent(gen_final_contract_baryon(r), 2))
    except Exception as e:
        print(f"  FAIL: {e}")


def meson(label, snk_op, src_op, cur, src_name="source", snk_name="sink"):
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    try:
        r = contract_meson_3pt(snk_op, src_op, cur)
        print(f"  Topologies: {r['n_topologies']}")
        print(f"  Current gamma:  {r['current_gamma']}")
        print(f"  Forward:    {r['fwd_flavor']} -> {r['fwd_var']}")
        for i, s in enumerate(r['sink_terms']):
            print(f"    [{i}] sign={s['sign']:+d}  {s.get('description','')}")
            print(f"        einsum: {s['einsum']}")
            print(f"        spectator: {s['flavor_a']} -> {s['var_a']}")

        print(f"\n-- gen_meson_3pt_code() --")
        print(_indent(gen_meson_3pt_code(src_op, snk_op, cur, src_name, snk_name), 2))
    except Exception as e:
        print(f"  FAIL: {e}")


def _indent(text, spaces):
    """Indent each line of text by `spaces` spaces."""
    pad = " " * spaces
    return "\n".join(f"{pad}{line}" for line in text.split("\n"))


if __name__ == "__main__":
    # ---- Baryon 3pt (original set) ----
    baryon("proton -> proton (V)", "p", "p", ("u","d","u"), ("u","d","u"), "u","u","g1")
#    baryon("proton -> proton (A)", "p", "p", ("u","d","u"), ("u","d","u"), "u","u","g5")
    baryon("Lambda -> Lambda (V)", "L","L", ("u","d","s"), ("u","d","s"), "s","s","g1")
#    baryon("Lambda -> Lambda (A)", "L","L", ("u","d","s"), ("u","d","s"), "s","s","g5")
    baryon("Lambda -> proton", "L","p", ("u","d","s"), ("u","d","u"), "u","s","g1")
#    baryon("Lambda_c -> Lambda", "Lc","L", ("u","d","c"), ("u","d","s"), "s","c","g1")
#    baryon("Xi_c -> Xi", "Xc","X", ("d","s","c"), ("d","s","s"), "s","c","g1")
#    baryon("Lambda_b -> Lambda_c", "Lb","Lc", ("u","d","b"), ("u","d","c"), "c","b","g1")
#    baryon("Lambda_b -> Lambda", "Lb","L", ("u","d","b"), ("u","d","s"), "s","b","g1")
#    baryon("Lambda_b -> proton", "Lb","p", ("u","d","b"), ("u","d","u"), "u","b","g1")
#    baryon("Xi_b -> Xi", "Xb","X", ("d","s","b"), ("d","s","s"), "s","b","g1")

#    baryon("Lambda -> proton", "L","p", ("u","d","s"), ("u","d","u"), "u","s","g1g5")
#    baryon("Lambda_c -> Lambda", "Lc","L", ("u","d","c"), ("u","d","s"), "s","c","g1g5")
#    baryon("Xi_c -> Xi", "Xc","X", ("d","s","c"), ("d","s","s"), "s","c","g1g5")
#    baryon("Lambda_b -> Lambda_c", "Lb","Lc", ("u","d","b"), ("u","d","c"), "c","b","g1g5")
#    baryon("Lambda_b -> Lambda", "Lb","L", ("u","d","b"), ("u","d","s"), "s","b","g1g5")
#    baryon("Lambda_b -> proton", "Lb","p", ("u","d","b"), ("u","d","u"), "u","b","g1g5")
#    baryon("Xi_b -> Xi", "Xc","X", ("d","s","b"), ("d","s","s"), "s","b","g1g5")

    # ---- Baryon 3pt (sigma/isospin transitions) ----
   # baryon("Sigma+ -> Sigma+ (V)", "Sp","Sp", ("u","u","s"), ("u","u","s"), "u","u","g1")
   # baryon("Sigma+ -> Lambda (V)", "Sp","L", ("u","u","s"), ("u","d","s"), "d","u","g1")
   # baryon("Sigma0 -> Lambda (V)", "S0","L", ("u","d","s"), ("u","d","s"), "d","d","g1")
   # baryon("Sigma_c++ -> Lambda_c+ (V)", "Scpp","Lc", ("u","u","c"), ("u","d","c"), "d","u","g1")

    # ---- Baryon 3pt (doubly/triply strange) ----
   # baryon("Xi0 -> Xi0 (V)", "X0","X0", ("u","s","s"), ("u","s","s"), "s","s","g1")
   # baryon("Xi0 -> Lambda (V)", "X0","L", ("u","s","s"), ("u","d","s"), "d","s","g1")
   # baryon("Omega- -> Omega- (V)", "Om","Om", ("s","s","s"), ("s","s","s"), "s","s","g1")
   # baryon("Omega- -> Xi0 (V)", "Om","X0", ("s","s","s"), ("u","s","s"), "u","s","g1")

    # ---- Baryon 3pt (double/triple charm) ----
#    baryon("Xi_cc++ -> Xi_c+ (c->d)", "Xcc","Xc", ("u","c","c"), ("u","d","c"), "d","c","g1")
#    baryon("Xi_cc++ -> Xi_c+ (c->d)", "Xcc","Xc", ("u","c","c"), ("u","d","c"), "d","c","g1g5")
   #  baryon("Xi_cc++ -> Lambda_c+ (c->d)", "Xcc","Lc", ("u","c","c"), ("u","d","c"), "d","c","g1")
   # baryon("Omega_ccc++ -> Omega_cc+ (c->s, 6 topol!)", "Occc","Occ", ("c","c","c"), ("c","c","s"), "s","c","g1")
   # baryon("Omega_cc+ -> Omega_c0 (c->s, 4 topol)", "Occ","Oc", ("s","c","c"), ("s","c","s"), "s","c","g1")

    # ---- Meson 3pt (original: light spectator) ----
    meson("D0 -> K+  (c->s, spectator=u)",
          meson_operator('u','s','g5'), meson_operator('u','c','g5'), current_operator('s','c','g1'))
#    meson("D0 -> pi+  (c->d, spectator=u)",
#          meson_operator('u','d','g5'), meson_operator('u','c','g5'), current_operator('d','c','g1'))
#    meson("B+ -> D0  (b->c, spectator=u)",
#          meson_operator('u','c','g5'), meson_operator('u','b','g5'), current_operator('c','b','g1'))
#    meson("B+ -> K+  (b->s, spectator=u)",
#          meson_operator('u','s','g5'), meson_operator('u','b','g5'), current_operator('s','b','g1'))
#    meson("B+ -> pi+  (b->d, spectator=u)",
#          meson_operator('u','d','g5'), meson_operator('u','b','g5'), current_operator('d','b','g1'))
#    meson("D0 -> K*+ (c->s, vector sink, spectator=u)",
#          meson_operator('u','s','g1'), meson_operator('u','c','g5'), current_operator('s','c','g1'))
#    meson("B+ -> K*+ (b->s, vector sink, spectator=u)",
#          meson_operator('u','s','g1'), meson_operator('u','b','g5'), current_operator('s','b','g1'))

    # ---- Meson 3pt (heavy spectator = s) ----
#    meson("Ds+ -> phi  (c->s, spectator=s, 2 topol!)",
#          meson_operator('s','s','g1'), meson_operator('s','c','g5'), current_operator('s','c','g1'))
#    meson("Ds+ -> Ds+ EM (c->c, spectator=s, 2 topol!)",
#          meson_operator('s','c','g5'), meson_operator('s','c','g5'), current_operator('c','c','g1'))
#    meson("Bs0 -> Ds-  (b->c, spectator=s, 1 topol)",
#          meson_operator('s','c','g5'), meson_operator('s','b','g5'), current_operator('c','b','g1'))
#    meson("K+ -> pi+  (s->d, spectator=u)",
#          meson_operator('u','d','g5'), meson_operator('u','s','g5'), current_operator('d','s','g1'))
#    meson("D0 -> eta (c->u, spectator=u, 2 topol)",
#          meson_operator('u','u','g5'), meson_operator('u','c','g5'), current_operator('u','c','g1'))

    # ---- Meson 3pt (Bc -> vector, vec source) ----
#    meson("Bc+ -> J/psi (b->c, spectator=c, ps->ps)",
#          meson_operator('c','c','g1'), meson_operator('c','b','g5'), current_operator('c','b','g1'))
