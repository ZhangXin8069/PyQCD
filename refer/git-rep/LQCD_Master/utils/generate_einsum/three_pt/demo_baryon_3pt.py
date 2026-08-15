#!/usr/bin/env python3
"""demo_baryon_3pt.py — Baryon 3pt codegen demo.

Uses gen_code(snk, src, cur) from codegen_baryon to generate
complete PyQUDA code (sink block → final contraction).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hadron_operator import baryon_operator, current_operator
from _wick_translate import _to_wicklib
from wicklib.correlator import Correlator
from wicklib.operator import SpinProjector
from codegen_baryon import gen_pyquda_code


def dump_einsum(snk_t, src_t, cur):
    """Print raw to_einsum() output (like dump_einsum.py)."""
    w_snk, si_snk = _to_wicklib(snk_t, "x")
    w_src, si_src = _to_wicklib(src_t, "y")
    w_cur, _ = _to_wicklib(cur[0] if isinstance(cur, tuple) else cur, "z")
    P = SpinProjector.P_plus(si_src, si_snk)
    corr = Correlator(P * w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)
    for term in corr.terms:
        fac, einsum, ops = term.to_einsum()
        einsum = einsum.replace("...", "wtzyx")
        print(f"    sign={fac:+d}")
        print(f"    einsum: {einsum}")
        for i, op in enumerate(ops):
            print(f"      [{i}] {op}")


def demo(label, src_flavors, snk_flavors, f_in, f_out, gamma):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    snk, _ = baryon_operator(*snk_flavors)
    src, _ = baryon_operator(*src_flavors)
    cur = current_operator(f_out, f_in, gamma)

    # ── Raw to_einsum (like dump_einsum.py) ──
    print(f"\n── Raw to_einsum() output ──")
    dump_einsum(snk[0] if isinstance(snk, tuple) else snk,
                src[0] if isinstance(src, tuple) else src, cur)

    # ── Generated code (gen_code) ──
    print(f"\n── gen_code() ──")
    print(gen_pyquda_code(snk, src, cur))


if __name__ == "__main__":
    # ── Octet (Jᴾ = 1/2⁺, diquark = Cγ₅) ──
    demo("proton → proton  (u→u, vector)",
         ("u","d","u"), ("u","d","u"), "u","u","g1")

    demo("proton → proton  (u→u, axial)",
         ("u","d","u"), ("u","d","u"), "u","u","g5")

    demo("Λ → Λ  (s→s, vector)",
         ("u","d","c"), ("u","d","s"),"c","s","g1")

#    demo("Ω⁻ → Ω⁻  (s→s, vector, 18 topol!)",
#         ("s","s","s"), ("s","s","s"), "s","s","g1")

#    demo("Λ꜀ → Λ  (c→s, vector)",
#         ("u","d","c"), ("u","d","s"), "s","c","g1")

#    demo("Λᵦ → Λ  (b→s, vector)",
#         ("u","d","b"), ("u","d","s"), "s","b","g1")

#    demo("Λᵦ → p  (b→u, vector)",
#         ("u","d","b"), ("u","d","u"), "u","b","g1")

    # ── Decuplet → Decuplet (Jᴾ = 3/2⁺, diquark = Cγ₁) ──
#    demo("Ω⁻ → Ω⁻  (s→s, vector, 18 topol!)",
#         ("s","s","s"), ("s","s","s"), "s","s","g1")
