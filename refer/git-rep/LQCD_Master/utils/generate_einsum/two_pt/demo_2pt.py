#!/usr/bin/env python3
"""
demo_2pt.py — PyQUDA contraction code generation for hadron 2pt functions.

Three steps:
  1. meson_operator / baryon_operator → Tensor[] (operator)
  2. wick_contract → Wick pairing + γ algebra + name substitution → ContractionTerm
  3. Add wtzyx prefix (propagators only) + contract() wrapper → PyQUDA code

Usage: python3 demo_2pt.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import wick_contract_2pt
from codegen import pyquda_format_contract
from hadron_operator import meson_operator, baryon_operator


if __name__ == "__main__":
    # ── Cases list ──
    #
    # Each entry: (label, sink_operator, source_operator)
    #
    #   sink   → first operator,  sits at t   (later time)
    #   source → second operator, sits at t=0 (earlier time)  †
    #
    # The engine computes:  ⟨ sink(t) | source†(0) ⟩
    #   - For mesons:  source† = Dirac adjoint (γ₄ · Γ† · γ₄) applied automatically
    #   - For baryons: source† = Dirac adjoint Ō = O†·γ₄ applied automatically
    #
    # ─────────────────────────────────────────────────────────────────
    #
    # === Meson operator ===
    #
    #   meson_operator(anti_flavor, quark_flavor, gamma):
    #     bar{q}(anti_flavor) · Γ(gamma) · q(quark_flavor)
    #
    #     anti_flavor: flavor of the barred quark  q̄
    #     quark_flavor: flavor of the unbarred quark q
    #     gamma:  "g1" (γ₁, vector), "g5" (γ₅, pseudoscalar), etc.
    #
    #     Example: meson_operator("u","d","g5") = ū·γ₅·d  (pion)
    #              meson_operator("c","c","g1") = c̄·γ₁·c  (J/ψ)
    #
    #   Propagator naming:
    #     "u","d" → prop_l (light, degenerate)
    #     "s"     → prop_s (strange)
    #     "c"     → prop_c (charm)
    #
    # ─────────────────────────────────────────────────────────────────
    #
    # === Baryon operator (Cγ₅ diquark, Jᴾ = 1/2⁺ octet) ===
    #
    #   baryon_operator(a_flavor, b_flavor, c_flavor):
    #     ε^{abc} (q_a^T · Cγ₅ · q_b) · q_c
    #
    #     a,b = diquark (first two flavors)
    #     c   = spectator (last flavor)
    #
    #     Example: baryon_operator("u","d","s") = ε(u·Cγ₅·d)·s  (Λ)
    #
    #   ⚠️  Cγ₅ produces a flavor-antisymmetric diquark. Baryons whose
    #       diquark has identical flavors (uu, dd, ss) vanish algebraically
    #       with this interpolator → labeled "zero" below.  For those, use
    #       a diquark gamma different from Cγ₅ (e.g. Cγ₁ for Δ/Ω/Σ_c).
    #
    #   ⚠️  Λ and Σ⁰ share the same (uds) flavors with Cγ₅ interpolator;
    #       their connected 2pt contractions are identical.
    #
    # ─────────────────────────────────────────────────────────────────
    #
    # Propagator naming:
    #   prop_l = u/d light (degenerate),  prop_s = strange,  prop_c = charm
    #
    cases = [
        # ── Meson 2pt (pseudoscalar) ──
        ("π      π = ū·γ₅·d",               meson_operator("u","d","g5"), meson_operator("u","d","g5")),
#        ("K      K = ū·γ₅·s",               meson_operator("u","s","g5"), meson_operator("u","s","g5")),
#        ("η_s    ηₛ = s̄·γ₅·s",              meson_operator("s","s","g5"), meson_operator("s","s","g5")),
#        ("η_c    η꜀ = c̄·γ₅·c",              meson_operator("c","c","g5"), meson_operator("c","c","g5")),
#        # ── Meson 2pt (vector) ──
        ("ρ      ρ = ū·γ₁·d",               meson_operator("u","d","g1"), meson_operator("u","d","g1")),
        ("η_s    ηₛ = s̄·γ₅·s",              meson_operator("s","s","g5"), meson_operator("s","s","g5")),
#        ("D      D⁺ = c̄·γ₅·d",             meson_operator("c","d","g5"), meson_operator("c","d","g5")),
#        ("D_s    Dₛ⁺ = c̄·γ₅·s",            meson_operator("c","s","g5"), meson_operator("c","s","g5")),
#        ("J/ψ    J/ψ = c̄·γ₁·c",            meson_operator("c","c","g1"), meson_operator("c","c","g1")),
        # ── Baryon 2pt (octet, Cγ₅ diquark, Jᴾ=1/2⁺) ──
        ("p      p = ε(u·Cγ₅·d)·u",         baryon_operator("u","d","u"), baryon_operator("u","d","u")),
#        ("Λ      Λ = ε(u·Cγ₅·d)·s",         baryon_operator("u","d","s"), baryon_operator("u","d","s")),
#        ("Ξ⁻     Ξ⁻ = ε(d·Cγ₅·s)·s",       baryon_operator("d","s","s"), baryon_operator("d","s","s")),
#        ("Σ⁺     Σ⁺ = ε(u·Cγ1·u)·s",  baryon_operator("u","u","s","Cg1"), baryon_operator("u","u","s","Cg1")),
#        ("Σ⁻     Σ⁻ = ε(d·Cγ1·d)·s",  baryon_operator("d","d","s","Cg1"), baryon_operator("d","d","s","Cg1")),
        # ── Charmed baryon 2pt (Cγ₅ diquark) ──
#        ("Λ_c    Λ꜀⁺ = ε(u·Cγ₅·d)·c",      baryon_operator("u","d","c"), baryon_operator("u","d","c")),
        ("Ω_c    Ω꜀⁰ = ε(s·Cγ1·s)·c",  baryon_operator("s","s","c","Cg1"), baryon_operator("s","s","c","Cg1")),
#        ("Ξ_c⁰   Ξ꜀⁰ = ε(d·Cγ₅·s)·c",      baryon_operator("d","s","c"), baryon_operator("d","s","c")),
#        ("Σ_c⁺   Σ꜀⁺ = ε(u·Cγ1·d)·c",     baryon_operator("u","d","c","Cg1"), baryon_operator("u","d","c","Cg1")),
#        ("Σ_c⁰   Σ꜀⁰ = ε(d·Cγ1·d)·c",  baryon_operator("d","d","c","Cg1"), baryon_operator("d","d","c","Cg1")),
#        ("Ω_c    Ω꜀⁰ = ε(s·Cγ1·s)·c",  baryon_operator("s","s","c","Cg1"), baryon_operator("s","s","c","Cg1")),
    ]

    for label, snk, src in cases:
        result = wick_contract_2pt(snk, src, "I")
        kind = "baryon" if result.is_baryon() else "meson"
        print(f"# {'='*60}")
        print(f"# {label}")
        print(f"# {kind}, {len(result.terms)} term(s)")
        print(f"# {'='*60}")
        for i, term in enumerate(result.terms):
            coeff = term.coefficient
            coeff_str = f"{coeff}" if abs(coeff) != 1 else ("" if coeff == 1 else "-")
            print(f"#   term[{i}]: coef={coeff_str}  subs={term.einsum_subs}")
            print(f"#             ops={term.operands}")
            print(pyquda_format_contract(term))
        print()
