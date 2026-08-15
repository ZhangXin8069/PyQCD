#!/usr/bin/env python3
"""
demo_multi_2pt.py — PyQUDA contraction code generation for multi-meson 2pt functions.

Multi-meson operators: products of N meson operators at the same point.
The wicklib engine automatically enumerates all N! topological contractions
from identical quarks (e.g., 3! = 6 for 3 charm anti-quarks).

Usage: python3 demo_multi_2pt.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_meson import multi_meson_operator, wick_contract_multi_2pt, pyquda_format_contract

if __name__ == "__main__":
    cases = [
        # ── 2 mesons ──
        (
            "π⁺·π⁺   ū·γ₅·d  ·  ū·γ₅·d",
            multi_meson_operator(("u","d","g5"), ("u","d","g5")),
            multi_meson_operator(("u","d","g5"), ("u","d","g5")),
        ),
        (
            "π⁺·π⁻   ū·γ₅·d  ·  d̄·γ₅·u",
            multi_meson_operator(("u","d","g5"), ("d","u","g5")),
            multi_meson_operator(("u","d","g5"), ("d","u","g5")),
        ),
        (
            "K⁺·K⁻   ū·γ₅·s  ·  s̄·γ₅·u",
            multi_meson_operator(("u","s","g5"), ("s","u","g5")),
            multi_meson_operator(("u","s","g5"), ("s","u","g5")),
        ),
        (
            "D⁰·D⁰   c̄·γ₅·u  ·  c̄·γ₅·d",
            multi_meson_operator(("c","u","g5"), ("c","d","g5")),
            multi_meson_operator(("c","u","g5"), ("c","d","g5")),
        ),
        (
            "D⁺·D⁻   c̄·γ₅·d  ·  d̄·γ₅·c",
            multi_meson_operator(("c","d","g5"), ("d","c","g5")),
            multi_meson_operator(("c","d","g5"), ("d","c","g5")),
        ),
        # ── 3 mesons ──
        (
            "π⁺·π⁺·π⁻   ūγ₅d · ūγ₅d · d̄γ₅u",
            multi_meson_operator(("u","d","g5"), ("u","d","g5"), ("d","u","g5")),
            multi_meson_operator(("u","d","g5"), ("u","d","g5"), ("d","u","g5")),
        ),
        (
            "D⁰·D⁰·D_s⁺   c̄γ₅u · c̄γ₅d · c̄γ₅s",
            multi_meson_operator(("c","u","g5"), ("c","d","g5"), ("c","s","g5")),
            multi_meson_operator(("c","u","g5"), ("c","d","g5"), ("c","s","g5")),
        ),
        # ── 4 mesons: 4-quark operator (tetraquark) ──
        (
            "cc̄cc̄ (4 charm)   c̄γ₅c · c̄γ₅c · c̄γ₅c · c̄γ₅c",
            multi_meson_operator(("c","c","g5"), ("c","c","g5"),
                                 ("c","c","g5"), ("c","c","g5")),
            multi_meson_operator(("c","c","g5"), ("c","c","g5"),
                                 ("c","c","g5"), ("c","c","g5")),
        ),
    ]

    for label, snk, src in cases:
        result = wick_contract_multi_2pt(snk, src)
        n_m = result.n_mesons
        n_t = len(result.terms)
        print(f"# {'='*65}")
        print(f"# {label}")
        print(f"# {n_m} meson(s), {n_t} topology(ies)  "
              f"({'identical quarks' if n_t > 1 else 'all distinct'})")
        print(f"# {'='*65}")
        for i, term in enumerate(result.terms):
            coeff = term.coefficient
            coeff_str = f"{coeff}" if abs(coeff) != 1 else ("" if coeff == 1 else "-")
            print(f"#   term[{i}]: coef={coeff_str}  subs={term.einsum_subs}")
            print(f"#             ops={term.operands}")
            print(pyquda_format_contract(term))
        print()
