#!/usr/bin/env python3
"""demo_meson_3pt.py — Meson 3pt codegen demo using codegen_meson.

Generates clean PyQUDA code for various meson 3pt processes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hadron_operator import meson_operator, current_operator
from codegen_meson import gen_meson_3pt_code
from codegen_meson_backup import gen_pyquda_code


def show(label, snk, src, cur, src_name="src", snk_name="snk"):
    """Generate meson 3pt code for one case."""
    print(f"\n{'#' * 72}")
    print(f"#  {label}")
    print(f"{'#' * 72}")

    code = gen_meson_3pt_code(snk, src, cur, src_name=src_name, snk_name=snk_name)
    print(code)


# ═══════════════════════════════════════════════════════════════════════
#  All test cases
# ═══════════════════════════════════════════════════════════════════════


CASES2 = [
    # ─── Light spectator (u) ───
    #   Source  Sink   current   transition
    ("D0 -> K+  (c->s, u spectator)",
     meson_operator('u','s','g5'), meson_operator('u','c','g5'),
     current_operator('s','c','g1'),
     "D0", "K+")
]    
CASES = [
    # ─── Light spectator (u) ───
    #   Source  Sink   current   transition
    ("D0 -> K+  (c->s, u spectator)",
     meson_operator('u','s','g5'), meson_operator('u','c','g5'),
     current_operator('s','c','g1'),
     "D0", "K+"),

    ("D0 -> pi+  (c->d, u spectator)",
     meson_operator('u','d','g5'), meson_operator('u','c','g5'),
     current_operator('d','c','g1'),
     "D0", "pi+"),

    ("B+ -> D0  (b->c, u spectator)",
     meson_operator('u','c','g5'), meson_operator('u','b','g5'),
     current_operator('c','b','g1'),
     "B+", "D0"),

    ("B+ -> K+  (b->s, u spectator)",
     meson_operator('u','s','g5'), meson_operator('u','b','g5'),
     current_operator('s','b','g1'),
     "B+", "K+"),

    ("B+ -> pi+  (b->d, u spectator)",
     meson_operator('u','d','g5'), meson_operator('u','b','g5'),
     current_operator('d','b','g1'),
     "B+", "pi+"),

    # ─── Light spectator, vector sink ───
    ("D0 -> K*+ (c->s, u spectator, vector sink)",
     meson_operator('u','s','g1'), meson_operator('u','c','g5'),
     current_operator('s','c','g1'),
     "D0", "K*+"),

    ("B+ -> K*+ (b->s, u spectator, vector sink)",
     meson_operator('u','s','g1'), meson_operator('u','b','g5'),
     current_operator('s','b','g1'),
     "B+", "K*+"),

    # ─── Heavy spectator (s) ───
    ("Ds+ -> phi  (c->s, s spectator, 2 topol!)",
     meson_operator('s','s','g1'), meson_operator('s','c','g5'),
     current_operator('s','c','g1'),
     "Ds+", "phi"),

    ("Ds+ -> Ds+ EM (c->c, s spectator, 2 topol!)",
     meson_operator('s','c','g5'), meson_operator('s','c','g5'),
     current_operator('c','c','g1'),
     "Ds+", "Ds+"),

    ("Bs0 -> Ds-  (b->c, s spectator, 1 topol)",
     meson_operator('s','c','g5'), meson_operator('s','b','g5'),
     current_operator('c','b','g1'),
     "Bs0", "Ds-"),

    ("K+ -> pi+  (s->d, u spectator)",
     meson_operator('u','d','g5'), meson_operator('u','s','g5'),
     current_operator('d','s','g1'),
     "K+", "pi+"),

    # ─── Heavy spectator (c) ───
    ("Bc+ -> J/psi (b->c, c spectator, 2 topol!)",
     meson_operator('c','c','g1'), meson_operator('c','b','g5'),
     current_operator('c','b','g1'),
     "Bc+", "J/psi"),

    # ─── Various gamma structures ───
    ("D0 -> K+  (c->s, axial current: g1g5)",
     meson_operator('u','s','g5'), meson_operator('u','c','g5'),
     current_operator('s','c','g1g5'),
     "D0", "K+"),

    ("D0 -> K+ (c->s, tensor current: gtg5)",
     meson_operator('u','s','g5'), meson_operator('u','c','g5'),
     current_operator('s','c','gtg5'),
     "D0", "K+"),
]

def show_backup(label, snk, src, cur):
    """Generate meson 3pt code using backup (baryon-style) codegen."""
    print(f"\n{'#' * 72}")
    print(f"#  {label}  [backup]")
    print(f"{'#' * 72}")

    code = gen_pyquda_code(snk, src, cur)
    print(code)


if __name__ == "__main__":
    print("=" * 72)
    print("  Original codegen_meson")
    print("=" * 72)
    for label, snk, src, cur, src_name, snk_name in CASES2:
        show(label, snk, src, cur, src_name=src_name, snk_name=snk_name)

    print("\n" + "=" * 72)
    print("  Backup codegen (baryon-style interface)")
    print("=" * 72)
    for label, snk, src, cur, src_name, snk_name in CASES2:
        show_backup(label, snk, src, cur)
