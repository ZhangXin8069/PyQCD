#!/usr/bin/env python3
"""Print raw wicklib operator structure and to_einsum() output for key 3pt cases.
"""
import sys
from pathlib import Path
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from hadron_operator import baryon_operator, current_operator
try:
    from utils.generate_einsum._wick_translate import _to_wicklib
except ImportError:
    from _wick_translate import _to_wicklib
from wicklib.correlator import Correlator
from wicklib.operator import SpinProjector


def _pp_op(name, obj, indent="  "):
    """Pretty-print a wicklib operator/block with field info."""
    print(f"{indent}── {name} ──")
    print(f"{indent}  type: {type(obj).__name__}")

    # Operator with blocks
    if hasattr(obj, "terms"):
        for ti, t in enumerate(obj.terms):
            print(f"{indent}  term[{ti}]: factor={t.factor}")
            for bi, blk in enumerate(t.blocks):
                tp = type(blk).__name__
                loc = getattr(blk, "x", "?")
                if tp == "DiquarkBlock":
                    f = getattr(blk, "f", "?")
                    q = getattr(blk, "q", "?")
                    g = getattr(blk, "gamma", "?")
                    col = getattr(blk, "color", "?")
                    ff = getattr(f, "flavor", "?")
                    qf = getattr(q, "flavor", "?")
                    print(f"{indent}    block[{bi}]: {tp}  x={loc}  "
                          f"f({ff})·{g}·q({qf})  color={col}")
                elif tp == "QuarkBlock":
                    q = getattr(blk, "q", "?")
                    g = getattr(blk, "gamma", "?")
                    sp = getattr(blk, "spin", "?")
                    col = getattr(blk, "color", "?")
                    qf = getattr(q, "flavor", "?")
                    print(f"{indent}    block[{bi}]: {tp}  x={loc}  "
                          f"q({qf})  gamma={g}  spin={sp}  color={col}")
                elif tp == "AntiQuarkBlock":
                    q = getattr(blk, "q", "?")
                    g = getattr(blk, "gamma", "?")
                    sp = getattr(blk, "spin", "?")
                    col = getattr(blk, "color", "?")
                    qf = getattr(q, "flavor", "?")
                    print(f"{indent}    block[{bi}]: {tp}  x={loc}  "
                          f"~q({qf})  gamma={g}  spin={sp}  color={col}")
                elif tp == "AntiDiquarkBlock":
                    f = getattr(blk, "f", "?")
                    q = getattr(blk, "q", "?")
                    g = getattr(blk, "gamma", "?")
                    col = getattr(blk, "color", "?")
                    ff = getattr(f, "flavor", "?")
                    qf = getattr(q, "flavor", "?")
                    print(f"{indent}    block[{bi}]: {tp}  x={loc}  "
                          f"~q({qf})·{g}·~q({ff})  color={col}")
                else:
                    print(f"{indent}    block[{bi}]: {tp}  x={loc}")
        return
    # Single block (QuarkBilinearBlock, SpinProjector, etc.)
    if hasattr(obj, "x") and hasattr(obj, "barred"):
        bf = getattr(getattr(obj, "barred", ""), "flavor", "?")
        uf = getattr(getattr(obj, "unbarred", ""), "flavor", "?")
        g = getattr(obj, "gamma", "?")
        print(f"{indent}  x={getattr(obj,'x','?')}  "
              f"~q({bf})·{g}·q({uf})")
    elif hasattr(obj, "x") and hasattr(obj, "q"):
        q = getattr(obj, "q", "?")
        qf = getattr(q, "flavor", "?")
        g = getattr(obj, "gamma", "?")
        sp = getattr(obj, "spin", "?")
        col = getattr(obj, "color", "?")
        print(f"{indent}  x={getattr(obj,'x','?')}  "
              f"q({qf})  gamma={g}  spin={sp}  color={col}")
    elif hasattr(obj, "factors"):
        print(f"{indent}  spin_snk={getattr(obj,'spin_snk','?')}  "
              f"spin_src={getattr(obj,'spin_src','?')}")


def showeinsum(label, sf, snkf, co, ci):
    cur = current_operator(co, ci, "g1")
    snk_t, _ = baryon_operator(*snkf)
    src_t, _ = baryon_operator(*sf)
    w_snk, si_snk = _to_wicklib(snk_t, "x")
    w_src, si_src = _to_wicklib(src_t, "y")
    w_cur, _ = _to_wicklib(cur[0], "z")
    T = SpinProjector.P_plus(si_src, si_snk)

    print(label)
    _pp_op("w_snk", w_snk)
    _pp_op("w_src", w_src)
    _pp_op("w_src.adj()", w_src.adjoint())
    _pp_op("w_cur", w_cur)
    _pp_op("T (proj)", T)

    corr = Correlator(T * w_snk * w_cur * w_src.adjoint())
    corr.simplify(degenerate=False)
    for ti, term in enumerate(corr.terms):
        fac, einsum, ops = term.to_einsum()
        einsum = einsum.replace("...", "wtzyx")
        print(f"  Term {ti}: factor={fac}  einsum={einsum}  ops={ops}")


if __name__ == "__main__":
    CASES = [
        ("Lambda_c -> Lambda  (c->s)", ("u","d","c"), ("u","d","s"), "s", "c"),
#        ("proton -> proton  (u->u)", ("u","d","u"), ("u","d","u"), "u", "u"),
#        ("Lambda -> proton  (s->u)", ("u","d","s"), ("u","d","u"), "u", "s"),
    ]
    for label, sf, snkf, co, ci in CASES:
        showeinsum(label, sf, snkf, co, ci)
