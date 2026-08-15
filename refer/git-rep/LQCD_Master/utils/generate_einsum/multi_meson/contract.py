"""contract — Multi-meson 2-point Wick contraction engine.

Converts a multi-meson operator (product of N mesons) to wicklib Blocks,
contracts with the adjoint, and extracts contraction terms.

Key difference from single-meson: N! topological contractions from
identical heavy quarks. The wicklib engine handles this automatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    from .._wick_translate import _rename_op
    from ..wicklib.quark import QuarkField
    from ..wicklib.gamma import Gamma, GAMMA_5, GAMMA_0, C
    from ..wicklib.operator import QuarkBilinear, AntiQuark, Quark
    from ..wicklib.index import ColorIndex, SpinIndex, SpinEinsumIndex, ColorEinsumIndex
    from ..wicklib.correlator import Correlator as _WickCorr
except ImportError:
    import sys
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    from _wick_translate import _rename_op
    from wicklib.quark import QuarkField
    from wicklib.gamma import Gamma, GAMMA_5, GAMMA_0, C
    from wicklib.operator import QuarkBilinear, AntiQuark, Quark
    from wicklib.index import ColorIndex, SpinIndex, SpinEinsumIndex, ColorEinsumIndex
    from wicklib.correlator import Correlator as _WickCorr

# Mapping from tensor gamma names to wicklib Gamma objects
_GAMMA_MAP = {
    "g1": Gamma(1),
    "g2": Gamma(2),
    "g3": Gamma(4),
    "g4": Gamma(8),
    "g5": GAMMA_5,
    "G5": GAMMA_5,
    "gtg5": Gamma(7),
    "I4": Gamma(0),
}

# ======================================================================
# Data structures
# ======================================================================

@dataclass
class Term:
    """A single contraction term from multi-meson 2pt."""
    coefficient: complex
    einsum_subs: str
    operands: List[str]


@dataclass
class Result:
    """Full multi-meson 2pt contraction result."""
    terms: List[Term]
    n_mesons: int
    has_identical_quarks: bool  # True if any flavor appears >1 in sink/source


# ======================================================================
# Operator → wicklib Block conversion
# ======================================================================

def _to_wicklib_multi(op, location: str = "x"):
    """Convert a multi-meson Operator → wicklib Operator (multiple QuarkBilinearBlocks).

    Groups tensors into (antiquark, gamma, quark) triplets,
    ignoring delta tensors (color trace is handled by wicklib).

    Each triplet becomes a QuarkBilinearBlock; all blocks are
    multiplied together via wicklib's __mul__.
    """
    tensors = op.tensors
    n_mesons = len(tensors) // 4  # 4 tensors per meson

    result = None
    for i in range(n_mesons):
        base = i * 4
        anti_t = tensors[base]
        gamma_t = tensors[base + 1]
        quark_t = tensors[base + 2]
        # tensors[base + 3] is delta — ignored (wicklib handles color)

        gamma_obj = _GAMMA_MAP.get(gamma_t.name, GAMMA_5)
        block = QuarkBilinear(
            barred=QuarkField(anti_t.flavor),
            unbarred=QuarkField(quark_t.flavor),
            gamma=gamma_obj,
        ).at(location)

        if result is None:
            result = block
        else:
            result = result * block

    return result


def _split_groups(tensors, n_per_group=4):
    """Split tensor list into groups of n_per_group."""
    return [tensors[i:i + n_per_group] for i in range(0, len(tensors), n_per_group)]


# ======================================================================
# Contraction entry point
# ======================================================================

def wick_contract_multi_2pt(sink_op, source_op) -> Result:
    """⟨ multi-meson-sink(x) · multi-meson-source†(y) ⟩ → Result.

    Parameters
    ----------
    sink_op : Operator
        Multi-meson sink operator (from multi_meson_operator).
    source_op : Operator
        Multi-meson source operator (from multi_meson_operator), same
        flavor content as sink (for connected 2pt).

    Returns
    -------
    Result
        List of contraction terms with einsum subscripts and
        PyQUDA variable names.
    """
    n_mesons = len(sink_op.tensors) // 4

    # Convert to wicklib Blocks
    w_snk = _to_wicklib_multi(sink_op, "x")
    w_src = _to_wicklib_multi(source_op, "y")

    # Contract: ⟨ O_snk · O_src† ⟩
    op = w_snk * w_src.adjoint()
    corr = _WickCorr(op)
    corr.simplify(degenerate=True)

    # Extract terms
    terms = []
    for term in corr.terms:
        SpinEinsumIndex.reset()
        ColorEinsumIndex.reset()
        factor, subs, ops = term.to_einsum()
        if not subs:
            continue

        parts = [
            p.replace("...", "").split("->")[0].strip()
            for p in subs.split(",")
            if p.strip()
        ]
        clean_subs = ",".join(p for p in parts if p)
        renamed = [_rename_op(o) for o in ops]
        coeff = getattr(term, "factor", 1)
        terms.append(Term(coeff, clean_subs, renamed))

    return Result(terms, n_mesons, has_identical_quarks=(len(terms) > 1))
