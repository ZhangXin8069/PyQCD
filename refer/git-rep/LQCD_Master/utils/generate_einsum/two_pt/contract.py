"""two_pt.contract — 2-point function Wick contraction engine.

Engine: wicklib Correlator (automatic Wick pairing + gamma algebra + fermion signs)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
try:
    from .._wick_translate import _to_wicklib, _rename_op
    from ..wicklib.operator import SpinProjector
    from ..wicklib.index import SpinEinsumIndex, ColorEinsumIndex
    from ..wicklib.correlator import Correlator as _WickCorr
except ImportError:
    # Allow standalone operation (no parent package)
    import sys
    _p = str(Path(__file__).resolve().parent.parent)
    if _p not in sys.path: sys.path.insert(0, _p)
    from _wick_translate import _to_wicklib, _rename_op
    from wicklib.operator import SpinProjector
    from wicklib.index import SpinEinsumIndex, ColorEinsumIndex
    from wicklib.correlator import Correlator as _WickCorr


# ======================================================================
# Intermediate data structures
# ======================================================================

@dataclass
class ContractionTerm:
    coefficient: complex
    einsum_subs: str
    operands: List[str]


@dataclass
class ContractionResult:
    sink_desc: str
    source_desc: str
    n_raw_terms: int
    terms: List[ContractionTerm]
    has_epsilon: bool
    projector: Optional[str] = None

    def __len__(self):
        return len(self.terms)

    def is_baryon(self):
        return self.has_epsilon

    def is_meson(self):
        return not self.has_epsilon


# ======================================================================
# Contraction entry point
# In this code, we formally used a projector for baryon as: Tmat
# 
# ======================================================================
## projector -> factors[]
## 
#factors = [0]*16; factors[0] = 1; factors[8] = 1
#proj = SpinProjector(factors, spin_snk, spin_src)
##
# gamma index → 4-bit index
GAMMA_INDEX = {
    "I":   0b0000,    # unit=I
    "g1":  0b0001,    # γ₁
    "g2":  0b0010,    # γ₂
    "g3":  0b0100,    # γ₃
    "g4":  0b1000,    # γ₄ (= γₜ)
    "gt":  0b1000,    # γ₄ (= γₜ)
    "g5":  0b1111,    # γ₅
    "g1g2": 0b0011,   # γ₁γ₂
    "g1g3": 0b0101,   # γ₁γ₃
    "g1g4": 0b1001,   # γ₁γ₄
    "g2g3": 0b0110,   # γ₂γ₃
    "g2g4": 0b1010,   # γ₂γ₄
    "g3g4": 0b1100,   # γ₃γ₄
    "g1g5": 0b1110,   # γ₁γ₅
    "g2g5": 0b1101,   # γ₂γ₅
    "g3g5": 0b1011,   # γ₃γ₅
    "g4g5": 0b0111,   # γ₄γ₅
}

def gamma_to_factor(gamma_name, coeff=1.0):
    """a single gamma → 16 factors array"""
    factors = [0.0] * 16
    factors[GAMMA_INDEX[gamma_name]] = coeff
    return factors
#
## 
##

def wick_contract_2pt(sink, source, projector: "I") -> ContractionResult:
    """<O_sink . O_source^dag, Tmat> -> ContractionResult."""

    is_baryon = sink[1]
    sink_op = sink[0]
    source_op = source[0]
    w_snk, si_snk = _to_wicklib(sink_op, "x")
    w_src, si_src = _to_wicklib(source_op, "y")
## the projector only works for the baryon 2pt

    if is_baryon and si_snk is not None:
### define the projector    
        factors    = [0.0] * 16
        factors[0] = 1 ## by default, projector is unit matrix
        Tmat = SpinProjector(factors,si_src, si_snk)

        if(projector =="I" or projector =="g1"):
            factors = gamma_to_factor(projector)
            Tmat = SpinProjector(factors,si_src, si_snk)
        elif(projector =="P_plus"):
            Tmat = SpinProjector.P_plus(si_src,si_snk) 
        ##in case of other cases! 
        op = Tmat * w_snk * w_src.adjoint()
    else:
        op = w_snk * w_src.adjoint()
    
    corr = _WickCorr(op)
    corr.simplify(degenerate=True)

    has_eps = any(
        "epsilon" in str(t) or "ColorEpsilon" in type(t).__name__
        for term in corr.terms
        for t in term.tensors
    )
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
        terms.append(ContractionTerm(coeff, clean_subs, renamed))

    return ContractionResult("", "", len(terms), terms, has_eps)
