from argparse import Namespace
"""蒸馏收缩引擎：自动 Wick 收缩、重子算符、顺序传播子、动态收缩。"""
from argparse import Namespace

from ._autowick import wick_contraction, identify_equivalent_diagrams
from ._baroperator import (
    split_hadrons, classify_structure, conjugate_operator, dagger_quark,
)
from ._seqperam import seq_peram
from ._dynamic import PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction

__all__ = [
    "wick_contraction", "identify_equivalent_diagrams",
    "split_hadrons", "classify_structure", "conjugate_operator", "dagger_quark",
    "seq_peram",
    "PeramRegistry", "VRegistry", "GammaRegistry", "dynamic_contraction",
]

Namespace.__module__ = "pyqcd.contraction"
