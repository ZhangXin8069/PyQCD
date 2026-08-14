"""蒸馏收缩引擎：自动 Wick 收缩、重子算符、顺序传播子、动态收缩。"""
from ._autowick import wick_contraction, identify_equivalent_diagrams
from ._baroperator import baroperator_conjugate
from ._seqperam import seq_peram
from ._dynamic import PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction

__all__ = [
    "wick_contraction", "identify_equivalent_diagrams",
    "baroperator_conjugate", "seq_peram",
    "PeramRegistry", "VRegistry", "GammaRegistry", "dynamic_contraction",
]

Namespace.__module__ = "pyqcd.contraction"
