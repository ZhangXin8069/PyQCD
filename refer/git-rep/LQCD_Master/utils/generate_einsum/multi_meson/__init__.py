"""multi_meson — Multi-meson 2-point function contraction engine.

Constructs product operators (e.g. π·π, D·D·D_s), performs Wick
contractions via wicklib, and generates PyQUDA contract() code.

The wicklib engine automatically handles all N! topological
contractions from identical quarks.
"""

from ._operator import multi_meson_operator
from .contract import wick_contract_multi_2pt, Term, Result
from .codegen import pyquda_format_contract
