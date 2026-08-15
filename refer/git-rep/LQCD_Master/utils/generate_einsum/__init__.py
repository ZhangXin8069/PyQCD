"""generate_einsum — Hadronic correlator contraction engine.

Sub-modules:
  two_pt/    2pt contraction: wick_contract_2pt, pyquda_format_contract
  three_pt/  3pt codegen: gen_code (baryon), gen_meson_3pt_code (meson)

Shared (one level up):
  hadron_operator.py  operator definitions
  wicklib/            Wick contraction engine
 _wick_translate.py   translation of einsum to wicklib 
"""

from .two_pt import wick_contract_2pt, pyquda_format_contract, ContractionTerm, ContractionResult
from .three_pt.codegen_baryon import gen_baryon_3pt_code as gen_pyquda_baryon
from .three_pt.codegen_meson import gen_meson_3pt_code
