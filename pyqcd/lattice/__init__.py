"""格点基础：物理常数、DeGrand-Rossi γ 矩阵、Pauli σ 矩阵。"""
from ._constants import Nc, Ns, Nd, fm2GeV, LATTICE_SPACING, INV_LATTICE_SPACING
from ._gamma import gamma, gamma_dagger, gamma_herm, gamma_smear, gamma_C, GAMMA_TYPES
from ._sigma import sigma

__all__ = [
    "Nc", "Ns", "Nd", "fm2GeV", "LATTICE_SPACING", "INV_LATTICE_SPACING",
    "gamma", "gamma_dagger", "gamma_herm", "gamma_smear", "gamma_C", "GAMMA_TYPES",
    "sigma",
]

Namespace.__module__ = "pyqcd.lattice"
