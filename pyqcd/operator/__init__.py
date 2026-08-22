from argparse import Namespace
"""胶子算符：Clover 场强张量 + Wilson 线的非定域胶子 OPE 算符（梯度流 TMD-PDF 的核心）。

本模块实现 donghx 的胶子 OPE 构造（``compute_ope.py`` 照抄逻辑），
并扩展提供 ``staple`` 型 Wilson 线算符（TMD 用，b_⊥ 方向位移）。
"""
from ._gluon_ope import (
    plaquette_clover,
    compute_dual_field_strength, gluon_ope_operator_z0, gluon_ff_operator_z0,
    get_ope_lorentz_pairs, staple_operator,
    read_gauge_lime,
)
from ._helicity import (
    plaquette_dual_stack, helicity_two_field_operator,
)

__all__ = [
    "plaquette_clover",
    "compute_dual_field_strength", "gluon_ope_operator_z0",
    "gluon_ff_operator_z0", "get_ope_lorentz_pairs", "staple_operator",
    "read_gauge_lime",
    "plaquette_dual_stack", "helicity_two_field_operator",
]

Namespace.__module__ = "pyqcd.operator"
