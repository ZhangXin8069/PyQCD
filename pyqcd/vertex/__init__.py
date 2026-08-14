from argparse import Namespace
"""顶点函数：VdV / VVV 动量投影顶点、相位因子。"""
from ._vertex import (
    phase_exp_2pt, phase_exp_3pt, Mom_VdV_sink_t, Mom_VVV_sink_t,
    VdV_sink_t_link, sink2src,
)

__all__ = [
    "phase_exp_2pt", "phase_exp_3pt", "Mom_VdV_sink_t", "Mom_VVV_sink_t",
    "VdV_sink_t_link", "sink2src",
]

Namespace.__module__ = "pyqcd.vertex"
