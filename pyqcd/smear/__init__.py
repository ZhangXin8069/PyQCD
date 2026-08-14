"""格点涂抹：HYP 涂抹（Hasenbusch 2001，梯度流的备选 UV 正则化方案）。"""
from argparse import Namespace

from ._hyp import hyp_smear, proj_su3

__all__ = ["hyp_smear", "proj_su3"]

Namespace.__module__ = "pyqcd.smear"
