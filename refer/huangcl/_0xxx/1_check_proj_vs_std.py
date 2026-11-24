#!/public/home/huangcl/.venv3_10/bin/python
"""比较投影矩阵方法与标准方法得到的四条通道."""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path("/public/home/huangcl/06_deuteron_gluon_pdf")
sys.path.insert(0, str(PROJECT_DIR / "98_tools"))
sys.path.insert(0, str(PROJECT_DIR / "07_get_channel_2pt"))
from comparison_data_tools import compare_data  # noqa: E402
from get_channel_2pt import (  # noqa: E402
    proj_deuteron_chan, proj_deuteron_chan_std,
    proj_dinucleon_chan, proj_dinucleon_chan_std,
)

DATA_FILE = ("/public/home/huangcl/00_data/00_NP_2pt/cg5/c24p29/4050/"
             "NPcorr_Px0Py0Pz0_conf4050.npy")

# para: dt, t_src, r, q, z, y
corr = np.load(DATA_FILE)

deut_proj = proj_deuteron_chan(corr)
deut_std = proj_deuteron_chan_std(corr)
din_proj = proj_dinucleon_chan(corr)
din_std = proj_dinucleon_chan_std(corr)

pairs = [(deut_std[..., i], deut_proj[..., i], f"deuteron sz={i - 1}")
         for i in range(3)]
pairs.append((din_std, din_proj, "dinucleon"))

compare_data(pairs, show_details=True)
