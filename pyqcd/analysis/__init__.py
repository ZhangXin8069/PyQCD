from argparse import Namespace
"""统计分析：Jackknife / Bootstrap 重采样、有效质量、3pt 比率。"""
from ._analyse import (
    Mom2GeV, Jackknife, Bootstrap, meff, ratio_3pt, loop_tsrc, solve_gevp,
)

__all__ = ["Mom2GeV", "Jackknife", "Bootstrap", "meff", "ratio_3pt", "loop_tsrc", "solve_gevp",
    "sem", "resample", "cov_mat", "model_ratio", "run_disconnected_ratio",
    "run_meff_jackknife", "run_3pt_ratio",
    "th_E0", "fit_dispersion", "dispersion_check", "pz_to_gev_lattice",
    "R_model", "covariance_matrix_inv", "fit_ratio",
    "AnaParams", "DirParams", "load_ratio", "load_corr2", "compute_eff_mass",
    "normalized_cov", "plot_histogram", "plot_ratio_histogram",
    "plot_corr2_histogram", "plot_eff_mass_histogram", "analyze_3dir"]

Namespace.__module__ = "pyqcd.analysis"

from ._disconnected import (
    sem, resample, cov_mat, model_ratio, run_disconnected_ratio,
)
from ._correlators import run_meff_jackknife, run_3pt_ratio
from ._dispersion import th_E0, fit_dispersion, dispersion_check, pz_to_gev_lattice
from ._ratio_fit import R_model, covariance_matrix_inv, fit_ratio
from ._ana_3dir import (
    AnaParams, DirParams, load_ratio, load_corr2, compute_eff_mass,
    normalized_cov, plot_histogram, plot_ratio_histogram, plot_corr2_histogram,
    plot_eff_mass_histogram, analyze_3dir,
)
