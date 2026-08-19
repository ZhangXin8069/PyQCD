from argparse import Namespace
"""统计分析：Jackknife / Bootstrap 重采样、有效质量、3pt 比率。"""
from ._analyse import (
    Mom2GeV, Jackknife, Bootstrap, meff, ratio_3pt, loop_tsrc, solve_gevp,
)

__all__ = ["Mom2GeV", "Jackknife", "Bootstrap", "meff", "ratio_3pt", "loop_tsrc", "solve_gevp",
    "sem", "resample", "cov_mat", "model_ratio", "run_disconnected_ratio",
    "run_disconnected_tmd_ratio", "plot_tmd_c0", "plot_tmd_ratio",
    "run_meff_jackknife", "run_3pt_ratio",
    "th_E0", "fit_dispersion", "dispersion_check", "pz_to_gev_lattice",
    "R_model", "covariance_matrix_inv", "fit_ratio",
    "AnaParams", "DirParams", "load_ratio", "load_corr2", "compute_eff_mass",
    "normalized_cov", "plot_histogram", "plot_ratio_histogram",
    "plot_corr2_histogram", "plot_eff_mass_histogram", "analyze_3dir",
    "DEFAULT_PLOT_COLORS", "plot_errbar", "plot_scatter", "plot_hist",
    "plot_single_errbar", "plot_single_chi2",
    "plot_multi_errbars", "plot_multi_chi2", "plot_multi_scatter",
    "get_peak_memory_gb",
    "FitParams", "calc_chi2", "calc_chi2_dof", "fit", "make_summary_table",
    "fit_report_lines",
    "SampleParams2pt", "PlotParamsRatio", "ope_combine", "compute_ratio",
    "ratio_file_name", "fit_dir_name", "fit_x_coor", "do_fit_and_report",
    "plot_ratio_fits", "run_ratio2pt",
    "AnaRatioParams", "ana_load_ratio", "load_fit_result", "plot_ratio_one_z",
    "ana_ratio_plot_all",
    "run_bare_matrix",
    "EnergyParams", "energy_model", "compute_corr2", "energy_do_fit",
    "plot_eff_mass", "run_energy",
    "FHParams", "compute_fh", "fh_model", "plot_fh", "fh_do_fit",
    "plot_para", "plot_para_cmp", "run_fh"]

Namespace.__module__ = "pyqcd.analysis"

from ._disconnected import (
    sem, resample, cov_mat, model_ratio, run_disconnected_ratio,
)
from ._tmd_ratio import (
    run_disconnected_tmd_ratio, plot_tmd_c0, plot_tmd_ratio,
)
from ._correlators import run_meff_jackknife, run_3pt_ratio
from ._dispersion import th_E0, fit_dispersion, dispersion_check, pz_to_gev_lattice
from ._ratio_fit import R_model, covariance_matrix_inv, fit_ratio
from ._ana_3dir import (
    AnaParams, DirParams, load_ratio, load_corr2, compute_eff_mass,
    normalized_cov, plot_histogram, plot_ratio_histogram, plot_corr2_histogram,
    plot_eff_mass_histogram, analyze_3dir,
)
from ._plots import (
    DEFAULT_PLOT_COLORS, plot_errbar, plot_scatter, plot_hist,
    plot_single_errbar, plot_single_chi2,
    plot_multi_errbars, plot_multi_chi2, plot_multi_scatter,
    get_peak_memory_gb,
)
from ._fitter import (
    FitParams, calc_chi2, calc_chi2_dof, fit, make_summary_table,
    fit_report_lines,
)
from ._ratio2pt import (
    SampleParams2pt, PlotParamsRatio, ope_combine, compute_ratio,
    ratio_file_name, fit_dir_name, fit_x_coor, do_fit_and_report,
    plot_ratio_fits, run_ratio2pt,
)
from ._ana_ratio import (
    AnaRatioParams, load_ratio as ana_load_ratio, load_fit_result,
    plot_ratio_one_z, plot_all as ana_ratio_plot_all,
)
from ._bare_matrix import run_bare_matrix
from ._proton_energy import (
    EnergyParams, energy_model, compute_corr2, do_fit as energy_do_fit,
    plot_eff_mass, run_energy,
)
from ._fh import (
    FHParams, compute_fh, fh_model, plot_fh, do_fit_and_report as fh_do_fit,
    plot_para, plot_para_cmp, run_fh,
)
