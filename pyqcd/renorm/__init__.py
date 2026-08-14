from argparse import Namespace
"""重整化：自重整化 Z_R、混合方案、NLO 匹配、连续极限外推、梯度流、TMD 提取。

本子包实现核子胶子 TMD-PDF 计算的完整重整化链（核心目标）：
    裸矩阵元 → 混合方案（比值 + 自重整化）→ λ 外推 → 傅里叶变换
    → NLO 匹配 → 连续极限外推，
并以梯度流（Wilson flow，Monahan–Orginos 方案）为 UV 正规化手段。
"""
from ._const import CA, CF, gammaE, pi, alpha_s, A_s, b0
from ._ensembles import (
    fm_to_GeV, a_len_set, Nl_set, pion_mass_set, MPI_PHYSICAL, pz_to_gev,
)
from ._zr import Z_MS, th_hB, th_ZR, cost_function, cost_function_all, fit_ZR
from ._hybrid import (
    hR_z_Pz, hR_lambda_fit_form, fit_hR_lambda, hR_lambda, hR_x,
)
from ._matching import hR_PDF, C_gluon_ratio, Si
from ._extrapolate import hR_form, build_fit_data, fit_hR_PDF_extrap
from ._gradient_flow import (
    wilson_flow, wilson_flow_step, flow_derivative, staple_6,
    flow_action_density, scale_setting_t0, proj_su3,
)
from ._tmd import (
    staple_wilson_line, M_mu_lambda_nu_rho, gluon_tmd_operator,
    tmd_matrix_elements, gradient_flow_renormalized_tmd,
    self_renormalized_ratio, invariant_amplitude, collins_soper_kernel,
)
from ._tmdextract import (
    quasi_tmd_pdf, cs_kernel_from_ratio, soft_function_intrinsic,
    tmd_matching_hybrid, sftx_gluon_matching_coeff,
    sftx_energy_density_t0,
)

__all__ = [
    # 常数
    "CA", "CF", "gammaE", "pi", "alpha_s", "A_s", "b0",
    # 系综
    "fm_to_GeV", "a_len_set", "Nl_set", "pion_mass_set", "MPI_PHYSICAL", "pz_to_gev",
    # 自重整化
    "Z_MS", "th_hB", "th_ZR", "cost_function", "cost_function_all", "fit_ZR",
    # 混合方案
    "hR_z_Pz", "hR_lambda_fit_form", "fit_hR_lambda", "hR_lambda", "hR_x",
    # 匹配
    "hR_PDF", "C_gluon_ratio", "Si",
    # 外推
    "hR_form", "build_fit_data", "fit_hR_PDF_extrap",
    # 梯度流
    "wilson_flow", "wilson_flow_step", "flow_derivative", "staple_6",
    "flow_action_density", "scale_setting_t0", "proj_su3",
    # TMD
    "staple_wilson_line", "M_mu_lambda_nu_rho", "gluon_tmd_operator",
    "tmd_matrix_elements", "gradient_flow_renormalized_tmd",
    "self_renormalized_ratio", "invariant_amplitude", "collins_soper_kernel",
    "quasi_tmd_pdf", "cs_kernel_from_ratio", "soft_function_intrinsic",
    "tmd_matching_hybrid", "sftx_gluon_matching_coeff",
    "sftx_energy_density_t0",
]

Namespace.__module__ = "pyqcd.renorm"
