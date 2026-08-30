"""examples/pyqcd 测试入口：python examples/pyqcd/conftest.py 或 pytest。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqcd.testing import (  # noqa: F401
    test_gamma_basis,
    test_gevp_preserves_complex_hermitian_data,
    test_zr_parametrization,
    test_gradient_flow_su3_and_dissipation,
    test_tmd_operator_runs,
    test_tmd_staple_matches_explicit_three_segment_path,
    test_tmd_staple_is_gauge_covariant,
    test_tmd_matrix_element_is_gauge_invariant,
    test_tmd_operator_uses_tx_ty_xy_lorentz_pairs,
    test_matching_kernel,
    test_hybrid_ratio,
    test_tmd_extraction_chain,
    test_scale_setting_flow_behavior,
    test_hyp_smear,
    test_gradient_flow_tau_limit,
    test_ratio_fit_extraction,
    test_hyp_vs_flow_consistent,
    test_gpu_backend_consistency,
    test_torch_backend_consistency,
    test_end_to_end_synthetic_meff,
    test_matching_sum_rule,
    test_core_chain_integrated,
    test_tmd_matching_nlo,
    # 整合功能测试（~auto-all）
    test_stout_smear,
    test_eigvec_compress,
    test_cg_coefficients,
    test_hB_dataset_loader,
    test_hybrid_boot_covariance,
    test_tmd_plateau_and_cs_kernel,
    test_plot_tmd_pdf,
    test_pipeline_validate_and_2pt_resume,
    test_proton_energy_dirs,
    test_round2_integrations,
    # 第三轮整合测试（~auto-all 第三遍清查 20260822）
    test_matching_ratio_kernels,
    test_quasi_pdf_gluon_sin_transform,
    test_gluon_ope_directions_and_ff,
    test_parity_boundary_projection,
    test_zr_sample_refit_loop,
    test_extrapolate_boot_fit,
    test_group_aggregate_and_disconnect,
    test_check_files_existence_guard,
    test_wickplot_and_flop_analysis,
    test_vertex_product_readers,
    test_env_snapshot,
    test_cmp_primitives,
    test_read_gauge_lime_accepts_contents_directory,
)


if __name__ == '__main__':
    import traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except Exception:
                print(f"FAIL {name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
