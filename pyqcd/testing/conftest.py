"""PyQCD 集成测试入口：``python -m pyqcd.testing`` 或 pytest。"""

import os

from pyqcd.testing import (  # noqa: F401
    test_gamma_basis,
    test_gevp_preserves_complex_hermitian_data,
    test_zr_parametrization,
    test_gradient_flow_su3_and_dissipation,
    test_wilson_flow_is_gauge_covariant,
    test_wilson_flow_zero_time_is_identity,
    test_wilson_flow_hits_requested_time,
    test_wilson_flow_rejects_invalid_time_controls,
    test_gauge_observables_contracts,
    test_flow_action_density_is_gauge_invariant_and_nonnegative,
    test_flow_action_density_weak_field_normalization,
    test_gradient_flow_gluon_reference_contracts,
    test_tmd_operator_runs,
    test_tmd_staple_matches_explicit_three_segment_path,
    test_tmd_staple_is_gauge_covariant,
    test_tmd_matrix_element_is_gauge_invariant,
    test_tmd_operator_uses_tx_ty_xy_lorentz_pairs,
    test_tmd_batch_reuses_clover_fields_and_staples,
    test_tmd_empty_batch_skips_geometry_work,
    test_matching_kernel,
    test_hybrid_ratio,
    test_tmd_extraction_chain,
    test_tmd_soft_factor_ze_contracts,
    test_sftx_flow_time_units_contracts,
    test_scale_setting_flow_behavior,
    test_hyp_smear,
    test_hyp_smear_is_gauge_covariant,
    test_hyp_zero_outer_weight_returns_input,
    test_hyp_reduces_to_ape_when_inner_weights_zero,
    test_hyp_is_hypercubic_under_axis_relabeling,
    test_gradient_flow_tau_limit,
    test_ratio_fit_extraction,
    test_gpu_backend_consistency,
    test_torch_backend_consistency,
    test_end_to_end_synthetic_meff,
    test_matching_sum_rule,
    test_core_chain_integrated,
    test_tmd_matching_nlo,
    # 整合功能测试（~auto-all）
    test_stout_smear,
    test_stout_smear_is_gauge_covariant,
    test_stout_torch_cuda_device_contract,
    test_eigvec_compress,
    test_eigcompress_dtype_contracts,
    test_inner_product_returns_cross_gram_matrix,
    test_omega_uses_extract_count_per_input_block,
    test_omega_dim2_single_sampled_block_weights,
    test_omega_dim3_single_sampled_block_weights,
    test_omega_rejects_invalid_partition_contract,
    test_omega_n1_dim3_matches_hand_inclusion_probabilities,
    test_omega_normal_uses_symmetric_row_sum_balancing,
    test_omega_exact0_multiblock_dim2_hand_weights,
    test_omega_exact0_multiblock_dim3_hand_weights,
    test_cg_coefficients,
    test_hB_dataset_loader,
    test_zr_dataset_object_api,
    test_zr_dataset_object_reuses_prepared_inputs,
    test_hybrid_boot_covariance,
    test_tmd_plateau_and_cs_kernel,
    test_plot_tmd_pdf,
    test_pipeline_baryon_2pt_uses_trace_projection,
    test_pipeline_baryon_3pt_traces_spin_and_preserves_current,
    test_pipeline_pjnnjnp_4pt_traces_spin_and_preserves_current,
    test_pipeline_pion_projection_controls_are_unchanged,
    test_pipeline_projection_rejects_uncontracted_spin_shapes,
    test_pipeline_2pt_only_swallows_known_forbidden_neutron_channel,
    test_pjn_3pt_explicit_wick_oracle,
    test_staple_operator_public_contracts,
    test_tmd_su3_geometry_contracts,
    test_step_tmd_single_flow_contract,
    test_tmd_hybrid_renormalization_contracts,
    test_quasi_tmd_fourier_contracts,
    test_matching_grid_contracts,
    test_pipeline_persistence_contracts,
    test_pipeline_runtime_failure_contracts,
    test_correlated_fit_identifiability_guards,
    test_bootstrap_resampling_contracts,
    test_threads_contracts,
    test_random_equations,
    test_thread_safety,
    test_contract_edge_cases,
    test_docs_api_contracts,
    test_release_metadata_contracts,
    test_field_strength_cache_contracts,
    test_ope_channel_contracts,
    test_continuum_extrapolation_identifiability_contracts,
    test_fh_adaptive_window_contracts,
    test_vertex_product_binary_contracts,
    test_momentum_smearing_contracts,
    test_pipeline_validate_and_2pt_resume,
    test_pipeline_tmd_uses_dimensionless_flow_time,
    test_proton_energy_dirs,
    test_dispersion_identifiability_contracts,
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
    test_renorm_numerical_guards_contracts,
    test_gradient_flow_quasi_policy_contracts,
    test_tmd_physical_contracts,
    test_bare_sumratio_contracts,
    test_analysis_safety_contracts,
    test_infra_safety_contracts,
    test_einsum_native_contracts,
    test_einsum_plan_gate_contracts,
    test_einsum_python_hotpath_contracts,
    test_native_install_diagnostics_contracts,
    test_einsum_zero_extent_and_noncontiguous_contracts,
    test_einsum_mixed_native_backends,
    test_topology_kernel_contracts,
    test_specialized_query_contracts,
    test_production_perf_gate_contracts,
    test_bench_scene_distill_contracts,
    test_dedicated_routing_contracts,
    test_label_sizes_contracts,
    test_einsum_test_cli_contracts,
    test_opt_einsum_parity_contracts,
    test_einsum_semantics_contracts,
    test_einsum_dispatch_contracts,
    test_einsum_cache_contracts,
    test_einsum_optimize_callable_contracts,
)

# 强制指定内核契约（`kernel=` / `PYQCD_EINSUM_KERNEL_FORCE`）：
# 直接导入套件入口，`main()` 的 globals 扫描即可发现并运行它。
from pyqcd.testing._einsum_kernel_force_contract import (
    test_einsum_kernel_force_contracts,
)
from pyqcd.testing._einsum_as_equivalence_cache_contract import (
    test_einsum_as_equivalence_cache_contracts,
)
# 配置画像缓存契约：CSV 往返、选优、稀疏写入、compare 只写不读、关闭零 IO、
# 不可写降级、形状归一化与目录优先级（2026-11-17）。
from pyqcd.testing._einsum_profile_cache_contract import (
    test_einsum_profile_cache_contracts,
)
# 换路不换答案契约（R4）：generic/dedicated/backend 三路对拍，以及画像命中、
# 关闭、建议不可用降级三条路径都不许改变结果与 dtype/形状（2026-11-17）。
from pyqcd.testing._einsum_dispatch_invariance_contract import (
    test_einsum_dispatch_invariance_contracts,
)
# 回落留痕契约：形状特化/原生编译失败的可观测性，以及两个 `_warn_*_once`
# 从"进程级一次"改成"每类失败一次"（2026-11-16）。
from pyqcd.testing._einsum_fallback_visibility_contract import (
    test_native_expression_compile_failure_leaves_a_warning,
    test_native_fallback_warns_once_per_reason,
    test_shape_path_failure_is_recorded_and_announced_once,
    test_specialized_kernel_warns_once_per_distinct_failure,
)
# 画像缓存的并发/健壮性契约：fork 锁重建、坏行跳过、字段截断、每键上限、
# ensure_configured 语义、关闭零副作用（2026-11-17）。
from pyqcd.testing._einsum_profile_robustness_contract import (
    test_einsum_profile_robustness_contracts,
)
# 等价负标记契约 + `--tune` 遍历契约：state=none 跳过等价步骤、无标记照常查、
# `as_equivalence="force"` 刷新标记、`profile=False` 零 IO、`--tune` 与它自己
# 遍历的三维互斥、组合数硬上限、最快但 MISMATCH 的组合不得当选（2026-11-17）。
from pyqcd.testing._einsum_tune_equivalence_contract import (
    test_einsum_tune_equivalence_contracts,
)
# 配置表只读契约（TV 重设计 2026-11-17）：同键多行→报错并列出全部候选、缺文件→
# 警告"需要先 tune"+命令模板且继续执行、普通调用只读不写（文件字节不变）、命中与
# 用户显式配置冲突→`ProfileConflict`、数值 rel ≤ 1e-12。
from pyqcd.testing._einsum_profile_readonly_contract import (
    test_einsum_profile_readonly_contracts,
)
# 计时口径契约（§4 冻结）：只有 `wall_ms` 一列，量的是**整个** `contract()` 的墙钟
# （含规划与等价改写准备），不许拆回两列（旧的拆分列与更早的毫秒列都已删除，
# 名字见该模块的 `LEGACY_TIMING_COLUMNS`；源码里不留旧词，§8 的 grep 零残留）。
from pyqcd.testing._einsum_profile_timing_contract import (
    test_einsum_profile_timing_contracts,
)
# `optimize="none"` 契约：字符串拼写与 `False` 完全等价（path 相同、数值 rel≤1e-15），
# 且配置表的 `optimize` 列双向一致（`False` 写 `none`，表里的 `none` 读回 `False`）。
from pyqcd.testing._einsum_optimize_none_contract import (
    test_einsum_optimize_none_contracts,
)
# optimize 对外取值契约：对外**恰好** 6 个（auto/none/greedy/optimal/dp/branch），
# 注册表里另外 9 个名字逐条标注（退化别名 / 待清除），三份文档逐字一致（2026-11-17）。
from pyqcd.testing._einsum_optimize_surface_contract import (
    test_einsum_optimize_surface_contracts,
)
# 命中通知标签契约：通知里的内核名必须等于 `LAST_INFO['path']`，topology 提示出现
# dedicated 时必须同时给出 `[match: topology]` 上下文，且机器段跨语言逐字相同
# （2026-11-17 用户报告"topology 命中却打印 kupl_dedicated"）。
from pyqcd.testing._einsum_notice_kernel_label_contract import (
    test_einsum_notice_kernel_label_contracts,
)

#: 显式套件清单：供 ``from pyqcd.testing.conftest import TESTS`` 按名筛选
#: （``main()`` 仍按 ``test_*`` 名字扫描，两者指向同一条用例）。
TESTS = (
    test_einsum_kernel_force_contracts,
    test_einsum_as_equivalence_cache_contracts,
    test_einsum_profile_cache_contracts,
    test_einsum_dispatch_invariance_contracts,
    test_einsum_profile_robustness_contracts,
    test_einsum_tune_equivalence_contracts,
    test_einsum_profile_readonly_contracts,
    test_einsum_profile_timing_contracts,
    test_einsum_optimize_none_contracts,
    test_einsum_optimize_surface_contracts,
    test_einsum_notice_kernel_label_contracts,
    test_native_expression_compile_failure_leaves_a_warning,
    test_native_fallback_warns_once_per_reason,
    test_shape_path_failure_is_recorded_and_announced_once,
    test_specialized_kernel_warns_once_per_distinct_failure,
)


def main():
    """Run every test registered by the central testing package.

    跑测试期间**强制关掉画像缓存**（``PYQCD_EINSUM_PROFILE=0``）：它的缺省目录是
    CWD 下的 ``./.cache``，而普通 ``contract()`` 调用会**读**它来决定选路——如果
    上一次测试往仓库根写了行，下一次测试就可能因为"缓存建议 backend"而换了路径，
    表现为"单跑通过、全套偶发失败"。需要画像的契约自己用 ``profile=True`` + 临时
    目录显式开启（显式参数压过环境变量），所以关掉不损失覆盖。
    """
    import traceback
    import unittest

    saved_profile = os.environ.get("PYQCD_EINSUM_PROFILE")
    os.environ["PYQCD_EINSUM_PROFILE"] = "0"
    passed = skipped = failed = 0
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith('test_') and callable(fn):
                try:
                    fn()
                    print(f"PASS {name}")
                    passed += 1
                except unittest.SkipTest as exc:
                    print(f"SKIP {name}: {exc}")
                    skipped += 1
                except Exception:
                    print(f"FAIL {name}")
                    traceback.print_exc()
                    failed += 1
    finally:
        if saved_profile is None:
            os.environ.pop("PYQCD_EINSUM_PROFILE", None)
        else:
            os.environ["PYQCD_EINSUM_PROFILE"] = saved_profile
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
