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
# 显式强制契约（2026-11-18）：`kernel_force` 没有 `auto`——传 `auto` 报错并点名 tune，
# 缺省（`None`/空串/CLI 不给该选项/环境变量未设）一律 `generic`，四个合法标签按书写
# 次序解析。与 `_einsum_kernel_force_contract`（九项，选路接线）互补。
from pyqcd.testing._einsum_kernel_force_explicit_contract import (
    test_einsum_kernel_force_explicit_contracts,
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
# `--tune` **分支化有序遍历 + min_wall_ms 剪枝**：B1/B2 = 29、B3 = 28 组
# 且逐项顺序固定；上界初值 = --tune-timeout、被更小实测值立刻压低；超过上界的组被终止
# （原文 `耗时大于min_wall_ms,此组被终止`）、不写缓存；best = 最小耗时组。
from pyqcd.testing._einsum_tune_plan_contract import (
    test_einsum_tune_plan_contracts,
)
# `--tune` 报告的**用户实测**三缺陷（2026-11-18）：缺表提示必须复现本次实参（GPU 仍 GPU、
# `--shape` 带输出段、不替用户加 `--compare`）、准入不过的内核整档 `skip` 而非逐组
# `error`、`N` 用裁剪后组数、末行 `PASS` 带 `min_wall_ms=` 与 `best=`。
from pyqcd.testing._einsum_tune_report_contract import (
    test_einsum_tune_report_contracts,
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
# optimize 叶子名字契约：`branch_1`/`branch_2`/`branch_all`/`none` 都解得出来（连字符
# 别名与下划线主名是同一对象），七个名字都能跑，`none` 对 topology 是唯一合法取值、
# 对其余内核等价 `False`，主清单恰好那七个（2026-11-18）。
from pyqcd.testing._einsum_optimize_leaves_contract import (
    test_einsum_optimize_leaves_contracts,
)
# 命中通知标签契约：通知里的内核名必须等于 `LAST_INFO['path']`，topology 提示出现
# dedicated 时必须同时给出 `[match: topology]` 上下文，且机器段跨语言逐字相同
# （2026-11-17 用户报告"topology 命中却打印 kupl_dedicated"）。
from pyqcd.testing._einsum_notice_kernel_label_contract import (
    test_einsum_notice_kernel_label_contracts,
)
# 命中提示的**执行语义**契约（2026-11-18）：`kernel_force` 只接受单值强制，默认
# `generic` 不试专用内核，所以提示不许再说"先 dedicated 后 topology"；强制了哪个就必须
# 点名哪个，三档文案（强制跑成 / 未强制 / 强制失败回落）在 zh/en 下机器段逐字相同。
from pyqcd.testing._einsum_notice_forced_semantics_contract import (
    test_einsum_notice_forced_semantics_contracts,
)
# CLI 默认值契约（2026-11-18 用户实测）：`--kernel-force` 不给时 argparse 是 `None`
# （未指定，帮助仍显示回落值 generic），已 tune 的 `kernel=backend` 行因此能套用；
# `--profile-dir ~/.cache` 先展开 `~` 再取绝对路径（与 `_api` 侧同一张表）。
from pyqcd.testing._einsum_cli_defaults_contract import (
    test_einsum_cli_defaults_contracts,
)
# 省略 `--shape` 时的**语料等价匹配**（2026-11-18）：轴序变体（换操作数位置 / 转置某操作数）
# 必须经 `_equivalence.normalize_equation` 命中语料，形状按 mapping **逐轴还原**成用户原方程的
# 轴序，还原不可靠时仍报 `corpus_miss`（五项断言，见该模块）。
from pyqcd.testing._einsum_shape_from_corpus_equiv_contract import (
    test_einsum_shape_from_corpus_equiv_contracts,
)
# `--tune-timeout` 的**真实能力**契约（2026-11-19）：它是尽力而为上限，能打断 Python 侧
# 长循环、打不断原生 C/CUDA 代码——help / `tune note :` / 模块 docstring 三处同口径，
# 且不许再说"可能打断原生内核"或"建议在新进程里跑 tune"。
from pyqcd.testing._einsum_tune_timeout_doc_contract import (
    test_einsum_tune_timeout_doc_contracts,
)
# `--tune` 的**组级硬超时**（2026-11-19）：缺省每组跑在**全新解释器**里，超时由父进程
# kill 子进程强制生效——这是唯一能打断原生 C/CUDA 代码的办法（SIGALRM 只在字节码之间
# 派发）。六项断言钉住：真能打断、只杀子进程组、启动开销单列、崩溃记 error 不记
# timeout、组内 min_wall 剪枝仍在、`PYQCD_EINSUM_TUNE_INPROCESS=1` 才退回进程内。
from pyqcd.testing._einsum_tune_isolation_contract import (
    test_einsum_tune_isolation_contracts,
)
# 计时与参照的**三处失真**（2026-11-19 用户实测）：`--compare opt_einsum` 只走 numpy
# 后端、异步后端只量到入队时间（同一条方程脚本 900 ms / tune 表里 0.32 ms）、
# `--warmup 1` 盖不住 GPU 冷启动。八项断言钉住修复后的可观测证据（无 GPU 时 SKIP）。
from pyqcd.testing._einsum_timing_compare_contract import (
    test_einsum_timing_compare_contracts,
)
# `--tune` 的**硬上限在每条路线上都生效**（2026-11-19 用户实测）：GPU 侧曾缺
# `bypass_plan_gate`，`_api._attempt_kernel` 的 `getattr(..., None)` 静默不旁路 →
# `dedicated` 组被 arena 门槛判死记 `error`（59.5 s 才返回）。四项断言钉住两个模块
# 的旁路同名同义、共享同一标志、嵌套配平，以及组级窗口 = 单次上限 × 调用次数。
from pyqcd.testing._einsum_tune_gate_contract import (
    test_einsum_tune_gate_contracts,
)
# GPU 探测诊断契约（2026-11-19 用户实测）：探测失败的 stage/detail 必须如实留痕、
# 拒绝原因不许复述写死的"2x2 matmul disagreed with numpy"、probe_dtypes 覆盖四类、
# 端到端容差复用 _PROBE_TOLERANCE、"部分可用"的 verdict 必须如实（无 GPU 时 SKIP）。
from pyqcd.testing._einsum_gpu_probe_diagnosis_contract import (
    test_einsum_gpu_probe_diagnosis_contracts,
)
# 峰值内存契约（2026-11-19 用户要求"每组记录峰值内存/显存、进下一组前释放"）：
# 四个公开函数绝不抛异常、reset 生效、已知大小的分配能被测出来、release 清池、
# format_peak_mb 的三个边界、失败只记 LAST_ERROR 不改返回值。
from pyqcd.testing._einsum_peak_memory_contract import (
    test_einsum_peak_memory_contracts,
)
# 记录表契约（2026-11-20）：`pyqcd.einsum.record.csv` 只作参考、绝不参与命中——
# 整表重写而非追加、列是配置表的超集、`read_records` 只许出现在"更省内存"提示里、
# 关闭开关时不落盘、旧位置 `./.cache` 只读回退（不复制不移动）。
from pyqcd.testing._einsum_record_table_contract import (
    test_einsum_record_table_contracts,
)
# 暂存落点与新口径契约（2026-11-20 用户实测）：生产形状操作数 1.95 GiB 而 `/tmp`
# 只剩 1.3 GiB → 写第 3 个操作数 ENOSPC → 29 组全部 error，且失败后没清理、残留
# 2.0 GiB 毒害下一轮。八项断言钉住：tmpfs 落点优先、显式配置压过一切、空间不足时
# 跳过该落点、`required_bytes` 含余量、`_ensure_room` 的报错说清"要多少/剩多少/
# 怎么办"、写盘失败自清、操作数生成不绕道 complex128。
from pyqcd.testing._einsum_tune_stage_contract import (
    test_einsum_tune_stage_contracts,
)
# 九项需求端到端验收（2026-11-20）：GPU 诊断真实 stage / `original` 七种 optimize /
# 家目录配置表 / 逐组峰值 / 2× 剪枝截止 / 记录表不参与命中 / warmup=0+rounds=1 /
# "更省内存"提示的半数门槛 / 自动 tune 同键只试一次。一条需求对一条断言。
from pyqcd.testing._einsum_nine_items_acceptance_contract import (
    test_einsum_nine_items_acceptance_contracts,
)

#: 显式套件清单：供 ``from pyqcd.testing.conftest import TESTS`` 按名筛选
#: （``main()`` 仍按 ``test_*`` 名字扫描，两者指向同一条用例）。
TESTS = (
    test_einsum_kernel_force_contracts,
    test_einsum_kernel_force_explicit_contracts,
    test_einsum_as_equivalence_cache_contracts,
    test_einsum_profile_cache_contracts,
    test_einsum_dispatch_invariance_contracts,
    test_einsum_profile_robustness_contracts,
    test_einsum_tune_equivalence_contracts,
    test_einsum_tune_plan_contracts,
    test_einsum_tune_report_contracts,
    test_einsum_tune_timeout_doc_contracts,
    test_einsum_tune_isolation_contracts,
    test_einsum_timing_compare_contracts,
    test_einsum_tune_gate_contracts,
    test_einsum_gpu_probe_diagnosis_contracts,
    test_einsum_peak_memory_contracts,
    test_einsum_record_table_contracts,
    test_einsum_tune_stage_contracts,
    test_einsum_nine_items_acceptance_contracts,
    test_einsum_profile_readonly_contracts,
    test_einsum_profile_timing_contracts,
    test_einsum_optimize_none_contracts,
    test_einsum_optimize_surface_contracts,
    test_einsum_optimize_leaves_contracts,
    test_einsum_notice_kernel_label_contracts,
    test_einsum_notice_forced_semantics_contracts,
    test_einsum_cli_defaults_contracts,
    test_native_expression_compile_failure_leaves_a_warning,
    test_native_fallback_warns_once_per_reason,
    test_shape_path_failure_is_recorded_and_announced_once,
    test_specialized_kernel_warns_once_per_distinct_failure,
)


def main():
    """Run every test registered by the central testing package.

    跑测试期间**强制关掉画像缓存**（``PYQCD_EINSUM_PROFILE=0``）：普通 ``contract()``
    调用会**读**它来决定选路——如果上一次测试往表里写了行，下一次测试就可能因为
    "缓存建议 backend"而换了路径，表现为"单跑通过、全套偶发失败"。需要画像的契约自己
    用 ``profile=True`` + 临时目录显式开启（显式参数压过环境变量），所以关掉不损失覆盖。

    **同时关掉自动 tune**（``PYQCD_EINSUM_AUTOTUNE=0``）：缺省行为是"第一次调用先
    tune 再真跑"，在套件里那是每组几十个子进程、几百毫秒起步——它会把一次回归从
    一分半拖成几十分钟，而且被测的是它自己。契约要验自动 tune 时用
    :func:`pyqcd.einsum._autotune.run_for` 的显式调用，不靠缺省路径。
    """
    import traceback
    import unittest

    saved_profile = os.environ.get("PYQCD_EINSUM_PROFILE")
    saved_autotune = os.environ.get("PYQCD_EINSUM_AUTOTUNE")
    os.environ["PYQCD_EINSUM_PROFILE"] = "0"
    os.environ["PYQCD_EINSUM_AUTOTUNE"] = "0"
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
        if saved_autotune is None:
            os.environ.pop("PYQCD_EINSUM_AUTOTUNE", None)
        else:
            os.environ["PYQCD_EINSUM_AUTOTUNE"] = saved_autotune
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
