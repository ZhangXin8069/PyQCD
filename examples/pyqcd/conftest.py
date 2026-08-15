"""examples/pyqcd 测试入口：python examples/pyqcd/conftest.py 或 pytest。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyqcd.testing import (  # noqa: F401
    test_gamma_basis,
    test_zr_parametrization,
    test_gradient_flow_su3_and_dissipation,
    test_tmd_operator_runs,
    test_matching_kernel,
    test_hybrid_ratio,
    test_tmd_extraction_chain,
    test_scale_setting_flow_behavior,
    test_hyp_smear,
    test_gradient_flow_tau_limit,
    test_ratio_fit_extraction,
    test_hyp_vs_flow_consistent,
    test_gpu_backend_consistency,
    test_end_to_end_synthetic_meff,
    test_matching_sum_rule,
    test_core_chain_integrated,
    test_tmd_matching_nlo,
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
