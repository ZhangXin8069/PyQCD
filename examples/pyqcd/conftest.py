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
