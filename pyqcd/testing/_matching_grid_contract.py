"""NLO 胶子匹配 x 网格的公共入口契约。"""
from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

import numpy as np

from pyqcd.renorm import hR_PDF


def _normal_grid_fixture():
    """返回已避开零节点的现有 NLO 匹配型输入。"""
    x = np.linspace(0.02, 1.48, 21)
    return x, np.exp(-x ** 2)


class MatchingGridContract(unittest.TestCase):
    def test_exact_zero_node_is_rejected_without_runtime_warning(self):
        """无主值离散化时，x=0 必须在数值核前被公共入口拒绝。"""
        x = np.array([-0.4, 0.0, 0.4])
        h0 = np.exp(-x ** 2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                    "pyqcd.renorm._matching.A_s_run",
                    side_effect=AssertionError(
                        "zero-grid validation must precede coupling work")), \
                    patch(
                        "pyqcd.renorm._matching._matching_kernels",
                        side_effect=AssertionError(
                            "zero-grid validation must precede kernel work")):
                with self.assertRaisesRegex(
                        ValueError, "zero.*avoid|avoid.*zero"):
                    hR_PDF(
                        x, Pz_=4, conf="L24x72",
                        hR_tilde_data=h0, mu_=2.0)

        runtime_warnings = [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])

    def test_normal_zero_avoiding_grid_remains_finite_without_runtime_warning(self):
        """正常避零网格仍给出有限 NLO 匹配结果且不泄漏 RuntimeWarning。"""
        x, h0 = _normal_grid_fixture()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = hR_PDF(x, Pz_=4, conf="L24x72", hR_tilde_data=h0, mu_=2.0)

        self.assertTrue(np.isfinite(out).all())
        runtime_warnings = [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])

    def test_zero_coupling_keeps_the_identity_matching_contract(self):
        """alpha_s=0 时 Z=I，匹配必须精确保留避零网格上的输入。"""
        x, h0 = _normal_grid_fixture()

        with patch("pyqcd.renorm._matching.A_s_run", return_value=0.0):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = hR_PDF(
                    x, Pz_=4, conf="L24x72", hR_tilde_data=h0, mu_=2.0)

        np.testing.assert_allclose(out, h0, rtol=0.0, atol=0.0)
        runtime_warnings = [
            item for item in caught if issubclass(item.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])

    def test_negative_xi_branch_has_common_plus_g0(self):
        """xi<0 必须使用 +g0+g1，而不是把共同 g0 和 g1 一起反号。"""
        from pyqcd.renorm._matching import Si, _matching_kernels

        xi = np.array([-3.2, -0.7, -0.01])
        y = np.array([1.0, 1.2, 0.8])
        pz = 2.3
        mu = 2.0
        lambda_s = 0.8
        polynomial = (
            11.0 - 28.0 * xi + 18.0 * xi ** 2.0 - 12.0 * xi ** 3.0
        )
        g1 = (
            -2.0 * (1.0 - xi + xi ** 2.0) ** 2.0
            / (1.0 - xi) * np.log(xi / (xi - 1.0))
            - polynomial / (6.0 * (1.0 - xi))
        )
        g0 = 5.0 / 6.0 * (
            -1.0 / np.abs(1.0 - xi)
            + 2.0 * Si((1.0 - xi) * np.abs(y) * lambda_s)
            / (np.pi * (1.0 - xi))
        )
        expected = g0 + g1
        actual = _matching_kernels(
            xi[:, None], y[:, None], pz, mu, lambda_s)[:, 0]

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
