"""生产分支接入契约：hR_PDF / tmd_matching_hybrid / hR_x / match_one。

只验"新增入口的行为边界 + 默认路径逐位不变"，不重复 `_nnlo_line_mass_tail_contract`
里的公式代数（那边管系数，这边管接线）。
"""
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pyqcd.renorm import (
    hR_PDF,
    hR_x,
    match_one,
    tmd_matching_hybrid,
)


def _hybrid_fixture():
    z = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    hR_pz = np.array([2.0, 1.8, 1.5, 1.2, 0.9, 0.7])
    hR_0 = np.array([1.0, 0.95, 0.85, 0.75, 0.65, 0.55])
    zr = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    return dict(
        Pz_=4, lambda_extra_=2.0, l1_fit_=1.0, a1_fit_=1.0, lambda0_fit_=0.5,
        zs=0.4, z_data=z, hR_pz=hR_pz, hR_0=hR_0, zr_fit=zr, conf="L24x72")


def _quasi_fixture():
    array = np.zeros((1, 2, 4, 6, 3, 2), dtype=np.complex128)
    array[:, :, :, 1, :, :] = np.asarray([2.0, 4.0, 6.0])[
        None, None, None, :, None]
    array[:, :, :, 2, :, :] = np.asarray([1.0, 2.0, 3.0])[
        None, None, None, :, None]
    array[:, :, :, 0, :, :] = array[:, :, :, 1, :, :] - array[:, :, :, 2, :, :]
    return array


class HRPDFEntryContract(unittest.TestCase):
    def test_default_path_is_bitwise_unchanged(self):
        x = np.linspace(0.02, 1.48, 21)
        h0 = np.exp(-x ** 2)
        actual = hR_PDF(x, Pz_=4, conf="L24x72", hR_tilde_data=h0, mu_=2.0)
        expected = h0 - h0 * 0.0
        # 默认 alpha_s 走 A_s_run；与显式传入同一值必须逐位一致
        explicit = hR_PDF(
            x, Pz_=4, conf="L24x72", hR_tilde_data=h0, mu_=2.0,
            alpha_s=float(hR_PDF(
                x, Pz_=4, conf="L24x72", hR_tilde_data=h0, mu_=2.0,
                return_metadata=True)[1]["alpha_s"]))
        assert_array_equal(actual, explicit)
        self.assertEqual(actual.shape, expected.shape)

    def test_explicit_zero_coupling_is_the_identity(self):
        x = np.linspace(0.02, 1.48, 21)
        h0 = np.exp(-x ** 2)
        assert_array_equal(
            hR_PDF(x, Pz_=4, conf="L24x72", hR_tilde_data=h0, alpha_s=0.0),
            h0)

    def test_metadata_declares_the_two_loop_kernel_gap(self):
        x = np.linspace(0.02, 1.48, 21)
        h0 = np.exp(-x ** 2)
        _, metadata = hR_PDF(
            x, Pz_=4, conf="L24x72", hR_tilde_data=h0, return_metadata=True)
        self.assertEqual(metadata["order"], "one_loop")
        self.assertEqual(metadata["two_loop_kernel"], "unavailable")
        self.assertIn("g^(2)", metadata["two_loop_kernel_reason"])
        self.assertEqual(
            metadata["two_loop_endpoint_hook"], "match_one(endpoint_order=2)")
        self.assertFalse(metadata["physical_acceptance"])
        self.assertEqual(metadata["alpha_s_source"], "A_s_run(mu)*4*pi")

    def test_alpha_s_is_validated_and_cannot_be_boolean(self):
        x = np.linspace(0.02, 1.48, 21)
        h0 = np.exp(-x ** 2)
        for bad in (True, -0.1, float("nan"), float("inf")):
            with self.subTest(alpha_s=bad):
                with self.assertRaises(ValueError):
                    hR_PDF(
                        x, Pz_=4, conf="L24x72", hR_tilde_data=h0,
                        alpha_s=bad)

    def test_explicit_coupling_changes_the_result_monotonically(self):
        x = np.linspace(0.02, 1.48, 21)
        h0 = np.exp(-x ** 2)
        zero = hR_PDF(x, Pz_=4, conf="L24x72", hR_tilde_data=h0, alpha_s=0.0)
        weak = hR_PDF(x, Pz_=4, conf="L24x72", hR_tilde_data=h0, alpha_s=0.1)
        strong = hR_PDF(x, Pz_=4, conf="L24x72", hR_tilde_data=h0, alpha_s=0.4)
        self.assertGreater(
            float(np.max(np.abs(strong - zero))),
            float(np.max(np.abs(weak - zero))))


class TmdHybridEntryContract(unittest.TestCase):
    def test_default_path_is_unchanged(self):
        x = np.array([0.2, 0.4, 0.6])
        x_tmd = np.exp(-x)
        first = tmd_matching_hybrid(
            x, pz_gev=2.0, pz_scale=2.0, cs_kernel=0.0,
            soft_factor=1.0, x_tmd=x_tmd)
        second = tmd_matching_hybrid(
            x, pz_gev=2.0, pz_scale=2.0, cs_kernel=0.0,
            soft_factor=1.0, x_tmd=x_tmd)
        assert_array_equal(first[1], second[1])

    def test_alpha_s_source_is_reported(self):
        x = np.array([0.2, 0.4, 0.6])
        x_tmd = np.exp(-x)
        _, _, default_meta = tmd_matching_hybrid(
            x, cs_kernel=0.0, soft_factor=1.0, x_tmd=x_tmd,
            return_metadata=True)
        _, _, explicit_meta = tmd_matching_hybrid(
            x, cs_kernel=0.0, soft_factor=1.0, x_tmd=x_tmd,
            alpha_s=0.3, return_metadata=True)
        self.assertEqual(default_meta["alpha_s_source"], "A_s_run(mu)*4*pi")
        self.assertEqual(explicit_meta["alpha_s_source"], "explicit_input")
        self.assertEqual(meta_value := explicit_meta["alpha_s"], 0.3)
        self.assertEqual(meta_value, 0.3)

    def test_x_outside_y_support_is_rejected_by_the_matrix_gate(self):
        """x 超出 y 支撑时离散矩阵先失败，不靠 interp 静默夹值。"""
        y = np.linspace(0.3, 0.9, 4)
        x = np.linspace(0.1, 0.8, 3)
        with self.assertRaisesRegex(
                ValueError, "outside y-grid support|support"):
            tmd_matching_hybrid(
                x, y_grid=y, cs_kernel=0.0, soft_factor=1.0,
                x_tmd=np.exp(-y))

    def test_default_metadata_keeps_proxy_semantics(self):
        x = np.array([0.2, 0.4, 0.6])
        _, _, meta = tmd_matching_hybrid(
            x, cs_kernel=0.0, soft_factor=1.0, x_tmd=np.exp(-x),
            return_metadata=True)
        self.assertEqual(
            meta["status"], "one_loop_hybrid_matching_diagnostic")
        self.assertFalse(meta["physical_acceptance"])
        self.assertEqual(meta["proxy_terms"],
                         ["cs_kernel_zero", "soft_factor_one"])

    def test_matching_order_two_fails_closed_instead_of_silently_downgrading(self):
        x = np.array([0.2, 0.4, 0.6])
        with self.assertRaisesRegex(NotImplementedError, "g\\^\\(2\\)"):
            tmd_matching_hybrid(
                x, cs_kernel=0.0, soft_factor=1.0, x_tmd=np.exp(-x),
                matching_order=2)
        _, _, meta = tmd_matching_hybrid(
            x, cs_kernel=0.0, soft_factor=1.0, x_tmd=np.exp(-x),
            return_metadata=True)
        self.assertEqual(meta["matching_order"], 1)
        self.assertEqual(meta["two_loop_kernel"], "unavailable")
        self.assertEqual(
            meta["two_loop_endpoint_hook"], "match_one(endpoint_order=2)")
        self.assertEqual(
            meta["two_loop_local_normalization"],
            "local_ratio_factor(order=2)")


class HRxTailContract(unittest.TestCase):
    def test_tail_none_is_bitwise_identical(self):
        fixture = _hybrid_fixture()
        x = np.array([0.1, 0.3, 0.5])
        explicit_none = hR_x(x, tail_model='none', **fixture)
        default = hR_x(x, **fixture)
        assert_array_equal(default, explicit_none)

    def test_power_exp_tail_adds_only_the_beyond_cutoff_piece(self):
        fixture = _hybrid_fixture()
        x = np.array([0.1, 0.3, 0.5])
        base = hR_x(x, **fixture)
        full, metadata = hR_x(
            x, tail_model='power_exp', return_metadata=True, **fixture)
        self.assertTrue(metadata["tail_applied"])
        self.assertEqual(metadata["tail_model"], "power_exp")
        delta = full - base
        self.assertGreater(
            float(np.max(np.abs(delta))),
            float(np.max(np.abs(base))) * 1.0e-14)
        self.assertGreater(metadata["tail_max_abs"], 0.0)
        # λ0=0.5、λ_max≈1.27：尾部在 λ_max 之后已衰减，贡献远小于主体
        self.assertLess(metadata["tail_relative_to_cut"], 1.0)

    def test_la_nla_tail_is_larger_for_a_slowly_decaying_shape(self):
        fixture = _hybrid_fixture()
        x = np.array([0.1, 0.3, 0.5])
        base = hR_x(x, **fixture)
        _, power_meta = hR_x(
            x, tail_model='power_exp', return_metadata=True, **fixture)
        _, nla_meta = hR_x(
            x, tail_model='la_nla', tail_params=(1.0, 0.2, 0.5),
            return_metadata=True, **fixture)
        # λ0=0.5 的指数衰减比 η=0.5 的 LA/NLA 更快 → power_exp 贡献更小
        self.assertGreater(
            nla_meta["tail_max_abs"], power_meta["tail_max_abs"])

    def test_tail_parameters_are_echoed_in_metadata(self):
        fixture = _hybrid_fixture()
        x = np.array([0.2])
        _, metadata = hR_x(
            x, tail_model='la_nla', tail_params=(1.0, 0.25, 0.4),
            return_metadata=True, **fixture)
        self.assertEqual(
            [complex(value) for value in metadata["tail_parameters"]],
            [complex(1.0), complex(0.25), complex(0.4)])

    def test_tail_model_and_parameters_are_validated(self):
        fixture = _hybrid_fixture()
        x = np.array([0.2])
        with self.assertRaisesRegex(ValueError, "tail_model"):
            hR_x(x, tail_model='nonexistent', **fixture)
        with self.assertRaisesRegex(ValueError, "tail_params"):
            hR_x(x, tail_model='la_nla', **fixture)
        with self.assertRaisesRegex(ValueError, "tail_params"):
            hR_x(x, tail_model='la_nla', tail_params=(1.0, 0.5), **fixture)

    def test_hermitian_tail_keeps_both_channels(self):
        z = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        hR_pz = np.array([2.0, 1.9 + 0.1j, 1.7 + 0.2j, 1.4 + 0.25j,
                          1.1 + 0.3j, 0.9 + 0.32j])
        hR_0 = np.array([1.0, 0.97, 0.92, 0.86, 0.80, 0.74])
        zr = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        x = np.array([0.1, 0.3])
        base = hR_x(x, transform='hermitian', Pz_=4, lambda_extra_=2.0,
                    l1_fit_=1.0, a1_fit_=1.0, lambda0_fit_=0.5, zs=0.4,
                    z_data=z, hR_pz=hR_pz, hR_0=hR_0, zr_fit=zr,
                    conf="L24x72")
        tailed = hR_x(x, transform='hermitian', tail_model='la_nla',
                      tail_params=(0.8 + 0.1j, 0.2 - 0.05j, 0.5),
                      Pz_=4, lambda_extra_=2.0, l1_fit_=1.0, a1_fit_=1.0,
                      lambda0_fit_=0.5, zs=0.4, z_data=z, hR_pz=hR_pz,
                      hR_0=hR_0, zr_fit=zr, conf="L24x72")
        self.assertTrue(np.iscomplexobj(base))
        self.assertTrue(np.iscomplexobj(tailed))
        delta = tailed - base
        self.assertGreater(abs(complex(delta[0].imag)), 0.0)


class MatchOneEndpointOrderContract(unittest.TestCase):
    def test_default_is_the_resummed_one_loop_form(self):
        array = _quasi_fixture()
        matched, metadata = match_one(
            array, 0.5, 0.0775, 2.0, 0.25)
        self.assertEqual(metadata["scheme_conversion_details"]["order"],
                         "one_loop")
        self.assertEqual(
            metadata["scheme_conversion_details"]["endpoint_form"],
            "resummed_one_loop")
        self.assertEqual(metadata["scheme_conversion_details"]["endpoint_order"],
                         1)

    def test_second_order_uses_the_strict_two_loop_series(self):
        array = _quasi_fixture()
        first, _ = match_one(array.copy(), 0.5, 0.0775, 2.0, 0.25)
        second, metadata = match_one(
            array.copy(), 0.5, 0.0775, 2.0, 0.25, endpoint_order=2)
        self.assertEqual(metadata["scheme_conversion_details"]["order"],
                         "two_loop")
        self.assertEqual(
            metadata["scheme_conversion_details"]["endpoint_form"],
            "strict_two_loop_series")
        self.assertEqual(
            metadata["scheme_conversion"],
            "two_loop_endpoint_applied_without_kappa_F")
        # 默认 z0_policy='nan' 让 z=0 整列 NaN，比较前必须剔除
        finite = np.isfinite(first) & np.isfinite(second)
        self.assertTrue(finite.any())
        relative = float(np.max(np.abs(second[finite] - first[finite]))) / float(
            np.max(np.abs(first[finite])))
        # 与重求和形式差 O(a_s²)：不能逐位相同，也不能是量级错误
        self.assertGreater(relative, 1.0e-5)
        self.assertLess(relative, 0.5)

    def test_second_order_one_loop_part_matches_the_resummed_expansion(self):
        """order=2 减 order=1 只差 O(a_s²)；差异必须等于 3(a_s C_A L_f)²。"""
        import math

        from pyqcd.renorm._gradient_flow_quasi import (
            coefficients, match_one as local_match)
        from pyqcd.renorm._nnlo import flow_log, gluon_endpoint_factor

        tau, a_fm, mu, alpha_s, ca = 0.5, 0.0775, 2.0, 0.25, 3.0
        c_parallel, c_perp, _, _ = coefficients(
            tau, a_fm, mu, alpha_s, ca)
        L_f = flow_log(tau, a_fm, mu)
        strict_one = gluon_endpoint_factor("U", alpha_s, L_f, order=1)
        resummed = 1.0 / (c_perp * c_perp)
        u = alpha_s / (4.0 * math.pi) * ca * L_f
        # 1/(1+u)² = 1 − 2u + 3u² − 4u³ + …，严格一圈 1−2u
        # ⇒ 差 = 3u² − 4u³ + …，主项 3u²；u≈5.6e-3 时下一项贡献 ~0.75%
        gap = resummed - strict_one
        self.assertAlmostEqual(gap / (3.0 * u * u), 1.0, delta=0.01)

    def test_invalid_endpoint_order_is_rejected(self):
        array = _quasi_fixture()
        for bad in (0, 3, True):
            with self.subTest(endpoint_order=bad):
                with self.assertRaises(ValueError):
                    match_one(array, 0.5, 0.0775, 2.0, 0.25,
                              endpoint_order=bad)


def test_production_branch_wiring_contracts():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (HRPDFEntryContract, TmdHybridEntryContract,
                 HRxTailContract, MatchOneEndpointOrderContract):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("production branch wiring contracts failed")


if __name__ == "__main__":
    test_production_branch_wiring_contracts()
