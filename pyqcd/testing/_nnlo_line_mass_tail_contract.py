"""NNLO 系数、H3 线质量核、解析尾 Fourier 的数值契约。"""
from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

from pyqcd.renorm._line_mass import (
    effective_slope,
    finite_segment_kernel,
    flow_radius,
    h3_log_difference,
    line_mass,
    line_mass_kernel,
    line_mass_kernel_remainder,
    one_loop_amplitude,
)
from pyqcd.renorm._nnlo import (
    flow_log,
    gluon_endpoint_coefficients,
    gluon_endpoint_factor,
    local_ratio_coefficients,
    local_ratio_factor,
    short_distance_c0,
)
from pyqcd.renorm._tail import (
    abrupt_cutoff_error,
    analytic_tail_fourier,
    tail_amplitude_fit,
    tail_shape,
)


def _kernel_by_definition(x):
    """``F(x)`` 的定义式直接求值（x 不能太小，用于中等 x 的独立对照）。"""
    g = -math.expm1(-x * x)
    return erf(x) + (g * (1.0 - 2.0 * x * x) - x * x) / (
        2.0 * math.sqrt(math.pi) * x ** 3)


def _tail_quadrature(x, C, B, eta, lower, upper=None):
    def integrand(lam):
        return (C + B / lam) * math.exp(-eta * lam) * math.cos(x * lam) / math.pi

    return quad(integrand, lower, math.inf if upper is None else upper,
                limit=800, epsabs=1.0e-14, epsrel=1.0e-13)[0]


class NnloCoefficientContract(unittest.TestCase):
    def test_quark_vector_reproduces_the_published_shape(self):
        # D_V^NLO = 1 + a_s(4L_f + 10/3) 与 NNLO 三系数（arXiv:2610.40225）
        alpha_s = 0.29812517844614284
        a_s = alpha_s / (4.0 * math.pi)
        for L in (-0.5, 0.0, 1.0, 2.5):
            nlo = local_ratio_factor("V", alpha_s, L, order=1)
            self.assertAlmostEqual(nlo, 1.0 + a_s * (4.0 * L + 10.0 / 3.0),
                                   places=14)
            nnlo = local_ratio_factor("V", alpha_s, L, order=2)
            expected = (1.0 + a_s * (4.0 * L + 10.0 / 3.0)
                        + a_s ** 2 * (26.0 * L ** 2
                                      + 85.3591794985557 * L
                                      + 30.92223469310622))
            self.assertAlmostEqual(nnlo, expected, places=13)

    def test_rg_consistency_fixes_the_double_logarithm(self):
        # L^2 系数 = (f11^2 + b0*f11)/2，f11 = 3C_F，b0 = 11 - 2nf/3
        f11 = 3.0 * (4.0 / 3.0)
        b0 = 11.0 - 2.0 * 3.0 / 3.0
        self.assertAlmostEqual((f11 ** 2 + b0 * f11) / 2.0, 26.0, places=13)
        # 由 local_ratio_factor 的二阶系数在 L 上的二次项反解同样得到 26
        alpha_s = 0.3
        a_s = alpha_s / (4.0 * math.pi)
        low = local_ratio_factor("V", alpha_s, 0.0, order=2)
        high = local_ratio_factor("V", alpha_s, 1.0, order=2)
        # 二次项：a2(L) 的 L^2 系数 = 26
        second = local_ratio_coefficients("V", L=2.0)[1]
        zero = local_ratio_coefficients("V", L=0.0)[1]
        one = local_ratio_coefficients("V", L=1.0)[1]
        curvature = second - 2.0 * one + zero
        self.assertAlmostEqual(curvature, 2.0 * 26.0, places=9)
        self.assertGreater(high, low)
        self.assertGreater(a_s, 0.0)

    def test_ringed_two_loop_constant_cancels_in_the_ratio(self):
        for channel in ("V", "T"):
            base = local_ratio_coefficients(channel)
            for ringed in (-50.0, 0.0, 100.0):
                other = local_ratio_coefficients(
                    channel, ringed_constant=ringed)
                self.assertEqual(base, other)

    def test_transversity_keeps_the_flow_logarithm(self):
        # 手征奇通道无 Ward 恒等式保护，L_f 不能相消：一圈斜率 = 4C_F
        alpha_s = 0.3
        a_s = alpha_s / (4.0 * math.pi)
        low = local_ratio_factor("T", alpha_s, 0.0, order=1)
        high = local_ratio_factor("T", alpha_s, 2.0, order=1)
        self.assertAlmostEqual((high - low) / (2.0 * a_s), 4.0 * 4.0 / 3.0,
                               places=12)

    def test_gluon_endpoint_products_match_the_rounded_reference(self):
        for nf in (0, 3):
            pairs = gluon_endpoint_coefficients(nf)
            # 平行对只有两圈项（p1=0），c2 斜率与常数与参考三位小数一致
            self.assertEqual(pairs["P"][0], 0.0)
            self.assertAlmostEqual(pairs["P"][1], -171.62092757651598, places=10)
            # U/H 的 L^0 常数随 nf 的线性系数（参考 -13.1788 / -6.58938）
            self.assertAlmostEqual(
                pairs["U"][1], -203.76273091296028 - 13.1788 * (nf - 3),
                places=1)
            self.assertAlmostEqual(
                pairs["H"][1], -187.69182924473813 - 6.58938 * (nf - 3),
                places=1)

    def test_gluon_endpoint_inverse_is_a_faithful_series(self):
        alpha_s = 0.3
        a_s = alpha_s / (4.0 * math.pi)
        for combination, (c1, c2) in gluon_endpoint_coefficients(3).items():
            factor = gluon_endpoint_factor(
                combination, alpha_s, 0.0, order=2)
            self.assertAlmostEqual(
                factor, 1.0 - c1 * a_s + (c1 * c1 - c2) * a_s ** 2, places=14)
            # 逆因子 1/ζ = 1 - c1 a_s + (c1²-c2)a_s²，L=0 时 c1=0 只剩 c2 项
            self.assertGreater(factor, 1.0)
            # 一圈与两圈可加性：order=1 即 1 - c1 a_s
            self.assertAlmostEqual(
                gluon_endpoint_factor(combination, alpha_s, 0.0, order=1),
                1.0, places=14)

    def test_short_distance_c0_uses_the_correct_prefactor(self):
        alpha_s, mu = 0.298, 2.0
        z_fm = 0.2
        L = math.log((z_fm / 0.1973269804) ** 2 * mu ** 2 / 4.0) + 2.0 * 0.5772156649015329
        nlo = short_distance_c0(z_fm, mu, alpha_s, order=1)
        expected = 1.0 + alpha_s * (4.0 / 3.0) / (2.0 * math.pi) * (
            1.5 * L + 2.5)
        self.assertAlmostEqual(nlo, expected, places=13)
        # NNLO 前因子是 alpha_s^2/(2 pi^2)，不是 (alpha_s/(2 pi))^2（差 2 倍）
        quad_only = short_distance_c0(z_fm, mu, alpha_s, order=2) - nlo
        naive = (alpha_s / (2.0 * math.pi)) ** 2
        self.assertAlmostEqual(quad_only / naive, 2.0 * 42.943457131817404,
                               places=6)

    def test_short_distance_rejects_the_local_point(self):
        with self.assertRaises(ValueError):
            short_distance_c0(0.0, 2.0, 0.3)
        with self.assertRaises(ValueError):
            short_distance_c0([-0.1], 2.0, 0.3)

    def test_flow_log_uses_gev_minus_two_flow_time(self):
        # L_f = ln[2 mu^2 t_f] + gammaE，t_f = tau*a^2（fm -> GeV^-1）
        tau, a_fm, mu = 3.0, 0.105, 2.0
        t_gev = tau * (a_fm / 0.1973269804) ** 2
        expected = math.log(2.0 * mu * mu * t_gev) + 0.5772156649015329
        self.assertAlmostEqual(flow_log(tau, a_fm, mu), expected, places=13)
        with self.assertRaises(ValueError):
            flow_log(0.0, a_fm, mu)


class LineMassKernelContract(unittest.TestCase):
    def test_kernel_matches_the_definition_at_moderate_x(self):
        for x in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            self.assertAlmostEqual(
                line_mass_kernel(x), _kernel_by_definition(x), places=13)

    def test_kernel_is_singularity_free_at_small_x(self):
        # 0.75x/sqrt(pi) 的领头行为，且不像直接求值那样抵消
        small = 1.0e-12
        self.assertAlmostEqual(
            line_mass_kernel(small) / (0.75 * small / math.sqrt(math.pi)),
            1.0, places=12)
        self.assertEqual(line_mass_kernel(0.0), 0.0)
        # 与定义式的比值在小 x 应为 1（定义式在 x=1e-3 已损失 ~1e-10）
        self.assertAlmostEqual(
            line_mass_kernel(1.0e-3) / _kernel_by_definition(1.0e-3),
            1.0, places=8)

    def test_kernel_saturates_with_the_known_asymptote(self):
        # F(x) = 1 - 3/(2 sqrt(pi) x) + O(x^-3)，收敛很慢，必须用渐近式核验
        for x in (1.0e4, 1.0e6, 1.0e8):
            expected = 1.0 - 3.0 / (2.0 * math.sqrt(math.pi) * x)
            self.assertAlmostEqual(
                line_mass_kernel(x) / expected, 1.0, places=12)
        self.assertLess(line_mass_kernel(20.0), 0.96)
        self.assertGreater(line_mass_kernel(1000.0), 0.999)

    def test_kernel_is_scalar_and_array_consistent(self):
        values = np.array([0.0, 0.05, 0.5, 5.0])
        batch = line_mass_kernel(values)
        self.assertEqual(batch.shape, values.shape)
        for index, value in enumerate(values):
            self.assertAlmostEqual(float(batch[index]),
                                   line_mass_kernel(value), places=15)

    def test_remainder_reproduces_the_long_distance_expansion(self):
        # K = 1/r - 3/(2 sqrt(pi) z) + r^2/(2 sqrt(pi) z^3) + R_K
        for x in (5.0, 8.0, 12.0):
            r = 0.18
            z = x * r
            k_exact = line_mass_kernel(x) / r
            expanded = (1.0 / r - 3.0 / (2.0 * math.sqrt(math.pi) * z)
                        + r ** 2 / (2.0 * math.sqrt(math.pi) * z ** 3))
            remainder = line_mass_kernel_remainder(x) / r
            self.assertAlmostEqual(k_exact, expanded + remainder, places=14)
            # R_K 在 x≫1 时指数小，且首项为 -3r^4 e^-x^2/(4 sqrt(pi) z^5)
            self.assertLess(abs(remainder), 1.0e-6 / r)

    def test_line_mass_uses_dimensionally_consistent_units(self):
        alpha_s, r_fm = 0.3, 0.18
        d1 = one_loop_amplitude(alpha_s)
        # z -> 0 时 mu_R -> 0；z -> 无穷时 mu_R -> d1/r（GeV），收敛是 1/x 慢
        self.assertAlmostEqual(float(line_mass(0.0, r_fm, alpha_s)), 0.0,
                              places=15)
        plateau = float(line_mass(1.0e6 * r_fm, r_fm, alpha_s))
        self.assertAlmostEqual(
            plateau, d1 / (r_fm / 0.1973269804), places=5)

    def test_flow_radius_and_amplitude_are_positive(self):
        self.assertAlmostEqual(flow_radius(3.0, 0.105),
                               math.sqrt(24.0) * 0.105, places=15)
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                flow_radius(bad, 0.105)
            with self.assertRaises(ValueError):
                one_loop_amplitude(bad)

    def test_effective_slope_limits(self):
        # d_eff = d - k3·r_i r_j (r_i+r_j)/z³ 从下方趋于 d（k3>0）
        r_i, r_j = 0.16, 0.1965
        slope = effective_slope([10.0, 1000.0], r_i, r_j, 0.8, 0.3)
        self.assertLess(slope[0], slope[1])
        self.assertAlmostEqual(float(slope[1]), 0.8, places=5)
        self.assertAlmostEqual(
            float(slope[0]),
            0.8 - 0.3 * r_i * r_j * (r_i + r_j) / 10.0 ** 3, places=12)
        # k3 = 0 时与 z 无关
        flat = effective_slope([10.0, 1000.0], r_i, r_j, 0.8, 0.0)
        np.testing.assert_allclose(flat, 0.8, rtol=0.0, atol=1e-15)

    def test_finite_segment_kernel_differs_from_the_midpoint(self):
        r_i, r_j = 0.16, 0.1965
        z_left, z_right = 0.6, 1.0
        exact = finite_segment_kernel(z_left, z_right, r_i, r_j)
        midpoint = ((r_i ** 2 - r_j ** 2) / (1.0 / r_i - 1.0 / r_j)
                    * 0.5 * (z_left + z_right) ** -3)
        self.assertNotAlmostEqual(exact, midpoint, places=4)
        with self.assertRaises(ValueError):
            finite_segment_kernel(z_right, z_left, r_i, r_j)
        with self.assertRaises(ValueError):
            finite_segment_kernel(z_left, z_right, r_i, r_i)

    def test_h3_shape_and_reference_row(self):
        z = np.array([0.5, 0.7, 0.9])
        radii = np.array([0.1605, 0.1794, 0.1965])
        out = h3_log_difference(z, radii, 0.8, 0.3)
        self.assertEqual(out.shape, (2, z.size))
        # 参考行 0：第 0 个非参考 flow 就是 index 1
        expected = (-0.8 * z * (1.0 / radii[1] - 1.0 / radii[0])
                    + 0.3 * (radii[1] ** 2 - radii[0] ** 2) / (2.0 * z ** 2))
        np.testing.assert_allclose(out[0], expected, rtol=0.0, atol=1e-15)
        # 全零幅度给出全零对数差
        zeros = h3_log_difference(z, radii, 0.0, 0.0)
        np.testing.assert_allclose(zeros, 0.0, rtol=0.0, atol=1e-15)

    def test_h3_parameter_count_is_nflow_plus_one(self):
        radii = np.array([0.1605, 0.1794, 0.1965])
        z = np.array([0.5, 0.7, 0.9, 1.1])
        rows = h3_log_difference(z, radii, 0.8, 0.3).size
        # Nf-1 组独立对数差 x Nz 数据点；参数为 d、k3 与 Nf-1 个截距（差掉参考）
        self.assertEqual(rows, z.size * (radii.size - 1))
        self.assertEqual(2 + (radii.size - 1), radii.size + 1)


class AnalyticTailContract(unittest.TestCase):
    def test_closed_form_matches_direct_quadrature(self):
        x = np.array([0.0, 0.2, 0.5, 1.5, 3.0])
        analytic = analytic_tail_fourier(x, 1.0, 0.3, 0.5, 1.0)
        for index, value in enumerate(x):
            reference = _tail_quadrature(value, 1.0, 0.3, 0.5, 1.0)
            self.assertAlmostEqual(float(analytic[index]), reference, places=10)

    def test_la_limit_drops_the_single_pole_term(self):
        x = 1.0
        la = analytic_tail_fourier(x, 2.0, 0.0, 0.4, 0.8)
        reference = _tail_quadrature(x, 2.0, 0.0, 0.4, 0.8)
        self.assertAlmostEqual(float(la), reference, places=10)
        nla = analytic_tail_fourier(x, 2.0, -0.5, 0.4, 0.8)
        self.assertNotAlmostEqual(float(la), float(nla), places=3)

    def test_cutoff_error_is_the_dropped_piece(self):
        x = np.array([0.1, 0.7, 2.0])
        dropped = abrupt_cutoff_error(x, 1.0, 0.3, 0.5, 20.0)
        for index, value in enumerate(x):
            reference = _tail_quadrature(value, 1.0, 0.3, 0.5, 20.0)
            self.assertAlmostEqual(float(dropped[index]), reference, places=12)
            self.assertLess(abs(float(dropped[index])), 1.0e-3)

    def test_closed_form_rejects_an_uncontrolled_lower_limit(self):
        with self.assertRaises(ValueError):
            analytic_tail_fourier(0.5, 1.0, 0.3, 0.5, 0.0)
        with self.assertRaises(ValueError):
            analytic_tail_fourier(0.5, 1.0, 0.3, -0.5, 1.0)
        with self.assertRaises(ValueError):
            abrupt_cutoff_error(0.5, 1.0, 0.3, 0.5, 0.0)

    def test_scalar_inputs_return_floats(self):
        value = analytic_tail_fourier(0.5, 1.0, 0.2, 0.6, 1.2)
        self.assertIsInstance(value, float)
        shape = analytic_tail_fourier(np.zeros((2, 3)), 1.0, 0.2, 0.6, 1.2)
        self.assertEqual(shape.shape, (2, 3))

    def test_tail_shape_decays_and_is_positive(self):
        lam = np.array([1.0, 2.0, 4.0])
        values = tail_shape(lam, 0.5)
        self.assertTrue(np.all(np.diff(values) < 0.0))
        self.assertTrue(np.all(values > 0.0))
        with np.errstate(all="ignore"):
            self.assertTrue(np.isinf(tail_shape(0.0, 0.5, exponent=1.0)))
        with self.assertRaises(ValueError):
            tail_shape(lam, 0.0)

    def test_single_amplitude_fit_recovers_a_known_slope(self):
        lam = np.linspace(0.5, 3.0, 12)
        true_amplitude, eta = 1.7, 0.45
        data = true_amplitude * tail_shape(lam, eta)
        errors = np.full(lam.size, 0.01)
        amplitude, chi2, sigma = tail_amplitude_fit(lam, data, errors, eta)
        self.assertAlmostEqual(amplitude, true_amplitude, places=11)
        self.assertLess(chi2, 1.0e-18)
        self.assertAlmostEqual(sigma, 0.01 * math.sqrt(
            1.0 / np.sum(tail_shape(lam, eta) ** 2)), places=12)
        with self.assertRaises(ValueError):
            tail_amplitude_fit(lam, data, np.zeros_like(lam), eta)


def test_nnlo_line_mass_tail_contracts():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (NnloCoefficientContract, LineMassKernelContract,
                 AnalyticTailContract):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("NNLO / line-mass / tail contracts failed")


if __name__ == "__main__":
    test_nnlo_line_mass_tail_contracts()
