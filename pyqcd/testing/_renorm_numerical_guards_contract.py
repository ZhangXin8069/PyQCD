"""Regression contracts for fail-closed renormalization numerical guards."""
from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

import numpy as np

from pyqcd.renorm import (
    alpha_s,
    fit_hR_PDF_extrap_boot,
    hR_form,
    hR_PDF,
    hR_z_Pz,
    prepare_zr_dataset,
    summarize_ZR_samples,
    tmd_matching_hybrid,
)
from pyqcd.renorm import cs_kernel_from_ratio, cs_kernel_two_momentum
from pyqcd.renorm import _tmd


class HybridComplexContract(unittest.TestCase):
    def test_complex_hR_keeps_the_imaginary_channel(self):
        z = np.array([0.0, 0.2, 0.8])
        hR_pz = np.array([2.0 + 3.0j, 4.0 + 5.0j, 6.0 + 7.0j])
        hR_0 = np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j])
        zr = np.array([1.0, 1.5, 2.0 + 0.25j])

        _, actual = hR_z_Pz(
            z, 4, hR_pz, hR_0, zs=0.4, zr_fit=zr, conf="L24x72")

        expected = hR_pz / hR_0
        expected[2] = (hR_pz[2] / zr[2]) * (zr[2] / hR_0[2])
        self.assertTrue(np.iscomplexobj(actual))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)
        self.assertGreater(float(np.max(np.abs(actual.imag))), 0.0)

    def test_lambda_fit_rejects_complex_data_instead_of_dropping_imaginary(self):
        lam = np.linspace(0.5, 3.0, 8)
        with self.assertRaisesRegex(ValueError, "complex|real"):
            from pyqcd.renorm import fit_hR_lambda

            fit_hR_lambda(
                [1.0, 1.0, 2.0], (0.5, 3.0), lam,
                np.ones(8, dtype=np.complex128) * (1.0 + 0.25j),
                cov_kind="diag")

    def test_complex64_hybrid_outputs_are_not_silently_upcast(self):
        from pyqcd.renorm import hR_x

        z = np.array([0.0, 0.2, 0.8], dtype=np.float32)
        hR_pz = np.array(
            [2.0 + 3.0j, 4.0 + 5.0j, 6.0 + 7.0j],
            dtype=np.complex64,
        )
        hR_0 = np.array(
            [1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j],
            dtype=np.complex64,
        )
        zr = np.array([1.0, 1.5, 2.0 + 0.25j], dtype=np.complex64)

        _, joined = hR_z_Pz(
            z, 4, hR_pz, hR_0, zs=0.4, zr_fit=zr, conf="L24x72")
        self.assertEqual(joined.dtype, np.dtype(np.complex64))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed = hR_x(
                np.array([0.1, 0.2]), 4, 0.3,
                1.0, 1.0, 0.5,
                zs=0.4, z_data=z, hR_pz=hR_pz, hR_0=hR_0,
                zr_fit=zr, conf="L24x72",
            )
        self.assertEqual(transformed.dtype, np.dtype(np.complex64))
        self.assertFalse(any(
            "divide by zero" in str(item.message) for item in caught
        ))


class CouplingDomainContract(unittest.TestCase):
    def test_valid_coupling_formula_is_unchanged(self):
        mu, lam_qcd, nf = 2.0, 0.23, 3.0
        expected = 2.0 * np.pi / (
            (11.0 - 2.0 * nf / 3.0) * np.log(mu / lam_qcd))
        self.assertEqual(alpha_s(mu, lam_qcd, nf), expected)

    def test_out_of_domain_coupling_inputs_are_rejected(self):
        invalid = (
            {"mu": 0.23, "Lambda_QCD": 0.23},
            {"mu": 0.20, "Lambda_QCD": 0.23},
            {"mu": 2.0, "Lambda_QCD": 0.0},
            {"mu": 2.0, "Lambda_QCD": np.inf},
            {"mu": np.nan, "Lambda_QCD": 0.23},
            {"mu": 2.0, "Lambda_QCD": 0.23, "nf": 16.5},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    alpha_s(**arguments)

    def test_boolean_coupling_inputs_are_rejected(self):
        invalid = (
            {"mu": True, "Lambda_QCD": 0.23, "nf": 3.0},
            {"mu": 2.0, "Lambda_QCD": 0.23, "nf": True},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    alpha_s(**arguments)


class KernelInvalidRatioContract(unittest.TestCase):
    def test_positive_ratio_kernel_regression(self):
        h1 = np.array([2.0, 4.0, 8.0])
        h2 = np.array([1.0, 2.0, 4.0])
        expected = np.log(h1[0] / h2[0]) / np.log(4.0 / 2.0)
        np.testing.assert_allclose(
            cs_kernel_from_ratio(h1, h2, 4.0, 2.0),
            expected, rtol=0.0, atol=0.0)

        z = np.arange(3)[:, None]
        c02 = np.exp(-z / 0.3)
        c01 = c02 * 2.0 ** z
        np.testing.assert_allclose(
            cs_kernel_two_momentum(c01, c02, 4.0, 2.0, z_ref=1),
            np.array([1.0]), rtol=0.0, atol=1e-15)

    def test_nonpositive_or_nonfinite_ratios_are_rejected_before_clipping(self):
        for first in (-1.0, 0.0, np.nan):
            with self.subTest(first=first):
                with self.assertRaises(ValueError):
                    cs_kernel_from_ratio(
                        np.array([first]), np.array([1.0]), 4.0, 2.0)
                with self.assertRaises(ValueError):
                    cs_kernel_two_momentum(
                        np.array([[first]]), np.array([[1.0]]),
                        4.0, 2.0, k_clip=(-3.0, 3.0))

    def test_boolean_scalars_and_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            cs_kernel_from_ratio(
                np.array([1.0]), np.array([1.0]), True, 2.0)
        with self.assertRaises(ValueError):
            cs_kernel_two_momentum(
                np.array([[True]]), np.array([[1.0]]),
                4.0, 2.0, z_ref=0)

    def test_zero_node_and_singular_matching_matrix_fail_closed(self):
        x_zero = np.array([0.0, 0.2, 0.4])
        with self.assertRaisesRegex(ValueError, "zero|avoid"):
            tmd_matching_hybrid(
                x_zero, pz_gev=2.0, pz_scale=2.0, cs_kernel=0.0,
                soft_factor=1.0, x_tmd=np.ones(3))

        x = np.array([0.2, 0.4])
        alpha = 0.32
        t = (x[1] - x[0]) * alpha * 3.0 / (2.0 * np.pi)
        g_singular = np.array([
            [1.0, x[1] ** 2 / (t * x[0])],
            [0.0, 1.0],
        ])
        with patch(
                "pyqcd.renorm._tmdextract.A_s_run",
                return_value=alpha / (4.0 * np.pi)), \
                patch(
                    "pyqcd.renorm._tmdextract._matching_kernels",
                    return_value=g_singular):
            with self.assertRaisesRegex(ValueError, "singular|singularity"):
                tmd_matching_hybrid(
                    x, pz_gev=2.0, pz_scale=2.0, cs_kernel=0.0,
                    soft_factor=1.0, x_tmd=np.ones(2))


class MatchingGridContract(unittest.TestCase):
    def test_uniform_grid_regression_is_unchanged(self):
        x = np.linspace(0.02, 1.48, 21)
        h0 = np.exp(-x ** 2)
        with patch("pyqcd.renorm._matching.A_s_run", return_value=0.0):
            actual = hR_PDF(
                x, Pz_=4, conf="L24x72", hR_tilde_data=h0, mu_=2.0)
        np.testing.assert_array_equal(actual, h0)

    def test_nonuniform_grid_is_rejected_instead_of_using_first_spacing(self):
        x = np.array([0.1, 0.2, 0.45])
        with self.assertRaisesRegex(ValueError, "uniform"):
            hR_PDF(
                x, Pz_=4, conf="L24x72",
                hR_tilde_data=np.ones_like(x), mu_=2.0)


class TmdImaginaryResidualContract(unittest.TestCase):
    @staticmethod
    def _complex_value(value):
        return np.full((2, 2, 2, 2), value, dtype=np.complex128)

    def test_real_and_roundoff_scale_imaginary_values_are_preserved(self):
        gauge = np.zeros((2, 1, 1, 1, 4, 3, 3), dtype=np.complex128)
        value = self._complex_value(1.25 + 1.0e-14j)
        with patch(
                "pyqcd.renorm._tmd._tmd_clover_fields",
                return_value=(None, None, None)), \
                patch(
                    "pyqcd.renorm._tmd._gluon_tmd_operator_from_fields",
                    return_value=value):
            averaged = _tmd.tmd_matrix_elements(
                gauge, [0], [0], L=0)
            timed = _tmd.tmd_matrix_elements_time(
                gauge, [0], [0], L=0)
        self.assertTrue(np.isrealobj(averaged))
        self.assertTrue(np.isrealobj(timed))
        np.testing.assert_allclose(averaged, 10.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            timed, np.full((1, 1, 2), 10.0), rtol=0.0, atol=1e-12)

    def test_material_imaginary_residual_is_rejected(self):
        gauge = np.zeros((2, 1, 1, 1, 4, 3, 3), dtype=np.complex128)
        value = self._complex_value(1.25 + 0.1j)
        with patch(
                "pyqcd.renorm._tmd._tmd_clover_fields",
                return_value=(None, None, None)), \
                patch(
                    "pyqcd.renorm._tmd._gluon_tmd_operator_from_fields",
                    return_value=value):
            with self.assertRaisesRegex(ValueError, "imaginary|imag"):
                _tmd.tmd_matrix_elements(gauge, [0], [0], L=0)
            with self.assertRaisesRegex(ValueError, "imaginary|imag"):
                _tmd.tmd_matrix_elements_time(gauge, [0], [0], L=0)


class ZRCovarianceAndSummaryContract(unittest.TestCase):
    def test_diag_covariance_regression_uses_actual_positive_variances(self):
        samples = np.array([[0.0, 1.0], [2.0, 4.0]])
        prepared = prepare_zr_dataset(
            samples, np.array([0.1, 0.2]), 0.5, kind="diag")
        np.testing.assert_array_equal(
            prepared.c_inv, np.diag([4.0, 1.0]))

    def test_exact_zero_diag_variance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero|positive|singular"):
            prepare_zr_dataset(
                np.ones((2, 4)), np.array([0.1, 0.2]), 0.5, kind="diag")

    def test_summary_reports_finite_and_usable_status(self):
        rows = [
            {"sample_i": 0, "k": 1.0, "chi2": 0.1},
            {"sample_i": 1, "k": 2.0, "chi2": 0.2},
            {"sample_i": 2, "k": 3.0, "chi2": 0.3},
        ]
        summary = summarize_ZR_samples(rows)
        self.assertEqual(summary["n_rows"], 3)
        self.assertEqual(summary["n_usable"], 3)
        self.assertEqual(summary["finite_counts"]["k"], 3)
        self.assertTrue(summary["usable"])
        self.assertEqual(summary["status"], "usable")
        np.testing.assert_allclose(summary["k"], (2.0, np.std([1, 2, 3])))

    def test_empty_summary_returns_auditable_empty_status(self):
        summary = summarize_ZR_samples([])
        self.assertEqual(summary["n_rows"], 0)
        self.assertEqual(summary["n_usable"], 0)
        self.assertFalse(summary["usable"])
        self.assertEqual(summary["status"], "empty")

    def test_summary_uses_union_of_row_keys(self):
        rows = [
            {"sample_i": 0, "k": 1.0},
            {"sample_i": 1, "k": 2.0, "chi2": 0.1},
        ]
        summary = summarize_ZR_samples(rows)
        self.assertEqual(set(summary["finite_counts"]), {"k", "chi2"})
        self.assertEqual(summary["finite_counts"]["chi2"], 1)
        self.assertEqual(summary["n_usable"], 1)
        self.assertFalse(summary["usable"])
        self.assertEqual(summary["status"], "partial")


class ExtrapolateCovarianceContract(unittest.TestCase):
    @staticmethod
    def _rows(n_rep, noisy):
        rng = np.random.default_rng(20261013)
        x = np.array([0.25])
        truth = np.array([0.37, -0.21, 0.0, 0.0,
                          0.12, 0.0, 0.08, 0.0])
        ensembles = (
            (0.31, 1.1, 0.15, 24.0),
            (0.34, 1.7, 0.18, 28.0),
            (0.38, 2.1, 0.22, 32.0),
            (0.42, 1.4, 0.16, 36.0),
            (0.47, 1.9, 0.25, 24.0),
            (0.52, 2.4, 0.19, 40.0),
            (0.57, 1.3, 0.28, 28.0),
            (0.63, 2.8, 0.23, 48.0),
        )
        rows = []
        for a_, pz_, mpi, L_ in ensembles:
            value = hR_form((a_, pz_, mpi, L_), truth)
            data = np.full((1, n_rep), value, dtype=float)
            if noisy:
                data += 1.0e-3 * rng.standard_normal(data.shape)
            rows.append({
                "x": x, "hR": data, "a": a_, "pz": pz_,
                "mpi": mpi, "L": L_,
            })
        return rows

    def test_positive_definite_covariance_regression_has_no_fallback_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, mean, _ = fit_hR_PDF_extrap_boot(
                self._rows(n_rep=16, noisy=True))
        self.assertTrue(np.all(np.isfinite(mean)))
        fallback = [
            item for item in caught
            if "identity" in str(item.message).lower()
        ]
        self.assertEqual(fallback, [])

    def test_nonpositive_covariance_fallback_is_explicitly_reported(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, mean, _ = fit_hR_PDF_extrap_boot(
                self._rows(n_rep=2, noisy=True))
        self.assertTrue(np.all(np.isfinite(mean)))
        messages = [str(item.message).lower() for item in caught]
        self.assertTrue(
            any("identity" in message and "covariance" in message
                for message in messages),
            messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
