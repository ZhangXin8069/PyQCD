"""Fail-closed policy contracts for gradient-flow quasi-operator matching."""
from __future__ import annotations

import math
import unittest

import numpy as np

from pyqcd.renorm._gradient_flow_quasi import (
    CA,
    EULER_GAMMA,
    GEV_FM,
    match_one,
)


_TAU = 0.5
_A_FM = 0.0775
_MU_GEV = 2.0
_ALPHA_S = 0.25


def _fixture():
    """One OPE with independent Mtiti/Mijij and all four orientations."""
    array = np.zeros((1, 2, 4, 6, 3, 2), dtype=np.complex128)
    array[:, :, :, 1, :, :] = np.asarray([2.0, 4.0, 6.0])[
        None, None, None, :, None
    ]
    array[:, :, :, 2, :, :] = np.asarray([1.0, 2.0, 3.0])[
        None, None, None, :, None
    ]
    array[:, :, :, 0, :, :] = (
        array[:, :, :, 1, :, :] - array[:, :, :, 2, :, :]
    )
    array[:, :, :, 3, :, :] = (
        array[:, :, :, 1, :, :] - array[:, :, :, 2, :, :]
    )
    return array


def _endpoint_factors():
    t_gev = _TAU * (_A_FM * GEV_FM) ** 2
    c_perp = 1.0 + _ALPHA_S * CA / (4.0 * math.pi) * (
        math.log(2.0 * _MU_GEV ** 2 * t_gev) + EULER_GAMMA
    )
    delta_m = -_ALPHA_S * CA / (4.0 * math.pi) * math.sqrt(
        2.0 * math.pi / t_gev
    )
    line = np.exp(-delta_m * np.arange(3) * _A_FM * GEV_FM)
    return 1.0 / c_perp ** 2, 1.0 / c_perp, line


def _match(array, **kwargs):
    return match_one(
        array, _TAU, _A_FM, _MU_GEV, _ALPHA_S, **kwargs
    )


class GradientFlowQuasiPolicyContract(unittest.TestCase):
    def test_default_z0_is_nan_while_nonlocal_values_are_preserved(self):
        array = _fixture()
        factor_u, factor_h, line = _endpoint_factors()

        matched, metadata = _match(array)

        self.assertTrue(np.isnan(matched[..., 0, :]).all())
        np.testing.assert_allclose(
            matched[:, 0, :, 0, 1:, :],
            array[:, 0, :, 0, 1:, :]
            * factor_u * line[1:, None],
            rtol=0, atol=3e-14,
        )
        np.testing.assert_allclose(
            matched[:, 1, :, 0, 1:, :],
            array[:, 1, :, 0, 1:, :]
            * factor_h * line[1:, None],
            rtol=0, atol=3e-14,
        )
        self.assertEqual(metadata["z0_status"], "nan_fail_closed")
        self.assertEqual(metadata["window_valid"], False)
        self.assertEqual(metadata["physical_acceptance"], False)
        self.assertEqual(
            metadata["scheme_conversion"],
            "one_loop_endpoint_applied_without_kappa_F",
        )

    def test_z0_spin2_local_conversion_matches_analytic_oracle(self):
        array = _fixture()
        matched, metadata = _match(array, z0_policy="spin2_local")

        t_gev = _TAU * (_A_FM * GEV_FM) ** 2
        alpha = _ALPHA_S * CA / (4.0 * math.pi)
        c_loc = 1.0 + alpha * (
            11.0 / 3.0
            * (math.log(2.0 * _MU_GEV ** 2 * t_gev) + EULER_GAMMA)
            + 7.0 / 3.0
        )
        expected_local = (
            (2.0 - c_loc)
            * (array[:, 0, :, 1, 0, :] - array[:, 0, :, 2, 0, :])
        )

        np.testing.assert_allclose(
            matched[:, 0, :, 0, 0, :], expected_local,
            rtol=0, atol=3e-14,
        )
        self.assertTrue(
            np.isnan(matched[:, 0, :, 1:, 0, :]).all()
        )
        self.assertTrue(
            np.isnan(matched[:, 1, :, :, 0, :]).all()
        )
        self.assertEqual(
            metadata["z0_status"],
            "spin2_local_unpolarized_combined",
        )
        self.assertAlmostEqual(
            metadata["c_local_spin2"], c_loc, places=15
        )
        self.assertAlmostEqual(
            metadata["local_spin2_factor"], 2.0 - c_loc, places=15
        )

    def test_window_mask_separates_validity_from_acceptance(self):
        array = _fixture()
        partial = np.array([False, True, True], dtype=bool)
        matched, metadata = _match(array, window_mask=partial)

        self.assertTrue(np.isnan(matched[..., 0, :]).all())
        self.assertTrue(np.isfinite(matched[..., 1:, :]).all())
        self.assertEqual(metadata["window_valid"], False)
        self.assertEqual(metadata["window"]["invalid_count"], 1)
        self.assertEqual(metadata["physical_acceptance"], False)

        all_valid = np.ones(3, dtype=bool)
        _, accepted_metadata = _match(array, window_mask=all_valid)
        self.assertEqual(accepted_metadata["window_valid"], True)
        self.assertEqual(accepted_metadata["physical_acceptance"], False)

    def test_helicity_physical_minus_and_legacy_plus_are_separate(self):
        array = _fixture()
        _, factor_h, line = _endpoint_factors()

        physical, physical_metadata = _match(array)
        legacy, legacy_metadata = _match(
            array, helicity_mode="legacy_plus_combined"
        )

        expected_minus = (
            (array[:, 1, :, 1, 1:, :] - array[:, 1, :, 2, 1:, :])
            * factor_h * line[1:, None]
        )
        expected_plus = (
            (array[:, 1, :, 1, 1:, :] + array[:, 1, :, 2, 1:, :])
            * factor_h * line[1:, None]
        )
        np.testing.assert_allclose(
            physical[:, 1, :, 0, 1:, :], expected_minus,
            rtol=0, atol=3e-14,
        )
        np.testing.assert_allclose(
            legacy[:, 1, :, 0, 1:, :], expected_plus,
            rtol=0, atol=3e-14,
        )
        self.assertEqual(
            physical_metadata["helicity_status"],
            "physical_T_minus_S",
        )
        self.assertEqual(
            legacy_metadata["helicity_status"],
            "legacy_plus_combined_diagnostic",
        )
        self.assertGreater(
            float(np.max(np.abs(
                physical[:, 1, :, 0, 1:, :]
                - legacy[:, 1, :, 0, 1:, :]
            ))),
            1e-3,
        )

    def test_legacy_plus_cannot_be_declared_physical(self):
        array = _fixture()
        with self.assertRaisesRegex(
                ValueError, "legacy_plus_combined.*physical_acceptance"):
            _match(
                array,
                helicity_mode="legacy_plus_combined",
                window_mask=np.ones(3, dtype=bool),
                kappa_F=1.0,
                physical_acceptance=True,
            )

    def test_legacy_z0_factor_is_explicit_and_remains_diagnostic(self):
        array = _fixture()
        factor_u, _, line = _endpoint_factors()
        matched, metadata = _match(array, z0_policy="legacy_endpoint")

        np.testing.assert_allclose(
            matched[:, 0, :, 0, 0, :],
            array[:, 0, :, 0, 0, :] * factor_u * line[0],
            rtol=0, atol=3e-14,
        )
        self.assertEqual(
            metadata["z0_status"],
            "legacy_endpoint_factor_explicit_opt_in",
        )
        with self.assertRaisesRegex(ValueError, "legacy z=0"):
            _match(
                array,
                z0_policy="legacy_endpoint",
                window_mask=np.ones(3, dtype=bool),
                kappa_F=1.0,
                physical_acceptance=True,
            )

    def test_reject_policy_and_acceptance_prerequisites_fail_closed(self):
        array = _fixture()
        with self.assertRaisesRegex(ValueError, "reject"):
            _match(array, z0_policy="reject")
        with self.assertRaisesRegex(ValueError, "kappa_F"):
            _match(
                array,
                window_mask=np.ones(3, dtype=bool),
                physical_acceptance=True,
            )

        matched, metadata = _match(
            array,
            window_mask=np.ones(3, dtype=bool),
            kappa_F=1.25,
            physical_acceptance=True,
        )
        factor_u, _, line = _endpoint_factors()
        np.testing.assert_allclose(
            matched[:, 0, :, 0, 1:, :],
            array[:, 0, :, 0, 1:, :]
            * factor_u * line[1:, None] * 1.25 ** 2,
            rtol=0, atol=3e-14,
        )
        self.assertEqual(metadata["physical_acceptance"], True)
        self.assertEqual(metadata["window_valid"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
