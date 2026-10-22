"""Contracts for forward Sumratio-difference AIC and fallback selection."""
from __future__ import annotations

import unittest
import os
import tempfile

import numpy as np

from pyqcd.analysis._bare_sumratio import (
    fit_forward_sumratio_difference,
    forward_difference,
    physical_tmin,
    sumratio_values,
)


def _constant_fixture(nboot=80, nt=20, noise=0.01):
    rng = np.random.default_rng(20261022)
    samples = np.full((nboot, nt, nt, 1, 1), np.nan)
    original = np.full((nt, nt, 1, 1), np.nan)
    for tsep in range(nt):
        for tau in range(tsep + 1):
            samples[:, tsep, tau] = 2.0 + rng.normal(
                0.0, noise, size=(nboot, 1, 1))
            original[tsep, tau] = 2.0
    return samples, original


class BareSumratioContract(unittest.TestCase):
    def test_physical_tmin_is_strict_for_all_registered_ensembles(self):
        self.assertEqual(physical_tmin(0.1050), 6)
        self.assertEqual(physical_tmin(0.0897), 7)
        self.assertEqual(physical_tmin(0.0775), 8)
        for a_fm, tmin in ((0.1050, 6), (0.0897, 7), (0.0775, 8)):
            self.assertGreater(tmin * a_fm, 0.6)

    def test_sumratio_and_forward_difference_keep_typed_axes(self):
        ratio = np.arange(8 * 8 * 2 * 3).reshape(8, 8, 2, 3)
        summed = sumratio_values(ratio, 2, np.array([4, 5, 6]), tsep_axis=0)
        self.assertEqual(summed.shape, (7, 2, 3))
        np.testing.assert_array_equal(
            summed[4], ratio[4, 2:3].sum(axis=0))

        sampled = np.stack((ratio, ratio + 1.0), axis=0)
        sampled_sum = sumratio_values(
            sampled, 2, np.array([4, 5, 6]), tsep_axis=1)
        self.assertEqual(sampled_sum.shape, (2, 7, 2, 3))
        np.testing.assert_array_equal(sampled_sum[0], summed)
        difference = forward_difference(
            sampled_sum, 4, 6, tsep_axis=1)
        self.assertEqual(difference.shape, (2, 2, 2, 3))
        np.testing.assert_array_equal(
            difference, sampled_sum[:, 5:7] - sampled_sum[:, 4:6])

    def test_constant_sumratio_difference_recovers_absolute_normalization(self):
        samples, original = _constant_fixture()
        result = fit_forward_sumratio_difference(
            samples, original, tmin=4, cuts=(1, 2, 3), tsep_max=14)
        self.assertTrue(result["quality_accepted"][0, 0])
        self.assertTrue(result["estimate_available"][0, 0])
        self.assertFalse(result["central_fallback_used"][0, 0])
        self.assertAlmostEqual(float(result["M"][0, 0]), 2.0, places=2)
        self.assertGreater(float(result["statistical_error"][0, 0]), 0.0)
        self.assertGreaterEqual(int(result["usable_cut_count"][0, 0, 0]), 1)
        self.assertEqual(result["candidate_M"].shape[:2], (3, 10))
        accepted_weights = result["candidate_aic_weight"][0, :, 0, 0]
        self.assertAlmostEqual(float(np.sum(accepted_weights)), 1.0, places=12)
        boot_weights = result["candidate_aic_weight_boot"][
            :, 0, :, 0, 0]
        boot_sums = np.sum(boot_weights, axis=1)
        active = boot_sums > 0.0
        self.assertTrue(np.all(active))
        np.testing.assert_allclose(
            boot_sums[active], np.ones(np.count_nonzero(active)), rtol=1e-12)

    def test_nonflat_central_data_uses_labelled_fallback_envelope(self):
        rng = np.random.default_rng(2718)
        nboot, nt = 80, 20
        samples = np.full((nboot, nt, nt, 1, 1), np.nan)
        original = np.full((nt, nt, 1, 1), np.nan)
        for tsep in range(nt):
            value = 2.0 + 0.3 * tsep
            for tau in range(tsep + 1):
                samples[:, tsep, tau] = value + rng.normal(
                    0.0, 0.02, size=(nboot, 1, 1))
                original[tsep, tau] = value
        result = fit_forward_sumratio_difference(
            samples, original, tmin=4, cuts=(1, 2), tsep_max=12)
        self.assertFalse(result["quality_accepted"][0, 0])
        self.assertTrue(result["central_fallback_used"][0, 0])
        self.assertIn(
            "minimum_candidate_statistical_error",
            str(result["central_fallback_status"][0, 0]))
        self.assertEqual(
            float(result["replica_fallback_fraction"][0, 0]), 1.0)
        self.assertTrue(result["estimate_available"][0, 0])

    def test_replica_selection_is_not_gated_by_failed_central_fit(self):
        """中心失败不得把本可通过 quality gate 的 replica 一并强制 fallback。"""
        samples, original = _constant_fixture(nboot=80, nt=20)
        for tsep in range(20):
            for tau in range(tsep + 1):
                original[tsep, tau] = 2.0 + 0.3 * tsep
        result = fit_forward_sumratio_difference(
            samples, original, tmin=4, cuts=(1, 2, 3), tsep_max=14)
        self.assertFalse(result["quality_accepted"][0, 0])
        self.assertTrue(result["central_fallback_used"][0, 0])
        self.assertGreater(
            float(np.mean(result["bootstrap_quality_accepted"][:, 0, 0])),
            0.5)
        self.assertLess(
            float(result["replica_fallback_fraction"][0, 0]), 0.5)
        self.assertTrue(result["estimate_available"][0, 0])

    def test_nonfinite_or_unmatched_inputs_fail_closed(self):
        samples, original = _constant_fixture(nboot=10, nt=12)
        samples[:, 4, :, :, :] = np.nan
        result = fit_forward_sumratio_difference(
            samples, original, tmin=4, tsep_max=10)
        self.assertFalse(result["quality_accepted"][0, 0])
        self.assertFalse(result["estimate_available"][0, 0])
        with self.assertRaises(ValueError):
            fit_forward_sumratio_difference(
                samples, original[:-1], tmin=4, tsep_max=10)

    def test_tmd_ratio_persists_sumratio_aic_artifact(self):
        from pyqcd.analysis._tmd_ratio import run_disconnected_tmd_ratio

        nt = 24
        corr = {}
        ope = {}
        for cid, shift in ((1, 0.0), (2, 0.2)):
            t = np.arange(nt, dtype=float)
            c2 = np.exp(-0.2 * t) * (1.0 + 0.01 * shift * np.cos(t)) + 0.01
            corr[cid] = {
                'corr_pp_P200': np.broadcast_to(
                    c2[:20][None, :], (nt, 20)).copy()
            }
            ope[cid] = {
                'tmd': ((1.0 + 0.05 * shift) * np.sin(0.3 * t) + 2.0)[
                    None, None, :]
            }
        with tempfile.TemporaryDirectory() as root:
            result = run_disconnected_tmd_ratio(
                corr, ope, [1, 2], root, logger=lambda *_args: None,
                NT=nt, nz=1, nb=1, dt_max=20, dt_start=7, dt_end=10,
                cut=6, momentum='P200')
            path = os.path.join(
                root, 'analysis', 'tmd_ratio', 'sumratio_difference_P200.npz')
            self.assertTrue(os.path.isfile(path))
            with np.load(path, allow_pickle=False) as artifact:
                self.assertIn('candidate_range_aic', artifact.files)
                self.assertIn('candidate_aic_weight', artifact.files)
                self.assertIn('quality_accepted', artifact.files)
                self.assertEqual(int(artifact['Nsample']), 200)
                self.assertEqual(int(artifact['bootstrap_samples']), 200)
            self.assertIsNotNone(result['proton']['sumratio_difference'])


def test_bare_sumratio_contracts():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        BareSumratioContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("bare Sumratio-difference contract failed")


if __name__ == "__main__":
    test_bare_sumratio_contracts()
