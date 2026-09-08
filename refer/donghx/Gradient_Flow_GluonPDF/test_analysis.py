#!/usr/bin/env python3
"""Small deterministic tests for the flow-time analysis contract."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from flow_time_analysis_v1.analyze import (
    COMPONENTS,
    TAUS,
    _geometry_mask,
    _nanmax_or_nan,
    fit_flow_line,
    load_products,
)


class FlowTimeAnalysisTests(unittest.TestCase):
    def test_geometry_is_strict_and_excludes_local_point(self) -> None:
        taus = np.asarray((0.1, 0.5, 0.6), dtype=float)
        z = np.asarray((0, 1, 2, 3), dtype=int)
        mask = _geometry_mask(taus, z, radius_factor=1.0)
        # z=0 is never scale-separated.
        self.assertFalse(bool(mask[:, 0].any()))
        # tau=0.5 gives r/a=2 exactly and is rejected at z/a=2 by '<'.
        self.assertFalse(bool(mask[1, 2]))
        self.assertTrue(bool(mask[0, 2]))
        self.assertTrue(bool(mask[2, 3]))

    def test_nanmax_preserves_all_nan_slice(self) -> None:
        values = np.asarray([[np.nan, 1.0], [np.nan, 2.0]])
        reduced = _nanmax_or_nan(values, axis=0)
        self.assertTrue(np.isnan(reduced[0]))
        self.assertEqual(float(reduced[1]), 2.0)

    def test_correlated_line_fit_recovers_intercept_and_slope(self) -> None:
        rng = np.random.default_rng(271828)
        tau = np.asarray((0.1, 0.2, 0.3, 0.4), dtype=float)
        expected_intercept, expected_slope = 1.25, -0.70
        central = expected_intercept + expected_slope * tau
        samples = central[None, :] + rng.normal(0.0, 0.01, size=(240, len(tau)))
        fit = fit_flow_line(central, samples, tau)
        self.assertTrue(fit.fit_computed)
        self.assertTrue(fit.quality_pass)
        self.assertAlmostEqual(fit.intercept, expected_intercept, places=10)
        self.assertAlmostEqual(fit.slope_tau, expected_slope, places=10)
        self.assertEqual(fit.intercept_boot.shape, (240,))
        self.assertTrue(np.isfinite(fit.intercept_boot).all())

    def test_all_production_flow_products_share_axes(self) -> None:
        fit_root = Path(
            "/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/"
            "bare_fit_sumdiff_aic_v1"
        )
        if not fit_root.is_dir():
            self.skipTest(f"production fit root is unavailable: {fit_root}")
        products = load_products(fit_root)
        np.testing.assert_allclose(products.taus, np.asarray(TAUS, dtype=float))
        self.assertEqual(products.M.shape, (len(TAUS), len(COMPONENTS), 3, 25))
        self.assertEqual(products.M_boot.shape, (len(TAUS), 1500, len(COMPONENTS), 3, 25))
        self.assertEqual(products.fit_accepted.shape, (len(TAUS), len(COMPONENTS), 3, 25))
        self.assertEqual(products.boot_indices.shape, (1500, 231))
        self.assertEqual(len(products.confs), 231)


if __name__ == "__main__":
    unittest.main()
