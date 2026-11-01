"""Contracts for the full lattice-only TMD analysis workflow."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np


class TmdFullAnalysisContract(unittest.TestCase):
    def test_direction_momentum_map_matches_flow_geometry(self):
        from pyqcd.testing.tmd.analyze_full_production import (
            DIRECTION_MOMENTUM,
        )

        self.assertEqual(DIRECTION_MOMENTUM[0], (0, 0, 2))
        self.assertEqual(DIRECTION_MOMENTUM[1], (0, 2, 0))
        self.assertEqual(DIRECTION_MOMENTUM[2], (2, 0, 0))

    def test_two_point_input_preserves_tsep_source_and_boundary(self):
        from pyqcd.testing.tmd.analyze_full_production import (
            build_two_point_ratio_input,
        )

        source_dt = np.arange(2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5)
        result = build_two_point_ratio_input(
            source_dt, [2, 4], n_conf=2, n_polarization=2)
        self.assertEqual(result.shape, (2, 1, 2, 2, 3, 1))
        np.testing.assert_array_equal(
            result[0, 0, 0, 0, :, 0], source_dt[0, :, 2])
        np.testing.assert_array_equal(
            result[0, 0, 0, 1, :, 0], source_dt[0, :, 4])
        np.testing.assert_array_equal(
            result[1, 0, 1, 0, :, 0], source_dt[1, :, 2])

        from pyqcd.testing.tmd.analyze_full_production import (
            TSEPS, _build_two_point_for_direction,
        )
        loaded = {
            conf_id: {
                'nopol': {'corr_pp_P200': np.ones((72, 72), complex)},
                'pol35': {'corr_pp_P200': np.ones((72, 72), complex)},
            }
            for conf_id in (10, 20)
        }
        mapping = _build_two_point_for_direction(
            loaded, (10, 20), 2, momentum=(2, 0, 0))
        self.assertEqual(set(mapping), {10, 20})
        self.assertEqual(mapping[10].shape, (2, 1, len(TSEPS), 72, 1))

    def test_select_ratio_observable_uses_named_axes(self):
        from pyqcd.testing.tmd.analyze_full_production import (
            select_ratio_observable,
        )

        shape = (2, 4, 6, 1, 3, 2, 2, 3, 1)
        ratio = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        samples = np.arange(
            np.prod(shape + (4,)), dtype=np.float64).reshape(shape + (4,))
        err = np.ones(shape)
        selected = select_ratio_observable(
            ratio,
            samples,
            err,
            channel='unpolarized',
            orientation='even_sum',
            component='combined',
            direction=0,
        )
        np.testing.assert_array_equal(selected['ratio'], ratio[0, 2, 0, 0, ..., 0])
        np.testing.assert_array_equal(
            selected['samples'],
            np.moveaxis(samples[0, 2, 0, 0, ..., 0, :], -1, 0))
        self.assertEqual(selected['ratio'].shape, (3, 2, 2, 3))
        self.assertEqual(selected['samples'].shape, (4, 3, 2, 2, 3))

        shape_2d = (2, 4, 6, 1, 3, 2, 2, 2, 3, 1)
        ratio_2d = np.arange(
            np.prod(shape_2d), dtype=np.float64).reshape(shape_2d)
        samples_2d = np.arange(
            np.prod(shape_2d + (4,)), dtype=np.float64).reshape(
                shape_2d + (4,))
        selected_2d = select_ratio_observable(
            ratio_2d,
            samples_2d,
            np.ones(shape_2d),
            channel='unpolarized',
            orientation='even_sum',
            component='combined',
            direction=0,
        )
        self.assertEqual(selected_2d['ratio'].shape, (3, 2, 2, 2, 3))
        self.assertEqual(selected_2d['samples'].shape, (4, 3, 2, 2, 2, 3))

    def test_plateau_samples_keep_replica_axis(self):
        from pyqcd.testing.tmd.analyze_full_production import (
            plateau_from_ratio_samples,
        )

        samples = (
            np.arange(4, dtype=np.float64)[:, None, None, None, None]
            + 0.25 + 0.5j
        )
        samples = np.broadcast_to(samples, (4, 3, 2, 16, 17)).copy()
        result = plateau_from_ratio_samples(
            samples, tsep_start=5, tsep_end=8, cut=4)
        self.assertEqual(result.shape, (4, 3, 2))
        expected = np.arange(4, dtype=np.float64) + 0.25 + 0.5j
        np.testing.assert_allclose(
            result, np.broadcast_to(expected[:, None, None], result.shape))

    def test_safe_self_renormalization_is_per_b_and_keeps_nan_columns(self):
        from pyqcd.testing.tmd.analyze_full_production import (
            safe_self_renormalize,
        )

        c0 = np.ones((3, 4, 2), dtype=np.complex128)
        c0[:, :, 0] *= np.array([2.0, 4.0, 8.0, 16.0])[None, :]
        c0[:, 2, 1] = np.nan
        hR, norm = safe_self_renormalize(c0)
        np.testing.assert_allclose(
            hR[:, :, 0], c0[:, :, 0] / c0[:, 0:1, 0])
        self.assertTrue(np.isnan(hR[:, :, 1]).all())
        np.testing.assert_allclose(norm[:, 0, 0], [2.0, 2.0, 2.0])
        self.assertTrue(np.isnan(norm[:, 0, 1]).all())

    def test_pdf_stage_runs_from_synthetic_selected_ratio(self):
        from pyqcd.testing.tmd.analyze_full_production import analyze_pdf
        from pyqcd.testing.tmd.run_full_production import B_LIST, Z_LIST

        ratio_1d = {}
        tseps = np.arange(5, 21)
        insertions = np.arange(21)
        valid = insertions[None, :] <= tseps[:, None]
        sample_factor = np.array([1.0, 2.0])[:, None, None, None, None]
        z_factor = 1.0 / (np.asarray(Z_LIST)[None, :, None, None, None] + 1.0)
        b_factor = 1.0 + 0.01 * np.asarray(B_LIST)[
            None, None, :, None, None]
        samples = sample_factor * z_factor * b_factor
        samples = np.broadcast_to(
            samples, (2, len(Z_LIST), len(B_LIST), len(tseps), 21)).copy()
        samples = np.where(valid[None, None, None, :, :], samples, np.nan)
        for tau in (1.0, 2.0, 3.0):
            for tag in ('P200', 'P400'):
                ratio_1d[(tau, 2, tag)] = {
                    'result': {
                        'ratio': samples[0].copy(),
                        'samples': samples,
                        'error': np.ones(samples.shape[1:]),
                    },
                }
        with tempfile.TemporaryDirectory() as root:
            result = analyze_pdf(
                root, (6250, 6450), ratio_1d, (1.0, 2.0, 3.0),
                logger=lambda *_args: None)
            self.assertEqual(len(result), 9)
            self.assertTrue((Path(root) / 'analysis/pdf' / 'pdf_summary.json').is_file())
            self.assertTrue((Path(root) / 'analysis/pdf' / 'quasi_tau1_P200.npz').is_file())

    def test_ratio_axes_expand_to_physical_tsep_indices(self):
        from pyqcd.testing.tmd.analyze_full_production import expand_ratio_axes

        values = np.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)
        expanded = expand_ratio_axes(values, tseps=(1, 3), sample_axis=False)
        self.assertEqual(expanded.shape, (4, 4, 2, 3))
        np.testing.assert_array_equal(
            expanded[1, :2], np.moveaxis(values[:, :, 0, :2], -1, 0))
        np.testing.assert_array_equal(
            expanded[3, :4], np.moveaxis(values[:, :, 1, :4], -1, 0))

    def test_lattice_only_status_never_promotes_physical_acceptance(self):
        from pyqcd.testing.tmd.analyze_full_production import (
            make_lattice_only_status,
        )

        status = make_lattice_only_status(
            conf_ids=[6250, 6450], tau_values=[1, 2, 3], n_good_fit=5)
        self.assertEqual(status['status'], 'diagnostic_only')
        self.assertFalse(status['physical_acceptance'])
        self.assertFalse(status['perturbative_matching_applied'])
        self.assertIn('soft_rapidity_not_measured', status['reasons'])
        self.assertIn('single_lattice_spacing', status['reasons'])

    def test_npz_publish_is_atomic(self):
        from pyqcd.testing.tmd.analyze_full_production import _write_npz

        with tempfile.TemporaryDirectory() as root:
            path = _write_npz(Path(root) / 'sample.npz', value=np.arange(3))
            self.assertTrue(path.is_file())
            self.assertFalse(list(path.parent.glob('*.tmp.npz')))


if __name__ == '__main__':
    unittest.main()
