"""Physical boundary contracts for the TMD workflow state and flow window."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyqcd.pipeline._tmd9 import self_renormalize
from pyqcd.renorm import flow_scale_mask


class TmdPhysicalContract(unittest.TestCase):
    def test_flow_scale_mask_uses_sqrt_eight_tau(self):
        z = np.array([0, 1, 2, 3], dtype=float)
        expected = np.array([True, False, False, True])
        np.testing.assert_array_equal(
            flow_scale_mask(z, 0.5, include_local=True), expected)
        np.testing.assert_array_equal(
            flow_scale_mask(z, 0.5, include_local=False),
            np.array([False, False, False, True]))
        np.testing.assert_array_equal(
            flow_scale_mask(z, 0.0, include_local=False),
            np.array([False, True, True, True]))
        with self.assertRaises(ValueError):
            flow_scale_mask(z, -0.1)

    def test_flow_window_scan_keeps_local_and_nonlocal_separate(self):
        from pyqcd.testing.tmd.flow_window_scan import flow_window_indices

        masks = flow_window_indices(
            np.array([0, 2, 3, 4]), np.array([3, 4]), 0.5)
        self.assertEqual(masks['radius_over_a'], 2.0)
        np.testing.assert_array_equal(masks['z_local'], [True, False, False, False])
        np.testing.assert_array_equal(
            masks['z_nonlocal'], [False, False, True, True])
        np.testing.assert_array_equal(masks['b_nonlocal'], [True, True])

    def test_self_renormalize_preserves_complex_and_rejects_zero_norm(self):
        c0 = np.array([[[2.0 + 1.0j], [1.0 - 0.5j]],
                       [[3.0 - 1.0j], [0.5 + 0.25j]]])
        hR, norm = self_renormalize(c0)
        self.assertTrue(np.iscomplexobj(hR))
        np.testing.assert_allclose(norm[:, 0, :], c0[:, 0, :])
        np.testing.assert_allclose(
            hR, c0 / c0[:, 0:1, :])
        with self.assertRaisesRegex(ValueError, "z=0|归一化|为零"):
            self_renormalize(np.zeros((2, 2, 1), dtype=complex))
        with self.assertRaises(ValueError):
            self_renormalize(np.ones((2, 2)))

    def test_cyclic_transverse_labels_match_reference_for_zdirection_one(self):
        from pyqcd.operator._gradient_flow_gluon_ope import (
            _field_dictionary, _ordered_field, _straight_bilocal,
            flowed_gluon_ope,
        )
        from pyqcd.testing._gradient_flow_gluon_contract import _near_identity_su3

        gauge = _near_identity_su3(seed=81277)
        data = flowed_gluon_ope(gauge, z_count=2, z_dir=1)
        fields = _field_dictionary(gauge, 'traceless')

        def raw_ti(direction):
            first = _ordered_field(fields, 3, direction)
            plus = _straight_bilocal(gauge, first, first, 1, 2, +1)
            minus = _straight_bilocal(gauge, first, first, 1, 2, -1)
            return np.stack((plus, minus, plus + minus, plus - minus), axis=0)

        np.testing.assert_allclose(data[1, 0, :, 3], raw_ti(2),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(data[1, 0, :, 4], raw_ti(0),
                                   rtol=0, atol=1e-12)

    def test_local_single_channel_compute_dtype_is_enforced(self):
        from pyqcd.renorm._tmd import M_mu_lambda_nu_rho
        from pyqcd.testing import random_su3_gauge

        gauge = random_su3_gauge(L=2, seed=9123).astype(np.complex64)
        value = M_mu_lambda_nu_rho(
            gauge, 3, 0, 3, 0, 1, 0, compute_dtype=np.complex128)
        self.assertEqual(value.dtype, np.dtype('complex128'))
        with self.assertRaises(ValueError):
            M_mu_lambda_nu_rho(
                gauge, 3, 0, 3, 0, 1, 0, compute_dtype=np.float32)

    def test_torch_compute_dtype_mapping_does_not_call_numpy_dtype(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from pyqcd.renorm._tmd import M_mu_lambda_nu_rho
        from pyqcd.testing import random_su3_gauge
        from pyqcd.tools import set_backend

        gauge = torch.as_tensor(
            random_su3_gauge(L=2, seed=9124), dtype=torch.complex64)
        set_backend('torch')
        try:
            value = M_mu_lambda_nu_rho(
                gauge, 3, 0, 3, 0, 1, 0,
                compute_dtype=torch.complex128)
            self.assertEqual(value.dtype, torch.complex128)
            value64 = M_mu_lambda_nu_rho(
                gauge, 3, 0, 3, 0, 1, 0,
                compute_dtype='complex64')
            self.assertEqual(value64.dtype, torch.complex64)
        finally:
            set_backend('numpy')

    def test_spatial_sum_cannot_hide_local_imaginary_residual(self):
        from pyqcd.renorm import _tmd

        gauge = np.zeros((1, 1, 1, 2, 4, 3, 3), dtype=complex)
        value = np.array([[[[1.0 + 0.1j, 1.0 - 0.1j]]]], dtype=complex)
        with patch.object(_tmd, '_tmd_clover_fields',
                          return_value=(None, None, None)), \
                patch.object(_tmd, '_gluon_tmd_operator_from_fields',
                             return_value=value):
            with self.assertRaisesRegex(ValueError, 'imaginary'):
                _tmd.tmd_matrix_elements(gauge, [0], [0], L=0)
            with self.assertRaisesRegex(ValueError, 'imaginary'):
                _tmd.tmd_matrix_elements_time(gauge, [0], [0], L=0)

    def test_su3_exp_large_step_retains_unitarity(self):
        from scipy.linalg import expm
        from pyqcd.renorm._gradient_flow import su3_exp
        from pyqcd.tools import set_backend

        set_backend('numpy')
        rng = np.random.default_rng(441)
        matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        generator = matrix - matrix.conj().T
        generator -= np.trace(generator) * np.eye(3) / 3.0
        generator = 0.6 * generator / np.max(np.abs(generator))
        batch = generator[None, None, None, None, ...]
        actual = np.asarray(su3_exp(batch))[0, 0, 0, 0]
        expected = expm(generator)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        unitary_error = np.max(
            np.abs(actual @ actual.conj().T - np.eye(3)))
        self.assertLess(unitary_error, 2e-6)

    def test_chain_with_unidentified_plateau_only_writes_diagnostic_status(self):
        from pyqcd.testing.tmd.pdf_workflow import run_tmd_pdf_chain

        with tempfile.TemporaryDirectory() as root:
            analysis = os.path.join(root, 'analysis', 'tmd_ratio')
            os.makedirs(analysis)
            np.save(os.path.join(analysis, 'c0_plateau_P200.npy'),
                    np.full((1, 3, 1), np.nan))
            np.savez(os.path.join(analysis, 'c0_plateau_status_P200.npz'),
                     plateau_status=np.asarray('statistically_unidentifiable'))
            with open(os.path.join(
                    analysis, 'zero_momentum_denominator_status.json'),
                    'w', encoding='utf-8') as stream:
                import json
                json.dump({
                    'physical_status': 'unverified',
                    'used_for_pdf': False,
                    'strict_gate_passed': False,
                    'status': 'quality_gate_failed',
                }, stream)
            run_tmd_pdf_chain(
                [6250], root, lambda *_args: None,
                SimpleNamespace(tau=3.0, eps=0.05),
                momenta=[(2, 0, 0)])
            path = os.path.join(analysis, 'chain_status.json')
            self.assertTrue(os.path.isfile(path))
            import json
            with open(path, encoding='utf-8') as stream:
                status = json.load(stream)
            self.assertEqual(status['status'], 'diagnostic_only')
            self.assertFalse(status['physical_acceptance'])
            self.assertIn('no_verified_positive_longitudinal_plateau',
                          status['reasons'])
            self.assertIn(
                'zero_momentum_denominator_quality_gate_failed',
                status['reasons'])
            self.assertIn(
                'flow_radius_not_smaller_than_all_nonzero_b',
                status['reasons'])

    def test_source_resolved_2pt_cache_is_a_distinct_physical_contract(self):
        from pyqcd.pipeline._tmd9 import _multi_2pt_cache_specs

        common = (6250, '/tmp/unused', [(2, 0, 0)], ('pp',),
                  'complex128', 'VVV')
        legacy = _multi_2pt_cache_specs(*common)
        source = _multi_2pt_cache_specs(*common, source_resolved=True)
        key = 'corr_pp_P200'
        self.assertEqual(legacy[key]['shape'], (72,))
        self.assertEqual(source[key]['shape'], (72, 72))
        self.assertEqual(
            source[key]['contract']['algorithm_version'],
            'pyqcd.pipeline.multi-2pt.v2-source-resolved')
        self.assertEqual(
            source[key]['contract']['source_axis_order'], 'source,dt')
        self.assertNotEqual(
            legacy[key]['contract'], source[key]['contract'])

    def test_gauge_content_identity_changes_flow_cache_contract(self):
        from pyqcd.pipeline import _tmd9

        with tempfile.TemporaryDirectory() as root:
            gauge = os.path.join(root, 'gauge.lime')
            with open(gauge, 'wb') as stream:
                stream.write(b'first-gauge-content')
            with patch.object(_tmd9, 'get_gauge_path', return_value=gauge):
                first = _tmd9._gauge_source_identity(6250)
                first_contract = _tmd9._flow_cache_contract(
                    6250, 3.0, 0.05, 'complex64', input_source=first)
                with open(gauge, 'wb') as stream:
                    stream.write(b'second-gauge-content')
                second = _tmd9._gauge_source_identity(6250)
                second_contract = _tmd9._flow_cache_contract(
                    6250, 3.0, 0.05, 'complex64', input_source=second)
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertNotEqual(
            _tmd9._contract_sha256(first_contract),
            _tmd9._contract_sha256(second_contract))

    def test_tmd_ratio_rejects_source_averaged_2pt_instead_of_rolling_it(self):
        from pyqcd.analysis import run_disconnected_tmd_ratio

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'source-resolved|source'):
                run_disconnected_tmd_ratio(
                    {6250: {'corr_pp_P200': np.ones(72)}},
                    {6250: {'tmd': np.ones((2, 1, 72))}},
                    [6250], root, logger=lambda *_args: None,
                    NT=72, nz=2, nb=1, dt_max=8,
                    dt_start=3, dt_end=5, cut=2, momentum='P200')

    def test_plateau_rank_and_flatness_are_independent_status_layers(self):
        from pyqcd.analysis._tmd_ratio import (
            _plateau_status_metadata, plateau_c0,
        )

        rng = np.random.default_rng(6250)
        ratio = rng.normal(scale=0.02, size=(12, 20, 20, 1, 1))
        for dt in range(20):
            for dtau in range(dt + 1):
                ratio[:, dt, dtau, 0, 0] += 0.08 * dt + 0.03 * dtau
        c0 = plateau_c0(
            ratio, dt_max=20, dt_start=7, dt_end=10, cut=6)
        status = _plateau_status_metadata(
            ratio, c0, dt_max=20, dt_start=7, dt_end=10, cut=6)
        self.assertEqual(str(status['plateau_status']), 'identifiable')
        self.assertNotEqual(
            str(status['plateau_fit_quality_status']), 'good')
        self.assertNotEqual(
            str(status['plateau_physical_result_status']), 'supported')


def test_tmd_physical_contracts():
    """Run the physical boundary unit tests without pytest."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TmdPhysicalContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('TMD physical boundary contract failed')
