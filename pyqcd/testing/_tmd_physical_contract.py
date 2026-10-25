"""Physical boundary contracts for the TMD workflow state and flow window."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
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
        source = _multi_2pt_cache_specs(
            *common, source_resolved=True, polarization='nopol')
        source_pol35 = _multi_2pt_cache_specs(
            *common, source_resolved=True, polarization='pol35')
        key = 'corr_pp_P200'
        self.assertEqual(legacy[key]['shape'], (72,))
        self.assertEqual(source[key]['shape'], (72, 72))
        self.assertEqual(
            source[key]['contract']['algorithm_version'],
            'pyqcd.pipeline.multi-2pt.v3-source-resolved-polarized')
        self.assertEqual(
            source[key]['contract']['source_axis_order'], 'source,dt')
        self.assertEqual(
            source[key]['contract']['polarization'], 'nopol')
        self.assertEqual(
            source_pol35[key]['contract']['polarization'], 'pol35')
        self.assertEqual(source[key]['dtype'], 'complex128')
        self.assertEqual(source_pol35[key]['dtype'], 'complex128')
        self.assertNotEqual(source[key]['path'], source_pol35[key]['path'])
        self.assertNotEqual(
            legacy[key]['contract'], source[key]['contract'])

    def test_polarized_source_resolved_2pt_computes_both_channels(self):
        from pyqcd.pipeline import _steps, _tmd9
        from pyqcd.tools import set_backend

        set_backend('numpy')
        marker = np.array(
            [[1.0 + 2.0j, 0.25 - 0.5j, 0.0, 0.0],
             [0.0, 0.0, -0.75 + 0.125j, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.5j]], dtype=np.complex128)

        def fake_contract(_peram, vvv_sink, _vvv_source, **_kwargs):
            scale = vvv_sink[0, 0, 0]
            return scale * marker

        def fake_peram(*_args, **_kwargs):
            return np.zeros((2, 4, 4, 1, 1), dtype=np.complex128)

        vvv = np.zeros((2, 1, 1, 1, 1), dtype=np.complex128)
        vvv[0, 0, 0, 0, 0] = 1.0
        vvv[1, 0, 0, 0, 0] = 2.0

        with tempfile.TemporaryDirectory() as root, \
                patch.object(_steps, 'NT', 2), \
                patch.object(_steps, 'NEV', 1), \
                patch.object(_tmd9, 'NT', 2), \
                patch.object(_tmd9, 'NX', 2), \
                patch.object(_tmd9, 'NEV', 1), \
                patch.object(_tmd9, 'NEV1', 1), \
                patch.object(_steps, 'readin_peram_time_slice',
                             side_effect=fake_peram), \
                patch.object(_steps, 'contract_donghx_2pt_pair',
                             side_effect=fake_contract):
            Path(root, 'data', 'conf7').mkdir(parents=True)
            result = _tmd9.compute_2pt_polarizations_multi(
                7, root, lambda *_args: None, {'VVV': vvv},
                [(2, 0, 0)], precision='complex128')
            with patch.object(
                    _steps, 'contract_donghx_2pt_pair',
                    side_effect=AssertionError(
                        'complete polarized cache repeated contraction')):
                cached = _tmd9.compute_2pt_polarizations_multi(
                    7, root, lambda *_args: None, {'VVV': vvv},
                    [(2, 0, 0)], precision='complex128')
            loaded = _tmd9.load_multi_2pt_polarizations(
                root, [7], ['P200'], logger=lambda *_args: None,
                precision='complex128')

        from pyqcd.lattice import gamma

        g0 = np.asarray(gamma(0))
        g3 = np.asarray(gamma(3))
        g4 = np.asarray(gamma(4))
        g5 = np.asarray(gamma(5))
        pplus = 0.5 * (g0 + g4)
        projectors = np.stack((
            pplus,
            pplus @ (1j * g3 @ g5),
        ))
        expected = {
            polarization: np.zeros((2, 2), dtype=np.complex128)
            for polarization in _tmd9.TWO_POINT_POLARIZATIONS
        }
        for source in range(2):
            for dt in range(2):
                sink = (source + dt) % 2
                sign = -1.0 if sink < source else 1.0
                contract = sign * vvv[sink, 0, 0, 0, 0] * marker
                projected = np.einsum(
                    'pli,il->p', projectors, contract)
                for index, polarization in enumerate(
                        _tmd9.TWO_POINT_POLARIZATIONS):
                    expected[polarization][source, dt] = projected[index]

        for polarization in _tmd9.TWO_POINT_POLARIZATIONS:
            key = 'corr_pp_P200'
            actual = result[polarization][key]
            self.assertEqual(actual.dtype, np.dtype(np.complex128))
            np.testing.assert_allclose(
                actual, expected[polarization], atol=1e-14, rtol=0.0)
            np.testing.assert_array_equal(
                loaded[7][polarization][key], actual)
            np.testing.assert_array_equal(
                cached[polarization][key], actual)
        self.assertFalse(np.array_equal(
            result['nopol']['corr_pp_P200'],
            result['pol35']['corr_pp_P200']))

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

    def test_finite_b_input_audit_distinguishes_ready_and_blocked(self):
        import h5py
        import json
        from pyqcd.operator import (
            GRADIENT_FLOW_CHANNELS,
            GRADIENT_FLOW_ORIENTATIONS,
            GRADIENT_FLOW_COMPONENTS,
        )
        from pyqcd.pipeline import _tmd9
        from pyqcd.testing.tmd.audit_finite_b_inputs import (
            audit_finite_b_ratio_inputs,
        )

        def write_cache(path, shape, contract, dtype=np.float64):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, 'w') as handle:
                handle.create_dataset(
                    'data', data=np.zeros(shape, dtype=dtype))
                handle.attrs['pyqcd_cache_contract_json'] = json.dumps(
                    contract, sort_keys=True)

        with tempfile.TemporaryDirectory() as root:
            conf_dir = os.path.join(root, 'data', 'conf6250')
            write_cache(
                os.path.join(conf_dir, 'corr_pp_P200.h5'),
                (8,),
                {'conf_id': 6250},
            )
            write_cache(
                os.path.join(conf_dir, 'tmd_ope_P200.h5'),
                (2, 2, 8),
                {'conf_id': 6250},
            )
            blocked = audit_finite_b_ratio_inputs(root)
        self.assertEqual(blocked['status'], 'blocked')
        self.assertIn(
            'source_resolved_c2_missing', blocked['blocking_reasons'])
        self.assertIn(
            'full_channel_orientation_component_tmd_ope_missing',
            blocked['blocking_reasons'],
        )

        with tempfile.TemporaryDirectory() as root:
            for conf_id in (6250, 6450):
                conf_dir = os.path.join(root, 'data', f'conf{conf_id}')
                for polarization in _tmd9.TWO_POINT_POLARIZATIONS:
                    write_cache(
                        os.path.join(
                            conf_dir,
                            f'corr_pp_P200_{polarization}.h5'),
                        (8, 8),
                        {
                            'algorithm_version':
                                _tmd9._POLARIZED_2PT_ALGORITHM_VERSION,
                            'antiperiodic_boundary': True,
                            'artifact': 'tmd9_multi_2pt_polarized',
                            'channel': 'pp',
                            'conf_id': conf_id,
                            'dirac_interpolator': 'Cg5g4',
                            'dtype': 'complex128',
                            'momentum': [2, 0, 0],
                            'polarization': polarization,
                            'shape': [8, 8],
                            'source_axis_order': 'source,dt',
                            'time_axis_order': 'source,sink',
                        },
                        dtype=np.complex128,
                    )
                write_cache(
                    os.path.join(conf_dir, 'tmd_ope_P200.h5'),
                    (2, 4, 6, 1, 2, 8),
                    {
                        'algorithm_version':
                            _tmd9._FULL_OPE_ALGORITHM_VERSION,
                        'artifact': 'tmd_ope_time_full_channel',
                        'channel_labels': list(GRADIENT_FLOW_CHANNELS),
                        'component_labels': list(GRADIENT_FLOW_COMPONENTS),
                        'conf_id': conf_id,
                        'orientation_labels':
                            list(GRADIENT_FLOW_ORIENTATIONS),
                        'shape': [2, 4, 6, 1, 2, 8],
                    },
                )
                write_cache(
                    os.path.join(conf_dir, 'tmd_ope_legacy_P200.h5'),
                    (1, 2, 8),
                    {
                        'algorithm_version':
                            _tmd9._OPE_ALGORITHM_VERSION,
                        'artifact': 'tmd_ope_time',
                        'conf_id': conf_id,
                        'shape': [1, 2, 8],
                    },
                )
                write_cache(
                    os.path.join(conf_dir, 'tmd_ope_2d_P200.h5'),
                    (2, 4, 6, 1, 2, 2, 8),
                    {
                        'algorithm_version':
                            _tmd9._FULL_2D_OPE_ALGORITHM_VERSION,
                        'artifact': 'tmd_ope_time_full_channel_2d',
                        'b_x_dir': 0,
                        'b_x_list': [0, 1],
                        'b_y_dir': 1,
                        'b_y_list': [0, 1],
                        'channel_labels': list(GRADIENT_FLOW_CHANNELS),
                        'component_labels': list(GRADIENT_FLOW_COMPONENTS),
                        'conf_id': conf_id,
                        'orientation_labels':
                            list(GRADIENT_FLOW_ORIENTATIONS),
                        'shape': [2, 4, 6, 1, 2, 2, 8],
                        'transverse_order': [0, 1],
                        'z_dir': 2,
                    },
                )
            ready = audit_finite_b_ratio_inputs(root)
            ready_2d = audit_finite_b_ratio_inputs(
                root, dimension='2d', b_x_list=(0, 1),
                b_y_list=(0, 1), transverse_order=(0, 1))
            Path(
                root, 'data', 'conf6250',
                'corr_pp_P200_pol35.h5').unlink()
            missing_pol35 = audit_finite_b_ratio_inputs(root)
        self.assertEqual(ready['status'], 'ready')
        self.assertTrue(ready['source_resolved_c2'])
        self.assertTrue(ready['full_channel_tmd_ope'])
        self.assertEqual(ready['configurations'], [6250, 6450])
        self.assertTrue(all(
            len(item['tmd_ope']) == 3
            for item in ready['per_configuration'].values()
        ))
        self.assertEqual(ready_2d['status'], 'ready')
        self.assertEqual(ready_2d['dimension'], '2d')
        self.assertEqual(missing_pol35['status'], 'blocked')
        self.assertIn(
            'source_resolved_c2_missing',
            missing_pol35['blocking_reasons'])

    def test_full_channel_tmd_ope_cache_roundtrip_and_audit(self):
        from pyqcd.pipeline import _tmd9
        from pyqcd.testing.tmd.audit_finite_b_inputs import (
            audit_finite_b_ratio_inputs,
        )
        from pyqcd.tools import set_backend

        set_backend('numpy')
        gauge = np.broadcast_to(
            np.eye(3, dtype=np.complex128),
            (3, 3, 3, 3, 4, 3, 3),
        ).copy()

        def encode(_gauge, z_list, b_list, **_kwargs):
            return np.full(
                (2, 4, 6, len(z_list), len(b_list), _gauge.shape[0]),
                0.5j,
                dtype=np.complex128,
            )

        with tempfile.TemporaryDirectory() as root, \
                patch.object(_tmd9, 'NT', 3), \
                patch.object(_tmd9, 'NX', 3), \
                patch.object(_tmd9, '_gauge_source_identity',
                             return_value=None), \
                patch.object(_tmd9, 'flow_gauge_for_config',
                             return_value=gauge), \
                patch.object(_tmd9, 'full_channel_tmd_ope_time',
                             side_effect=encode):
            first = _tmd9.compute_tmd_ope_time(
                11, root, lambda *_args: None,
                [0, 1], [0, 1], tau=1.0, eps=0.1,
                precision='complex128', staple_length=1,
                full_channels=True,
            )
            second = _tmd9.compute_tmd_ope_time(
                22, root, lambda *_args: None,
                [0, 1], [0, 1], tau=1.0, eps=0.1,
                precision='complex128', staple_length=1,
                full_channels=True,
            )
            loaded = _tmd9.load_full_channel_tmd_ope_all(
                root, [11, 22], [0, 1], [0, 1],
                logger=lambda *_args: None,
                tau=1.0, eps=0.1, precision='complex128',
                staple_length=1,
            )
            report = audit_finite_b_ratio_inputs(root)

        expected = np.full((2, 4, 6, 2, 2, 3), 0.5j)
        np.testing.assert_array_equal(first['tmd_full'], expected)
        np.testing.assert_array_equal(second['tmd_full'], expected)
        self.assertEqual(sorted(loaded), [11, 22])
        np.testing.assert_array_equal(loaded[11]['tmd_full'], expected)
        self.assertTrue(report['full_channel_tmd_ope'])
        self.assertIn('source_resolved_c2_missing', report['blocking_reasons'])

    def test_synthetic_two_configuration_audit_to_ratio_chain(self):
        from pyqcd.analysis import (
            calculate_gradient_flow_tmd_ratios_from_configurations,
            physical_helicity_ratio,
            validate_gradient_flow_tmd_ratio_results,
        )
        from pyqcd.pipeline import _tmd9
        from pyqcd.pipeline._steps import conf_data_dir
        from pyqcd.testing import random_su3_gauge
        from pyqcd.testing.tmd.audit_finite_b_inputs import (
            audit_finite_b_ratio_inputs,
        )
        from pyqcd.tools import set_backend

        set_backend('numpy')
        conf_ids = [11, 22]
        z_list = [0, 1]
        b_list = [0, 1]
        b_values = [0, 2]
        momenta = [(2, 0, 0)]
        gauges = {
            11: random_su3_gauge(L=3, seed=7411),
            22: random_su3_gauge(L=3, seed=7422),
        }

        with tempfile.TemporaryDirectory() as root, \
                patch.object(_tmd9, 'NT', 3), \
                patch.object(_tmd9, 'NX', 3), \
                patch.object(_tmd9, '_gauge_source_identity',
                             return_value=None), \
                patch.object(_tmd9, 'flow_gauge_for_config',
                             side_effect=lambda cid, *_args, **_kwargs:
                             gauges[cid]):
            for conf_id in conf_ids:
                Path(conf_data_dir(root, conf_id)).mkdir(
                    parents=True, exist_ok=True)
                _tmd9.compute_tmd_ope_time(
                    conf_id, root, lambda *_args: None,
                    z_list, b_list, tau=1.0, eps=0.1,
                    precision='complex128', staple_length=1,
                    full_channels=True,
                )
                specs = _tmd9._multi_2pt_cache_specs(
                    conf_id, root, momenta, ('pp',),
                    'complex128', 'VVV', source_resolved=True,
                    polarization='nopol')
                pol35_specs = _tmd9._multi_2pt_cache_specs(
                    conf_id, root, momenta, ('pp',),
                    'complex128', 'VVV', source_resolved=True,
                    polarization='pol35')
                t = np.arange(3, dtype=np.float64)
                corr = 2.0 + np.exp(-0.3 * t)
                nopol = np.stack([
                    np.roll(corr, source) for source in range(3)
                ]).astype(np.complex128)
                pol35 = (0.5 + 0.25j) * np.conj(nopol)
                _tmd9._save_contract_h5(
                    nopol, specs['corr_pp_P200']['path'],
                    specs['corr_pp_P200']['contract'])
                _tmd9._save_contract_h5(
                    pol35, pol35_specs['corr_pp_P200']['path'],
                    pol35_specs['corr_pp_P200']['contract'])

            audit = audit_finite_b_ratio_inputs(root)
            self.assertEqual(audit['status'], 'ready')
            ope_all = _tmd9.load_full_channel_tmd_ope_all(
                root, conf_ids, z_list, b_list,
                logger=lambda *_args: None,
                tau=1.0, eps=0.1, precision='complex128',
                staple_length=1,
            )
            c2_all = _tmd9.load_multi_2pt_polarizations(
                root, conf_ids, ['P200'], channels=('pp',),
                logger=lambda *_args: None, precision='complex128',
                v_kind='VVV',
            )

        ope_by_conf = {
            conf_id: ope_all[conf_id]['tmd_full'] for conf_id in conf_ids
        }
        twopt_by_conf = {}
        for conf_id in conf_ids:
            source_nopol = c2_all[conf_id]['nopol']['corr_pp_P200']
            source_pol35 = c2_all[conf_id]['pol35']['corr_pp_P200']
            nopol = source_nopol[:, 1][None, None, None, :, None]
            pol35 = source_pol35[:, 1][None, None, None, :, None]
            twopt_by_conf[conf_id] = np.concatenate(
                (nopol, pol35), axis=0).astype(np.complex128)

        result = calculate_gradient_flow_tmd_ratios_from_configurations(
            ope_by_conf, twopt_by_conf, conf_ids, b_values, [1],
            do_jackknife=True,
        )
        self.assertTrue(validate_gradient_flow_tmd_ratio_results(
            result, nconf=2))
        ratio = result['ratio']
        phase = physical_helicity_ratio(ratio)
        expected = np.imag(ratio[1, 3, 1] - ratio[1, 3, 2])
        np.testing.assert_array_equal(phase, expected)
        valid = result['valid_insertion_mask']
        expanded = np.broadcast_to(
            valid[None, None, None, None, None, None, :, :, None],
            ratio.shape,
        )
        self.assertTrue(np.isfinite(ratio[expanded]).all())

    def test_two_dimensional_full_channel_cache_roundtrip_and_isolation(self):
        import h5py
        import json
        from pyqcd.pipeline import _tmd9
        from pyqcd.tools import set_backend

        set_backend('numpy')
        gauge = np.broadcast_to(
            np.eye(3, dtype=np.complex128),
            (3, 3, 3, 3, 4, 3, 3),
        ).copy()

        def encode(_gauge, z_list, b_x_list, b_y_list, **_kwargs):
            return np.full(
                (2, 4, 6, len(z_list), len(b_x_list), len(b_y_list),
                 _gauge.shape[0]),
                0.25 + 0.5j,
                dtype=np.complex128,
            )

        with tempfile.TemporaryDirectory() as root, \
                patch.object(_tmd9, 'NT', 3), \
                patch.object(_tmd9, 'NX', 3), \
                patch.object(_tmd9, '_gauge_source_identity',
                             return_value=None), \
                patch.object(_tmd9, 'flow_gauge_for_config',
                             return_value=gauge), \
                patch.object(_tmd9, 'full_channel_tmd_ope_time_2d',
                             side_effect=encode):
            with self.assertRaisesRegex(ValueError, 'transverse_order'):
                _tmd9.compute_tmd_ope_time_2d(
                    11, root, None, [0], [0], [0],
                    tau=1.0, eps=0.1, precision='complex128',
                    transverse_order=(False, True), staple_length=0)
            with self.assertRaisesRegex(ValueError, 'b_x_list'):
                _tmd9.compute_tmd_ope_time_2d(
                    11, root, None, [0], [0, 0], [0],
                    tau=1.0, eps=0.1, precision='complex128',
                    staple_length=0)
            base = _tmd9.compute_tmd_ope_time_2d(
                11, root, None, [0, 1], [0, 1], [0, 1],
                tau=1.0, eps=0.1, precision='complex128',
                transverse_order=(0, 1), staple_length=1)
            reversed_path = _tmd9.compute_tmd_ope_time_2d(
                11, root, None, [0, 1], [0, 1], [0, 1],
                tau=1.0, eps=0.1, precision='complex128',
                transverse_order=(1, 0), staple_length=1)
            changed_grid = _tmd9.compute_tmd_ope_time_2d(
                11, root, None, [0, 1], [0, 1], [0, 2],
                tau=1.0, eps=0.1, precision='complex128',
                transverse_order=(0, 1), staple_length=1)
            files = sorted(Path(
                _tmd9._strict_conf_path(root, 11)).glob('tmd_ope_2d_*.h5'))
            contracts = []
            for path in files:
                with h5py.File(path, 'r') as handle:
                    contracts.append(json.loads(
                        handle.attrs['pyqcd_cache_contract_json']))
            loaded = _tmd9.load_full_channel_tmd_ope_2d_all(
                root, [11], [0, 1], [0, 1], [0, 1],
                logger=lambda *_args: None, tau=1.0, eps=0.1,
                precision='complex128', transverse_order=(0, 1),
                staple_length=1)
            legacy_loader = _tmd9.load_full_channel_tmd_ope_all(
                root, [11], [0, 1], [0, 1],
                logger=lambda *_args: None, tau=1.0, eps=0.1,
                precision='complex128', staple_length=1)

        expected = np.full((2, 4, 6, 2, 2, 2, 3), 0.25 + 0.5j)
        np.testing.assert_array_equal(base['tmd_full_2d'], expected)
        np.testing.assert_array_equal(
            reversed_path['tmd_full_2d'], expected)
        np.testing.assert_array_equal(changed_grid['tmd_full_2d'], expected)
        self.assertEqual(len(files), 3)
        self.assertEqual(len({path.name for path in files}), 3)
        self.assertEqual(
            {(tuple(item['transverse_order']), tuple(item['b_y_list']))
             for item in contracts},
            {((0, 1), (0, 1)), ((1, 0), (0, 1)), ((0, 1), (0, 2))},
        )
        np.testing.assert_array_equal(
            loaded[11]['tmd_full_2d'], expected)
        self.assertEqual(legacy_loader, {})

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


if __name__ == "__main__":
    test_tmd_physical_contracts()
