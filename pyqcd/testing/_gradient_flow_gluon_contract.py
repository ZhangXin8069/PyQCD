"""参考 donghx schema-v2 梯度流胶子 OPE/ratio 的独立契约。"""
from __future__ import annotations

import tempfile
import unittest

import numpy as np
from scipy.linalg import expm


_SHAPE = (2, 2, 2, 2)


def _near_identity_su3(seed=20260909, strength=0.12):
    rng = np.random.default_rng(seed)
    gauge = np.empty(_SHAPE + (4, 3, 3), dtype=np.complex128)
    for index in np.ndindex(*_SHAPE, 4):
        matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        antihermitian = matrix - matrix.conj().T
        antihermitian -= (
            np.trace(antihermitian) * np.eye(3, dtype=np.complex128) / 3.0
        )
        gauge[index] = expm(strength * antihermitian)
    return gauge


def _site_transform(seed=20260910):
    return _near_identity_su3(seed=seed, strength=0.2)[..., 0, :, :]


def _gauge_transform(gauge, transformation):
    transformed = np.empty_like(gauge)
    for direction in range(4):
        axis = 3 - direction
        forward = np.roll(transformation, -1, axis=axis)
        transformed[..., direction, :, :] = (
            transformation @ gauge[..., direction, :, :]
            @ forward.conj().swapaxes(-1, -2)
        )
    return transformed


class GradientFlowGluonContract(unittest.TestCase):
    def setUp(self):
        from pyqcd.tools import set_backend

        set_backend("numpy")

    def test_schema_components_orientations_and_metadata(self):
        from pyqcd.operator import (
            GRADIENT_FLOW_COMPONENTS,
            GRADIENT_FLOW_FIELD_PROJECTIONS,
            GRADIENT_FLOW_OPE_AXES,
            GRADIENT_FLOW_OPE_SCHEMA,
            GRADIENT_FLOW_ORIENTATIONS,
            gradient_flow_gluon_ope,
            validate_gradient_flow_gluon_ope,
        )

        data, metadata = gradient_flow_gluon_ope(
            _near_identity_su3(), tau=0.0, epsilon=0.01, z_count=2,
            conf_id=4150,
        )
        validate_gradient_flow_gluon_ope(
            data, metadata, conf_id=4150, tau=0.0, epsilon=0.01
        )
        self.assertEqual(data.shape, (2, 2, 4, 6, 2, 2))
        self.assertEqual(data.dtype, np.dtype("complex128"))
        self.assertEqual(metadata["schema"], GRADIENT_FLOW_OPE_SCHEMA)
        self.assertEqual(metadata["axes"], list(GRADIENT_FLOW_OPE_AXES))
        self.assertEqual(
            metadata["axis_labels"]["field_projection"],
            list(GRADIENT_FLOW_FIELD_PROJECTIONS),
        )
        self.assertEqual(
            metadata["axis_labels"]["z_orientation"],
            list(GRADIENT_FLOW_ORIENTATIONS),
        )
        self.assertEqual(
            metadata["axis_labels"]["component"],
            list(GRADIENT_FLOW_COMPONENTS),
        )
        self.assertEqual(metadata["input_scheme"], "thin_link_no_hyp_no_smear")
        self.assertEqual(metadata["flow"]["n_steps"], 0)

    def test_default_z_count_metadata_uses_resolved_periodic_extent(self):
        """z_count=None 必须记录实际输出 z 轴，而不是对 None 调用 int()."""
        from pyqcd.operator import (
            gradient_flow_gluon_ope,
            validate_gradient_flow_gluon_ope,
        )

        data, metadata = gradient_flow_gluon_ope(
            _near_identity_su3(seed=4150), tau=0.0, z_count=None,
            conf_id=4150,
        )
        self.assertEqual(data.shape[4], 2)
        self.assertEqual(metadata["operator"]["z_values_a"], [0, 1])
        validate_gradient_flow_gluon_ope(
            data, metadata, conf_id=4150, tau=0.0, epsilon=0.01)

    def test_traceless_projection_matches_gauge_observable(self):
        from pyqcd.gauge import clover_field_strength
        from pyqcd.operator import flowed_gluon_ope, plaquette_clover

        gauge = _near_identity_su3(seed=7311)
        data = flowed_gluon_ope(gauge, z_count=1)
        raw = np.asarray(plaquette_clover(gauge, 3, 0))
        traceless = np.asarray(clover_field_strength(
            gauge, 3, 0, traceless=True
        ))
        expected = np.sum(
            np.einsum("...ab,...ba->...", traceless, traceless),
            axis=(1, 2, 3),
        )
        actual = data[1, 0, 0, 3, 0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
        self.assertGreater(
            float(np.max(np.abs(raw - traceless))), 1e-10,
            "随机有限-a 组态未体现 legacy 与 traceless 的区别",
        )

    def test_straight_wilson_line_full_sum_is_gauge_invariant(self):
        from pyqcd.operator import flowed_gluon_ope

        gauge = _near_identity_su3(seed=7312)
        transformed = _gauge_transform(gauge, _site_transform())
        original = flowed_gluon_ope(gauge, z_count=2)
        rotated = flowed_gluon_ope(transformed, z_count=2)
        np.testing.assert_allclose(rotated, original, rtol=0, atol=3e-11)
        np.testing.assert_allclose(
            original[:, :, 0, :, 0],
            original[:, :, 1, :, 0],
            rtol=0, atol=2e-12,
        )

    def test_six_component_identities_and_helicity_selector(self):
        from pyqcd.operator import (
            flowed_gluon_ope, select_gradient_flow_gluon_component,
        )

        data = flowed_gluon_ope(_near_identity_su3(seed=7313), z_count=2)
        np.testing.assert_allclose(
            data[:, :, :, 1], data[:, :, :, 3] + data[:, :, :, 4],
            rtol=0, atol=2e-12,
        )
        np.testing.assert_allclose(
            data[:, :, :, 2], 2.0 * data[:, :, :, 5],
            rtol=0, atol=2e-12,
        )
        np.testing.assert_allclose(
            data[:, 0, :, 0], data[:, 0, :, 1] - data[:, 0, :, 2],
            rtol=0, atol=2e-12,
        )
        np.testing.assert_allclose(
            data[:, 1, :, 0], data[:, 1, :, 1] + data[:, 1, :, 2],
            rtol=0, atol=2e-12,
        )
        physical = select_gradient_flow_gluon_component(
            data, channel="helicity", orientation="odd_difference",
            component="helicity_physical_T_minus_S",
        )
        expected = data[1, 1, 3, 1] - data[1, 1, 3, 2]
        np.testing.assert_allclose(physical, expected, rtol=0, atol=2e-12)
        with self.assertRaisesRegex(ValueError, "legacy|显式选择"):
            select_gradient_flow_gluon_component(
                data, channel="helicity", orientation="odd_difference",
                component="combined",
            )
        np.testing.assert_allclose(
            data[:, :, 2], data[:, :, 0] + data[:, :, 1],
            rtol=0, atol=2e-12,
        )
        np.testing.assert_allclose(
            data[:, :, 3], data[:, :, 0] - data[:, :, 1],
            rtol=0, atol=2e-12,
        )

    def test_flow_step_default_and_epsilon_convergence(self):
        from pyqcd.operator import gradient_flow_gluon_ope

        gauge = _near_identity_su3(seed=7314)
        coarse, metadata = gradient_flow_gluon_ope(
            gauge, tau=0.02, z_count=2
        )
        fine, fine_metadata = gradient_flow_gluon_ope(
            gauge, tau=0.02, epsilon=0.005, z_count=2
        )
        self.assertEqual(metadata["flow"]["epsilon"], 0.01)
        self.assertEqual(metadata["flow"]["n_steps"], 2)
        self.assertEqual(fine_metadata["flow"]["n_steps"], 4)
        relative = np.linalg.norm((coarse - fine).ravel()) / max(
            np.linalg.norm(fine.ravel()), 1e-300
        )
        self.assertLess(relative, 1e-4)

    def test_memory_efficient_staple_matches_precomputed_roll_oracle(self):
        """On-demand rolls must preserve the precomputed six-staple result."""
        from pyqcd.renorm._gradient_flow import staple_6
        from pyqcd.tools import set_backend

        gauge = _near_identity_su3(
            seed=20261025, strength=0.2).astype(np.complex64)
        np.testing.assert_array_equal(
            np.asarray(staple_6(gauge)),
            np.asarray(staple_6(gauge, precompute_rolls=True)),
        )

        import torch

        set_backend('torch', device='cpu')
        try:
            tensor = torch.as_tensor(gauge, dtype=torch.complex64)
            actual = staple_6(tensor).detach().cpu().numpy()
            expected = staple_6(
                tensor, precompute_rolls=True).detach().cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
        finally:
            set_backend('numpy')

    def test_periodic_z_count_cannot_wrap_past_lattice_extent(self):
        """z_count 不得超过对应周期方向长度，避免静默回绕到同一格点。"""
        from pyqcd.operator import flowed_gluon_ope

        with self.assertRaisesRegex(ValueError, "z_count.*周期长度"):
            flowed_gluon_ope(
                _near_identity_su3(seed=7318), z_count=3, z_dir=2)

    def test_explicit_steps_cannot_exceed_requested_epsilon(self):
        """n_steps 覆盖不得把 RK3 实际步长放大到 epsilon 之上。"""
        from pyqcd.operator import gradient_flow_gluon_ope

        with self.assertRaisesRegex(ValueError, "实际步长.*超过"):
            gradient_flow_gluon_ope(
                _near_identity_su3(seed=7319),
                tau=0.1, epsilon=0.01, n_steps=1, z_count=1)

    def test_flow_to_quasi_matches_reference_component_factors(self):
        from pyqcd.renorm import coefficients, match_one
        from pyqcd.renorm._gradient_flow_quasi import FLOW_TO_QUASI_STATUS

        array = np.zeros((1, 2, 4, 6, 3, 1), dtype=np.complex128)
        array[0, 0, 0, 1, :, 0] = 2.0
        array[0, 0, 0, 2, :, 0] = 1.0
        matched, metadata = match_one(
            array, tau=0.5, a_fm=0.0775, mu_gev=2.0, alpha_s=0.25
        )
        _, c_perp, delta_m, _ = coefficients(
            0.5, 0.0775, 2.0, 0.25
        )
        line = np.exp(
            -delta_m * np.arange(3) * 0.0775 * 5.067730716
        )
        self.assertTrue(np.isnan(matched[0, 0, 0, 1, 0, 0]))
        np.testing.assert_allclose(
            matched[0, 0, 0, 1, 1:, 0],
            2.0 * line[1:] / (c_perp * c_perp),
            rtol=0, atol=2e-14,
        )
        np.testing.assert_allclose(
            matched[0, 0, 0, 0, 1:, 0],
            matched[0, 0, 0, 1, 1:, 0]
            - matched[0, 0, 0, 2, 1:, 0],
            rtol=0, atol=2e-14,
        )
        self.assertEqual(metadata["status"], FLOW_TO_QUASI_STATUS)
        self.assertEqual(metadata["z0_status"], "nan_fail_closed")
        self.assertFalse(metadata["physical_acceptance"])
        self.assertTrue(np.isfinite([
            metadata["c_parallel_perp"],
            metadata["c_perp_perp"],
            metadata["delta_m_GeV"],
            metadata["t_GeV_minus2"],
        ]).all())
        array[0, 1, 0, 1, :, 0] = 2.0
        array[0, 1, 0, 2, :, 0] = 1.0
        matched_h, metadata_h = match_one(
            array, tau=0.5, a_fm=0.0775, mu_gev=2.0, alpha_s=0.25
        )
        self.assertEqual(
            metadata_h["component_factors_helicity"],
            1.0 / metadata_h["c_perp_perp"],
        )
        np.testing.assert_allclose(
            matched_h[0, 1, 0, 0, 1:, 0],
            line[1:] / metadata_h["c_perp_perp"],
            rtol=0, atol=2e-14,
        )
        self.assertEqual(
            metadata_h["helicity_status"], "physical_T_minus_S"
        )

    def test_complex_ratio_uses_full_pol35_and_shared_nopol_denominator(self):
        from pyqcd.analysis import covariance_ratio_estimator

        rng = np.random.default_rng(7315)
        operator = rng.normal(size=(6, 3, 5)) + 1j * rng.normal(size=(6, 3, 5))
        pol35 = rng.normal(size=(6, 5, 2)) + 1j * rng.normal(size=(6, 5, 2))
        nopol = 4.0 + rng.normal(size=(6, 5, 2)) + 1j * rng.normal(size=(6, 5, 2))
        got = covariance_ratio_estimator(
            operator, pol35, True, denominator_correlator=nopol
        )
        mean_o = operator.mean(axis=0)
        mean_pol = pol35.mean(axis=0)
        c3 = (
            (operator[..., None] * pol35[:, None]).mean(axis=0)
            - mean_o[..., None] * mean_pol[None]
        ).mean(axis=1)
        c2 = nopol.mean(axis=(0, 1))
        np.testing.assert_allclose(got[0], c3, rtol=0, atol=3e-14)
        np.testing.assert_allclose(got[1], c2, rtol=0, atol=3e-14)
        complex_division = c3 / c2[None, :]
        np.testing.assert_allclose(
            got[0] / got[1][None, :],
            complex_division,
            rtol=0, atol=3e-14,
        )
        wrong = covariance_ratio_estimator(
            operator, pol35.imag, True, denominator_correlator=nopol
        )
        self.assertGreater(
            float(np.max(np.abs(wrong[0] - got[0]))), 1e-8,
            "丢弃 pol35 实部没有暴露 estimator 差异",
        )

    def test_ratio_axes_nopol_denominator_and_final_phase_projection(self):
        from pyqcd.analysis import (
            calculate_gradient_flow_ratios,
            physical_helicity_ratio,
            validate_gradient_flow_ratio_results,
        )

        rng = np.random.default_rng(7316)
        nconf, nt, nz = 5, 4, 2
        raw = rng.normal(size=(nconf, 2, 2, 3, nz, nt))
        raw = raw + 1j * rng.normal(size=raw.shape)
        raw[:, :, 1, :, 0] = raw[:, :, 0, :, 0]
        ope = np.empty((nconf, 2, 4, 3, nz, nt), dtype=np.complex128)
        for channel in range(2):
            for orientation in range(2):
                mt = raw[:, channel, orientation, 0]
                mi = raw[:, channel, orientation, 1]
                ope[:, channel, orientation, 1] = mt
                ope[:, channel, orientation, 2] = mi
                ope[:, channel, orientation, 0] = (
                    mt - mi if channel == 0 else mt + mi
                )
            ope[:, channel, 2] = ope[:, channel, 0] + ope[:, channel, 1]
            ope[:, channel, 3] = ope[:, channel, 0] - ope[:, channel, 1]
        twopt = (
            3.0 + rng.normal(size=(2, 1, nconf, 2, nt, 2))
            + 1j * rng.normal(size=(2, 1, nconf, 2, nt, 2))
        )
        result = calculate_gradient_flow_ratios(
            ope, twopt, [1, 2], do_jackknife=True
        )
        validate_gradient_flow_ratio_results(result, nconf=nconf)
        self.assertEqual(result["ratio"].shape, (2, 4, 3, 1, nz, 2, 3, 2))
        np.testing.assert_allclose(
            result["c2_mean"][0], result["c2_mean"][1], rtol=0, atol=0
        )
        phase = physical_helicity_ratio(result["ratio"])
        self.assertEqual(phase.shape, result["ratio"].shape[3:])
        self.assertTrue(np.isrealobj(phase))
        combined_phase = physical_helicity_ratio(
            result["ratio"], component="helicity_diagnostic_T_plus_S"
        )
        self.assertEqual(combined_phase.shape, phase.shape)
        with self.assertRaisesRegex(ValueError, "legacy|显式选择"):
            physical_helicity_ratio(result["ratio"], component="combined")

    def test_finite_b_ratio_v2_keeps_b_axis_isolated_and_complex_phase(self):
        from pyqcd.analysis import (
            calculate_gradient_flow_ratios,
            calculate_gradient_flow_tmd_ratios,
            physical_helicity_ratio,
            validate_gradient_flow_tmd_ratio_results,
        )

        rng = np.random.default_rng(7318)
        nconf, nt, nz, nb = 6, 5, 2, 3
        raw = (
            rng.normal(size=(nconf, 2, 2, 3, nz, nb, nt))
            + 1j * rng.normal(size=(nconf, 2, 2, 3, nz, nb, nt))
        )
        raw[:, :, 1, :, 0] = raw[:, :, 0, :, 0]
        ope = np.empty(
            (nconf, 2, 4, 3, nz, nb, nt), dtype=np.complex128)
        for channel in range(2):
            for orientation in range(2):
                mt = raw[:, channel, orientation, 0]
                mi = raw[:, channel, orientation, 1]
                ope[:, channel, orientation, 1] = mt
                ope[:, channel, orientation, 2] = mi
                ope[:, channel, orientation, 0] = (
                    mt - mi if channel == 0 else mt + mi
                )
            ope[:, channel, 2] = (
                ope[:, channel, 0] + ope[:, channel, 1])
            ope[:, channel, 3] = (
                ope[:, channel, 0] - ope[:, channel, 1])
        twopt = (
            3.0 + rng.normal(size=(2, 1, nconf, 2, nt, 2))
            + 1j * rng.normal(size=(2, 1, nconf, 2, nt, 2))
        )

        result = calculate_gradient_flow_tmd_ratios(
            ope, twopt, [1, 2], [0, 2, 4], do_jackknife=True)
        self.assertTrue(validate_gradient_flow_tmd_ratio_results(
            result, nconf=nconf))
        expected_shape = (2, 4, 3, 1, nz, nb, 2, 3, 2)
        self.assertEqual(result["ratio"].shape, expected_shape)
        self.assertEqual(result["axes"], (
            "channel", "z_orientation", "component", "direction",
            "z", "b", "tsep", "insertion", "momentum_abs",
        ))
        self.assertEqual(result["b_values"], (0, 2, 4))
        self.assertEqual(result["b_unit"], "lattice_spacing_a")
        np.testing.assert_array_equal(
            result["c2_mean"][0], result["c2_mean"][1])

        flattened = calculate_gradient_flow_ratios(
            ope.reshape(nconf, 2, 4, 3, nz * nb, nt),
            twopt,
            [1, 2],
            do_jackknife=True,
        )
        np.testing.assert_allclose(
            result["ratio"],
            flattened["ratio"].reshape(expected_shape),
            rtol=0,
            atol=0,
        )

        odd = 3
        physical = physical_helicity_ratio(result["ratio"])
        expected_phase = np.imag(
            result["ratio"][1, odd, 1] - result["ratio"][1, odd, 2])
        np.testing.assert_array_equal(physical, expected_phase)

        perturbed = ope.copy()
        perturbed[..., 2, :] += (
            (0.03 + 0.02j)
            * np.arange(1, perturbed[..., 2, :].size + 1).reshape(
                perturbed[..., 2, :].shape)
        )
        changed = calculate_gradient_flow_tmd_ratios(
            perturbed, twopt, [1, 2], [0, 2, 4], do_jackknife=True)
        np.testing.assert_array_equal(
            changed["ratio"][..., :2, :, :, :],
            result["ratio"][..., :2, :, :, :],
        )
        valid = result["valid_insertion_mask"]
        changed_b2 = changed["ratio"][..., 2, :, :, :]
        original_b2 = result["ratio"][..., 2, :, :, :]
        changed_valid = np.broadcast_to(
            valid[None, None, None, None, None, :, :, None],
            changed_b2.shape,
        )
        self.assertGreater(
            float(np.max(np.abs(
                (changed_b2 - original_b2)[changed_valid]))),
            1e-8,
        )
        central_only = calculate_gradient_flow_tmd_ratios(
            ope, twopt, [1, 2], [0, 2, 4], do_jackknife=False)
        self.assertFalse(central_only["do_jackknife"])
        self.assertTrue(validate_gradient_flow_tmd_ratio_results(
            central_only, nconf=nconf))
        for key in (
                "ratio_jackknife_mean",
                "ratio_jackknife_bias_corrected",
                "ratio_jackknife_error_real",
                "ratio_jackknife_error_imag"):
            self.assertTrue(np.isnan(central_only[key]).all())
        with self.assertRaisesRegex(ValueError, "b_values"):
            calculate_gradient_flow_tmd_ratios(
                ope, twopt, [1, 2], [0, 1], do_jackknife=True)
        with self.assertRaisesRegex(ValueError, "b_values"):
            calculate_gradient_flow_tmd_ratios(
                ope, twopt, [1, 2], [0, 2, 2], do_jackknife=True)

    def test_finite_b_delete_one_jackknife_matches_direct_resampling(self):
        from pyqcd.analysis import covariance_ratio_estimator

        rng = np.random.default_rng(7319)
        nconf, nz, nb, nsource, nmom = 6, 2, 3, 4, 2
        nobs = nz * nb
        operator = (
            rng.normal(size=(nconf, nobs, nsource))
            + 1j * rng.normal(size=(nconf, nobs, nsource))
        )
        pol35 = (
            rng.normal(size=(nconf, nsource, nmom))
            + 1j * rng.normal(size=(nconf, nsource, nmom))
        )
        nopol = (
            4.0 + rng.normal(size=(nconf, nsource, nmom))
            + 1j * rng.normal(size=(nconf, nsource, nmom))
        )
        result = covariance_ratio_estimator(
            operator, pol35, True, denominator_correlator=nopol)

        def direct_central(selected):
            op = operator[selected]
            corr = pol35[selected]
            denominator = nopol[selected]
            nselected = len(selected)
            cross = np.einsum(
                "nos,nsp->op", op, corr, optimize=True
            ) / nsource / nselected
            disconnected = np.einsum(
                "os,sp->op", op.sum(axis=0), corr.sum(axis=0),
                optimize=True,
            ) / nsource / (nselected * nselected)
            return (
                cross - disconnected,
                denominator.mean(axis=(0, 1)),
            )

        all_conf = np.arange(nconf)
        c3, c2 = direct_central(all_conf)
        jackknife = []
        for omitted in range(nconf):
            c3_jk, c2_jk = direct_central(np.delete(all_conf, omitted))
            jackknife.append(c3_jk / c2_jk[None, :])
        jackknife = np.asarray(jackknife)
        jackknife_mean = jackknife.mean(axis=0)
        central_ratio = c3 / c2[None, :]
        bias_corrected = (
            nconf * central_ratio - (nconf - 1) * jackknife_mean)
        prefactor = (nconf - 1) / nconf
        error_real = np.sqrt(prefactor * np.sum(
            (jackknife.real - jackknife_mean.real) ** 2, axis=0))
        error_imag = np.sqrt(prefactor * np.sum(
            (jackknife.imag - jackknife_mean.imag) ** 2, axis=0))

        np.testing.assert_allclose(result[0], c3, rtol=0, atol=3e-14)
        np.testing.assert_allclose(result[1], c2, rtol=0, atol=3e-14)
        np.testing.assert_allclose(
            result[2], jackknife_mean, rtol=0, atol=3e-14)
        np.testing.assert_allclose(
            result[3], bias_corrected, rtol=0, atol=3e-14)
        np.testing.assert_allclose(
            result[4], error_real, rtol=0, atol=3e-14)
        np.testing.assert_allclose(
            result[5], error_imag, rtol=0, atol=3e-14)
        self.assertTrue(all(np.isfinite(np.asarray(value)).all()
                            for value in result))

    def test_two_dimensional_ratio_v3_axes_and_cross_configuration_gate(self):
        import hashlib

        from pyqcd.analysis import (
            calculate_gradient_flow_ratios,
            calculate_gradient_flow_tmd_ratios_2d,
            calculate_gradient_flow_tmd_ratios_2d_from_configurations,
            physical_helicity_ratio,
            validate_gradient_flow_tmd_ratio_2d_results,
        )

        rng = np.random.default_rng(7321)
        nconf, nt, nz, nbx, nby, ncomp = 4, 4, 2, 2, 2, 3
        raw = (
            rng.normal(size=(nconf, 2, 2, ncomp, nz, nbx, nby, nt))
            + 1j * rng.normal(size=(nconf, 2, 2, ncomp, nz, nbx, nby, nt))
        )
        raw[:, :, 1, :, 0] = raw[:, :, 0, :, 0]
        ope = np.empty(
            (nconf, 2, 4, ncomp, nz, nbx, nby, nt),
            dtype=np.complex128,
        )
        for channel in range(2):
            for orientation in range(2):
                mt = raw[:, channel, orientation, 1]
                mi = raw[:, channel, orientation, 2]
                ope[:, channel, orientation, 1] = mt
                ope[:, channel, orientation, 2] = mi
                ope[:, channel, orientation, 0] = (
                    mt - mi if channel == 0 else mt + mi)
            ope[:, channel, 2] = (
                ope[:, channel, 0] + ope[:, channel, 1])
            ope[:, channel, 3] = (
                ope[:, channel, 0] - ope[:, channel, 1])
        twopt = (
            3.0 + rng.normal(size=(2, 1, nconf, 2, nt, 2))
            + 1j * rng.normal(size=(2, 1, nconf, 2, nt, 2))
        )

        result = calculate_gradient_flow_tmd_ratios_2d(
            ope,
            twopt,
            [1, 2],
            [0, 2],
            [0, 1],
            z_dir=2,
            b_x_dir=0,
            b_y_dir=1,
            transverse_order=(0, 1),
            do_jackknife=True,
        )
        self.assertTrue(validate_gradient_flow_tmd_ratio_2d_results(
            result, nconf=nconf))
        expected_shape = (2, 4, ncomp, 1, nz, nbx, nby, 2, 3, 2)
        self.assertEqual(result["ratio"].shape, expected_shape)
        flattened = calculate_gradient_flow_ratios(
            ope.reshape(nconf, 2, 4, ncomp, nz * nbx * nby, nt),
            twopt,
            [1, 2],
            do_jackknife=True,
        )
        np.testing.assert_allclose(
            result["ratio"],
            flattened["ratio"].reshape(expected_shape),
            rtol=0,
            atol=0,
        )
        phase = physical_helicity_ratio(result["ratio"])
        expected_phase = np.imag(
            result["ratio"][1, 3, 1] - result["ratio"][1, 3, 2])
        np.testing.assert_array_equal(phase, expected_phase)

        ope_by_conf = {cid: ope[index] for index, cid in enumerate((40, 10, 30, 20))}
        twopt_by_conf = {
            cid: twopt[:, :, index]
            for index, cid in enumerate((40, 10, 30, 20))
        }
        conf_ids = [40, 10, 30, 20]
        cross = calculate_gradient_flow_tmd_ratios_2d_from_configurations(
            ope_by_conf,
            twopt_by_conf,
            conf_ids,
            [0, 2],
            [0, 1],
            [1, 2],
            z_dir=2,
            b_x_dir=0,
            b_y_dir=1,
            transverse_order=(0, 1),
            do_jackknife=True,
        )
        np.testing.assert_allclose(
            cross["ratio"], result["ratio"], rtol=0, atol=0)
        self.assertEqual(tuple(cross["conf_ids"]), tuple(conf_ids))
        self.assertEqual(
            cross["input_index_sha256"],
            hashlib.sha256(
                np.asarray(conf_ids, dtype='<i8').tobytes(order='C')
            ).hexdigest(),
        )
        tampered = dict(cross)
        tampered["transverse_order"] = (0, 0)
        with self.assertRaisesRegex(ValueError, 'transverse_order'):
            validate_gradient_flow_tmd_ratio_2d_results(
                tampered, nconf=nconf)
        source_averaged = dict(twopt_by_conf)
        source_averaged[10] = np.ones(nt)
        with self.assertRaisesRegex(ValueError, 'source-resolved'):
            calculate_gradient_flow_tmd_ratios_2d_from_configurations(
                ope_by_conf, source_averaged, conf_ids,
                [0, 2], [0, 1], [1, 2])
        with self.assertRaisesRegex(ValueError, '重复'):
            calculate_gradient_flow_tmd_ratios_2d_from_configurations(
                ope_by_conf, twopt_by_conf, [40, 10, 10],
                [0, 2], [0, 1], [1, 2])

    def test_configuration_ratio_v2_requires_source_resolved_shared_order(self):
        import hashlib

        from pyqcd.analysis import (
            calculate_gradient_flow_tmd_ratios,
            calculate_gradient_flow_tmd_ratios_from_configurations,
            validate_gradient_flow_tmd_ratio_results,
        )

        rng = np.random.default_rng(7320)
        conf_ids = [40, 10, 30, 20]
        nt, nz, nb, ncomp = 4, 1, 2, 3
        ope_by_conf = {}
        twopt_by_conf = {}
        for sample, conf_id in enumerate(conf_ids):
            raw = (
                rng.normal(size=(2, 2, ncomp, nz, nb, nt))
                + 1j * rng.normal(size=(2, 2, ncomp, nz, nb, nt))
            )
            raw[:, :, :, 0, :, :] += sample * 0.01
            raw[:, 1, :, 0] = raw[:, 0, :, 0]
            ope = np.empty(
                (2, 4, ncomp, nz, nb, nt), dtype=np.complex128)
            for channel in range(2):
                for orientation in range(2):
                    mt = raw[channel, orientation, 1]
                    mi = raw[channel, orientation, 2]
                    ope[channel, orientation, 1] = mt
                    ope[channel, orientation, 2] = mi
                    ope[channel, orientation, 0] = (
                        mt - mi if channel == 0 else mt + mi
                    )
                ope[channel, 2] = (
                    ope[channel, 0] + ope[channel, 1])
                ope[channel, 3] = (
                    ope[channel, 0] - ope[channel, 1])
            ope_by_conf[conf_id] = ope
            twopt_by_conf[conf_id] = (
                3.0 + rng.normal(size=(2, 1, 2, nt, 2))
                + 1j * rng.normal(size=(2, 1, 2, nt, 2))
            )

        actual = calculate_gradient_flow_tmd_ratios_from_configurations(
            ope_by_conf,
            twopt_by_conf,
            conf_ids,
            [0, 2],
            [1, 2],
            do_jackknife=True,
        )
        self.assertTrue(validate_gradient_flow_tmd_ratio_results(
            actual, nconf=len(conf_ids)))
        expected = calculate_gradient_flow_tmd_ratios(
            np.stack([ope_by_conf[cid] for cid in conf_ids]),
            np.stack([twopt_by_conf[cid] for cid in conf_ids], axis=2),
            [1, 2],
            [0, 2],
            do_jackknife=True,
        )
        for key in (
                "c3_mean", "ratio", "ratio_jackknife_mean",
                "ratio_jackknife_bias_corrected",
                "ratio_jackknife_error_real",
                "ratio_jackknife_error_imag"):
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=0, atol=0)
        expected_hash = hashlib.sha256(
            np.asarray(conf_ids, dtype='<i8').tobytes(order='C')
        ).hexdigest()
        self.assertEqual(tuple(actual["conf_ids"]), tuple(conf_ids))
        self.assertEqual(actual["input_index_sha256"], expected_hash)
        self.assertTrue(actual["source_resolved_c2"])
        bad_hash = dict(actual)
        bad_hash["input_index_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "index_sha256|索引"):
            validate_gradient_flow_tmd_ratio_results(
                bad_hash, nconf=len(conf_ids))
        unaveraged_claim = dict(actual)
        unaveraged_claim["source_resolved_c2"] = False
        with self.assertRaisesRegex(ValueError, "source-resolved"):
            validate_gradient_flow_tmd_ratio_results(
                unaveraged_claim, nconf=len(conf_ids))

        source_averaged = dict(twopt_by_conf)
        source_averaged[10] = np.ones(nt)
        with self.assertRaisesRegex(ValueError, "source-resolved"):
            calculate_gradient_flow_tmd_ratios_from_configurations(
                ope_by_conf, source_averaged, conf_ids, [0, 2], [1, 2])
        with self.assertRaisesRegex(ValueError, "conf_ids.*重复|重复"):
            calculate_gradient_flow_tmd_ratios_from_configurations(
                ope_by_conf, twopt_by_conf, [40, 10, 10], [0, 2], [1, 2])
        with self.assertRaisesRegex(ValueError, "缺少|missing"):
            calculate_gradient_flow_tmd_ratios_from_configurations(
                ope_by_conf, twopt_by_conf, [40, 99], [0, 2], [1, 2])

    def test_complex64_ratio_uses_precision_aware_orientation_gate(self):
        from pyqcd.analysis import (
            calculate_gradient_flow_tmd_ratios,
            validate_gradient_flow_tmd_ratio_results,
        )

        rng = np.random.default_rng(20261025)
        ope = (
            rng.normal(size=(2, 2, 4, 3, 2, 4))
            + 1j * rng.normal(size=(2, 2, 4, 3, 2, 4))
        ).astype(np.complex64)
        ope[:, :, 2] = ope[:, :, 0] + ope[:, :, 1]
        ope[:, :, 3] = ope[:, :, 0] - ope[:, :, 1]
        ope[:, 1, 3, :, 0, :] = 0.0
        ope = ope[:, :, :, :, :, None, :]
        twopt = (
            rng.normal(size=(2, 1, 2, 1, 4, 1))
            + 1j * rng.normal(size=(2, 1, 2, 1, 4, 1))
        ).astype(np.complex64)
        actual = calculate_gradient_flow_tmd_ratios(
            ope, twopt, [1], [0], do_jackknife=True,
            component_labels=('combined', 'Mtiti', 'Mijij'),
        )
        self.assertEqual(actual['input_precision'], 'complex64')
        self.assertTrue(validate_gradient_flow_tmd_ratio_results(
            actual, nconf=2))

    def test_npz_json_roundtrip_is_atomic_and_strict(self):
        from pyqcd.operator import (
            gradient_flow_gluon_ope, load_gradient_flow_gluon_ope,
            save_gradient_flow_gluon_ope,
        )

        data, metadata = gradient_flow_gluon_ope(
            _near_identity_su3(seed=7317), tau=0.0, z_count=1
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = f"{temporary}/conf4150_tau0p000_eps0.010_zdir2"
            paths = save_gradient_flow_gluon_ope(base, data, metadata)
            self.assertTrue(all(path.endswith(suffix)
                                for path, suffix in zip(paths, (".npy", ".json"))))
            loaded, loaded_metadata = load_gradient_flow_gluon_ope(base)
        np.testing.assert_array_equal(loaded, data)
        self.assertEqual(loaded_metadata, metadata)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        GradientFlowGluonContract
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
