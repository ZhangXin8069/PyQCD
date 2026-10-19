from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.linalg import expm

from .core import (FLOW_SOURCE, atomic_npz, canonical_hash, digest, dslash_central, free_noise_reference,
                   free_wilson_kinetic, gamma_dr, kinetic_sample, laplace, metadata_array,
                   unit_gauge, validate_plan, z4_noise)
from .collect import CHECKS, collect, read_raw, validate_review
from .measure import check_shard, input_identity
from quark_sixflow_ringed_sfte_v1 import load_bare, load_kinetic

HERE = Path(__file__).resolve().parent


def matrix_of(operation, shape):
    n = int(np.prod(shape))
    basis = np.eye(n, dtype=complex)
    return np.column_stack([operation(basis[:, j].reshape(shape)).ravel() for j in range(n)])


class StencilTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shape = (2, 2, 2, 2, 4, 3)
        cls.gauge = unit_gauge(cls.shape[:4])
        cls.derivative = matrix_of(lambda x: dslash_central(x, cls.gauge), cls.shape)
        cls.lap = matrix_of(lambda x: laplace(x, cls.gauge), cls.shape)
        cls.mass = .13
        cls.dirac = cls.mass*np.eye(np.prod(cls.shape)) - cls.lap/2 + cls.derivative

    def test_gamma_matches_production(self):
        sys.path.insert(0, str(FLOW_SOURCE))
        from flow_common_l32x96 import gamma_matrices
        np.testing.assert_array_equal(gamma_dr(), gamma_matrices()[1:5])
        for i, a in enumerate(gamma_dr()):
            np.testing.assert_array_equal(a, a.conj().T)
            for j, b in enumerate(gamma_dr()):
                np.testing.assert_array_equal(a@b+b@a, 2*np.eye(4) if i == j else np.zeros((4, 4)))

    def test_antiperiodic_plane_wave(self):
        shape = (4, 2, 2, 2, 4, 3)
        v = np.ones(shape, complex)*np.exp(1j*np.pi*np.arange(4)/4).reshape(4, 1, 1, 1, 1, 1)
        expected = 1j*np.sin(np.pi/4)*np.einsum("rs,...sc->...rc", gamma_dr()[3], v)
        np.testing.assert_allclose(dslash_central(v, unit_gauge(shape[:4])), expected, atol=1e-14)

    def test_antihermitian_and_no_spin_trace_contact(self):
        np.testing.assert_allclose(self.derivative.conj().T, -self.derivative, atol=1e-14)
        k = expm(.1*self.lap)
        self.assertLess(abs(np.trace(self.derivative@k@k.conj().T)), 1e-13)

    def test_exact_trace_and_minus_two_volume_factor(self):
        k = expm(.2*self.lap)
        st = k@np.linalg.solve(self.dirac, k.conj().T)
        got = -2*np.trace(self.derivative@st)/np.prod(self.shape[:4])
        exact = free_wilson_kinetic([2, 2, 2, 2], self.mass, [.2])[0]
        np.testing.assert_allclose(got, exact, rtol=1e-12, atol=1e-13)
        self.assertLess(exact, 0)

    def test_noise_estimator_matches_fourier_same_noise(self):
        eta = z4_noise(self.shape, 32, 10000, 0)
        k = expm(.1*self.lap)
        rho = (k@eta.ravel()).reshape(self.shape)
        phi = (k@np.linalg.solve(self.dirac, eta.ravel())).reshape(self.shape)
        got = kinetic_sample(rho, phi, self.gauge)
        exact = free_noise_reference(eta, self.mass, [.1])[0]
        np.testing.assert_allclose(got, exact, atol=1e-13, rtol=1e-12)

    def test_gauge_covariance(self):
        rng = np.random.default_rng(17)
        shape = (4, 2, 2, 2)
        r = rng.normal(size=(*shape, 3, 3)) + 1j*rng.normal(size=(*shape, 3, 3))
        g, _ = np.linalg.qr(r)
        g[..., :, 0] /= np.linalg.det(g)[..., None]
        gauge = unit_gauge(shape)
        transformed = np.stack([g@np.roll(g, -1, axis=3-mu).swapaxes(-1, -2).conj() for mu in range(4)])
        eta = z4_noise((*shape, 4, 3), 2, 3, 4)
        rotate = lambda x: np.einsum("...ab,...sb->...sa", g, x)
        np.testing.assert_allclose(dslash_central(rotate(eta), transformed), rotate(dslash_central(eta, gauge)), atol=3e-14)
        np.testing.assert_allclose(laplace(rotate(eta), transformed), rotate(laplace(eta, gauge)), atol=3e-14)

    def test_rk3_step_halving(self):
        eta = z4_noise(self.shape, 12, 45, 0).ravel()
        target = expm(.2*self.lap)@eta
        errors = []
        for eps in (.02, .01):
            y = eta.copy()
            for _ in range(round(.2/eps)):
                a = self.lap@y
                b = self.lap@(y+eps*a/4)
                c = self.lap@(y+eps*(8*b-2*a)/9)
                y += eps*(a/4+3*c/4)
            errors.append(np.linalg.norm(y-target))
        self.assertGreater(errors[0]/errors[1], 7)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ringed-test-")
        self.root = Path(self.tmp.name)
        with (HERE/"l32x96_pilot.json").open() as f:
            self.plan = json.load(f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan_and_checkpoint_guards(self):
        np.testing.assert_array_equal(validate_plan(self.plan), [5, 10, 20, 50, 80, 100])
        bad = copy.deepcopy(self.plan)
        bad["flow_times"][0] = .051
        with self.assertRaises(ValueError):
            validate_plan(bad)
        bad["flow_times"] = [.1, .1]
        with self.assertRaises(ValueError):
            validate_plan(bad)

    def test_noise_reproducible_full_trace_normalization(self):
        a = z4_noise((2, 2, 2, 2, 4, 3), 77, 10000, 1)
        np.testing.assert_array_equal(a, z4_noise(a.shape, 77, 10000, 1))
        self.assertFalse(np.array_equal(a, z4_noise(a.shape, 77, 10100, 1)))
        self.assertEqual(np.vdot(a, a).real, a.size)

    def test_no_overwrite_and_resume_identity(self):
        path = self.root/"noise.npz"
        atomic_npz(path, metadata_json=metadata_array({"run_identity_sha256": "abc"}), kinetic_a4_noise=np.array([-1., -2.]))
        with self.assertRaises(FileExistsError):
            atomic_npz(path, data=[0])
        check_shard(path, "abc", 2)
        with self.assertRaises(ValueError):
            check_shard(path, "changed", 2)

    def make_bare(self):
        path = self.root/"bare.npz"
        m = np.ones((6, 1, 2))
        atomic_npz(path, schema=np.array("quark_flow_unpolarized_sixflow_bare_sumdiff_aic_v1"),
                   flow_times=np.array(self.plan["flow_times"]), confs=np.array([10000, 10100]),
                   bootstrap_indices=np.array([[0, 0], [1, 1], [0, 1]], dtype=int),
                   momenta_integer_pz=np.array([3]), delta_z_values=np.array([0, 1]), a_fm=np.array(.0775),
                   M=m, M_filled=m, M_boot=np.ones((6, 3, 1, 2)), M_boot_filled=np.ones((6, 3, 1, 2)),
                   fit_valid=np.ones_like(m, bool), estimate_available=np.ones_like(m, bool),
                   central_fallback_used=np.zeros_like(m, bool), replica_fallback_used=np.zeros((6, 3, 1, 2), bool),
                   estimator_id=np.array("test"), q_min=np.array(1e-6), correlated_chi2_dof_max=np.array(10.),
                   fixed_tmin=np.array(8), fixed_tmin_fm=np.array(.62))
        return path

    def make_raw(self, conf, offset, metadata_updates=None):
        identity = {"plan": self.plan, "conf": conf, "noise_count": 4}
        meta = {"schema": "quark_ringed_configuration_raw_v1", "run_identity": identity,
                "run_identity_sha256": canonical_hash(identity), "trace_space": "full_spin_color_spacetime",
                "status": "measured_unvalidated", "axes_kinetic": ["flow", "noise"],
                "temporal_truncation": "none", "noise": "full_volume_Z4_E_eta_eta_dagger_equals_I",
                "fermion_normalization": "D_mass_inverse_no_extra_2kappa",
                "derivative_discretization": "4D_symmetric_covariant_nearest_neighbor_aD",
                "flavor_count": 1}
        meta.update(metadata_updates or {})
        values = -np.ones((6, 4))*(1+offset) + np.array([-.02, .02, -.01, .01])[None]
        path = self.root/f"raw_{conf}.npz"
        atomic_npz(path, metadata_json=metadata_array(meta), configuration=np.array(conf),
                   flow_times=np.array(self.plan["flow_times"]), kinetic_a4_noise=values.astype(complex),
                   kinetic_a4_config=values.mean(axis=1).astype(complex), true_residual=np.full(4, 1e-12))
        return path

    def test_collection_pairs_indices_and_blocks_unvalidated_export(self):
        bare = self.make_bare()
        raw = [self.make_raw(10100, .5), self.make_raw(10000, 0)]
        out = self.root/"collected.npz"
        collect(raw, bare, out)
        with np.load(out, allow_pickle=False) as d:
            np.testing.assert_array_equal(d["confs"], [10000, 10100])
            np.testing.assert_allclose(d["kinetic_a4_bootstrap"][0], [-1, -1.5, -1.25])
            self.assertTrue(np.isnan(d["Zchi"]).all())
        with self.assertRaisesRegex(ValueError, "status"):
            load_kinetic(out, load_bare(bare))

    def test_duplicate_missing_and_template_rejected(self):
        bare = self.make_bare()
        raw = self.make_raw(10000, 0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            collect([raw, raw], bare, self.root/"out.npz")
        with self.assertRaisesRegex(ValueError, "exactly"):
            collect([raw], bare, self.root/"out.npz")
        with self.assertRaises(ValueError):
            validate_review(HERE/"validation_review.template.json", {}, self.plan)

    def test_raw_normalization_and_temporal_window_rejected(self):
        wrong_norm = self.make_raw(10000, 0, {"fermion_normalization": "unit_norm_source"})
        truncated = self.make_raw(10100, 0, {"temporal_truncation": "windowed"})
        for path in (wrong_norm, truncated):
            with self.assertRaisesRegex(ValueError, "Raw metadata"):
                read_raw(path)

    def test_synthetic_review_export_matches_existing_consumer(self):
        # Temporary synthetic evidence tests the interface, not physics acceptance.
        self.plan["mass_treatment"] = "synthetic_direct_critical_mass_TEST_ONLY"
        bare = self.make_bare()
        raw = [self.make_raw(10100, .5), self.make_raw(10000, 0)]
        hashes = {str(read_raw(path)[0]): digest(path) for path in raw}
        evidence = self.root/"synthetic_evidence.npz"
        atomic_npz(evidence, notice=np.array("TEST_ONLY_NOT_PHYSICS_EVIDENCE"))
        review = {"schema": "quark_ringed_validation_review_v1", "status": "validated",
                  "input_sha256_by_conf": hashes,
                  "mass_prescription": {"kind": "direct_critical_mass", "bare_mass": self.plan["mass"]},
                  "checks": {name: {"status": "passed", "summary": "synthetic unit test only",
                                    "evidence_path": str(evidence), "evidence_sha256": digest(evidence)}
                             for name in CHECKS}}
        path = self.root/"synthetic_review.json"
        path.write_text(json.dumps(review))
        output = self.root/"accepted_test_only.npz"
        collect(raw, bare, output, path)
        result, meta = load_kinetic(output, load_bare(bare))
        self.assertEqual(meta["status"], "validated")
        with np.load(output, allow_pickle=False) as d:
            np.testing.assert_array_equal(result[0], d["Zchi"])
            np.testing.assert_array_equal(result[1], d["Zchi_bootstrap"])
        pilot = dict(self.plan, mass_treatment="finite_mass_pilot_NOT_chiral")
        with self.assertRaisesRegex(ValueError, "Finite-mass pilot"):
            validate_review(path, hashes, pilot)
        evidence.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or changed evidence"):
            validate_review(path, hashes, self.plan)

    def test_gauge_payload_shape_guard(self):
        gauge = self.root/"gauge.raw"
        gauge.touch()
        self.plan["gauge_path_template"] = str(gauge)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            input_identity(self.plan, 10000, False)


if __name__ == "__main__":
    unittest.main()
