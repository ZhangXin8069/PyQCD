"""Safety contracts for analysis and contraction audit fixes."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np


class Test9MomentumSafetyContractTests(unittest.TestCase):
    @staticmethod
    def _write_corr(root: Path, conf_ids):
        t = np.arange(72, dtype=np.float64)
        for offset, conf_id in enumerate(conf_ids):
            data_dir = root / "data" / f"conf{conf_id}"
            data_dir.mkdir(parents=True)
            corr = np.exp(-(0.25 + 0.002 * offset) * t) + 0.01
            np.save(data_dir / f"corr_pp_P200_{conf_id}.npy", corr)

    def test_missing_pz6_is_not_copied_from_another_momentum(self):
        from pyqcd.analysis._test9_extended import generate_test6_style_plots

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "test9"
            self._write_corr(root, [11, 12])
            messages = []
            generate_test6_style_plots(
                root,
                Path(tmpdir) / "out",
                [11, 12],
                ["P200"],
                logger=messages.append,
            )
            result_root = Path(tmpdir) / "out" / "1_result" / "L24x72"

            self.assertTrue((result_root / "Pz2").is_dir())
            self.assertFalse((result_root / "Pz6").exists())
            self.assertTrue(any(
                "Pz6" in message and "refusing to copy" in message
                for message in messages))

    def test_explicit_missing_pz6_raises_without_fabricating_output(self):
        from pyqcd.analysis._test9_extended import generate_test6_style_plots

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(FileNotFoundError, "Pz6.*missing"):
                generate_test6_style_plots(
                    root / "test9",
                    root / "out",
                    [11],
                    ["P600"],
                    logger=lambda _message: None,
                )
            self.assertFalse(
                (root / "out" / "1_result" / "L24x72" / "Pz6").exists())

    def test_invalid_momentum_label_fails_instead_of_becoming_pz0(self):
        from pyqcd.analysis._test9_extended import generate_test6_style_plots

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "invalid momentum tag"):
                generate_test6_style_plots(
                    root / "test9",
                    root / "out",
                    [11],
                    ["Pz2"],
                    logger=lambda _message: None,
                )
            self.assertFalse(
                (root / "out" / "1_result" / "L24x72" / "Pz0").exists())


class RatioAxisSafetyContractTests(unittest.TestCase):
    def test_unequal_sink_and_tau_axes_contract_in_2d_mode(self):
        from pyqcd.analysis import ratio_3pt

        c3 = np.arange(2 * 2 * 4 * 4, dtype=np.float64).reshape(2, 2, 4, 4)
        c2 = np.ones((2, 4, 2, 4), dtype=np.float64)
        result = ratio_3pt(
            c3,
            c2,
            t_sep=2,
            Nconf_axes=0,
            tau_axes=2,
            t_sink_axes=1,
            t_src_axes=3,
        )

        self.assertEqual(result["data_sample"].shape, c3.shape)
        np.testing.assert_allclose(
            result["data_sample"], c3, rtol=0.0, atol=0.0)

    def test_unequal_sink_and_tau_axes_contract_in_1d_mode(self):
        from pyqcd.analysis import ratio_3pt

        c3 = np.arange(2 * 2 * 4, dtype=np.float64).reshape(2, 2, 4)
        c2 = np.ones((2, 4, 2), dtype=np.float64)
        result = ratio_3pt(
            c3,
            c2,
            t_sep=2,
            Nconf_axes=0,
            tau_axes=2,
            t_sink_axes=1,
        )

        self.assertEqual(result["data_sample"].shape, c3.shape)
        np.testing.assert_allclose(
            result["data_sample"], c3, rtol=0.0, atol=0.0)


class AnalysisInputSafetyContractTests(unittest.TestCase):
    def test_meff_rejects_implicit_complex_absolute_value(self):
        from pyqcd.analysis import meff

        t = np.arange(6, dtype=np.float64)
        real = np.exp(-0.2 * t)[None, :] * np.ones((3, 1))
        expected = meff(real, 1.0)

        with self.assertRaisesRegex(ValueError, "complex data"):
            meff(real.astype(np.complex128), 1.0)

        projected = meff(
            real.astype(np.complex128), 1.0, projection="abs")
        np.testing.assert_allclose(
            projected["data_sample"],
            expected["data_sample"],
            rtol=0.0,
            atol=0.0,
        )

    def test_fit_constant_window_rejects_unknown_kind(self):
        from pyqcd.analysis import fit_constant_window

        data = np.array([
            [1.00, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00],
        ])
        valid = fit_constant_window(data, kind="boot")
        self.assertEqual(valid["fit_status"], "identifiable")
        with self.assertRaisesRegex(ValueError, "kind"):
            fit_constant_window(data, kind="diagonal")


class WickEquivalenceSafetyContractTests(unittest.TestCase):
    @staticmethod
    def _base_result():
        from pyqcd.contraction import wick_contraction

        return wick_contraction(
            ["|", "u", "u", "gamma_7", "d", "|"],
            ["|", "u^d", "gamma_7", "d^d", "u^d", "|"],
            [],
            "2pt",
        )

    def test_duplicate_complete_diagrams_are_still_merged(self):
        from pyqcd.contraction import identify_equivalent_diagrams

        base = self._base_result()
        groups = identify_equivalent_diagrams(
            base, copy.deepcopy(base))

        self.assertEqual([len(group) for group in groups], [2, 2])

    def test_v_gamma_time_and_topology_are_part_of_equivalence_proof(self):
        from pyqcd.contraction import identify_equivalent_diagrams

        base = self._base_result()

        def mutate(which):
            changed = copy.deepcopy(base)
            if which == "gamma":
                old = changed["gamma_pos"][0]
                changed["gamma_pos"][0] = (
                    old[0], "gamma_5", old[2], old[3])
            elif which == "vertex":
                changed["V"][0][3] = "other_time"
            elif which == "topology":
                changed["V"][0][0] = (0, 4)
            elif which == "time":
                changed["peram"][0][0][4][0] = "other_time"
            return changed

        for which in ("gamma", "vertex", "topology", "time"):
            with self.subTest(which=which):
                groups = identify_equivalent_diagrams(
                    base, mutate(which))
                target_group = next(
                    group for group in groups
                    if (0, 0) in [entry[:2] for entry in group])
                self.assertFalse(any(
                    entry[0] == 1 for entry in target_group))

    def test_incomplete_wick_result_is_rejected(self):
        from pyqcd.contraction import identify_equivalent_diagrams

        incomplete = self._base_result()
        del incomplete["V"]
        with self.assertRaisesRegex(ValueError, "missing fields.*V"):
            identify_equivalent_diagrams(incomplete)


class HelicityArgumentSafetyContractTests(unittest.TestCase):
    @staticmethod
    def _kwargs(**overrides):
        values = {
            "gauge": None,
            "pla": None,
            "pla_tilde": None,
            "z_dir": 2,
            "delta_z": 1,
            "mu": 0,
            "nu": 1,
            "mu2": 0,
            "nu2": 1,
        }
        values.update(overrides)
        return values

    def test_boolean_delta_z_is_rejected(self):
        from pyqcd.operator import helicity_two_field_operator

        with self.assertRaisesRegex(TypeError, "delta_z"):
            helicity_two_field_operator(**self._kwargs(delta_z=True))
        with self.assertRaisesRegex(TypeError, "delta_z"):
            helicity_two_field_operator(**self._kwargs(delta_z=np.bool_(False)))

    def test_string_minus_is_rejected(self):
        from pyqcd.operator import helicity_two_field_operator

        with self.assertRaisesRegex(TypeError, "minus"):
            helicity_two_field_operator(**self._kwargs(minus="minus"))
        with self.assertRaisesRegex(TypeError, "minus"):
            helicity_two_field_operator(**self._kwargs(minus=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
