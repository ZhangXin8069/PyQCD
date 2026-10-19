"""Public regression contracts for the scalar/multi-b invariant amplitude."""
from __future__ import annotations

import unittest

import numpy as np

from pyqcd.renorm import invariant_amplitude


class InvariantAmplitudeContract(unittest.TestCase):
    """Keep legacy 1-D output while supporting an explicit b_perp axis."""

    def test_one_dimensional_input_keeps_legacy_shape(self):
        matrix = np.array([1.0 + 0.2j, 0.8 - 0.1j, 0.4 + 0.05j, 0.2])
        result = invariant_amplitude(
            matrix, np.array([0.0, 0.37]), np.array([0.1]),
            nu_grid=np.arange(4, dtype=float) * 0.4)

        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.isfinite(result).all())

    def test_two_dimensional_input_returns_x_by_b_shape(self):
        matrix = np.array([1.0 + 0.2j, 0.8 - 0.1j, 0.4 + 0.05j, 0.2])
        matrix_2d = np.column_stack((matrix, 2.0 * matrix))
        x_grid = np.array([0.0, 0.37])

        nu_grid = np.arange(4, dtype=float) * 0.4
        scalar = invariant_amplitude(
            matrix, x_grid, np.array([0.1]), nu_grid=nu_grid)
        result = invariant_amplitude(
            matrix_2d, x_grid, np.array([0.1, 0.2]), nu_grid=nu_grid)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result[:, 0], scalar)
        np.testing.assert_allclose(result[:, 1], 2.0 * scalar)

    def test_single_b_and_zero_x_boundary_keeps_two_axes(self):
        matrix = np.ones((4, 1), dtype=complex)
        result = invariant_amplitude(
            matrix, np.array([0.0]), np.array([0.0]),
            nu_grid=np.arange(4, dtype=float) * 0.4)

        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(np.isfinite(result).all())

    def test_full_line_fourier_convention_has_factor_two(self):
        """invariant_amplitude 必须实现 -2/pi ∫ dnu cos(x nu) Re M_pp。"""
        matrix = np.array([2.0, 3.0, 4.0], dtype=complex)
        nu_grid = np.array([0.0, 0.5, 1.0])
        x_grid = np.array([0.0, 0.4])

        actual = invariant_amplitude(
            matrix, x_grid, np.array([0.1]), nu_grid=nu_grid)
        integrand = (
            np.cos(np.outer(x_grid, nu_grid))
            * matrix.real[None, :]
        )
        expected = -2.0 / np.pi * np.trapezoid(
            integrand, nu_grid, axis=1)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)

    def test_implicit_ioffe_grid_is_rejected(self):
        """缺少物理 ν 节点时不得用任意默认周期猜测 Fourier 变换。"""
        with self.assertRaisesRegex(ValueError, "nu_grid"):
            invariant_amplitude(
                np.ones(4), np.array([0.0]), np.array([0.1]))

    def test_ioffe_grid_must_be_nonnegative_half_axis_from_zero(self):
        """偏移正网格和对称全实轴都不能套用半轴 -2/pi 约定。"""
        for nu_grid in (
                np.array([1.0, 2.0, 3.0]),
                np.array([-1.0, 0.0, 1.0])):
            with self.subTest(nu_grid=nu_grid):
                with self.assertRaisesRegex(ValueError, "非负半轴"):
                    invariant_amplitude(
                        np.ones(3), np.array([0.0]),
                        np.array([0.1]), nu_grid=nu_grid)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        InvariantAmplitudeContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
