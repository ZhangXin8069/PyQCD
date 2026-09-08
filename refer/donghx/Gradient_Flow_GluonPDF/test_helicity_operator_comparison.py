from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "analysis" / "helicity_operator_comparison.py"
SPEC = importlib.util.spec_from_file_location("helicity_operator_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def direct(operator: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    o_mean = operator.mean(axis=0)
    n_mean = numerator.mean(axis=0)
    covariance = (operator[:, :, :, None] * numerator[:, None]).mean(axis=0)
    covariance -= o_mean[:, :, None] * n_mean[None]
    c3 = covariance.mean(axis=1)
    c2 = denominator.mean(axis=(0, 1))
    return c3 / c2[None]


def test_one_estimator_matches_direct_delete_one() -> None:
    rng = np.random.default_rng(9173)
    nconf, nselector, nsource, npmom = 7, 3, 5, 2
    operator = rng.normal(size=(nconf, nselector, nsource)) + 1j * rng.normal(
        size=(nconf, nselector, nsource)
    )
    numerator = rng.normal(size=(nconf, nsource, npmom)) + 1j * rng.normal(
        size=(nconf, nsource, npmom)
    )
    denominator = 2.0 + rng.normal(size=(nconf, nsource, npmom)) + 1j * rng.normal(
        size=(nconf, nsource, npmom)
    )
    central, samples = MODULE.one_estimator(operator, numerator, denominator)
    np.testing.assert_allclose(central, direct(operator, numerator, denominator))
    for omitted in range(nconf):
        keep = np.arange(nconf) != omitted
        np.testing.assert_allclose(
            samples[omitted], direct(operator[keep], numerator[keep], denominator[keep])
        )


def test_selector_coefficients_are_exact() -> None:
    raw = np.asarray([[[1.0], [4.0]], [[2.0], [7.0]]], dtype=np.complex128)
    got = np.einsum("kr,nrs->nks", MODULE.COEFFICIENTS, raw)
    expected = np.asarray([[[-3.0], [5.0], [1.0]], [[-5.0], [9.0], [2.0]]])
    np.testing.assert_array_equal(got, expected)


def test_midpoint_projection_preserves_real_and_imaginary_errors() -> None:
    rng = np.random.default_rng(714)
    nconf, nselector, ntsep, nins, npmom = 9, 3, 2, 5, 2
    samples = rng.normal(size=(nconf, nselector, ntsep, nins, npmom))
    samples = samples + 1j * rng.normal(size=samples.shape)
    central = samples.mean(axis=0)
    tseps = np.asarray([3, 4])
    mid, mid_samples, error_real, error_imag = MODULE.midpoint_projection(
        central, samples, tseps
    )
    expected_samples = np.stack(
        [samples[:, :, 0, [1, 2]].mean(axis=2), samples[:, :, 1, [2, 2]].mean(axis=2)],
        axis=2,
    )
    expected_central = np.stack(
        [central[:, 0, [1, 2]].mean(axis=1), central[:, 1, [2, 2]].mean(axis=1)],
        axis=1,
    )
    np.testing.assert_allclose(mid, expected_central)
    np.testing.assert_allclose(mid_samples, expected_samples)
    mean = expected_samples.mean(axis=0)
    prefactor = (nconf - 1) / nconf
    np.testing.assert_allclose(
        error_real,
        np.sqrt(prefactor * ((expected_samples.real - mean.real) ** 2).sum(axis=0)),
    )
    np.testing.assert_allclose(
        error_imag,
        np.sqrt(prefactor * ((expected_samples.imag - mean.imag) ** 2).sum(axis=0)),
    )
