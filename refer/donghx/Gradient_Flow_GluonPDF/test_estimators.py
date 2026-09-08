#!/usr/bin/env python3
"""Independent small-array checks for central and delete-one estimators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calc_ratio_gradient_flow import (  # noqa: E402
    calculate_ratios,
    covariance_ratio_estimator,
)


def direct_ratio(
    operator: np.ndarray,
    correlator: np.ndarray,
    denominator: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if denominator is None:
        denominator = correlator
    mean_o = np.mean(operator, axis=0)
    mean_c = np.mean(correlator, axis=0)
    mean_oc = np.mean(operator[..., None] * correlator[:, None], axis=0)
    c3 = np.mean(mean_oc - mean_o[..., None] * mean_c[None], axis=1)
    c2 = np.mean(np.mean(denominator, axis=0), axis=0)
    return c3, c2, c3 / c2[None]


def test_delete_one() -> None:
    rng = np.random.default_rng(20260903)
    operator = rng.normal(size=(5, 3, 4)) + 1j * rng.normal(size=(5, 3, 4))
    correlator = 2.0 + rng.normal(size=(5, 4, 2)) + 1j * rng.normal(size=(5, 4, 2))
    result = covariance_ratio_estimator(operator, correlator, True)
    c3, c2, jk_mean, jk_bias, err_real, err_imag = result
    direct_c3, direct_c2, direct_central = direct_ratio(operator, correlator)
    np.testing.assert_allclose(c3, direct_c3, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(c2, direct_c2, rtol=2e-14, atol=2e-14)

    samples = []
    for omitted in range(operator.shape[0]):
        keep = np.arange(operator.shape[0]) != omitted
        samples.append(direct_ratio(operator[keep], correlator[keep])[2])
    samples = np.asarray(samples)
    expected_mean = np.mean(samples, axis=0)
    expected_bias = operator.shape[0] * direct_central - (operator.shape[0] - 1) * expected_mean
    prefactor = (operator.shape[0] - 1) / operator.shape[0]
    expected_real = np.sqrt(prefactor * np.sum((samples.real - expected_mean.real) ** 2, axis=0))
    expected_imag = np.sqrt(prefactor * np.sum((samples.imag - expected_mean.imag) ** 2, axis=0))
    np.testing.assert_allclose(jk_mean, expected_mean, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(jk_bias, expected_bias, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(err_real, expected_real, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(err_imag, expected_imag, rtol=2e-14, atol=2e-14)


def test_separate_unpolarized_denominator() -> None:
    rng = np.random.default_rng(220708733)
    operator = rng.normal(size=(6, 4, 5)) + 1j * rng.normal(size=(6, 4, 5))
    projected = rng.normal(size=(6, 5, 2)) + 1j * rng.normal(size=(6, 5, 2))
    denominator = 4.0 + rng.normal(size=(6, 5, 2)) + 1j * rng.normal(size=(6, 5, 2))
    got = covariance_ratio_estimator(
        operator, projected, True, denominator_correlator=denominator
    )
    expected_c3, expected_c2, expected_ratio = direct_ratio(
        operator, projected, denominator
    )
    np.testing.assert_allclose(got[0], expected_c3, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got[1], expected_c2, rtol=2e-14, atol=2e-14)
    samples = []
    for omitted in range(operator.shape[0]):
        keep = np.arange(operator.shape[0]) != omitted
        samples.append(direct_ratio(operator[keep], projected[keep], denominator[keep])[2])
    samples = np.asarray(samples)
    mean = np.mean(samples, axis=0)
    prefactor = (operator.shape[0] - 1) / operator.shape[0]
    np.testing.assert_allclose(got[2], mean, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        got[3], operator.shape[0] * expected_ratio - (operator.shape[0] - 1) * mean,
        rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(
        got[4], np.sqrt(prefactor * np.sum((samples.real - mean.real) ** 2, axis=0)),
        rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(
        got[5], np.sqrt(prefactor * np.sum((samples.imag - mean.imag) ** 2, axis=0)),
        rtol=2e-14, atol=2e-14,
    )


def test_axis_and_linear_identities() -> None:
    rng = np.random.default_rng(41)
    nconf, nt, nz = 4, 6, 2
    # Input component order is combined, Mtiti, Mijij.
    raw = rng.normal(size=(nconf, 2, 2, 2, nz, nt)) + 1j * rng.normal(
        size=(nconf, 2, 2, 2, nz, nt)
    )
    ope = np.empty((nconf, 2, 4, 3, nz, nt), dtype=np.complex128)
    for channel in range(2):
        for orientation in range(2):
            mt = raw[:, channel, orientation, 0]
            mi = raw[:, channel, orientation, 1]
            ope[:, channel, orientation, 1] = mt
            ope[:, channel, orientation, 2] = mi
            ope[:, channel, orientation, 0] = mt - mi if channel == 0 else mt + mi
        ope[:, channel, 2] = ope[:, channel, 0] + ope[:, channel, 1]
        ope[:, channel, 3] = ope[:, channel, 0] - ope[:, channel, 1]
    twopt = 3.0 + rng.normal(size=(2, 1, nconf, 2, nt, 1)) + 1j * rng.normal(
        size=(2, 1, nconf, 2, nt, 1)
    )
    out = calculate_ratios(ope, twopt, [1, 2], do_jackknife=False)
    ratio = out["ratio"]
    assert ratio.shape == (2, 4, 3, 1, nz, 2, 3, 1)
    np.testing.assert_allclose(ratio[:, 2], ratio[:, 0] + ratio[:, 1], equal_nan=True)
    np.testing.assert_allclose(ratio[:, 3], ratio[:, 0] - ratio[:, 1], equal_nan=True)
    np.testing.assert_allclose(ratio[0, :, 0], ratio[0, :, 1] - ratio[0, :, 2], equal_nan=True)
    np.testing.assert_allclose(ratio[1, :, 0], ratio[1, :, 1] + ratio[1, :, 2], equal_nan=True)


def main() -> None:
    test_delete_one()
    test_separate_unpolarized_denominator()
    test_axis_and_linear_identities()
    print("ESTIMATOR_TESTS_OK")


if __name__ == "__main__":
    main()
