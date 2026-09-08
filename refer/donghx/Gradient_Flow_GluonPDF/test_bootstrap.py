from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from construct_bootstrap import (  # noqa: E402
    bootstrap_summed,
    pointwise_central,
    shared_bootstrap_indices,
)


def direct_sum(op, c2, indices, T, cut, channel, projection):
    picked_op = op[indices]
    picked_c2 = c2[:, indices]
    source = np.arange(op.shape[-1])
    op_sum = sum(
        np.take(picked_op, (source + insertion) % op.shape[-1], axis=-1)
        for insertion in range(cut, T - cut + 1)
    )
    own = np.einsum(
        "nzs,nsp->nzp", op_sum, picked_c2[channel, :, 0], optimize=True
    ).mean(axis=0) / op.shape[-1]
    disconnected = np.einsum(
        "zs,sp->zp", op_sum.mean(axis=0), picked_c2[channel, :, 0].mean(axis=0),
        optimize=True,
    ) / op.shape[-1]
    c3 = own - disconnected
    denominator = picked_c2[0, :, 0].mean(axis=(0, 1)).real
    numerator = c3.real if projection == "real" else c3.imag
    return c3, numerator / denominator[None]


def test_bootstrap_recomputes_covariance_and_denominator():
    rng = np.random.default_rng(27)
    nconf, nz, nt, npmom = 7, 2, 6, 2
    op = rng.normal(size=(nconf, 3, nz, nt)) + 1j * rng.normal(
        size=(nconf, 3, nz, nt)
    )
    c2 = 3.0 + rng.normal(size=(2, nconf, 1, nt, npmom))
    c2 = c2 + 1j * rng.normal(size=c2.shape)
    indices, weights = shared_bootstrap_indices(nconf, 9, 1115)
    got = bootstrap_summed(op, c2, weights, np.asarray([5]), np.asarray([1]))
    for boot in range(9):
        for operator, (channel, projection) in enumerate(((0, "real"), (1, "imag"), (1, "imag"))):
            c3, ratio = direct_sum(
                op[:, operator], c2, indices[boot], 5, 1, channel, projection
            )
            np.testing.assert_allclose(got["c3_sum_bootstrap"][boot, operator, 0, :, 0], c3)
            np.testing.assert_allclose(got["ratio_sum_bootstrap"][boot, operator, 0, :, 0], ratio)


def test_pointwise_sum_matches_summed_central():
    rng = np.random.default_rng(35)
    nconf, nz, nt, npmom = 6, 2, 6, 2
    all_op = rng.normal(size=(nconf, 6, nz, nt)) + 1j * rng.normal(
        size=(nconf, 6, nz, nt)
    )
    c2 = 4.0 + rng.normal(size=(2, nconf, 1, nt, npmom))
    c2 = c2 + 1j * rng.normal(size=c2.shape)
    _, weights = shared_bootstrap_indices(nconf, 5, 1115)
    point = pointwise_central(all_op, c2, np.asarray([5]))
    summed = bootstrap_summed(all_op[:, [0, 3, 4]], c2, weights, np.asarray([5]), np.asarray([1]))
    for target, all_index in enumerate((0, 3, 4)):
        expected = point["ratio_original"][all_index, :, 0, 1:5].sum(axis=1)
        np.testing.assert_allclose(summed["ratio_sum_original"][target, 0, :, 0], expected)
