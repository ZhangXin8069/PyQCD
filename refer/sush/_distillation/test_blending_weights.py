"""Small independent checks; run: python -m unittest -v test_blending_weights."""

import ast
from itertools import combinations, product
from math import prod
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from lqcddb.eigvectors.blending import BlendingWeights, create_omega_block


def scalar_reference(weights, coordinates):
    """Independent scalar formula: count distinct indices, then falling factorials."""
    result = 1.0
    for selection, population, sample in zip(
            weights.part_slices, weights.space_sizes, weights.sample_sizes):
        distinct = len({i for i in coordinates if selection.start <= i < selection.stop})
        result *= prod(population - r for r in range(distinct))
        result /= prod(sample - r for r in range(distinct))
    return result


class TestBlendingWeights(unittest.TestCase):
    def setUp(self):
        # Includes splitting one compressed group into two independent parts.
        self.weights = BlendingWeights(Nx=2, exact=2, N_eigen=(8,),
                                       N_sum=(4,), N_extract=(2,), noise=3)

    def test_orders_against_independent_scalar_formula(self):
        for dim in range(1, 5):
            with self.subTest(dim=dim):
                actual = self.weights.block(dim=dim)
                expected = np.empty_like(actual)
                for coordinate in np.ndindex(actual.shape):
                    expected[coordinate] = scalar_reference(self.weights, coordinate)
                assert_allclose(actual, expected, rtol=2e-15)

    def test_all_fourth_order_equality_patterns(self):
        weights = BlendingWeights(Nx=2, exact=4, noise=4)
        # One noise subspace: 20 population vectors, 4 retained vectors.
        for coordinate, d in [((4, 4, 4, 4), 1), ((4, 4, 4, 5), 2),
                              ((4, 4, 5, 5), 2), ((4, 4, 5, 6), 3),
                              ((4, 5, 6, 7), 4)]:
            expected = prod((20-r)/(4-r) for r in range(d))
            actual = weights.block(dim=4, indices=coordinate).item()
            self.assertAlmostEqual(actual, expected)
        # Simultaneous repeated pairs in DIFFERENT stochastic subspaces.
        self.assertAlmostEqual(self.weights.block(dim=4, indices=(2, 2, 6, 6)).item(),
                               (4/2) * (14/3))

    def test_arbitrary_global_selections_and_axis_permutations(self):
        selectors = ([8, 0, 3, 3], slice(None, None, -2), [2, 6, 1], -1)
        full = self.weights.block(dim=4)
        arrays = [np.atleast_1d(np.arange(self.weights.nvec)[s]) for s in selectors]
        actual = self.weights.block(dim=4, indices=selectors)
        assert_allclose(actual, full[np.ix_(*arrays)])
        permuted = (selectors[2], selectors[0], selectors[3], selectors[1])
        assert_allclose(self.weights.block(dim=4, indices=permuted),
                        actual.transpose(2, 0, 3, 1))
        self.assertEqual(self.weights.block(dim=3, indices=([], [1, 2], 3)).shape,
                         (0, 2, 1))

    def test_independent_blocks_and_distributed_coverage(self):
        shape = (self.weights.nvec,) * 4
        result = np.zeros(shape)
        visits = np.zeros(shape, dtype=int)
        for rank in range(5):
            for selectors, weight in self.weights.iter_blocks(
                    dim=4, block_size=(2, 3, 4, 5), rank=rank, size=5):
                visits[selectors] += 1
                result[selectors] = weight
                self.assertLessEqual(weight.size, 2*3*4*5)
        assert_array_equal(visits, np.ones(shape))
        assert_allclose(result, self.weights.block(dim=4))
        restricted = np.zeros((3, 2))
        for (a, b), weight in self.weights.iter_blocks(
                dim=2, block_size=2, axis_slices=(slice(2, 5), slice(6, 8))):
            restricted[a.start-2:a.stop-2, b.start-6:b.stop-6] = weight
        assert_allclose(restricted, self.weights.block(dim=2, indices=(slice(2, 5), slice(6, 8))))

    def test_equal_time_complex_contraction_with_reference_axis_order(self):
        rng = np.random.default_rng(8261)
        n, spin, nsrc, nmom, ngamma = self.weights.nvec, 2, 2, 2, 2

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        u = random((spin, spin, n, nsrc))
        sep = random((spin, spin, n, n))
        down = random((spin, spin, nsrc, n))
        source = random((nsrc, nsrc))
        vdv = random((nmom, n, n))
        sink = random((nmom, n, n))
        gamma5 = random((spin, spin))
        gamma0 = random((nmom, ngamma, spin, spin))
        gamma1 = random((nmom, ngamma, spin, spin))
        from opt_einsum import contract
        equation = 'fnph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG'
        direct = contract(equation, self.weights.block(dim=4), u, sep, down,
                          source, vdv, sink, gamma5, gamma0, gamma1, optimize='greedy')
        blocked = np.zeros_like(direct)
        for (f, n, p, h), weight in self.weights.iter_blocks(dim=4, block_size=4):
            blocked += contract(equation, weight, u[:, :, n, :], sep[:, :, h, f],
                                down[:, :, :, p], source, vdv[:, f, n], sink[:, p, h],
                                gamma5, gamma0, gamma1, optimize='greedy')
        assert_allclose(blocked, direct, rtol=3e-13, atol=3e-10)
        # In general separate second-order current weights miss shared-index terms.
        omega2 = self.weights.block(dim=2)
        separate = np.einsum('fn,ph->fnph', omega2, omega2)
        self.assertGreater(np.max(np.abs(separate-self.weights.block(dim=4))), 1.0)

    def test_sampling_interpretation_by_exact_enumeration(self):
        # Two independent simple-random samples without replacement.  Averaging
        # weighted observed tensor entries must reproduce the population sum.
        weights = BlendingWeights(Nx=1, Nc=11, exact=0, N_eigen=(5,),
                                  N_sum=(4,), N_extract=(4,), noise=4)
        population_tensor = np.random.default_rng(381).normal(size=(11,)*4)
        total = 0.0
        nsamples = 0
        omega = weights.block(dim=4)
        for first, second in product(combinations(range(5), 4), combinations(range(5, 11), 4)):
            observed = first + second
            total += np.sum(population_tensor[np.ix_(*([observed]*4))] * omega)
            nsamples += 1
        self.assertAlmostEqual(total/nsamples, population_tensor.sum(), places=11)

    def test_small_subspaces_and_exact_only(self):
        # Legacy code forms unused 0/0 factors here.  The block formula does not.
        with np.errstate(all='raise'):
            weights = BlendingWeights(Nx=1, exact=1, noise=1)
            self.assertEqual(weights.block(dim=4, indices=(1, 1, 1, 1)).item(), 2)
            assert_array_equal(BlendingWeights(Nx=1, exact=3).block(dim=4),
                               np.ones((3,)*4))
            full = BlendingWeights(Nx=1, Nc=6, exact=0, noise=6)
            assert_array_equal(full.block(dim=4), np.ones((6,)*4))

    def test_legacy_current_and_old_reference_implementations(self):
        # Execute only the real legacy method definition, without importing its
        # unrelated MPI/GPU modules or running the production contraction script.
        root = Path(__file__).resolve().parent.parent
        sources = [root / 'lqcddb_update/src/lqcddb/eigvectors/vertex.py',
                   root / 'lqcddb/src/lqcddb/eigvectors/vertex.py',
                   root / 'function_contraction/corr_eigvecs.py']
        if not all(path.exists() for path in sources):
            self.skipTest('legacy source trees are not present beside this workspace')
        parameters = dict(exact=4, N_eigen=[16, 12], N_sum=[8, 4],
                          N_extract=[4, 4], noise=4)
        weights = BlendingWeights(Nx=3, **parameters)
        for path in sources:
            tree = ast.parse(path.read_text())
            method = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.FunctionDef) and node.name == 'create_omega_accelerate')
            module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
            namespace = dict(np=np, get_backend=lambda: np, Nc=3, combinations=combinations)
            exec(compile(module, str(path), 'exec'), namespace)
            legacy = namespace['create_omega_accelerate']
            for dim in range(1, 5):
                with self.subTest(source=path.parent.name, dim=dim):
                    expected = legacy(SimpleNamespace(Nx=3, Nc=3), dim=dim, **parameters)
                    assert_allclose(weights.block(dim=dim), expected, rtol=3e-15)

    def test_validation_and_memory_guard(self):
        for invalid in [dict(Nx=0), dict(Nx=1, exact=4),
                        dict(Nx=1, exact=2, noise=2),
                        dict(Nx=3, N_eigen=(5,), N_sum=(4,), N_extract=(2,)),
                        dict(Nx=3, N_eigen=(8,), N_sum=(4,), N_extract=(3,)),
                        dict(Nx=3, N_eigen=(8,), N_sum=(0,), N_extract=(0,))]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BlendingWeights(**invalid)
        with self.assertRaises(MemoryError):
            self.weights.block(dim=4, max_block_bytes=10)
        with self.assertRaises(ValueError):
            self.weights.block(dim=5)
        with self.assertRaises(IndexError):
            self.weights.block(indices=([100], [1]))
        with self.assertRaises(TypeError):
            self.weights.block(indices=([1.2], [1]))
        with self.assertRaises(ValueError):
            list(self.weights.iter_blocks(rank=2, size=2))
        with self.assertRaises(ValueError):
            list(self.weights.iter_blocks(block_size=0))
        assert_allclose(create_omega_block(Nx=2, exact=2, N_eigen=(8,), N_sum=(4,),
                                          N_extract=(2,), noise=3, dim=3,
                                          indices=(slice(1, 4), [2, 7], 8)),
                        self.weights.block(dim=3, indices=(slice(1, 4), [2, 7], 8)))


if __name__ == '__main__':
    unittest.main()
