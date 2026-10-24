"""Exact output comparisons against the untouched lqcddb implementation."""

from copy import deepcopy
import unittest

import numpy as np

from lqcddb import set_backend
from reference_lqcddb import reference
Reference = reference.vertex_creator
from omega_optimized import create_omega_accelerate, vertex_creator as Optimized
from lqcddb.eigvectors import blending as implementation


MODELS = (
    dict(Nx=2, exact=4),
    dict(Nx=2, exact=4, noise=4),
    dict(Nx=3, exact=4, N_eigen=[12], N_sum=[8], N_extract=[4]),
    dict(Nx=3, exact=4, N_eigen=[16, 12], N_sum=[8, 4], N_extract=[4, 4], noise=4),
    dict(Nx=4, exact=8, N_eigen=[24, 16], N_sum=[8, 8], N_extract=[4, 4], noise=8),
)

# Optional audit sink, enabled only by validate_omega_optimized.py.
COMPARISON_RECORDS = None


def compare_exact(test, model, **options):
    params = deepcopy(model)
    Nx = params.pop('Nx')
    params.update(options)
    reference, optimized = Reference(Nx=Nx), Optimized(Nx=Nx)
    before = deepcopy(params)
    expected = reference.create_omega_accelerate(**params)
    actual = optimized.create_omega_accelerate(**params)
    test.assertEqual(expected.dtype, np.dtype('complex128'))
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    test.assertTrue(np.isfinite(expected).all())
    test.assertTrue(np.isfinite(actual).all())
    # Compare every output bit, including real/imaginary values and zero signs.
    test.assertTrue(actual.tobytes() == expected.tobytes(),
                    f'Outputs differ byte for byte; max absolute error '
                    f'{np.max(np.abs(actual-expected))}')
    test.assertEqual(params, before)
    if COMPARISON_RECORDS is not None:
        COMPARISON_RECORDS.append(dict(Nx=Nx, parameters=deepcopy(params),
                                       shape=actual.shape, elements=actual.size,
                                       dtype=str(actual.dtype), bitwise_equal=True))
    return actual, reference, params


class TestOptimizedOmega(unittest.TestCase):
    def setUp(self):
        set_backend('numpy')

    def test_all_orders_and_models_exactly(self):
        for model in MODELS:
            for dim in range(1, 5):
                with self.subTest(model=model, dim=dim):
                    compare_exact(self, model, dim=dim)

    def test_normalization_and_conserved_modes_exactly(self):
        for model in MODELS:
            for normal in (False, True):
                for conserved in (False, True):
                    with self.subTest(model=model, normal=normal, conserved=conserved):
                        compare_exact(self, model, dim=2, normal=normal, conserved=conserved)
            for requested_dim in (1, 3, 4, 7):
                with self.subTest(model=model, forced_dim=requested_dim):
                    compare_exact(self, model, dim=requested_dim, conserved=True)

    def test_fixed_singletons_ranges_and_cached_group_plans_exactly(self):
        model = MODELS[3]
        # Five subspaces, each retained size four: exact, three compressed, noise.
        selectors = [[start] for start in range(0, 20, 4)]
        selectors += [[start+3] for start in range(0, 20, 4)]
        selectors += [[start+1, start+2] for start in range(0, 20, 4)]
        selectors += [list(range(start, start+4)) for start in range(0, 20, 4)]
        for fixed in selectors:
            for dim in range(1, 5):
                with self.subTest(fixed=fixed, dim=dim):
                    compare_exact(self, model, dim=dim, fixed_first_pos=fixed)
        for fixed in selectors:
            for conserved, normal in ((False, True), (True, False), (True, True)):
                with self.subTest(fixed=fixed, conserved=conserved, normal=normal):
                    compare_exact(self, model, dim=2, fixed_first_pos=fixed,
                                  conserved=conserved, normal=normal)

    def test_randomized_valid_sampling_models_exactly(self):
        rng = np.random.default_rng(310729)
        for seed in range(8):
            population, retained, extracted = [], [], []
            for _ in range(int(rng.integers(0, 3))):
                split = int(rng.integers(1, 3))
                sample = int(rng.integers(4, 6))
                space = sample + int(rng.integers(0, 12))
                population.append(split*space)
                retained.append(split*sample)
                extracted.append(sample)
            model = dict(Nx=4, exact=int(rng.integers(4, 7)),
                         N_eigen=population, N_sum=retained, N_extract=extracted,
                         noise=int(rng.integers(4, 7)))
            for dim in range(1, 5):
                with self.subTest(seed=seed, model=model, dim=dim):
                    compare_exact(self, model, dim=dim)
            compare_exact(self, model, dim=2, normal=True)
            compare_exact(self, model, dim=2, normal=True, conserved=True)

    def test_reference_sized_second_order_and_fourth_order_slice_exactly(self):
        model = dict(Nx=24, exact=50, N_eigen=[100, 200, 300, 500, 800],
                     N_sum=[20]*5, N_extract=[10]*5, noise=500)
        for normal in (False, True):
            compare_exact(self, model, dim=2, normal=normal)
        compare_exact(self, model, dim=3, fixed_first_pos=[60])
        compare_exact(self, dict(model, noise=0), dim=4, fixed_first_pos=[60])

    def test_cache_keys_and_return_arrays_do_not_share_mutable_state(self):
        model = deepcopy(MODELS[3])
        result, reference, arguments = compare_exact(self, model, dim=2)
        original = result.copy()
        result[...] = -999
        repeated = create_omega_accelerate(reference, **arguments)
        self.assertEqual(repeated.tobytes(), original.tobytes())
        model['N_eigen'][0] += 4
        changed, _, _ = compare_exact(self, model, dim=2)
        self.assertNotEqual(changed.tobytes(), original.tobytes())
        compare_exact(self, dict(model, Nx=4), dim=2)
        for model in MODELS:
            compare_exact(self, model, dim=4)
        self.assertLessEqual(implementation._metadata.cache_info().currsize, 32)
        self.assertLessEqual(implementation._cached_cells.cache_info().currsize, 16)

    def test_standalone_function_with_unmodified_library_creator(self):
        model = deepcopy(MODELS[3])
        ref = Reference(Nx=model.pop('Nx'))
        expected = ref.create_omega_accelerate(dim=4, **model)
        actual = create_omega_accelerate(ref, dim=4, **model)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_large_group_plan_streams_without_unbounded_cache(self):
        # Ten groups -> 10**4 cells, exceeding the 4096-cell cache threshold.
        model = dict(Nx=4, exact=4, N_eigen=[40], N_sum=[32], N_extract=[4], noise=4)
        before = implementation._cached_cells.cache_info().currsize
        compare_exact(self, model, dim=4)
        self.assertEqual(implementation._cached_cells.cache_info().currsize, before)

    def test_legacy_modes_and_irregular_fixed_selectors(self):
        # Retain original results/errors for its legacy, nongeneral selector API.
        model = dict(MODELS[3])
        Nx = model.pop('Nx')
        for options in (dict(dim=4, fixed_first_pos=[3, 4]),
                        dict(dim=4, fixed_first_pos=[0, 2]),
                        dict(dim=2, fixed_first_pos=[20]),
                        dict(dim=1, normal=True),
                        dict(dim=3, normal=True),
                        dict(dim=4, normal=True)):
            with self.subTest(options=options):
                reference, optimized = Reference(Nx=Nx), Optimized(Nx=Nx)
                try:
                    expected = reference.create_omega_accelerate(**model, **options)
                except Exception as error:
                    with self.assertRaises(type(error)):
                        optimized.create_omega_accelerate(**model, **options)
                else:
                    actual = optimized.create_omega_accelerate(**model, **options)
                    self.assertEqual(actual.tobytes(), expected.tobytes())
        for cls in (Reference, Optimized):
            with self.assertRaises(ValueError):
                cls(Nx=2).create_omega_accelerate(exact=0, noise=4)
            with self.assertRaises(ZeroDivisionError):
                cls(Nx=2).create_omega_accelerate(exact=1, noise=4, dim=4)
        compare_exact(self, dict(Nx=2, exact=1, noise=1), conserved=True, normal=True)


if __name__ == '__main__':
    unittest.main()
