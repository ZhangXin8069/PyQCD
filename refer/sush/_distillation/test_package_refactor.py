"""Independent compatibility, block-allocation and package-routing checks."""

import importlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from lqcddb import set_backend
from reference_lqcddb import reference
Reference = reference.vertex_creator
import lqcddb as local
from lqcddb.eigvectors import vertex as managed
from lqcddb.eigvectors import blending as engine
from test_blending_dynamic import make_dynamic, dense_reference, VERTICES


class TestPackageRefactor(unittest.TestCase):
    def setUp(self):
        set_backend('numpy')

    def test_public_imports_use_native_package_modules(self):
        from omega_optimized import create_omega_accelerate, vertex_creator
        self.assertIs(local.vertex_creator, vertex_creator)
        self.assertIs(local.create_omega_accelerate, create_omega_accelerate)
        self.assertIs(local.create_omega_accelerate, managed.create_omega_accelerate)
        self.assertIs(local.vertex_creator.create_omega_accelerate, managed.create_omega_accelerate)
        self.assertIs(local.BlendingWeights, engine.BlendingWeights)
        self.assertTrue(local.vertex_creator.__module__.startswith('lqcddb.'))
        self.assertNotEqual(local.vertex_creator, Reference)
        self.assertNotEqual(local.get_backend, reference.get_backend)

    def test_public_function_and_bound_method_signatures_unchanged(self):
        self.assertEqual(inspect.signature(Reference.create_omega_accelerate),
                         inspect.signature(managed.create_omega_accelerate))
        self.assertEqual(inspect.signature(Reference.create_omega_accelerate),
                         inspect.signature(local.vertex_creator.create_omega_accelerate))

    def test_blocks_are_byte_equal_to_original_full_tensor_slices(self):
        cases = [dict(Nx=2, exact=4, noise=4),
                 dict(Nx=3, exact=4, N_eigen=[16, 12], N_sum=[8, 4], N_extract=[4, 4], noise=4),
                 # A huge population with a small retained sample also tests
                 # legacy float rounding, without a huge output allocation.
                 dict(Nx=200000, exact=4, noise=4)]
        for parameters in cases:
            weights = local.BlendingWeights(**parameters)
            kwargs = {k: v for k, v in parameters.items() if k != 'Nx'}
            creator = Reference(Nx=parameters['Nx'])
            n = weights.nvec
            for dim in range(1, 5):
                full = creator.create_omega_accelerate(dim=dim, **kwargs)
                patterns = [(slice(None),)*dim,
                            ([n-1, 0, n//2, n//2],)*dim,
                            (n-1,)*dim,
                            tuple((3*a+2) % n for a in range(dim)),
                            (slice(None, None, -2),)*dim,
                            tuple(slice(a % 3, min(a % 3+5, n)) for a in range(dim)),
                            ([],)+(slice(None),)*(dim-1)]
                for indices in patterns:
                    with self.subTest(parameters=parameters, dim=dim, indices=indices):
                        axes = [np.atleast_1d(np.arange(n)[s]) for s in indices]
                        expected = full[np.ix_(*axes)]
                        actual = weights.block(dim=dim, indices=indices, dtype=np.complex128)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.tobytes(), expected.tobytes())
                        real = weights.block(dim=dim, indices=indices)
                        self.assertEqual(real.tobytes(), expected.real.tobytes())

    def test_disjoint_blocks_reconstruct_original_byte_for_byte(self):
        parameters = dict(Nx=3, exact=4, N_eigen=[16, 12], N_sum=[8, 4], N_extract=[4, 4], noise=4)
        weights = local.BlendingWeights(**parameters)
        full = Reference(Nx=3).create_omega_accelerate(
            dim=4, **{k: v for k, v in parameters.items() if k != 'Nx'})
        reconstructed = np.empty_like(full)
        visits = np.zeros(full.shape, dtype=np.uint8)
        for selection, values in weights.iter_blocks(dim=4, block_size=(3, 5, 4, 6),
                                                     dtype=np.complex128):
            reconstructed[selection] = values
            visits[selection] += 1
        self.assertTrue(np.all(visits == 1))
        self.assertEqual(reconstructed.tobytes(), full.tobytes())

    def test_every_block_calls_the_managed_public_generator(self):
        weights = local.BlendingWeights(Nx=2, exact=4, noise=4)
        with patch.object(managed, 'create_omega_accelerate',
                          wraps=managed.create_omega_accelerate) as called:
            count = 0
            for _, _ in weights.iter_blocks(dim=4, block_size=4):
                count += 1
            self.assertEqual(called.call_count, count)
            self.assertTrue(all(isinstance(call.args[0], engine._OmegaBlockRequest)
                                for call in called.call_args_list))

    def test_dynamic_calls_local_functions_and_keeps_original_result(self):
        module = importlib.import_module('lqcddb.contraction.dynamic')
        dc = make_dynamic(implementation=module)
        weights = local.BlendingWeights(Nx=2, exact=4, noise=4)
        expected = sum(dense_reference(dc, i, VERTICES) for i in range(len(dc)))
        original = dc.calculate_all().copy()
        with patch.object(Reference, 'create_omega_accelerate',
                          side_effect=AssertionError('installed generator used by local block path')):
            with patch('_lqcddb_reference.contraction.dynamic.calculate_contraction',
                       side_effect=AssertionError('installed calculator used')):
                with patch.object(managed, 'create_omega_accelerate',
                                  wraps=managed.create_omega_accelerate) as called:
                    actual = dc.calculate_all(weights=weights, block_size=4)
                    self.assertGreater(called.call_count, 0)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-8)
        self.assertEqual(dc.calculate_all().tobytes(), original.tobytes())

    def test_only_selected_four_dimensional_output_is_allocated(self):
        weights = local.BlendingWeights(Nx=24, exact=50,
                                       N_eigen=[100, 200, 300, 500, 800],
                                       N_sum=[20]*5, N_extract=[10]*5, noise=500)
        selected = (slice(120, 152), slice(140, 172), slice(149, 181), slice(150, 182))
        allocations = []
        native_empty = np.empty

        def tracked(shape, *args, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 4:
                allocations.append(shape)
            return native_empty(shape, *args, **kwargs)

        with patch.object(engine.np, 'empty', side_effect=tracked):
            block = weights.block(dim=4, indices=selected)
        self.assertEqual(block.shape, (32,)*4)
        self.assertEqual(allocations, [(32,)*4])

    def test_dynamic_validates_vector_lengths_before_entering_the_hot_loop(self):
        module = importlib.import_module('lqcddb.contraction.dynamic')
        dc = make_dynamic(implementation=module)
        weights = local.BlendingWeights(Nx=2, exact=4)
        with patch.object(managed, 'create_omega_accelerate',
                          wraps=managed.create_omega_accelerate) as called:
            with self.assertRaisesRegex(ValueError, 'sample size'):
                dc.calculate_all(weights=weights, block_size=4)
            self.assertEqual(called.call_count, 0)


if __name__ == '__main__':
    unittest.main()
