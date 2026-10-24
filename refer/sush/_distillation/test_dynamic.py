"""Optional per-call weights in the native lqcddb dynamic module.

MPLCONFIGDIR="$PWD/.matplotlib-cache" python -B -m unittest -v test_dynamic
"""

import ast
import inspect
from hashlib import sha256
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from reference_lqcddb import DYNAMIC_SHA256
from numpy.testing import assert_allclose, assert_array_equal

from lqcddb.contraction import dynamic as local
from lqcddb.eigvectors.blending import BlendingWeights
from test_blending_dynamic import VERTICES, dense_reference, make_dynamic, package_mpi


class TestLocalDynamic(unittest.TestCase):
    def setUp(self):
        self.weights = BlendingWeights(Nx=2, exact=4, noise=4)

    def test_unweighted_constructor_and_original_method_bodies_preserved(self):
        source = Path('/public/home/sush/distillation/lqcddb_update/src/lqcddb/contraction/dynamic.py')
        self.assertEqual(sha256(source.read_bytes()).hexdigest(), DYNAMIC_SHA256)
        original = ast.parse(source.read_text())
        copied = ast.parse(Path(local.__file__).read_text())

        def methods(tree):
            cls = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == 'dynamic_contraction')
            return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}

        before, after = methods(original), methods(copied)
        self.assertEqual(set(before), set(after))  # no added class method
        self.assertEqual(ast.dump(before['__init__']), ast.dump(after['__init__']))
        for name in ('calculate', 'calculate_all'):
            # Exclude docstrings and the single new "if weights is not None" branch.
            expected = ast.Module(body=before[name].body[1:], type_ignores=[])
            actual = ast.Module(body=after[name].body[2:], type_ignores=[])
            self.assertEqual(ast.dump(expected), ast.dump(actual))

    def test_default_matches_lqcddb_for_pion_and_baryon(self):
        for options in ({}, {'oindex': 'M'}, {'pion_three_point': True},
                        {'baryon': True, 'projection': True},
                        {'baryon': True, 'projection': True, 'oindex': 'M'}):
            with self.subTest(options=options):
                native = make_dynamic(**options)
                copied = make_dynamic(**options, implementation=local)
                self.assertIs(type(copied), local.dynamic_contraction)
                self.assertIs(type(copied._v_registry), local.VRegistry)
                # The default path must not even construct the weight engine.
                with patch('lqcddb.eigvectors.blending.BlendedDynamicContraction',
                           side_effect=AssertionError('unweighted path entered weight engine')):
                    for i in range(len(native)):
                        assert_array_equal(copied.calculate(i), native.calculate(i))
                    assert_array_equal(copied.calculate_all(), native.calculate_all())

    def test_weighted_parameters_match_full_fourth_order_without_persistent_state(self):
        copied = make_dynamic(oindex='MHG', implementation=local)
        baseline = copied.calculate_all().copy()
        expected = 0
        for i in range(len(copied)):
            direct = dense_reference(copied, i, VERTICES)
            weighted = copied.calculate(i, weights=self.weights, block_size=3)
            assert_allclose(weighted, direct, rtol=2e-12, atol=2e-8)
            expected = expected + direct
        actual = copied.calculate_all(weights=self.weights, block_size=3)
        assert_allclose(actual, expected, rtol=2e-12, atol=2e-8)
        self.assertGreater(np.max(np.abs(actual-baseline)), 1.0)
        assert_array_equal(copied.calculate_all(), baseline)
        assert_array_equal(copied.calculate_all(weights=None), baseline)
        for i in range(len(copied)):
            assert_array_equal(copied.calculate(i), copied.calculate(i, weights=None))

    def test_weighted_path_uses_copied_contraction_and_projector_helpers(self):
        copied = make_dynamic(baryon=True, projection=True, implementation=local)
        selected = [VERTICES[0]]
        expected = sum(dense_reference(copied, i, selected) for i in range(len(copied)))
        # The local calculator must execute local helpers, including projection.
        with patch('_lqcddb_reference.contraction.dynamic.calculate_contraction',
                   side_effect=AssertionError('weighted path called installed calculator')):
            with patch.object(local, 'calculate_contraction', wraps=local.calculate_contraction) as called:
                actual = copied.calculate_all(weights=self.weights, block_size=4)
                self.assertGreater(called.call_count, 0)
        assert_allclose(actual, expected, rtol=2e-12, atol=2e-8)
        projector = copied._gamma_registry.resolve('Projector')
        copied._gamma_registry.register('Projector', {'sink': projector[0], 'source': projector[1],
                                                     'contraction': ['0-0']})
        expected = sum(dense_reference(copied, i, selected) for i in range(len(copied)))
        assert_allclose(copied.calculate_all(weights=self.weights, block_size=4),
                        expected, rtol=2e-12, atol=2e-8)

    def test_automatic_current_selection_and_mpi_distribution(self):
        # One current means Omega2. An equal-size sink is not evidence that it
        # shares the current's physical time or sample, so it stays unweighted.
        copied = make_dynamic(pion_three_point=True, implementation=local)
        selected = [('VDV_0', 'tcur0')]
        expected = sum(dense_reference(copied, i, selected) for i in range(len(copied)))
        assert_allclose(copied.calculate_all(weights=self.weights, block_size=4),
                        expected, rtol=2e-12, atol=2e-8)
        copied = make_dynamic(oindex='M', implementation=local)
        expected = sum(dense_reference(copied, i, VERTICES) for i in range(len(copied)))
        total = 0
        for rank in range(3):
            with package_mpi(rank, 3) as mpi:
                total = total + copied.calculate_all(weights=self.weights, block_size=4)
                mpi.getMPIRank.assert_called_once_with()
                mpi.getMPISize.assert_called_once_with()
        assert_allclose(total, expected, rtol=2e-12, atol=2e-8)

    def test_simple_signatures_and_serial_fallback(self):
        expected = ['weights', 'block_size', 'max_block_bytes']
        for name in ('calculate', 'calculate_all'):
            signature = inspect.signature(getattr(local.dynamic_contraction, name))
            self.assertEqual([n for n, p in signature.parameters.items()
                              if p.kind == inspect.Parameter.KEYWORD_ONLY], expected)
        copied = make_dynamic(implementation=local)
        direct = dense_reference(copied, 0, VERTICES)
        with patch.dict(sys.modules, {'lqcddb.base.mpi_init': None}):
            self.assertIsNone(local._loaded_mpi_module())
            assert_allclose(copied.calculate(0, weights=self.weights, block_size=4),
                            direct, rtol=2e-12, atol=2e-8)

    def test_registry_order_and_unrelated_vertices_do_not_change_selection(self):
        copied = make_dynamic(implementation=local)
        direct = dense_reference(copied, 0, VERTICES)
        # Registries can hold data for other plans; discover only this plan's
        # current operands instead of treating every stored VDV as a current.
        copied._v_registry.register('VDV_99', 'tcur99', np.zeros((2, 99, 99)))
        copied._v_registry._entries = dict(reversed(list(copied._v_registry._entries.items())))
        assert_allclose(copied.calculate(0, weights=self.weights, block_size=4),
                        direct, rtol=2e-12, atol=2e-8)

    def test_selection_is_derived_for_each_diagram_not_cached_globally(self):
        copied = make_dynamic(implementation=local)
        other = make_dynamic(pion_three_point=True, implementation=local)
        copied.plan = [copied.plan[0], other.plan[0]]
        for i, selected in enumerate((VERTICES, [('VDV_0', 'tcur0')])):
            assert_allclose(copied.calculate(i, weights=self.weights, block_size=4),
                            dense_reference(copied, i, selected), rtol=2e-12, atol=2e-8)

    def test_invalid_mpi_context_is_rejected_and_unweighted_calls_ignore_it(self):
        copied = make_dynamic(implementation=local)
        with package_mpi(2, 2):
            with self.assertRaisesRegex(ValueError, 'rank'):
                copied.calculate(0, weights=self.weights)
        baseline = copied.calculate_all().copy()
        with package_mpi(0, 3) as mpi:
            assert_array_equal(copied.calculate_all(), baseline)
            mpi.getMPIRank.assert_not_called()
            mpi.getMPISize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
