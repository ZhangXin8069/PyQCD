"""Test the public weighted calculator in the sibling lqcddb_new package.

Run: python -B test_blending_dynamic.py
For a combined suite, put ../lqcddb_new/src on PYTHONPATH before importing tests.
"""

from copy import deepcopy
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from opt_einsum import contract

# Standalone runs must test the requested source, not the installed old copy.
_PACKAGE_SOURCE = Path(__file__).resolve().parent.parent / 'lqcddb_new' / 'src'
if 'lqcddb' not in sys.modules:
    sys.path.insert(0, str(_PACKAGE_SOURCE))
import lqcddb
if Path(lqcddb.__file__).resolve().parent != _PACKAGE_SOURCE / 'lqcddb':
    raise ImportError('Tests require lqcddb_new; put its src directory on PYTHONPATH '
                      'before importing lqcddb or this test module')

from lqcddb import BlendingWeights, set_backend
from lqcddb.contraction import dynamic as current
from reference_lqcddb import reference

# Shared reference helpers retain their old defaults for the other audit files.
# Every test below explicitly builds the current calculator instead.
PeramRegistry, VRegistry, GammaRegistry = (reference.PeramRegistry, reference.VRegistry,
                                         reference.GammaRegistry)
dynamic_contraction = reference.dynamic_contraction
run_wick_analysis, vertex_creator, gamma = (reference.run_wick_analysis,
                                           reference.vertex_creator, reference.gamma)
_per_region_v_names = import_module('_lqcddb_reference.contraction.dynamic')._per_region_v_names


VERTICES = [('VDV_0', 'tcur0'), ('VDV_0', 'tcur1')]


@contextmanager
def package_mpi(rank, size):
    """Test the native getter route without initializing an MPI runtime."""
    marker = SimpleNamespace(getMPIRank=Mock(return_value=rank),
                             getMPISize=Mock(return_value=size))
    with patch.dict(sys.modules, {'lqcddb.base.mpi_init': marker}):
        yield marker


def make_dynamic(*, baryon=False, projection=False, oindex=None, ncurr=6,
                 pion_three_point=False, implementation=None):
    """Build a fixed-seed fixture; None selects the untouched reference package."""
    set_backend('numpy')
    if baryon:
        sink = ['|', 'u', 'u', 'gamma_7', 'd', '|']
        source = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|']
    else:
        sink = ['|', 'd^d', 'gamma_5', 'u', '|']
        source = ['|', 'u^d', 'gamma_5', 'd', '|']
    current = ['|', 'u^d', 'gamma_curr', 'u', '|']
    three_point = baryon or pion_three_point
    groups = [(sink, source, current if three_point else current+current)]
    options = dict(Cpt='3pt' if three_point else '4pt', Vindex=['M']*(3 if three_point else 4),
                   Gindex=['', 'G', ''] if three_point else ['', 'G', 'H', ''],
                   use_equivalence=False, ignore_dis=False, verbose=False,
                   Projection=projection, Oindex=oindex, optimize='greedy')
    # Derive required registrations from an ACTUAL Wick plan first.
    discovery = dict(options)
    discovery['Projection'] = False
    analyze = run_wick_analysis if implementation is None else implementation.run_wick_analysis
    construct = dynamic_contraction if implementation is None else implementation.dynamic_contraction
    registries = ((PeramRegistry, VRegistry, GammaRegistry) if implementation is None else
                  (implementation.PeramRegistry, implementation.VRegistry, implementation.GammaRegistry))
    plan = analyze(groups, **discovery)
    pr, vr, gr = (registry() for registry in registries)
    rng = np.random.default_rng(82131)

    def random(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)

    def nev(time):
        return ncurr if time.startswith('tcur') or (pion_three_point and time == 'tsink') else 2

    registered_p, registered_v = set(), set()
    for entry in plan:
        wick, diagram = entry[2], entry[3]
        for p in wick['peram'][diagram]:
            key = (p[2][0], tuple(p[4]))
            if key not in registered_p:
                pr.register(*key, random((4, 4, nev(p[4][0]), nev(p[4][1]))))
                registered_p.add(key)
        for v, key in zip(wick['V'], _per_region_v_names(wick['V'])):
            if key not in registered_v:
                legs = 3 if key[0].startswith('VVV') else 2
                vr.register(*key, random((2,)+(nev(key[1]),)*legs))
                registered_v.add(key)
    gr.register('gamma_5', np.asarray(gamma(5)))
    gr.register('gamma_7', np.asarray(gamma(7)))
    gr.register('gamma_curr', np.asarray([gamma(1), gamma(4)]))
    if projection:
        gr.register('Projector', [random((4, 3)), random((4, 3))])
    return construct(groups, peram_registry=pr, v_registry=vr,
                     gamma_registry=gr, **options)


def dense_reference(dc, index, vertices, *, mappings=None, exact_only=False):
    """Independent full-weight einsum, then project its complete spin output."""
    entry = dc[index]
    wick, diagram = entry[2], entry[3]
    equation = wick['result_indx'][diagram][0]
    lhs, rhs = equation.split('->')
    spin = ''.join(c for c in rhs if c.islower())
    if len(entry) > 5:
        rhs = spin + entry[5]
    pvars = [dc._peram_registry.resolve(p[2], p[4]) for p in wick['peram'][diagram]]
    gvars = [dc._gamma_registry.resolve(g[1]) for g in wick['gamma_pos']]
    vkeys = _per_region_v_names(wick['V'])
    vvars = [dc._v_registry.resolve(*key) for key in vkeys]
    labels, axes = '', []
    for key in vertices:
        position = vkeys.index(key)
        labels += wick['V'][position][2][-2:]
        selectors = None if mappings is None else mappings.get(key)
        if selectors is None:
            selectors = tuple(slice(0, length) for length in vvars[position].shape[-2:])
        axes.extend(np.atleast_1d(np.arange(8)[s]) for s in selectors)
    creator = vertex_creator(Nx=2)
    omega = creator.create_omega_accelerate(exact=8 if exact_only else 4,
                                          noise=0 if exact_only else 4, dim=len(axes))
    selected_weight = omega[np.ix_(*axes)]
    coeff = sum(c for _, _, c in entry[0])
    output = coeff * contract(lhs+','+labels+'->'+rhs,
                              *pvars, *gvars, *vvars, selected_weight, optimize='greedy')
    if dc._projection is not False:
        projector = dc._gamma_registry.resolve('Projector')
        if isinstance(projector, dict):
            # Independently identify sink/source free spin ports from quark
            # positions rather than using lqcddb's projection/einsum builder.
            roles = {}
            for (lo, hi), kind, _, time in wick['V']:
                if kind.startswith('VVV') and time in ('tsink', 'tsrc'):
                    roles[time] = next(ports[0] for pos, _, ports in wick['quark_pos']
                                       if lo < pos < hi and ports[0] in spin)
            sink_axis, source_axis = spin.index(roles['tsink']), spin.index(roles['tsrc'])
            ordered = np.moveaxis(output, (sink_axis, source_axis), (0, 1))
            if projector.get('contraction'):
                return np.einsum('ab...,ax,bx->...', ordered,
                                 projector['sink'], projector['source'])
            return np.einsum('ab...,ax,by->xy...', ordered,
                             projector['sink'], projector['source'])
        if len(entry) > 6 and entry[6]:
            return np.einsum('ab...,ax,bx->...', output, *projector)
        return np.einsum('ab...,ax,by->xy...', output, *projector)
    return output


class TestDynamicAdapter(unittest.TestCase):
    def setUp(self):
        self.weights = BlendingWeights(Nx=2, exact=4, noise=4)

    def make_current(self, **options):
        dc = make_dynamic(implementation=current, **options)
        self.assertIs(type(dc), current.dynamic_contraction)
        self.assertIs(type(dc._v_registry), current.VRegistry)
        return dc

    def test_real_four_point_plan_all_diagrams_and_registry_updates(self):
        dc = self.make_current(oindex='MHG')
        plan_before = deepcopy(dc.plan)
        arrays_before = {key: value.copy() for key, value in dc._v_registry._entries.items()}
        options = dict(weights=self.weights, block_size=3)
        expected = 0
        for i in range(len(dc)):
            direct = dense_reference(dc, i, VERTICES)
            assert_allclose(dc.calculate(i, **options), direct, rtol=2e-12, atol=2e-8)
            expected = expected + direct
        assert_allclose(dc.calculate_all(**options), expected, rtol=2e-12, atol=2e-8)
        self.assertEqual(dc.plan, plan_before)
        for key, value in arrays_before.items():
            assert_array_equal(dc._v_registry.resolve(*key), value)
        key = VERTICES[0]
        dc._v_registry.register(*key, 2 * dc._v_registry.resolve(*key))
        assert_allclose(dc.calculate(0, **options), dense_reference(dc, 0, VERTICES),
                        rtol=2e-12, atol=2e-8)

    def test_package_mpi_distribution_and_empty_rank(self):
        dc = self.make_current(oindex='M')
        expected = dense_reference(dc, 0, VERTICES)
        total = 0
        for rank in range(3):
            with package_mpi(rank, 3) as mpi:
                total = total + dc.calculate(0, weights=self.weights, block_size=4)
                mpi.getMPIRank.assert_called_once_with()
                mpi.getMPISize.assert_called_once_with()
        assert_allclose(total, expected, rtol=2e-12, atol=2e-8)
        with package_mpi(1, 2):
            empty = dc.calculate(0, weights=self.weights, block_size=20)
        assert_array_equal(empty, np.zeros_like(expected))

    def test_three_point_selects_current_without_weighting_sink(self):
        dc = self.make_current(pion_three_point=True)
        # The equal-size sink is not an automatically selected current.
        selected = [('VDV_0', 'tcur0')]
        for i in range(len(dc)):
            assert_allclose(dc.calculate(i, weights=self.weights, block_size=4),
                            dense_reference(dc, i, selected),
                            rtol=2e-12, atol=2e-8)

    def test_baryon_projection_paths_and_second_order_weight(self):
        selected = [VERTICES[0]]
        for oindex in (None, 'M'):
            with self.subTest(oindex=oindex):
                dc = self.make_current(baryon=True, projection=True, oindex=oindex)
                # Both diagrams have real Wick coefficients and open baryon spins.
                for i in range(len(dc)):
                    assert_allclose(dc.calculate(i, weights=self.weights, block_size=4),
                                    dense_reference(dc, i, selected),
                                    rtol=2e-12, atol=2e-8)
        dc = self.make_current(baryon=True, projection=True)
        projector = dc._gamma_registry.resolve('Projector')
        dc._gamma_registry.register('Projector', {'sink': projector[0], 'source': projector[1],
                                                 'contraction': []})
        for i in range(len(dc)):
            assert_allclose(dc.calculate(i, weights=self.weights, block_size=4),
                            dense_reference(dc, i, selected),
                            rtol=2e-12, atol=2e-8)
        dc._gamma_registry.resolve('Projector')['contraction'] = ['0-0']
        for i in range(len(dc)):
            assert_allclose(dc.calculate(i, weights=self.weights, block_size=4),
                            dense_reference(dc, i, selected),
                            rtol=2e-12, atol=2e-8)

    def test_unity_weights_match_original_dynamic(self):
        dc = self.make_current()
        expected = make_dynamic().calculate_all()
        actual = dc.calculate_all(weights=BlendingWeights(Nx=2, exact=8), block_size=3)
        assert_allclose(actual, expected, rtol=2e-12, atol=2e-8)

    def test_repeated_peram_registry_key_uses_distinct_operand_views(self):
        # Force all u time pairs to resolve to ONE key, while their Wick vector
        # labels remain distinct. The adapter must slice each occurrence itself.
        dc = self.make_current(ncurr=2)
        dc.plan = deepcopy(dc.plan)  # keep lqcddb's shared Wick-plan cache intact
        array = np.random.default_rng(113).normal(size=(4, 4, 2, 2)).astype(complex)
        dc._peram_registry.register('u', ('shared', 'shared'), array)
        for entry in dc.plan:
            for p in entry[2]['peram'][entry[3]]:
                if p[2][0] == 'u':
                    p[4] = ['shared', 'shared']
        for i in range(len(dc)):
            assert_allclose(dc.calculate(i, weights=self.weights, block_size=1),
                            dense_reference(dc, i, VERTICES),
                            rtol=2e-12, atol=2e-8)


if __name__ == '__main__':
    unittest.main()
