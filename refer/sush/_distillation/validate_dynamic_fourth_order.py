"""Audit every diagram against direct contraction with lqcddb's FULL Omega4.

MPLCONFIGDIR="$PWD/.matplotlib-cache" python -B validate_dynamic_fourth_order.py

This is a synthetic small-tensor validation. It never calls the block engine
to construct the dense reference, never executes a production measurement,
and does not modify lqcddb or the local contraction implementations.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from reference_lqcddb import DYNAMIC_SHA256
from opt_einsum import contract

from lqcddb.contraction import dynamic as local
from lqcddb.eigvectors.blending import BlendingWeights, BlendedDynamicContraction
from lqcddb import gamma, set_backend
from reference_lqcddb import reference
vertex_creator = reference.vertex_creator


RTOL, ATOL = 1e-12, 1e-8
MODELS = {
    'exact-noise': dict(Nx=2, exact=4, noise=4),
    'compressed-noise': dict(Nx=3, exact=4, N_eigen=[12], N_sum=[8],
                             N_extract=[4], noise=4),
}


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    model: str = 'exact-noise'
    baryon: bool = False
    coincident_sink: bool = False
    projection: str = ''
    oindex: str | None = None
    subset: str = ''
    block_size: object = 4


CASES = (
    Case('pion-4pt-all-eight-vectors', 8261, oindex='MHG', block_size=3),
    Case('pion-4pt-compressed-subspaces', 137, model='compressed-noise',
         block_size=(5, 7, 6, 8)),
    Case('pion-4pt-reordered-global-subsets', 945, model='compressed-noise',
         subset='reordered', oindex='M', block_size=(2, 3, 4, 5)),
    Case('pion-4pt-all-noise-subset', 1203, subset='noise', block_size=3),
    Case('pion-3pt-coincident-sink-current', 1717, model='compressed-noise',
         coincident_sink=True, block_size=(5, 7, 6, 8)),
    Case('proton-4pt-legacy-projectors', 2024, baryon=True, projection='legacy'),
    Case('proton-4pt-legacy-projectors-explicit-output', 2025, baryon=True,
         projection='legacy', oindex='M', block_size=5),
    Case('proton-4pt-role-projectors', 2026, baryon=True, projection='roles',
         block_size=(3, 4, 5, 6)),
    Case('proton-4pt-role-projector-contraction', 2027, baryon=True,
         projection='roles-contracted'),
)


def vertex_keys(v_info):
    """Independently translate Wick's global names into registry names."""
    counts, keys = {}, []
    for _, kind, _, time_label in v_info:
        prefix = 'VVV' if kind.startswith('VVV') else 'VDV'
        number = counts.get((prefix, time_label), 0)
        keys.append((f'{prefix}_{number}', time_label))
        counts[(prefix, time_label)] = number + 1
    return keys


def make_case(case):
    parameters = MODELS[case.model]
    weights = BlendingWeights(**parameters)
    selected = ([('VDV_0', 'tsink'), ('VDV_0', 'tcur0')] if case.coincident_sink else
                [('VDV_0', 'tcur0'), ('VDV_0', 'tcur1')])
    ncurr, mappings = weights.nvec, None
    if case.subset == 'reordered':
        ncurr = 6
        mappings = {
            selected[0]: ([15, 0, 4, 12, 8, 5], [4, 13, 1, 8, 14, 7]),
            selected[1]: ([12, 14, 0, 9, 15, 5], [15, 4, 12, 1, 8, 13]),
        }
    elif case.subset == 'noise':
        ncurr = parameters['noise']
        offset = weights.nvec - ncurr
        mappings = {key: (slice(offset, weights.nvec),) * 2 for key in selected}

    if case.baryon:
        sink = ['|', 'u', 'u', 'gamma_7', 'd', '|']
        source = ['|', 'u^d', 'gamma_7', 'd^d', 'u^d', '|']
    else:
        sink = ['|', 'd^d', 'gamma_5', 'u', '|']
        source = ['|', 'u^d', 'gamma_5', 'd', '|']
    current = ['|', 'u^d', 'gamma_curr', 'u', '|']
    count = 1 if case.coincident_sink else 2
    groups = [(sink, source, current * count)]
    # V/gamma order is sink, current(s), source. Keep every Wick diagram.
    options = dict(Cpt='3pt' if count == 1 else '4pt', Vindex=['M']*(count+2),
                   Gindex=['', 'G', ''] if count == 1 else ['', 'G', 'H', ''],
                   ignore_dis=False, use_equivalence=False, verbose=False,
                   Oindex=case.oindex, optimize='greedy')
    discovery = local.run_wick_analysis(groups, **options)
    pr, vr, gr = local.PeramRegistry(), local.VRegistry(), local.GammaRegistry()
    rng = np.random.default_rng(case.seed)

    def random(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)

    def nev(label):
        return ncurr if label.startswith('tcur') or (case.coincident_sink and label == 'tsink') else 2

    registered_p, registered_v = set(), set()
    for entry in discovery:
        wick, diagram = entry[2], entry[3]
        for p in wick['peram'][diagram]:
            key = (p[2][0], tuple(p[4]))
            if key not in registered_p:
                pr.register(*key, random((4, 4, nev(p[4][0]), nev(p[4][1]))))
                registered_p.add(key)
        for vertex, key in zip(wick['V'], vertex_keys(wick['V'])):
            if key not in registered_v:
                legs = 3 if vertex[1].startswith('VVV') else 2
                vr.register(*key, random((2,) + (nev(key[1]),)*legs))
                registered_v.add(key)
    gr.register('gamma_5', np.asarray(gamma(5)))
    gr.register('gamma_7', np.asarray(gamma(7)))
    gr.register('gamma_curr', np.asarray([gamma(1), gamma(4)]))
    if case.projection == 'legacy':
        gr.register('Projector', [random((4, 3)), random((4, 3))])
    elif case.projection.startswith('roles'):
        contracted = case.projection == 'roles-contracted'
        gr.register('Projector', dict(sink=random((4, 3 if contracted else 2)),
                                     source=random((4, 3)),
                                     contraction=['0-0'] if contracted else []))
    dc = local.dynamic_contraction(groups, peram_registry=pr, v_registry=vr,
                                    gamma_registry=gr, Projection=bool(case.projection), **options)
    # Independent reference: generate the ENTIRE tensor with the real library.
    omega_parameters = dict(parameters)
    creator = vertex_creator(Nx=omega_parameters.pop('Nx'))
    omega4 = creator.create_omega_accelerate(dim=4, **omega_parameters)
    assert omega4.shape == (weights.nvec,)*4
    return dc, weights, selected, mappings, omega4


def direct_full_weight(dc, i, selected, mappings, omega4):
    """Append full Omega4 to the original, unsliced tensor network exactly once."""
    entry = dc[i]
    wick, diagram = entry[2], entry[3]
    lhs, rhs = wick['result_indx'][diagram][0].split('->')
    spin = rhs[:next((j for j, label in enumerate(rhs) if not label.islower()), len(rhs))]
    if len(entry) > 5:
        rhs = spin + entry[5]
    pvars = [dc._peram_registry.resolve(p[2], p[4]) for p in wick['peram'][diagram]]
    gvars = [dc._gamma_registry.resolve(g[1]) for g in wick['gamma_pos']]
    keys = vertex_keys(wick['V'])
    vvars = [dc._v_registry.resolve(*key) for key in keys]
    labels, indices = '', []
    global_axis = np.arange(omega4.shape[0])
    for key in selected:
        position = keys.index(key)
        labels += wick['V'][position][2][-2:]
        selectors = None if mappings is None else mappings.get(key)
        if selectors is None:
            selectors = tuple(slice(0, length) for length in vvars[position].shape[-2:])
        indices.extend(np.atleast_1d(global_axis[s]) for s in selectors)
    assert len(labels) == len(set(labels)) == 4
    direct_weight = omega4[np.ix_(*indices)]
    coefficient = sum(coef for _, _, coef in entry[0])
    output = coefficient * contract(lhs+','+labels+'->'+rhs,
                                    *pvars, *gvars, *vvars, direct_weight, optimize='greedy')
    if dc._projection is False:
        return output
    # Project AFTER the independent full contraction, without using any local
    # or installed dynamic projection helper (the tested path projects inside).
    projector = dc._gamma_registry.resolve('Projector')
    if isinstance(projector, dict):
        roles = {}
        for (lo, hi), kind, _, label in wick['V']:
            if kind.startswith('VVV') and label in ('tsink', 'tsrc'):
                candidates = [ports[0] for pos, _, ports in wick['quark_pos']
                              if lo < pos < hi and ports[0] in spin]
                assert len(candidates) == 1
                roles[label] = spin.index(candidates[0])
        ordered = np.moveaxis(output, (roles['tsink'], roles['tsrc']), (0, 1))
        if projector['contraction']:
            return np.einsum('ab...,ax,bx->...', ordered, projector['sink'], projector['source'])
        return np.einsum('ab...,ax,by->xy...', ordered, projector['sink'], projector['source'])
    if len(entry) > 6 and entry[6]:
        return np.einsum('ab...,ax,bx->...', output, *projector)
    return np.einsum('ab...,ax,by->xy...', output, *projector)


def compare_arrays(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return dict(passed=False, absolute=float('inf'), scaled=float('inf'), normalized=float('inf'))
    difference = np.abs(actual-reference)
    absolute = float(np.max(difference, initial=0))
    scale = float(np.max(np.abs(reference), initial=0))
    normalized = float(np.max(difference/(ATOL + RTOL*np.abs(reference)), initial=0))
    return dict(passed=normalized <= 1, absolute=absolute,
                scaled=absolute/scale if scale else absolute, normalized=normalized)


def audit_case(case):
    dc, weights, selected, mappings, omega4 = make_case(case)
    baseline = np.asarray(dc.calculate_all()).copy()
    automatic = mappings is None and not case.coincident_sink
    if automatic:
        calculator = dc
        options = dict(weights=weights, block_size=case.block_size)
    else:
        # Deliberately broader internal-kernel audit, where the fixture knows
        # explicit physical coincidence or a global permutation. These cannot
        # be inferred from raw registry arrays by the simple current API.
        calculator = BlendedDynamicContraction(dc, weights, vertices=selected,
                    vector_indices=mappings, block_size=case.block_size)
        options = {}
    direct_total = 0
    blocked_total = 0
    rows, equations = [], []
    for i in range(len(dc)):
        reference = direct_full_weight(dc, i, selected, mappings, omega4)
        actual = calculator.calculate(i, **options)
        metrics = compare_arrays(actual, reference)
        coefficient = sum(coef for _, _, coef in dc[i][0])
        rows.append(dict(diagram=i, coefficient=coefficient, shape=reference.shape, **metrics))
        equations.append(dc[i][2]['result_indx'][dc[i][3]][0])
        direct_total = direct_total + reference
        blocked_total = blocked_total + actual
    totals = {
        'calculate_all versus direct sum': compare_arrays(calculator.calculate_all(**options), direct_total),
        'sum of calculate(i) versus direct sum': compare_arrays(blocked_total, direct_total),
    }
    ordinary_unchanged = np.array_equal(dc.calculate_all(), baseline)
    return dict(case=case, rows=rows, equations=equations, totals=totals, automatic=automatic,
                ordinary_unchanged=ordinary_unchanged, nvec=weights.nvec,
                samples=weights.sample_sizes, populations=weights.space_sizes,
                passed=all(row['passed'] for row in rows) and
                       all(value['passed'] for value in totals.values()) and ordinary_unchanged)


def audit_weight_entries():
    """Compare ALL entries, separating noise-sector and equality-pattern cases."""
    results = []
    for name, parameters in MODELS.items():
        weights = BlendingWeights(**parameters)
        options = dict(parameters)
        creator = vertex_creator(Nx=options.pop('Nx'))
        full = creator.create_omega_accelerate(dim=4, **options)
        reconstructed = np.zeros(full.shape, dtype=float)
        visits = np.zeros(full.shape, dtype=np.uint8)
        for slices, weight in weights.iter_blocks(dim=4, block_size=(3, 5, 4, 6)):
            reconstructed[slices] = weight
            visits[slices] += 1
        assert np.all(visits == 1)
        axes = np.ogrid[(slice(0, weights.nvec),)*4]
        noise_count = np.zeros(full.shape, dtype=np.uint8)
        distinct = np.ones(full.shape, dtype=np.uint8)
        offset = weights.nvec-parameters['noise']
        for a, axis in enumerate(axes):
            noise_count += axis >= offset
            if a:
                new = np.ones(full.shape, dtype=bool)
                for earlier in axes[:a]:
                    new &= axis != earlier
                distinct += new
        for label, mask in [(f'{count} noise indices', noise_count == count) for count in range(5)]:
            results.append(dict(model=name, category=label, entries=int(mask.sum()),
                                **compare_arrays(reconstructed[mask], full[mask])))
        for count in range(1, 5):
            mask = (noise_count == 4) & (distinct == count)
            results.append(dict(model=name, category=f'all noise, {count} distinct indices',
                                entries=int(mask.sum()), **compare_arrays(reconstructed[mask], full[mask])))
    return results


def write_report(path, results, weight_rows, elapsed, upstream_unchanged):
    diagrams = sum(len(result['rows']) for result in results)
    passed = (all(result['passed'] for result in results) and
              all(row['passed'] for row in weight_rows) and upstream_unchanged)
    lines = [
        '# Fourth-order weight: one-by-one validation', '',
        f'- Result: **{"PASS" if passed else "FAIL"}**.',
        f'- Cases: {len(results)}; individually compared Wick diagrams: {diagrams}.',
        f'- Simplified public API diagrams: {sum(len(r["rows"]) for r in results if r["automatic"])}; explicit internal-kernel audit diagrams: {sum(len(r["rows"]) for r in results if not r["automatic"])}.',
        f'- Runtime on this host: {elapsed:.3f} s.',
        '- Backend: NumPy, float64 weights and complex128 contractions.',
        '- Source preservation: ' + ('PASS' if upstream_unchanged else 'FAIL') + '.', '',
        'The reference constructs the entire fourth-order tensor using the actual',
        '`lqcddb_update` source method `vertex_creator.create_omega_accelerate(dim=4)`. It appends that',
        'tensor to each original unsliced Wick equation and contracts directly',
        'with opt_einsum. It does not call the block-weight helper or its adapter.',
        'Projectors are applied independently after the direct contraction.', '',
        'Automatic-current cases use the simplified public calculate methods.',
        'Explicit offset/permutation or coincident sink/current fixtures audit',
        'the internal adapter; they do not test automatic physical-time inference.',
        'Every diagram is compared element by element, including its Wick',
        'coefficient. Each case also checks `calculate_all(weights=...)` against',
        'the sum of independently computed direct diagrams and verifies unchanged',
        'ordinary results after weighted calls.', '',
        r'Tolerance: $|C_{\mathrm{block}}-C_{\mathrm{direct}}|\le10^{-8}+10^{-12}|C_{\mathrm{direct}}|$.',
        r'Scaled error is $\max|\Delta C|/\max|C_{\mathrm{direct}}|$.',
        'The normalized error column is the largest elementwise error divided by',
        'its tolerance; PASS requires it to be at most one.', '',
        '## All weight entries and individual sectors', '',
        '| Model | Sector/pattern | Entries checked | Max absolute error | Scaled error | Status |',
        '| --- | --- | ---: | ---: | ---: | --- |',
    ]
    for row in weight_rows:
        lines.append(f"| {row['model']} | {row['category']} | {row['entries']} | "
                     f"{row['absolute']:.6e} | {row['scaled']:.6e} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines += ['', 'The five noise-count sectors are disjoint and exhaustive.',
              'The four all-noise distinct-count rows are additional diagonal checks.',
              'Each full tensor entry was covered by exactly one computed block.', '']
    for result in results:
        case = result['case']
        lines += [f'## {case.name}', '',
                  f'- Seed: {case.seed}; full retained vector count: {result["nvec"]}.',
                  f'- Population sizes: `{result["populations"]}`; retained sizes: `{result["samples"]}`.',
                  f'- Block sizes: `{case.block_size}`; projection: `{case.projection or "none"}`; Oindex: `{case.oindex}`.',
                  '- Diagram numbering below is zero-based, exactly as in `dc.plan`.', '',
                  '| Diagram | Wick coefficient | Output shape | Max absolute error | Scaled error | Max normalized error | Status |',
                  '| ---: | --- | --- | ---: | ---: | ---: | --- |']
        for row in result['rows']:
            lines.append(f"| {row['diagram']} | `{row['coefficient']}` | `{row['shape']}` | "
                         f"{row['absolute']:.6e} | {row['scaled']:.6e} | {row['normalized']:.6e} | "
                         f"{'PASS' if row['passed'] else 'FAIL'} |")
        lines += ['', '| Sum check | Max absolute error | Scaled error | Max normalized error | Status |',
                  '| --- | ---: | ---: | ---: | --- |']
        for name, row in result['totals'].items():
            lines.append(f"| {name} | {row['absolute']:.6e} | {row['scaled']:.6e} | "
                         f"{row['normalized']:.6e} | {'PASS' if row['passed'] else 'FAIL'} |")
        lines += ['', '- Ordinary result after weighted calls: ' +
                  ('bitwise unchanged.' if result['ordinary_unchanged'] else '**CHANGED**.'), '',
                  '<details><summary>Original Wick equations, one per diagram</summary>', '', '```text']
        lines += [f'{i}: {equation}' for i, equation in enumerate(result['equations'])]
        lines += ['```', '', '</details>', '']
    lines += ['## Scope', '',
              'All connected and disconnected diagrams generated for these operators',
              'were retained (`ignore_dis=False`, `use_equivalence=False`). Proton',
              'cases contain TWO VDV currents and therefore test fourth-order weights',
              'with baryon projections. Compression cases include two independent',
              'retained subspaces, and noise-only cases include all four distinct',
              'sampled noise vectors. Noncontiguous/reordered global subsets are also tested.', '',
              'These are synthetic complex-tensor identities with four Dirac spin',
              'components and the library gamma matrices. They test contraction and',
              'weight algebra, not gauge-configuration data or the physical validity',
              'of the underlying stochastic estimator. Actual MPI and GPU execution',
              'were not run. No full 650-vector fourth-order tensor was allocated.', '']
    Path(path).write_text('\n'.join(lines))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='dynamic_fourth_order_validation.md')
    args = parser.parse_args()
    set_backend('numpy')
    started = time.perf_counter()
    weight_rows = audit_weight_entries()
    results = []
    for case in CASES:
        result = audit_case(case)
        results.append(result)
        worst = max(row['scaled'] for row in result['rows'])
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case.name}: "
              f"{len(result['rows'])} diagrams, max scaled error {worst:.3e}", flush=True)
    upstream = Path('/public/home/sush/distillation/lqcddb_update/src/lqcddb/contraction/dynamic.py')
    unchanged = sha256(upstream.read_bytes()).hexdigest() == DYNAMIC_SHA256
    passed = write_report(args.report, results, weight_rows,
                          time.perf_counter()-started, unchanged)
    print(f"{'PASS' if passed else 'FAIL'}: {sum(len(r['rows']) for r in results)} individual diagrams; "
          f"{sum(len(r['totals']) for r in results)} sum checks; "
          f"{len(weight_rows)} weight-sector checks. Report: {Path(args.report).resolve()}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
