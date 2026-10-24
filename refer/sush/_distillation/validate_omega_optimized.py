"""Record exact matrix equivalence and cold/warm CPU timings.

Run: python -B validate_omega_optimized.py
The default report is omega_optimized_validation.md in the current directory.
"""

import argparse
from hashlib import sha256
import io
from pathlib import Path
from statistics import median
from time import perf_counter
import tracemalloc
import unittest

import numpy as np
from lqcddb import set_backend
from reference_lqcddb import reference
Reference = reference.vertex_creator
from lqcddb.eigvectors import blending as optimized
from lqcddb import vertex_creator as Optimized
import test_omega_optimized as tests


def clear_plans():
    optimized._cached_cells.cache_clear()
    optimized._metadata.cache_clear()
    optimized._components.cache_clear()


def timed(function, repeats=5):
    output = function()
    del output
    measurements = []
    for _ in range(repeats):
        started = perf_counter()
        output = function()
        measurements.append(perf_counter()-started)
        del output
    return 1000*median(measurements)


def peak(function):
    tracemalloc.start()
    output = function()
    _, maximum = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del output
    return maximum/1024**2


def benchmarks():
    physical = dict(Nx=24, exact=50, N_eigen=[100, 200, 300, 500, 800],
                    N_sum=[20]*5, N_extract=[10]*5, noise=500)
    cases = [
        ('650 vectors, order 2', physical, dict(dim=2)),
        ('650 vectors, normalized order 2', physical, dict(dim=2, normal=True)),
        ('650 vectors, order 3, first index 60', physical, dict(dim=3, fixed_first_pos=[60])),
        ('150 vectors, order 4, first index 60', dict(physical, noise=0), dict(dim=4, fixed_first_pos=[60])),
        ('32 vectors, full order 4', dict(Nx=4, exact=8, N_eigen=[24, 16],
             N_sum=[8, 8], N_extract=[4, 4], noise=8), dict(dim=4)),
    ]
    rows = []
    for label, parameters, options in cases:
        arguments = {key: value for key, value in parameters.items() if key != 'Nx'}
        arguments.update(options)
        original = Reference(Nx=parameters['Nx'])
        local = Optimized(Nx=parameters['Nx'])
        native_call = lambda: original.create_omega_accelerate(**arguments)
        local_call = lambda: local.create_omega_accelerate(**arguments)
        reference = native_call()
        native_ms = timed(native_call)
        native_peak = peak(native_call)
        clear_plans()
        started = perf_counter()
        cold_result = local_call()
        cold_ms = 1000*(perf_counter()-started)
        assert cold_result.dtype == reference.dtype == np.dtype('complex128')
        assert cold_result.shape == reference.shape
        assert cold_result.tobytes() == reference.tobytes()
        del cold_result
        local_ms = timed(local_call)
        local_peak = peak(local_call)
        warm_result = local_call()
        assert warm_result.tobytes() == reference.tobytes()
        rows.append(dict(case=label, shape=reference.shape, original_ms=native_ms,
                         cold_ms=cold_ms, warm_ms=local_ms,
                         speedup=native_ms/local_ms, original_peak=native_peak,
                         optimized_peak=local_peak))
        del reference, warm_result
        print(f'PASS {label}: original={native_ms:.3f} ms, '
              f'optimized cold={cold_ms:.3f} ms, warm={local_ms:.3f} ms', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='omega_optimized_validation.md')
    args = parser.parse_args()
    set_backend('numpy')
    source = Path('/public/home/sush/distillation/lqcddb_update/src/lqcddb/eigvectors/vertex.py')
    before = sha256(source.read_bytes()).hexdigest()
    records = []
    tests.COMPARISON_RECORDS = records
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.TestOptimizedOmega)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    tests.COMPARISON_RECORDS = None
    rows = benchmarks() if result.wasSuccessful() else []
    unchanged = sha256(source.read_bytes()).hexdigest() == before
    passed = result.wasSuccessful() and unchanged
    elements = sum(row['elements'] for row in records)
    lines = [
        '# Optimized create_omega_accelerate: exact-equivalence validation', '',
        f'- Result: **{"PASS" if passed else "FAIL"}**.',
        f'- New test methods: {result.testsRun}.',
        f'- Recorded finite matrix comparisons: {len(records)}.',
        f'- Total compared matrix elements across those calls: {elements:,}.',
        '- Comparison: identical shape, complex128 dtype, finite values and output bytes.',
        '- Additional standalone/fallback/error/cache checks are included in the test methods.',
        f'- Unmodified upstream source SHA256: `{before}`; preservation: {unchanged}.',
        f'- NumPy: `{np.__version__}`. CPU measurements only.', '',
        '## Exactness and scientific conventions', '',
        'Every expected matrix is computed using the actual installed/imported',
        '`lqcddb_update` source method `vertex_creator.create_omega_accelerate`. No tolerance-based',
        'comparison substitutes for byte equality in this audit. All output',
        'elements, including real/imaginary values and zero signs, are checked.', '',
        'The optimized function retains the original np.prod results, group',
        'multiplication order, diagonal division order and overlapping-patch',
        'assignment order. It changes metadata reuse, group enumeration, diagonal',
        'index selection and output allocation. No signed partition expansion is used.', '',
        'Conserved mode forces dim=2. Normalization runs only at dim=2 and uses',
        'the same real-valued row operations and triangular assignment. As in the',
        'original, normal=True is ignored for other ordinary orders. No attempt',
        'was made to redefine the original normalization or stochastic estimator.', '',
        '## Cold/warm weight-generation benchmarks', '',
        '| Case | Output shape | Original median ms | Optimized first call ms | Optimized median ms | Warm speedup | Original peak MiB | Optimized peak MiB |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(f"| {row['case']} | `{row['shape']}` | {row['original_ms']:.3f} | "
                     f"{row['cold_ms']:.3f} | {row['warm_ms']:.3f} | {row['speedup']:.2f} | "
                     f"{row['original_peak']:.3f} | {row['optimized_peak']:.3f} |")
    lines += ['', 'Times are weight generation only. Medians use five timed calls after',
              'warm-up. First-call measurements clear all optimization caches.',
              'Peak traced allocations are measured separately with prepared caches;',
              'they exclude resident inputs/reference outputs and baseline RSS.',
              'Both implementations return the same complex128 output, including',
              'in cold and warm benchmark comparisons. These are host-specific',
              'CPU results, not end-to-end contraction speedups.', '',
              '## One matrix per row', '',
              '| Check | Nx | Requested order | Conserved | Normal | Fixed first indices | Sampling (exact / eigen / sum / extract / noise) | Output shape | Elements | Result |',
              '| ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- |']
    for i, record in enumerate(records):
        p = record['parameters']
        sampling = (p['exact'], p.get('N_eigen', [0]), p.get('N_sum', [0]),
                    p.get('N_extract', [0]), p.get('noise', 0))
        lines.append(f"| {i} | {record['Nx']} | {p.get('dim', 2)} | {p.get('conserved', False)} | "
                     f"{p.get('normal', False)} | `{p.get('fixed_first_pos', [-1])}` | "
                     f"`{sampling}` | `{record['shape']}` | {record['elements']} | BYTE-EQUAL |")
    lines += ['', '## Test output', '', '```text', stream.getvalue().rstrip(), '```', '',
              '## Scope', '',
              'Orders 1--4; exact-only, noise and compressed models; fixed singletons',
              'and contiguous ranges in each subspace; normalization/conserved modes;',
              'seeded random sampling parameters; realistic 650-vector second-order',
              'matrices and a 150-vector fourth-order slice; streamed large group plans;',
              'mutable argument/return/cache isolation; standalone function use; and',
              'legacy behavior for irregular fixed selectors are covered.', '',
              'Caches are bounded to 32 metadata entries and 16 group plans. Plans',
              'above 4096 cells are streamed. Index vectors are temporary per call;',
              'the cache does not store returned weight matrices.', '',
              'The implementation keeps the active backend interface. NumPy was',
              'tested; GPU and actual MPI execution were not run. Full dense tensors',
              'still require O(Nev**dim) output memory. The existing independent',
              'block API remains the interface for arbitrarily selected small tiles.', '']
    Path(args.report).write_text('\n'.join(lines))
    print(f"{'PASS' if passed else 'FAIL'}: {result.testsRun} test methods, "
          f"{len(records)} recorded byte-equal matrices, {elements:,} compared elements; "
          f"report {Path(args.report).resolve()}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
