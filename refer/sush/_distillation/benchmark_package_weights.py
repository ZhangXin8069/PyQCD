"""Profile the managed block generator and complete weighted calculation.

MPLCONFIGDIR="$PWD/.matplotlib-cache" python -B benchmark_package_weights.py
"""

import argparse
import cProfile
import io
from pathlib import Path
import pstats
from statistics import median
from time import perf_counter
import tracemalloc
from unittest.mock import patch

import numpy as np
from lqcddb import clear_cache

from lqcddb import BlendingWeights, set_backend
from lqcddb.eigvectors import blending as _omega, vertex as omega_optimized
from lqcddb.contraction import dynamic
from test_blending_dynamic import make_dynamic


def previous_vectorized_kernel(request):
    """Previous distinct-count kernel, retained here only as a timing baseline.

    Both alternatives are called through the same managed block interface.
    The actual original lqcddb routine remains the independent correctness
    reference in the separate exact-equivalence and full-contraction audits.
    """
    axes = request.axes
    meta = _omega._block_metadata(request.key)
    shape, dim = tuple(map(len, axes)), len(axes)
    output = np.ones(shape, dtype=request.dtype)
    if not output.size:
        return output
    grids, groups, unique = [], [], []
    for a, axis in enumerate(axes):
        current = axis.reshape((1,)*a+(len(axis),))
        group = request.groups[axis].reshape(current.shape)
        is_new = np.ones(shape[:a+1], dtype=bool)
        before = np.zeros(shape[:a+1], dtype=np.uint8)
        for b in range(a):
            previous = grids[b].reshape(grids[b].shape+(1,)*(a-b))
            previous_group = groups[b].reshape(groups[b].shape+(1,)*(a-b))
            previous_unique = unique[b].reshape(unique[b].shape+(1,)*(a-b))
            is_new &= current != previous
            before += previous_unique & (group == previous_group)
        factor = meta.ratios[group, before]
        factor[~is_new] = 1.0
        output *= factor.reshape(shape[:a+1]+(1,)*(dim-a-1))
        grids.append(current)
        groups.append(group)
        unique.append(is_new)
    return output


def timed(call, repeats=7):
    call()
    values = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        values.append(perf_counter()-start)
        del result
    return median(values)*1000


def peak(call):
    tracemalloc.start()
    result = call()
    _, maximum = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return maximum/1024**2


def clear_kernel_caches():
    for function in (_omega._metadata, _omega._cached_cells, _omega._components,
                     _omega._block_metadata, _omega._block_cell, _omega._block_lookup,
                     _omega._compact_lookup, _omega._equality_states):
        function.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='package_weight_benchmarks.md')
    args = parser.parse_args()
    set_backend('numpy')
    weights = BlendingWeights(Nx=24, exact=50, N_eigen=[100, 200, 300, 500, 800],
                              N_sum=[20]*5, N_extract=[10]*5, noise=500)
    managed = omega_optimized.create_omega_accelerate

    def baseline(self, *args, **kwargs):
        if isinstance(self, _omega._OmegaBlockRequest):
            return previous_vectorized_kernel(self)
        return managed(self, *args, **kwargs)

    cases = [(f'overlapping noise, {size} per axis', (slice(150, 150+size),)*4)
             for size in (1, 2, 4, 8, 16, 32)]
    cases += [('disjoint noise ranges, 32 per axis',
               tuple(slice(start, start+32) for start in (150, 200, 250, 300))),
              ('cross-subspace ranges, 32 per axis',
               (slice(120, 152), slice(140, 172), slice(149, 181), slice(150, 182))),
              ('reordered cross-subspace vectors',
               ([151, 4, 61, 150, 61, 140], [150, 61, 4, 140, 152, 4],
                [4, 151, 62, 140, 4, 152], [61, 140, 152, 4, 151, 61]))]
    rows = []
    for label, indices in cases:
        call = lambda: weights.block(dim=4, indices=indices)
        clear_kernel_caches()
        started = perf_counter()
        expected = call()
        cold_ms = (perf_counter()-started)*1000
        with patch.object(omega_optimized, 'create_omega_accelerate', side_effect=baseline):
            reference = call()
            reference_ms, reference_peak = timed(call), peak(call)
        np.testing.assert_allclose(expected, reference, rtol=3e-15, atol=0)
        managed_ms, managed_peak = timed(call), peak(call)
        rows.append((label, reference_ms, cold_ms, managed_ms, reference_peak, managed_peak))
        print(f'{label}: baseline {reference_ms:.4f} ms, managed {managed_ms:.4f} ms', flush=True)

    dc = make_dynamic(implementation=dynamic, oindex='M')
    small = BlendingWeights(Nx=2, exact=4, noise=4)
    kwargs = dict(weights=small, block_size=4)
    call = lambda: dc.calculate_all(**kwargs)
    clear_kernel_caches()
    clear_cache()
    started = perf_counter()
    managed_result = call()
    weighted_cold = (perf_counter()-started)*1000
    clear_kernel_caches()
    clear_cache()
    with patch.object(omega_optimized, 'create_omega_accelerate', side_effect=baseline):
        started = perf_counter()
        old_result = call()
        previous_cold = (perf_counter()-started)*1000
        old_time = timed(call, repeats=5)
    np.testing.assert_allclose(managed_result, old_result, rtol=2e-12, atol=2e-8)
    new_time = timed(call, repeats=5)
    profiler = cProfile.Profile()
    profiler.runcall(call)
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).strip_dirs().sort_stats('cumulative').print_stats(18)

    lines = ['# Managed package: block and weighted-path performance', '',
             '- CPU/NumPy, prepared metadata, float64 block weights.',
             '- Baseline: the previous vectorized distinct-count block algorithm.',
             '- Both kernels run behind the same public create_omega_accelerate dispatch.',
             '- The untouched original library is used separately for correctness audits.',
             '- Median of seven calls after warm-up; five calls for the complete path.', '',
             '| Case | Previous median ms | Managed first call ms | Managed median ms | Warm speedup | Previous peak MiB | Managed peak MiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, old, cold, new, old_peak, new_peak in rows:
        lines.append(f'| {label} | {old:.6f} | {cold:.6f} | {new:.6f} | {old/new:.2f} | {old_peak:.6f} | {new_peak:.6f} |')
    lines += ['', 'Peak is traced temporary allocation during the call, excluding',
              'existing inputs, caches and baseline process RSS. Host load affects',
              'these measurements; they are not GPU or production-data timings.', '',
              'First-call timings clear kernel metadata/lookup caches while retaining',
              'the already prepared sampling geometry. Mixed irregular lookup tables',
              'have a first-use construction cost; median timings show their reuse.', '',
              '## Complete six-diagram weighted calculation', '',
              f'- Previous vectorized weight kernel: {old_time:.6f} ms.',
              f'- Previous first call with cold kernel/contraction-path caches: {previous_cold:.6f} ms.',
              f'- Managed first call with cold kernel/contraction-path caches: {weighted_cold:.6f} ms.',
              f'- Managed weight kernel: {new_time:.6f} ms.',
              f'- Measured speedup: {old_time/new_time:.2f}.',
              '- Outputs agree at rtol=2e-12, atol=2e-8.',
              '- Per-diagram summation order and all input arrays are unchanged.', '',
              '## Managed-path profile', '', '```text', output.getvalue().rstrip(), '```', '']
    Path(args.report).write_text('\n'.join(lines))
    print(f'Complete weighted path: previous={old_time:.3f} ms, managed={new_time:.3f} ms; '
          f'report={Path(args.report).resolve()}', flush=True)


if __name__ == '__main__':
    main()
