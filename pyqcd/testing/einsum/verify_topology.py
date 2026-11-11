"""Corpus-wide correctness and performance gate for ``pyqcd.einsum.contract``.

Every ``result_indx`` equation of ``refer/sush/wick_12cases.md`` is contracted with
deterministic single-precision operands and compared against ``numpy.einsum``;
the canonical topology signature is checked for stability under a random
consistent renaming of the labels.

    python -m pyqcd.testing.einsum.verify_topology --limit 50
    python -m pyqcd.testing.einsum.verify_topology --case 4 --limit 100
    python -m pyqcd.testing.einsum.verify_topology --bench --jobs 1

``numpy.einsum(optimize='optimal')`` is available through ``--numpy-optimize`` but
is not the default: its planning cost explodes on the 13-16 operand topologies
(measured 110 s for one case-3 equation, >8 s for most others), while every path
yields the same reference values up to rounding.

By default the tolerance is applied against ``numpy`` evaluated on the operands
upcast to double precision, i.e. the exact result of the single-precision inputs.
The same-dtype numpy result is still computed and reported: for the 14-16 operand
topologies its own float32 accumulation error can reach ~1e-5 relative to the
largest output element, which is larger than ``contract``'s. Use
``--reference native`` to compare against same-dtype numpy instead.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pyqcd.einsum import contract
from pyqcd.testing.einsum import topology_corpus as corpus

TOLERANCES = {
    'complex64': 1e-5,
    'float32': 1e-5,
    'complex128': 1e-12,
    'float64': 1e-12,
}
DEFAULT_JOBS = min(os.cpu_count() or 1, 8)


@dataclass(frozen=True)
class Task:
    """One (equation, operands, tolerances) unit of work, pickle-friendly."""

    case: int
    topology: str
    equation: str
    sizes: tuple[int, ...]
    dtype: str
    numpy_optimize: str
    optimize: str
    reference: str
    tolerance: float
    reps: int
    rename_rounds: int
    seed: int


def _upcast(dtype: str) -> str:
    """Double-precision counterpart of ``dtype``."""
    return 'complex128' if np.dtype(dtype).kind == 'c' else 'float64'


def _deviation(actual: object, expected: object) -> float:
    """Scale-normalised max deviation, ``inf`` on shape mismatch."""
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape:
        return float('inf')
    if right.size == 0:
        return 0.0
    scale = float(np.max(np.abs(right)))
    diff = float(np.max(np.abs(left - right)))
    if scale == 0.0:
        return 0.0 if diff == 0.0 else float('inf')
    return diff / scale


def _timed(call, reps: int) -> tuple[object, float]:
    """Run ``call`` ``reps`` times, returning the last value and median millis."""
    value = None
    samples: list[float] = []
    for _ in range(max(1, reps)):
        start = time.perf_counter()
        value = call()
        samples.append((time.perf_counter() - start) * 1e3)
    return value, statistics.median(samples)


def _signature_stable(equation: str, rng: np.random.Generator, rounds: int) -> bool:
    expected = corpus.canonical_signature(equation)
    for _ in range(max(0, rounds)):
        if corpus.canonical_signature(corpus.relabel_equation(equation, rng)) != expected:
            return False
    return True


def _run_task(task: Task) -> dict[str, object]:
    result: dict[str, object] = {
        'case': task.case,
        'topology': task.topology,
        'equation': task.equation,
        'ok': False,
        'deviation': float('inf'),
        'native_deviation': float('inf'),
        'reference_deviation': float('inf'),
        'signature_ok': False,
        'pyqcd_ms': 0.0,
        'numpy_ms': 0.0,
        'error': '',
        'notes': (),
        'dtype_ok': False,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            rng = np.random.default_rng(task.seed)
            operands = corpus.make_operands(task.equation, rng, task.dtype, task.sizes)
            upcast = [operand.astype(_upcast(task.dtype)) for operand in operands]
            reference, numpy_ms = _timed(
                lambda: np.einsum(task.equation, *operands, optimize=task.numpy_optimize),
                task.reps)
            exact = np.einsum(task.equation, *upcast, optimize=task.numpy_optimize)
            actual, pyqcd_ms = _timed(
                lambda: contract(task.equation, *operands, optimize=task.optimize),
                task.reps)
            gated = exact if task.reference == 'upcast' else reference
            deviation = _deviation(actual, gated)
            result['deviation'] = deviation
            result['native_deviation'] = _deviation(actual, reference)
            result['reference_deviation'] = _deviation(reference, exact)
            result['dtype_ok'] = np.asarray(actual).dtype == np.asarray(reference).dtype
            result['signature_ok'] = _signature_stable(task.equation, rng, task.rename_rounds)
            result['ok'] = (
                deviation <= task.tolerance
                and bool(result['signature_ok'])
                and bool(result['dtype_ok'])
            )
        except Exception as exc:  # noqa: BLE001 - report any engine failure as a gate failure
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['pyqcd_ms'] = pyqcd_ms
        result['numpy_ms'] = numpy_ms
    result['notes'] = tuple(sorted({str(item.message) for item in caught}))
    return result


def _execute(tasks: Sequence[Task], jobs: int) -> list[dict[str, object]]:
    if jobs <= 1 or len(tasks) <= 1:
        return [_run_task(task) for task in tasks]
    workers = min(jobs, len(tasks))
    chunksize = max(1, len(tasks) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_task, tasks, chunksize=chunksize))


def _build_tasks(rows: Sequence[corpus.CorpusRow], args: argparse.Namespace) -> list[Task]:
    sizes = corpus.parse_sizes(args.size)
    tolerance = args.tolerance if args.tolerance is not None else TOLERANCES[args.dtype]
    reps = args.reps if args.bench else 1
    return [
        Task(
            case=row.case,
            topology=row.topology,
            equation=row.equation,
            sizes=sizes,
            dtype=args.dtype,
            numpy_optimize=args.numpy_optimize,
            optimize=args.optimize,
            reference=args.reference,
            tolerance=tolerance,
            reps=reps,
            rename_rounds=args.rename_rounds,
            seed=corpus.equation_seed(row.equation, args.seed),
        )
        for row in rows
    ]


def _bench_table(results: Sequence[dict[str, object]]) -> str:
    grouped: dict[int, dict[str, list[float]]] = {}
    for result in results:
        entry = grouped.setdefault(int(result['case']), {'pyqcd': [], 'numpy': []})
        entry['pyqcd'].append(float(result['pyqcd_ms']))
        entry['numpy'].append(float(result['numpy_ms']))
    lines = [
        '| case | rows | pyqcd median (ms) | numpy median (ms) | pyqcd total (s) | numpy total (s) |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    pyqcd_all: list[float] = []
    numpy_all: list[float] = []
    for case in sorted(grouped):
        entry = grouped[case]
        pyqcd_all.extend(entry['pyqcd'])
        numpy_all.extend(entry['numpy'])
        lines.append(
            f'| {case} | {len(entry["pyqcd"])} '
            f'| {statistics.median(entry["pyqcd"]):.3f} '
            f'| {statistics.median(entry["numpy"]):.3f} '
            f'| {sum(entry["pyqcd"]) / 1e3:.3f} '
            f'| {sum(entry["numpy"]) / 1e3:.3f} |')
    lines.append(
        f'| **all** | {len(pyqcd_all)} '
        f'| {statistics.median(pyqcd_all):.3f} '
        f'| {statistics.median(numpy_all):.3f} '
        f'| {sum(pyqcd_all) / 1e3:.3f} '
        f'| {sum(numpy_all) / 1e3:.3f} |')
    return '\n'.join(lines)


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    rows = corpus.load_corpus()
    stats = corpus.corpus_statistics(rows)
    problems = list(corpus.corpus_problems(rows))
    print(corpus.format_statistics(stats))
    declared = corpus.declared_summary()
    print(f'declared summary: {len(declared)} cases, '
          f'{sum(diagrams for diagrams, _ in declared.values())} rows')
    if stats['unique_signatures'] != stats['total']:
        problems.append(
            f'{len(stats["duplicate_signatures"])} topology signatures are shared by more '
            f'than one equation')
    if args.expect_total is not None and stats['total'] != args.expect_total:
        problems.append(f'expected {args.expect_total} corpus rows, found {stats["total"]}')
    for problem in problems:
        print(f'CORPUS {problem}')

    selected = corpus.select_rows(rows, args.case, args.limit)
    if not selected:
        print('no rows selected')
        return 1
    tasks = _build_tasks(selected, args)
    sizes = corpus.parse_sizes(args.size)
    tolerance = args.tolerance if args.tolerance is not None else TOLERANCES[args.dtype]
    print(f'selection: {len(selected)}/{stats["total"]} rows '
          f'(cases: {sorted({row.case for row in selected})})')
    print(f'dtype: {args.dtype}  tolerance: {tolerance:g}  sizes: {",".join(map(str, sizes))}')
    print(f'reference: numpy {args.reference} (optimize={args.numpy_optimize})  '
          f'pyqcd optimize={args.optimize}  jobs: {args.jobs}  reps: {tasks[0].reps}')
    if args.bench and args.jobs > 1:
        print('note: bench timings are meaningful with --jobs 1 (parallel load skews them)',
              file=sys.stderr)

    results = _execute(tasks, args.jobs)
    failures = [result for result in results if not result['ok']]
    passed_signatures = sum(1 for result in results if result['signature_ok'])
    notes = sorted({note for result in results for note in result['notes']})
    worst = max(results, key=lambda result: float(result['deviation']))
    worst_reference = max(results, key=lambda result: float(result['reference_deviation']))
    native_limited = sum(
        1 for result in results if float(result['reference_deviation']) > tolerance)

    if failures:
        print(f'FAIL {len(failures)}/{len(results)} rows')
        for result in failures[:10]:
            detail = str(result['error']) or (
                f'deviation {float(result["deviation"]):.3e} > {tolerance:g} '
                f'(numpy native {float(result["native_deviation"]):.3e}, '
                f'numpy native vs exact {float(result["reference_deviation"]):.3e})')
            if not result['signature_ok']:
                detail += '; signature unstable under relabelling'
            if not result['dtype_ok']:
                detail += '; dtype mismatch'
            print(f'  case {result["case"]} {result["topology"]} `{result["equation"]}`: {detail}')
    else:
        print(f'PASS {len(results)}/{len(results)} rows, '
              f'signature {passed_signatures}/{len(results)}, '
              f'max deviation {float(worst["deviation"]):.3e}')
    print(f'numpy native vs exact: max {float(worst_reference["reference_deviation"]):.3e} '
          f'(case {worst_reference["case"]} {worst_reference["topology"]}), '
          f'{native_limited}/{len(results)} rows above tolerance')
    print(f'margin: {sum(1 for result in results if float(result["deviation"]) > 0.5 * tolerance)}'
          f'/{len(results)} rows above half the tolerance')

    if args.bench:
        print(_bench_table(results))
    if notes:
        print(f'engine notes ({len(notes)} distinct):')
        for note in notes[:5]:
            print(f'  {note}')
    print(f'elapsed: {time.perf_counter() - started:.2f} s')
    return 0 if not failures and not problems else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify pyqcd.einsum.contract against numpy over the Wick corpus')
    parser.add_argument('--limit', type=int, default=None,
                        help='keep the first N rows after case filtering')
    parser.add_argument('--case', type=int, nargs='+', default=None,
                        help='restrict to these case numbers')
    parser.add_argument('--jobs', type=int, default=DEFAULT_JOBS,
                        help=f'multiprocessing workers (default {DEFAULT_JOBS}, 1 = serial)')
    parser.add_argument('--size', default=None,
                        help='operand dimensions, cycling over labels (default 2,3,4)')
    parser.add_argument('--dtype', choices=sorted(TOLERANCES), default='complex64')
    parser.add_argument('--tolerance', type=float, default=None,
                        help='override the dtype default relative tolerance')
    parser.add_argument('--numpy-optimize', default='greedy',
                        help='numpy reference path optimizer (greedy/optimal/...)')
    parser.add_argument('--optimize', default='auto',
                        help='path optimizer passed to pyqcd.einsum.contract')
    parser.add_argument('--reference', choices=('upcast', 'native'), default='upcast',
                        help='compare against numpy on double-precision operands (upcast, '
                             'default) or on same-dtype operands (native)')
    parser.add_argument('--bench', action='store_true',
                        help='time both engines (median of --reps) and print a per-case table')
    parser.add_argument('--reps', type=int, default=3, help='benchmark repetitions (default 3)')
    parser.add_argument('--rename-rounds', type=int, default=1,
                        help='random relabellings per equation (default 1, 0 disables)')
    parser.add_argument('--seed', type=int, default=20261111,
                        help='base RNG seed; per-equation seeds are derived deterministically')
    parser.add_argument('--expect-total', type=int, default=None,
                        help='fail unless the corpus holds exactly this many rows')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
