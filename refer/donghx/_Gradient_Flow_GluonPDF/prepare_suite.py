#!/usr/bin/env python3
"""Freeze source hashes and input headers only; no physical calculation."""
import json
from common import (HERE, OLD, CONFS, CURRENTS, WINDOWS, SCHEME, config_for,
                    original_config, source_times, fingerprint, sha256, atomic_json)


def main():
    if (HERE / 'manifest.json').exists():
        raise RuntimeError('manifest already exists; do not mutate a submitted campaign')
    previous = json.loads((OLD / 'manifest.json').read_text())
    rows = []
    for conf in CONFS:
        oldrow = next(r for r in previous['configurations'] if r['conf'] == conf)
        cfg = original_config(conf)
        paths = {i['path'] for i in oldrow['inputs']}
        paths.update(str(cfg.eigen_path(s)) for s in source_times())
        for tj in CURRENTS:
            receipt = OLD / 'results' / conf / f'tJ{tj:03d}' / 'receipt.json'
            if json.loads(receipt.read_text())['status'] != 'PASS':
                raise RuntimeError(f'old B reference not PASS: {receipt}')
            paths.add(str(receipt))
            paths.add(str(receipt.with_name('comparison.npz')))
        inputs = [fingerprint(p) for p in sorted(paths)]
        frozen = {i['path']: i for i in oldrow['inputs']}
        for i in inputs:
            if i['path'] in frozen:
                old = frozen[i['path']]
                if any(i[k] != old[k] for k in ('bytes', 'mtime_ns')):
                    raise RuntimeError(f'input changed since old comparison: {i["path"]}')
        for s in source_times():
            info = next(i for i in inputs if i['path'] == str(cfg.eigen_path(s)))
            if info['shape'] != [400, 32, 32, 32, 3] or info['dtype'] != 'complex128':
                raise ValueError(f'wrong original basis: {info}')
        rows.append(dict(conf=conf, inputs=inputs, gauge_index=oldrow['gauge_index']))
        print('HEADER_PASS', conf, 'flow_sources', len(source_times()), flush=True)
    paths = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.slurm'))
    paths += sorted((HERE / 'reference').glob('*.py'))
    atomic_json(HERE / 'manifest.json', dict(
        schema='l32x64_schemeB_window_test_v1', generated_date='2026-10-15',
        configurations=rows, currents=list(CURRENTS), windows=list(WINDOWS),
        source_times=source_times(), flow_time_a2=.5, epsilon=.01, nstep=50,
        nev_current=400, nev_src=200, a_fm=.0897, tsep=7, dt=3,
        momenta=[3, 4, 5], direction='z', delta_z=list(range(21)),
        weight_scheme=SCHEME, per_column_squared_norm_gate=1e-6,
        baseline_reflow_relative_l2_gate=2e-6, prior_B_C3_relative_l2_gate=2e-5,
        interpretation='Numerical window-convergence diagnostic, not stochastic-unbiasedness proof; w6 is finite, not exact full-time C3.',
        files_sha256={str(p.relative_to(HERE)): sha256(p) for p in paths}))


if __name__ == '__main__':
    main()
