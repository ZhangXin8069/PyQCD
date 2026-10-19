"""Summarize actual GPU artifacts; never emit an accepted Zchi or matching input."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import digest


def summarize(path):
    with np.load(path, allow_pickle=False) as d:
        m = json.loads(str(d['metadata_json']))
        values = np.array(d['kinetic_a4_noise'])
        if m['schema'] != 'quark_ringed_configuration_raw_v1' or not np.isfinite(values).all():
            raise ValueError('Invalid raw measurement')
        if values.ndim != 2 or values.shape[0] != len(d['flow_times']):
            raise ValueError('Expected (flow, noise) raw values')
        n = values.shape[1]
        if n < 2 or len(m['source_shards']) != n or not np.isfinite(d['true_residual']).all():
            raise ValueError('Incomplete noise/residual data')
        mean_real = values.real.mean(axis=1)
        mean_imag = values.imag.mean(axis=1)
        sem_real = values.real.std(axis=1, ddof=1)/np.sqrt(n)
        sem_imag = values.imag.std(axis=1, ddof=1)/np.sqrt(n)
        record = {'path': str(path.resolve()), 'sha256': digest(path),
                  'status': m['status'], 'conf': int(d['configuration']),
                  'flow_times': d['flow_times'].tolist(), 'noise_count': n,
                  'mean_real': mean_real.tolist(), 'mean_imag': mean_imag.tolist(),
                  'noise_sem_real': sem_real.tolist(), 'noise_sem_imag': sem_imag.tolist(),
                  'noise_error_scope': 'stochastic error on ONE configuration, not ensemble uncertainty',
                  'relative_noise_sem_real': [float(e/abs(v)) if v != 0 else None
                                              for v, e in zip(mean_real, sem_real)],
                  'imaginary_mean_in_noise_sem': [float(v/e) if e > 0 else None
                                                 for v, e in zip(mean_imag, sem_imag)],
                  'true_residual_max': float(np.max(d['true_residual'])),
                  'runtime': m['runtime'], 'mass_treatment': m['run_identity']['plan']['mass_treatment'],
                  'prefix_noise_means_real': {str(k): values.real[:, :k].mean(axis=1).tolist()
                                              for k in (2, 4, 8, 16, 32) if k <= n}}
        if 'free_exact_trace' in d.files:
            record['free_exact_trace'] = d['free_exact_trace'].tolist()
    records = []
    for i in range(n):
        shard = path.parent/f'noise_{i:05d}.npz'
        if digest(shard) != m['source_shards'][i]:
            raise ValueError('Changed noise artifact')
        with np.load(shard, allow_pickle=False) as s:
            if not np.array_equal(s['kinetic_a4_noise'], values[:, i]):
                raise ValueError('Aggregate does not match its noise shards')
            for key in ('gauge_unitarity_determinant_errors', 'solve_seconds', 'iterations'):
                if not np.isfinite(s[key]).all():
                    raise ValueError(f'Nonfinite noise diagnostic: {key}')
            records.append({key: float(np.max(s[key])) for key in
                            ('free_same_noise_max_relative_error', 'gauge_unitarity_determinant_errors',
                             'solve_seconds', 'iterations') if key in s.files})
    record['maxima_over_noise'] = {k: max(r[k] for r in records) for k in records[0]}
    return record


def summarize_comparison(path):
    with np.load(path, allow_pickle=False) as d:
        report = json.loads(str(d['metadata_json']))
        for side in ('left', 'right'):
            if digest(Path(report[side])) != report[side+'_sha256']:
                raise ValueError('Changed paired-check source')
        with np.load(report['left'], allow_pickle=False) as left, np.load(report['right'], allow_pickle=False) as right:
            delta = left['kinetic_a4_noise']-right['kinetic_a4_noise']
            if not np.array_equal(delta, d['difference']) or not np.isfinite(delta).all():
                raise ValueError('Inconsistent paired-check differences')
            if np.any(np.abs(right['kinetic_a4_noise']) == 0):
                raise ValueError('Zero reference requires an absolute-error interpretation')
            passed = bool(np.all(np.abs(delta) <= report['atol'] + report['rtol']*np.abs(right['kinetic_a4_noise'])))
            if passed != report['passed']:
                raise ValueError('Inconsistent numerical check status')
            report['max_relative_difference'] = float(np.max(np.abs(delta/right['kinetic_a4_noise'])))
            report['max_relative_difference_by_flow'] = np.max(np.abs(delta/right['kinetic_a4_noise']), axis=1).tolist()
        report['report_path'] = str(path.resolve())
        report['report_sha256'] = digest(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--real-checks-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    report = {'status': 'GPU_numerical_checks_NOT_massless_normalization', 'measurements': {}}
    for name in ('free_checkpoint', 'free_direct', 'free_fine', 'conf10000_pilot'):
        path = a.run_root/name/'kinetic_conf.npz'
        report['measurements'][name] = summarize(path) if path.exists() else {'status': 'missing_or_incomplete'}
    report['comparisons'] = {}
    for name in ('free_restart_check', 'free_step_check'):
        path = a.run_root/(name+'.npz')
        report['comparisons'][name] = summarize_comparison(path) if path.exists() else {'status': 'missing_or_incomplete'}
    if a.real_checks_root is not None:
        for name in ('checkpoint', 'direct', 'fine', 'stock_backend'):
            path = a.real_checks_root/name/'kinetic_conf.npz'
            report['measurements']['real_'+name] = summarize(path) if path.exists() else {'status': 'missing_or_incomplete'}
        for name in ('restart_check', 'step_check', 'backend_check'):
            path = a.real_checks_root/(name+'.npz')
            report['comparisons']['real_'+name] = summarize_comparison(path) if path.exists() else {'status': 'missing_or_incomplete'}
    report['all_requested_artifacts_present'] = all(r.get('status') != 'missing_or_incomplete'
        for group in ('measurements', 'comparisons') for r in report[group].values())
    report['all_recorded_numerical_comparisons_passed'] = all(r.get('passed', False)
        for r in report['comparisons'].values())
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(a.output)


if __name__ == '__main__':
    main()
