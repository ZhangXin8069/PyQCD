"""Compare a new solver/backend to accepted same-noise BiCGStab samples."""
import argparse
import json
from pathlib import Path

import numpy as np
from quark_ringed_kinetic_v1.core import digest
from quark_ringed_kinetic_v1.report_gpu import summarize


def compare_noise(left, right, tolerance):
    with np.load(left, allow_pickle=False) as l, np.load(right, allow_pickle=False) as r:
        lm, rm = (json.loads(str(d['metadata_json'])) for d in (l, r))
        li, ri = lm['run_identity'], rm['run_identity']
        for key in ('conf', 'seed', 'gauge', 'direct'):
            if li[key] != ri[key]:
                raise ValueError('Not paired: '+key)
        if lm['noise_id'] != rm['noise_id'] or li['code_sha256']['core.py'] != ri['code_sha256']['core.py']:
            raise ValueError('Noise generator or identity changed')
        for key in ('mass', 'csw', 'lattice_xyzt', 'stout_steps', 'stout_rho',
                    'stout_dir_ignore', 'epsilon', 'flow_times', 'fermion_boundary', 'mass_normalization'):
            if li['plan'][key] != ri['plan'][key]:
                raise ValueError('Physical parameters changed: '+key)
        for d, identity in ((l, li), (r, ri)):
            if not np.isfinite(d['true_residual']).all() or float(d['true_residual']) > identity['plan']['residual_max']:
                raise ValueError('Unconverged reference or check')
        lv, rv = l['kinetic_a4_noise'], r['kinetic_a4_noise']
        relative = np.abs(lv-rv)/np.abs(rv)
        if not np.isfinite(relative).all() or np.max(relative) > tolerance:
            raise ValueError(f'Solver observable mismatch: {relative}')
        return {'left': str(left), 'right': str(right), 'left_sha256': digest(left), 'right_sha256': digest(right),
                'max_relative_difference': float(np.max(relative)), 'rtol': tolerance,
                'left_measure_sha256': li['code_sha256']['measure.py'],
                'right_measure_sha256': ri['code_sha256']['measure.py'],
                'left_library_sha256': lm['runtime']['libquda_sha256'],
                'right_library_sha256': rm['runtime']['libquda_sha256'],
                'core_sha256': li['code_sha256']['core.py'], 'passed': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.run_root = a.run_root.resolve(strict=True)
    project = Path(__file__).resolve().parent.parent
    free_old = project/'quark_ringed_kinetic_v1/runs/gpu_20260918_BjcGfO/free_checkpoint'
    real_old = project/'quark_ringed_mass_pilot_v1/runs/critical_mass_20260918_13cmron4/central'
    checks = []
    for name, old, count in [('free_solver', free_old, 4), ('central_solver_check', real_old, 2)]:
        aggregate = summarize(a.run_root/name/'kinetic_conf.npz')
        if aggregate['noise_count'] != count:
            raise ValueError('Incomplete solver validation')
        for i in range(count):
            checks.append(compare_noise(a.run_root/name/f'noise_{i:05d}.npz', old/f'noise_{i:05d}.npz', 1e-7))
    report = {'status': 'same_noise_solver_observable_checks_passed_NOT_physics_acceptance',
              'checks': checks, 'passed': True,
              'code_sha256': digest(Path(__file__)),
              'new_measure_sha256': checks[0]['left_measure_sha256'],
              'old_measure_sha256': checks[0]['right_measure_sha256'],
              'new_library_sha256': checks[0]['left_library_sha256'],
              'old_library_sha256': checks[0]['right_library_sha256'],
              'core_sha256': checks[0]['core_sha256']}
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
