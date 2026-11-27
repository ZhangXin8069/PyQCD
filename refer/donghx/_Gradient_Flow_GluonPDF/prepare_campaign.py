"""Freeze the next 50 configurations and build the checked same-basis P0 manifest."""
import argparse
import ast
import csv
from datetime import datetime
import subprocess

from plan_common import *

OLD_MOVING = HERE.parent / 'quark_l32x64_B_expand100_P456_20261028'
OLD_COMBINED = HERE.parent / 'quark_l32x64_B_conf51to100_P0456_retry_20261030'


def source_hashes(root):
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.slurm', '.sh')}


def freeze():
    assert not (HERE / 'request.json').exists(), 'already frozen'
    current_moving = load_module('new_moving', MOVING)
    previous_request = json.loads((OLD_MOVING / 'request.json').read_text())
    predecessor = json.loads((OLD_COMBINED / 'complete.json').read_text())
    assert predecessor['status'] == 'PASS_CONF51_TO_100_B_P0_P456_8FLOW'
    assert predecessor['confs'] == [str(conf) for conf in range(12000, 14000, 40)]
    for root, source, changes in ((MOVING, OLD_MOVING, ('common.py', 'prepare.py')),
                                   (P0, OLD_COMBINED / 'pz0', ('common.py',))):
        for path, digest in source_hashes(root).items():
            if path not in changes:
                assert sha(source / path) == digest, path
    for script in HERE.rglob('*.py'):
        ast.parse(script.read_text(), filename=str(script))
    for script in (*HERE.glob('*.slurm'), *HERE.glob('*.sh')):
        subprocess.run(['bash', '-n', str(script)], check=True)
    rows = []
    assert current_moving.CONFS == CONFS and current_moving.WINDOWS == WINDOWS
    for conf in CONFS:
        for flow, window in zip(FLOWS, WINDOWS):
            row = dict(task=len(rows), conf=conf, flow_time_a2=flow, epsilon=.01,
                       nstep=round(flow/.01), window=window)
            row['tag'] = current_moving.config_for(row).tag
            rows.append(row)
    workers = [dict(worker=start//4, conf=rows[start]['conf'], tasks=list(range(start, start+4)))
               for start in range(0, len(rows), 4)]
    assert len(rows) == 400 and len(workers) == 100
    assert all(len({rows[index]['conf'] for index in worker['tasks']}) == 1 for worker in workers)
    with (HERE / 'task_manifest.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    upstream = {str(path): sha(path) for path in (OLD_MOVING/'request.json', OLD_COMBINED/'complete.json',
                                                  Path(previous_request['prior_manifest']))}
    request = dict(generated=datetime.now().astimezone().isoformat(), confs=list(CONFS), flows=list(FLOWS),
                   windows=list(WINDOWS), epsilon=.01, prior_confs=list(map(str, range(10000, 14000, 40))),
                   prior_manifest=previous_request['prior_manifest'], upstream_sha256=upstream,
                   workers=workers, tasks=rows, expected_case_count=100800)
    atomic_json(MOVING/'request.json', dict(**request, files_sha256=source_hashes(MOVING)))
    atomic_json(P0/'request.json', dict(**request, files_sha256=source_hashes(P0)))
    files = source_hashes(HERE)
    for path in ('moving/request.json', 'pz0/request.json', 'task_manifest.csv'):
        files[path] = sha(HERE/path)
    atomic_json(HERE/'request.json', dict(**request, files_sha256=files,
                momenta=[0, 4, -4, 5, -5, 6, -6], moving_and_P0_cases=201600))
    print('FROZEN_EXPANSION', CONFS[0], CONFS[-1], len(CONFS), len(rows), len(workers), flush=True)


def prepare():
    allocation()
    request = check_frozen()
    assert not (HERE/'status/gate_complete.json').exists(), 'input gate already completed'
    if not (MOVING/'manifest.json').exists():
        subprocess.run([sys.executable, '-B', '-u', str(MOVING/'prepare.py')], cwd=MOVING, check=True)
    moving = load_module('new_moving', MOVING)
    manifest = moving.load_manifest()
    assert manifest['tasks'] == request['tasks']
    assert tuple(manifest['confs']) == CONFS and tuple(manifest['momenta']) == (4, -4, 5, -5, 6, -6)
    moving_digest = moving.manifest_hash()
    pz0 = load_module('new_pz0', P0)
    subprocess.run([sys.executable, '-B', '-u', str(P0/'test_cpu.py')], cwd=P0, check=True)
    tasks = []
    for row in manifest['tasks']:
        basis = {str(sigma): dict(schema=manifest['basis_schema'], manifest_sha256=moving_digest)
                 for sigma in moving.source_times(row)}
        tasks.append(dict(row, base_window=row['window'], basis_work_root=str(MOVING/'basis_cache'),
                          gauge_work_root=str(MOVING/'basis_cache'), upstream_task=row['task'],
                          basis_source_root=str(MOVING), basis_metadata=basis,
                          inputs=manifest['configuration_inputs'][row['conf']],
                          original_gauge_history='thin synchronized 4D gauge+fermion flow'))
    atomic_json(P0/'manifest.json', dict(schema=pz0.SCHEMA, profile=pz0.PROFILE,
                confs=list(CONFS), flows=list(FLOWS), windows=list(WINDOWS), currents=list(CURRENTS),
                tseps=list(TSEPS), momenta_z=[0], channels=list(pz0.CHANNELS), tasks=tasks,
                reused_tasks=[], workers=request['workers'], request_sha256=sha(P0/'request.json'),
                epsilon=.01, a_fm=.0897, weight_scheme=pz0.SCHEME, weight_applied_count=1,
                expected_new_case_count=100800, expected_combined_case_count=100800,
                algebra_relative_l2_gate=2e-10, C2_cross_flow_relative_l2_gate=2e-10,
                basis_generation=False, persistent_vertex_cache=False, persistent_VVV_cache=False))
    pz0_digest = sha(P0/'manifest.json')
    assert len(pz0.load_manifest()['tasks']) == len(manifest['tasks'])
    atomic_json(P0/'status/cpu_gate.json', dict(status='PASS', manifest_sha256=pz0_digest, job=os.environ['SLURM_JOB_ID']))
    atomic_json(HERE/'status/gate_complete.json', dict(status='PASS', confs=list(CONFS), moving_manifest_sha256=moving_digest,
                P0_manifest_sha256=pz0_digest, job=os.environ['SLURM_JOB_ID']))
    print('PASS_INPUTS_AND_P0_CONTRACT', len(CONFS), 'configs,', len(tasks), 'conf-flow tasks per momentum branch', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'prepare'))
    args = parser.parse_args()
    (freeze if args.phase == 'freeze' else prepare)()
