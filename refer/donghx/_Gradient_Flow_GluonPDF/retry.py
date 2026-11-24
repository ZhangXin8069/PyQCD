"""Scheduler-only successor; execute the immutable, already-tested P0 kernels."""
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'quark_l32x64_B_Pz0_fill50_8flow_20261029'
OLD_ARRAY = '3881711'
OLD_AUDIT = '3881712'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def timestamp():
    return datetime.now().astimezone().isoformat()


def check_frozen():
    plan = json.loads((HERE/'retry_manifest.json').read_text())
    for path, expected in plan['files_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError(('Frozen retry or original source changed', path))
    return plan


def original_modules():
    sys.path.insert(0, str(ORIGINAL))
    import common
    assert common.HERE == ORIGINAL
    return common


def gate():
    plan = check_frozen(); c = original_modules(); m = c.load_manifest()
    oldgate = json.loads((ORIGINAL/'status/cpu_gate.json').read_text())
    tests = json.loads((ORIGINAL/'status/tests.json').read_text())
    assert oldgate['status'] == tests['status'] == 'PASS'
    assert oldgate['manifest_sha256'] == sha(ORIGINAL/'manifest.json')
    assert tuple(m['confs']) == c.CONFS and tuple(m['flows']) == c.FLOWS
    assert len(c.CONFS) == 50 and len(m['tasks']) == 260 and len(m['reused_tasks']) == 140
    assert plan['array_conf_indices'] == list(range(50))
    assert m['weight_scheme'] == c.SCHEME and m['momenta_z'] == [0]
    records = []; new_verified = 0
    for row in m['tasks']:
        c.check_inputs(row)
        count = 0
        for tj in c.CURRENTS:
            for sep in c.TSEPS:
                for dt in range(sep+1):
                    if (c.output_dir(row, tj, sep, dt)/'receipt.json').exists():
                        c.read_case(row, tj, sep, dt, verify_hash=True)
                        count += 1
        new_verified += count
        records.append(dict(conf=row['conf'], tau=row['flow_time_a2'], task=row['task'],
                            verified_existing_cases=count, pending_cases=252-count))
        print('RETRY_INPUT_PASS', row['conf'], row['flow_time_a2'], 'pending', 252-count, flush=True)
    for row in m['reused_tasks']:
        marker = json.loads((c.result_root(row)/'status'/f'task{row["task"]}.json').read_text())
        assert marker['status'] == 'PASS' and marker['case_count'] == 252 and not marker['smoke']
        assert marker['manifest_sha256'] == c.result_manifest_hash(row)
    with (HERE/'missing_cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    check_frozen()
    write(HERE/'gate_complete.json', dict(status='PASS_RETRY_FROZEN_INPUTS_AND_MISSING_CASES',
        timestamp=timestamp(), job=os.environ['SLURM_JOB_ID'], original_manifest_sha256=sha(ORIGINAL/'manifest.json'),
        nconf=50, conf_flow_targets=400, reused_conf_flows=140, new_conf_flows=260,
        reused_cases=35280, existing_new_cases_verified=new_verified, pending_new_cases=65520-new_verified,
        kernel_changes=False, original_cpu_algebra_tests=tests['status'], eligible_nodes='gpu[026-027,029]',
        exclude_nodes='gpu025,gpu028'))
    print('P0_RETRY_GATE_PASS', '50 conf', 'pending cases', 65520-new_verified, flush=True)


def run(conf_index):
    plan = check_frozen()
    assert json.loads((HERE/'gate_complete.json').read_text())['status'] == 'PASS_RETRY_FROZEN_INPUTS_AND_MISSING_CASES'
    if os.environ.get('SLURMD_NODENAME') not in ('gpu026', 'gpu027', 'gpu029'):
        raise RuntimeError(('Allocated outside explicitly allowed nodes', os.environ.get('SLURMD_NODENAME')))
    assert conf_index in plan['array_conf_indices']
    original_modules()
    import dispatch_conf
    path = HERE/'status'/f'conf{conf_index}.json'
    state = dict(status='RUNNING', timestamp=timestamp(), conf_index=conf_index,
        conf=plan['confs'][conf_index], job=os.environ['SLURM_JOB_ID'], node=os.environ['SLURMD_NODENAME'],
        result_root=str(ORIGINAL), scheduler_only_change=True)
    write(path, state)
    try:
        sys.argv = [str(ORIGINAL/'dispatch_conf.py'), '--conf-index', str(conf_index)]
        dispatch_conf.main()
        state.update(status='PASS', timestamp=timestamp())
        write(path, state)
        print('P0_RETRY_CONFIGURATION_PASS', conf_index, state['conf'], flush=True)
    except BaseException as error:
        state.update(status='FAILED_OR_INTERRUPTED', error=str(error), timestamp=timestamp())
        write(path, state)
        raise


def audit():
    check_frozen()
    subprocess.run([sys.executable, '-B', '-u', str(ORIGINAL/'audit_cpu.py')], cwd=str(ORIGINAL), check=True)
    result_path = ORIGINAL/'summary/complete.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'PASS_50CONF_8FLOW_B_PZ0' and result['case_count'] == 100800
    check_frozen()
    write(HERE/'complete.json', dict(status='PASS_RETRY_FIRST50_P0_8FLOW', timestamp=timestamp(),
        job=os.environ['SLURM_JOB_ID'], original_complete=str(result_path), original_complete_sha256=sha(result_path),
        nconf=50, confs=result['confs'], flows=result['flows'], case_count=100800,
        exclude_nodes='gpu025,gpu028', kernel_changes=False))
    print('P0_RETRY_FINAL_AUDIT_PASS', flush=True)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('gate', 'run', 'audit'))
    parser.add_argument('--conf-index', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All input checks, numerical work and production require compute allocation')
    if args.stage != 'run' and os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ('', '-1', 'NoDevFiles'):
        raise RuntimeError('CPU-only audit/gate')
    dict(gate=gate, run=lambda: run(args.conf_index), audit=audit)[args.stage]()


if __name__ == '__main__':
    cli()
