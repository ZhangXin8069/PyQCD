"""Read-only production inventory on a CPU allocation; no science changes."""
import argparse
import collections
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
L64 = WORK / 'quark_l32x64_B_conf51to100_P0456_retry_20261030'
L96 = WORK / 'quark_l32x96_B_expand50_matched_radius_20261028'
W96 = WORK / 'wilson_L32x96_matchedrf_20261101/runs/allconf_2026-11-01T154534.916707_0800'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    path = Path(path)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs_guard(module, manifest):
    # Check each distinct immutable input once, including NPY header/size checks.
    expected = {}
    for row in manifest['tasks']:
        for record in row['inputs']:
            old = expected.setdefault(record['path'], record)
            assert old == record, ('inconsistent input identities', record['path'])
    for index, (path, record) in enumerate(expected.items()):
        assert module.fingerprint(path) == record, ('input changed', path)
        if index % 500 == 0:
            print('INPUT_IDENTITY_PASS', module.HERE.name, index, len(expected), flush=True)
    return len(expected)


def quark_rows(module, manifest):
    digest = sha(module.HERE / 'manifest.json')
    records = []
    for row in manifest['tasks']:
        marker_path = module.HERE / 'status' / f'task{row["task"]}.json'
        marker = read(marker_path) if marker_path.exists() else None
        good = (marker is not None and marker.get('status') == 'PASS'
                and marker.get('case_count') == 252
                and marker.get('manifest_sha256') == digest
                and not marker.get('smoke', False))
        if marker is not None and not good:
            raise RuntimeError(('non-production or invalid task marker', str(marker_path)))
        if good:
            # Probe an algebra-oracle case and the last saved insertion. The
            # later original full audit still reads ALL cases and payloads.
            for tj, sep, dt in [(module.CURRENTS[0], 7, 3), (module.CURRENTS[-1], 12, 12)]:
                module.read_case(row, tj, sep, dt, verify_hash=True)
        records.append(dict(task=row['task'], conf=row['conf'], tau=row['flow_time_a2'],
                            committed=good, marker=str(marker_path),
                            marker_sha256=sha(marker_path) if good else None))
    return records


def l64():
    sys.path.insert(0, str(L64))
    plan = load('recovery_l64_plan', L64 / 'plan_common.py')
    plan.check_frozen()
    lock = read(L64 / 'scheduler_v2_lock.json')
    for name, digest in lock['files_sha256'].items():
        assert sha(L64 / name) == digest, name
    moving = plan.load_module('recovery_l64_moving', plan.MOVING)
    p0 = plan.load_module('recovery_l64_p0', plan.P0)
    mm, pm = moving.load_manifest(), p0.load_manifest()
    n_inputs = inputs_guard(moving, mm) + inputs_guard(p0, pm)
    a, b = quark_rows(moving, mm), quark_rows(p0, pm)
    retry, worker_records = [], []
    for worker in pm['workers']:
        tasks = worker['tasks']
        complete = all(a[t]['committed'] and b[t]['committed'] for t in tasks)
        marker_path = L64 / 'status' / f'worker{worker["worker"]}.json'
        worker_pass = marker_path.exists() and read(marker_path).get('status') == 'PASS'
        complete = complete and worker_pass
        if not complete:
            retry.append(worker['worker'])
        worker_records.append(dict(worker=worker['worker'], conf=worker['conf'],
            complete=complete, tasks=tasks,
            missing_moving=[t for t in tasks if not a[t]['committed']],
            missing_p0=[t for t in tasks if not b[t]['committed']]))
    confs = sorted({r['conf'] for r in a}, key=int)
    return dict(kind='l64', nconf_expected=len(confs), nflow=8,
        conf_flow_expected=400, moving_committed=sum(r['committed'] for r in a),
        p0_committed=sum(r['committed'] for r in b),
        fully_complete_confs=[c for c in confs if all(r['committed'] for r in a+b if r['conf']==c)],
        input_identity_checks=n_inputs, retry_workers=retry, workers=worker_records,
        moving_tasks=a, p0_tasks=b, partition='na100-ins', reservation='donghx_250',
        excluded_nodes=['gpu025', 'gpu028'],
        script=str(L64 / 'run_na100_ins_v2.slurm'),
        full_audit_script=str(L64 / 'cpu.slurm'), full_audit_arguments=['audit'])


def l96():
    sys.path.insert(0, str(L96))
    module = load('recovery_l96_common', L96 / 'common.py')
    manifest = module.load_manifest()
    n_inputs = inputs_guard(module, manifest)
    records = quark_rows(module, manifest)
    retry, workers = [], []
    for worker, conf in enumerate(module.CONFS):
        tasks = list(range(worker*7, worker*7+7))
        marker_path = L96 / 'status' / f'worker{worker}.json'
        good = all(records[t]['committed'] for t in tasks)
        good = good and marker_path.exists() and read(marker_path).get('status') == 'PASS'
        if not good:
            retry.append(worker)
        workers.append(dict(worker=worker, conf=conf, complete=good, tasks=tasks,
                            missing_tasks=[t for t in tasks if not records[t]['committed']]))
    return dict(kind='l96', nconf_expected=30, nflow=7, conf_flow_expected=210,
        committed=sum(r['committed'] for r in records),
        fully_complete_confs=[w['conf'] for w in workers if w['complete']],
        input_identity_checks=n_inputs, retry_workers=retry, workers=workers, tasks=records,
        partition='na800-sug', reservation='donghx_223',
        script=str(L96 / 'run_na800_sug.slurm'),
        full_audit_script=str(L96 / 'audit_cpu.slurm'), full_audit_arguments=[])


def w96():
    sys.path.insert(0, str(W96 / 'code'))
    contracts = load('contracts', W96 / 'code/contracts.py')
    manifest = contracts.load_manifest(W96)
    production_gate = read(W96 / 'validation/production_gate.json')
    digest = sha(W96 / 'manifest.json')
    assert production_gate['status'] == 'PASS' and production_gate['manifest_sha256'] == digest
    records = []
    for task in manifest['tasks']:
        contracts.gauge_header(task)
        directory = W96 / task['output']
        completion, validation = directory / 'complete.json', directory / 'validation.json'
        complete = completion.exists() and validation.exists()
        if complete:
            report = read(validation)
            assert report['status'] == 'PASS' and report['manifest_sha256'] == digest
            # Check frozen payload hashes. This is a file-integrity check, not
            # a substitute for the later full numerical measurement audit.
            for relative, expected in report['product_sha256'].items():
                assert sha(directory / relative) == expected, str(directory / relative)
        records.append(dict(conf=task['conf_id'], committed=complete,
            measurement_complete_marker=completion.exists(), directory=str(directory)))
    retry = [i for i, group in enumerate(manifest['worker_groups'])
             if any(not records[t]['committed'] for t in group)]
    return dict(kind='w96', nconf_expected=manifest['Ncfg'],
        committed=sum(r['committed'] for r in records),
        complete_marker_count=sum(r['measurement_complete_marker'] for r in records),
        input_header_checks=len(records), retry_workers=retry, tasks=records,
        partition='a192c1t,cpu6248R,cpueicc,i72c512g', root=str(W96),
        script=str(W96 / 'slurm/worker_cpu.slurm'),
        full_audit_script=str(W96 / 'slurm/audit_cpu.slurm'), full_audit_arguments=[])


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm allocation required'
    assert os.environ.get('SLURM_JOB_PARTITION') in ('a192c1t','cpu6248R','cpueicc','i72c512g')
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['l64', 'l96', 'w96'], required=True)
    args = parser.parse_args()
    lock = read(ROOT / 'source_lock.json')
    for path, digest in lock.items():
        assert sha(path) == digest, path
    report = {'l64':l64, 'l96':l96, 'w96':w96}[args.kind]()
    report.update(status='PASS_INVENTORY', generated=datetime.now().astimezone().isoformat(),
        cpu_job=os.environ['SLURM_JOB_ID'], node=os.environ.get('SLURMD_NODENAME'),
        verification_scope='immutable code/input identities; committed marker checks; quark two-case payload probes per committed flow; Wilson committed-file hashes; full numerical audit deferred to original audit scripts')
    dump(ROOT / f'inventory_{args.kind}.json', report)
    print('INVENTORY_PASS', args.kind, 'retry_workers', report['retry_workers'], flush=True)


if __name__ == '__main__':
    main()
