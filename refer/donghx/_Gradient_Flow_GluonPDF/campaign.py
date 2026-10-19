#!/usr/bin/env python3
"""Freeze an extension-only grid and audit the combined 30-conf metadata."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.with_name('quark_l32x64_flow_20261007')
CONFS = list(range(10000, 11200, 40))
NEW = CONFS[20:]
FLOWS = [(0.05, 2), (0.10, 2), (0.20, 3), (0.30, 3),
         (0.40, 4), (0.50, 4), (0.80, 6), (1.00, 7)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def old_auditor():
    spec = importlib.util.spec_from_file_location('old_l32x64_audit', OLD/'audit_missing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit():
    old = old_auditor()
    receipts = {}
    for source in (OLD, HERE):
        for path in (source/'status').glob('full_chain_*.json'):
            row = json.loads(path.read_text())
            if row.get('state') != 'COMPLETED':
                continue
            stages = {r['stage']: r['returncode'] for r in row.get('stages', [])}
            if stages.get('validate_outputs') != 0 or any(stages.values()):
                continue
            log = source/'logs'/f"{row['conf']}_{row['job']}"/'validate_outputs.log'
            if not log.is_file() or '[production-outputs-validate] PASS' not in log.read_text():
                continue
            key = (int(row['conf']), round(float(row['flow_time']), 2))
            receipts[key] = str(path)
    complete, missing = [], []
    for fi, (tau, window) in enumerate(FLOWS):
        for ci, conf in enumerate(CONFS):
            state = old.product_state(conf, tau, window, set(receipts))
            row = {'conf':conf, 'tau_a2':tau, 'window':window,
                   'new_array_task': fi*10+ci-20 if ci >= 20 else None}
            if state == 'complete':
                complete.append({**row, 'completion_receipt':receipts[conf,tau]})
            else:
                missing.append({**row, 'reason':state})
    return {'schema':'l32x64_quark_30conf_audit_v1',
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'metadata, file sizes and existing compute-node validation logs',
            'status':'PASS' if not missing else 'INCOMPLETE',
            'expected':240, 'complete_count':len(complete), 'missing_count':len(missing),
            'flow_summary':[{'tau_a2':tau, 'complete':sum(r['tau_a2']==tau for r in complete),
                             'expected':30} for tau,_ in FLOWS],
            'complete':complete, 'missing':missing,
            'missing_new_array_spec':','.join(str(r['new_array_task']) for r in missing
                                            if r['new_array_task'] is not None)}


def prepare():
    manifest_path = HERE/'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Existing frozen manifest: verify it instead of replacing it')
    sys.path.insert(0, str(HERE/'reference'))
    from flow_common_l32x64 import FlowConfig
    from validate_production_conf_l32x64 import validate_inputs
    # These are header-only diagnostic checks: no inversions or contractions.
    headers = []
    for conf in NEW:
        cfg = FlowConfig(conf=str(conf), flow_time=.05, epsilon=.01, temporal_window=2)
        headers.append(validate_inputs(cfg))
        print(f'INPUT_HEADERS_PASS conf={conf}', flush=True)
    baseline = audit()
    old_complete = sum(r['conf'] not in NEW for r in baseline['complete'])
    if old_complete != 160:
        raise RuntimeError(f'Original 20-conf baseline incomplete: {old_complete}/160')
    if any(r['conf'] in NEW for r in baseline['complete']):
        raise RuntimeError('New products already complete: generate missing-only array first')
    numerical_files = ['run_spin2.py','spin2_contractions.py','validate_and_prepare.py']
    numerical_files += [str(p.relative_to(HERE)) for p in sorted((HERE/'reference').glob('*.py'))]
    for name in numerical_files:
        if digest(HERE/name) != digest(OLD/name):
            raise RuntimeError(f'Numerical source differs from successful baseline: {name}')
    original = (OLD/'full_chain.py').read_text()
    successor = (HERE/'full_chain.py').read_text()
    if successor != original.replace('range(10000,10800,40)', 'range(10800,11200,40)'):
        raise RuntimeError('Unexpected full_chain change beyond configuration whitelist')
    task_rows = [{'task':fi*10+ci,'conf':conf,'tau_a2':tau,'window':window}
                 for fi,(tau,window) in enumerate(FLOWS) for ci,conf in enumerate(NEW)]
    frozen = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.slurm'))
    frozen += sorted((HERE/'reference').glob('*.py'))
    manifest = {'schema':'l32x64_quark_expand20_to30_v1',
                'created_utc':datetime.now(timezone.utc).isoformat(),
                'new_confs':NEW, 'all_confs':CONFS, 'tasks':task_rows,
                'epsilon':.01, 'a_fm':.0897, 'tseps':[5,6,7,8,9,10],
                'mom_labels':['Pp3_z','Pp4_z','Pp5_z'],
                'partition':'na800-sug','reservation':'donghx_223',
                'save_root':str(OLD/'results'),'flow_work_root':str(OLD/'flow_cache'),
                'cache_policy':'retain basis/gauge; vertex removed, VVV is node-local temporary',
                'input_headers':headers,'baseline_complete':old_complete,
                'numerical_sources_unchanged':True,
                'source_sha256':{str(p.relative_to(HERE)):digest(p) for p in frozen},
                'physics_scope':'ordinary flowed connected C2/C3; no ringed/SFTE/LaMET'}
    write_json(HERE/'validation'/'baseline_audit.json', baseline)
    write_json(manifest_path, manifest)
    print(f'PREPARED 80 new tasks; original baseline={old_complete}/160', flush=True)


def verify(task=None):
    manifest = json.loads((HERE/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if digest(HERE/name) != expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    if task is not None:
        row = manifest['tasks'][task]
        expected = (row['tau_a2'], .01, row['window'], manifest['save_root'],manifest['flow_work_root'])
        actual = (float(os.environ['FLOW_TIME']),float(os.environ['FLOW_EPSILON']),
                  int(os.environ['FLOW_TEMPORAL_WINDOW']),os.environ['SAVE_ROOT'],os.environ['FLOW_WORK_ROOT'])
        if actual != expected:
            raise RuntimeError(f'Task metadata mismatch: {actual!r} != {expected!r}')
        print('FROZEN_TASK_PASS ' + json.dumps(row), flush=True)
    else:
        print('FROZEN_SOURCES_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','verify','audit'])
    parser.add_argument('--task', type=int, choices=range(80))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'verify':
        verify(args.task)
    else:
        report = audit()
        if args.report:
            write_json(args.report, report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('complete','missing')},indent=2))
        return int(bool(report['missing']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
