"""Validate completed task markers against ALL published numerical cases on CPUs."""
import argparse
import csv
import os
import sys
import time
import traceback

from resume_common import (HERE, PRODUCTION, NSHARDS, atomic_json, load_plan, now,
                           read_json, require_allocation, sha256, verify_production)


def shard_check(shard, plan):
    import numpy as np
    sys.path.insert(0, str(PRODUCTION))
    import common as old

    started = time.monotonic()
    manifest = old.load_manifest()
    conf_index = {conf: index for index, conf in enumerate(manifest['confs'])}
    tasks = [manifest['tasks'][task] for task in plan['completed_marker_tasks']
             if conf_index[manifest['tasks'][task]['conf']] % NSHARDS == shard]
    seen_inputs, references, records = {}, {}, []
    max_c2_relative = 0.0
    for row in tasks:
        marker = PRODUCTION / 'status' / f'task{row["task"]}.json'
        if sha256(marker) != plan['completed_marker_sha256'][str(row['task'])]:
            raise RuntimeError(f'Completion marker changed during recovery: {marker}')
        for expected in row['inputs']:
            path = expected['path']
            if path in seen_inputs:
                if seen_inputs[path] != expected:
                    raise RuntimeError(f'Conflicting input identities: {path}')
                continue
            if old.fingerprint(path) != expected:
                raise RuntimeError(f'Input identity changed: {path}')
            seen_inputs[path] = expected
        cases = 0
        for tj in old.CURRENTS:
            for sep in old.TSEPS:
                for dt in range(sep + 1):
                    _, _, c2, _, receipt = old.read_case(row, tj, sep, dt, verify_hash=True)
                    key = (row['conf'], tj, sep, dt)
                    if key in references:
                        relative = float(np.linalg.norm(c2 - references[key]) /
                                         max(np.linalg.norm(references[key]), 1e-300))
                        if relative > 2e-10:
                            raise AssertionError(('C2 changed across flows', key, relative))
                        max_c2_relative = max(max_c2_relative, relative)
                    else:
                        references[key] = c2.copy()
                    if sep == 7 and dt == 3:
                        checks = receipt['real_H_and_Wick_regressions']
                        if len(checks) != 1 + 6 * len(old.MOMENTA):
                            raise RuntimeError(('Missing H/Wick regression', row['task'], tj))
                        if any(check['relative_L2'] > manifest['algebra_relative_l2_gate']
                               for check in checks):
                            raise RuntimeError(('Failed H/Wick regression', row['task'], tj))
                    cases += 1
        if cases != 252:
            raise AssertionError(cases)
        records.append(dict(task=row['task'], conf=row['conf'], tau24=row['flow_time_a2'],
                            matched_tau64=row['tau64'], verified_cases=cases, status='PASS'))
        print('RECOVERY_COMPLETED_TASK_VERIFIED', row['task'], row['conf'], cases, flush=True)
    with (HERE / 'status' / f'preflight_shard{shard}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'task', 'conf', 'tau24', 'matched_tau64', 'verified_cases', 'status'))
        writer.writeheader()
        writer.writerows(records)
    return dict(status='PASS', shard=shard, generated=now(), job=os.environ['SLURM_JOB_ID'],
                resume_plan_sha256=sha256(HERE / 'resume_plan.json'),
                manifest_sha256=plan['manifest_sha256'], task_results=records,
                verified_cases=sum(r['verified_cases'] for r in records),
                unique_input_identities_checked=len(seen_inputs),
                C2_cross_flow_relative_L2=max_c2_relative, seconds=time.monotonic() - started)


def gate(plan):
    request, manifest = verify_production()
    records, results = [], []
    for shard in range(NSHARDS):
        result = read_json(HERE / 'status' / f'preflight_shard{shard}.json')
        if (result['status'] != 'PASS' or result['shard'] != shard or
                result['resume_plan_sha256'] != sha256(HERE / 'resume_plan.json') or
                result['manifest_sha256'] != plan['manifest_sha256']):
            raise RuntimeError(('Bad recovery shard result', shard))
        records.extend(result['task_results'])
        results.append(result)
    if sorted(r['task'] for r in records) != plan['completed_marker_tasks']:
        raise RuntimeError('Recovery shards do not cover exactly the original completed markers')
    for task, expected in plan['completed_marker_sha256'].items():
        if sha256(PRODUCTION / 'status' / f'task{task}.json') != expected:
            raise RuntimeError(('Completion marker changed', task))
    for name in ('cpu_gate', 'epsilon_gate', 'smoke_gate'):
        prior = read_json(PRODUCTION / 'status' / f'{name}.json')
        if prior['status'] != 'PASS' or prior['manifest_sha256'] != plan['manifest_sha256']:
            raise RuntimeError(('Original validation gate no longer valid', name))
    if sha256(request['backend_library_path']) != request['backend_library_sha256']:
        raise RuntimeError('Private QUDA binary changed')
    return dict(status='PASS_EXISTING_RESULTS_VALIDATED_GPU_RESUME_ALLOWED', generated=now(),
                job=os.environ['SLURM_JOB_ID'], manifest_sha256=plan['manifest_sha256'],
                resume_plan_sha256=sha256(HERE / 'resume_plan.json'),
                completed_tasks_validated=len(records),
                verified_cases=sum(r['verified_cases'] for r in records),
                remaining_task_count=len(plan['missing_tasks']),
                maximum_C2_cross_flow_relative_L2=max(r['C2_cross_flow_relative_L2'] for r in results),
                scope='Existing PASS task markers fully verified; partial cases are checked by the frozen GPU worker before reuse')


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int)
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    if args.gate == (args.shard is not None):
        parser.error('Choose exactly one of --shard or --gate')
    if args.shard is not None and not 0 <= args.shard < NSHARDS:
        parser.error('Invalid shard')
    target = HERE / 'status' / ('preflight_complete.json' if args.gate
                              else f'preflight_shard{args.shard}.json')
    try:
        plan = load_plan()
        result = gate(plan) if args.gate else shard_check(args.shard, plan)
        atomic_json(target, result)
        print('RECOVERY_PREFLIGHT_PASS', target, flush=True)
    except BaseException as error:
        atomic_json(target, dict(status='FAILED', generated=now(), job=os.environ['SLURM_JOB_ID'],
                                 error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
