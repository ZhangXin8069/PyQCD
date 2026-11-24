"""Resume exactly one original conf-flow task using unchanged production kernels."""
import argparse
import os
import subprocess
import sys
import traceback

from resume_common import (HERE, PRODUCTION, atomic_json, load_plan, marker_snapshot,
                           now, read_json, require_allocation, sha256,
                           verify_production)


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--array-index', type=int)
    mode.add_argument('--final-audit', action='store_true')
    args = parser.parse_args()
    plan = load_plan()
    _, manifest = verify_production()
    gate = read_json(HERE / 'status/preflight_complete.json')
    if (gate['status'] != 'PASS_EXISTING_RESULTS_VALIDATED_GPU_RESUME_ALLOWED'
            or gate['resume_plan_sha256'] != sha256(HERE / 'resume_plan.json')
            or gate['manifest_sha256'] != plan['manifest_sha256']
            or gate['completed_tasks_validated'] != len(plan['completed_marker_tasks'])):
        raise RuntimeError('Missing or incompatible completed-results recovery gate')
    if args.final_audit:
        subprocess.run([sys.executable, '-B', '-u', str(PRODUCTION / 'audit_cpu.py')],
                       check=True)
        summary = read_json(PRODUCTION / 'summary/complete.json')
        if (summary['status'] != 'COMPLETE_L32X96_B_MATCHED_RADIUS_C3_C2'
                or summary['tasks'] != 210 or summary['case_count'] != 52920
                or summary['manifest_sha256'] != plan['manifest_sha256']):
            raise RuntimeError('Final production audit identity mismatch')
        atomic_json(HERE / 'status/complete.json', dict(
            status='COMPLETE_PREVIOUS20_PLUS_EXPANSION30', generated=now(),
            job=os.environ['SLURM_JOB_ID'], combined_nconf=50,
            resume_plan_sha256=sha256(HERE / 'resume_plan.json'),
            expansion_summary=str(PRODUCTION / 'summary/complete.json'),
            expansion_summary_sha256=sha256(PRODUCTION / 'summary/complete.json'),
            previous20_complete_sha256=plan['previous20_complete_sha256']))
        return
    if not 0 <= args.array_index < len(plan['missing_tasks']):
        parser.error('Array index is outside the immutable missing-task mapping')
    task = plan['missing_tasks'][args.array_index]
    row = manifest['tasks'][task]
    target = HERE / 'status' / f'resumed_task{task}.json'
    result = dict(status='RUNNING', generated=now(), job=os.environ['SLURM_JOB_ID'],
                  array_index=args.array_index, task=task, conf=row['conf'],
                  flow_time_a2=row['flow_time_a2'], matched_tau64=row['tau64'],
                  resume_plan_sha256=sha256(HERE / 'resume_plan.json'))
    atomic_json(target, result)
    print('RESUME_ORIGINAL_TASK', task, row['conf'], row['flow_time_a2'], flush=True)
    try:
        subprocess.run([sys.executable, '-B', '-u', str(PRODUCTION / 'launch.py'),
                        '--task', str(task)], check=True)
        marker_digest = marker_snapshot({'tasks': [row]}, plan['manifest_sha256'])
        if str(task) not in marker_digest:
            raise RuntimeError(('Original task did not publish a valid PASS marker', task))
        result.update(status='PASS', completed=now(),
                      production_marker_sha256=marker_digest[str(task)])
    except BaseException as error:
        result.update(status='FAILED', failed=now(), error=repr(error),
                      traceback=traceback.format_exc())
        atomic_json(target, result)
        raise
    atomic_json(target, result)
    print('RESUMED_ORIGINAL_TASK_PASS', task, flush=True)


if __name__ == '__main__':
    main()
