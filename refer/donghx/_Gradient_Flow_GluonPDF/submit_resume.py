"""Recover frozen production with metadata-only planning on the login node."""
import argparse
import csv
import os
import subprocess

from resume_common import (HERE, PRODUCTION, NSHARDS, NWORKERS, atomic_json, now,
                           marker_snapshot, sha256, verify_production)

SOURCE_FILES = ('resume_common.py', 'preflight_cpu.py', 'performance_cpu.py',
                'submit_resume.py', 'run_cpu.slurm', 'README.md')


def build_plan():
    request, manifest = verify_production()
    digest = sha256(PRODUCTION / 'manifest.json')
    completed = marker_snapshot(manifest, digest)
    missing = [row['task'] for row in manifest['tasks'] if str(row['task']) not in completed]
    if not missing:
        raise RuntimeError('No missing tasks; run the final CPU audit instead')
    plan = dict(schema='L24x72_B_cancelled_production_resume_20261105', generated=now(),
                production_root=str(PRODUCTION), production_request_sha256=sha256(PRODUCTION / 'request.json'),
                manifest_sha256=digest, completed_marker_tasks=sorted(map(int, completed)),
                completed_marker_sha256=completed, missing_tasks=missing,
                total_tasks=len(manifest['tasks']), nconf=len(request['confs']),
                matched_tau64=request['matched_tau64'], momenta_z=request['momenta_z'],
                tseps=request['tseps'], epsilon=request['epsilon'],
                preflight_cpu_shards=NSHARDS, workers=NWORKERS,
                resume_source_sha256={name: sha256(HERE / name) for name in SOURCE_FILES},
                resource=request['resource'], persistent_basis=True,
                persistent_vertex=False, persistent_VVV=False,
                numerical_kernel_changed=False,
                scope='Recover original 50 conf x 7 matched-radius tasks; no change to physics or numerical kernels')
    return plan, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    plan, manifest = build_plan()
    print('COMPLETE_MARKERS', len(plan['completed_marker_tasks']),
          'MISSING_CONF_FLOW_TASKS', len(plan['missing_tasks']), flush=True)
    if not args.submit:
        print('Metadata-only dry run; pass --submit to write the plan and queue recovery jobs.')
        return
    if (HERE / 'submission.json').exists() or (HERE / 'submission_attempt.json').exists():
        raise RuntimeError('Recovery was already submitted/attempted; inspect its receipt before submitting again')
    queue = subprocess.check_output(['squeue', '-u', os.environ.get('USER', 'donghx'),
                                     '-h', '-o', '%i|%j|%T'], text=True)
    active = [line for line in queue.splitlines() if any(
        name in line for name in ('q24_B50_flow', 'q24_B50_resume', 'q24_B50_smoke', 'q24_resume_'))]
    if active:
        raise RuntimeError(f'Existing L24 production/recovery jobs still active: {active}')
    lock = os.open(HERE / 'submission_attempt.json', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(lock, 'w') as stream:
        import json
        json.dump(dict(started=now(), scope=plan['scope']), stream, indent=2)
        stream.write('\n')
    atomic_json(HERE / 'resume_plan.json', plan)
    with (HERE / 'missing_tasks.csv').open('w', newline='') as stream:
        fields = ('task', 'worker', 'conf', 'tau24', 'matched_tau64', 'radius_fm', 'window', 'nstep')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in plan['missing_tasks']:
            row = manifest['tasks'][task]
            writer.writerow(dict(task=task, worker=task % NWORKERS, conf=row['conf'],
                                 tau24=row['flow_time_a2'], matched_tau64=row['tau64'],
                                 radius_fm=row['radius_fm'], window=row['window'], nstep=row['nstep']))
    receipt = dict(status='SUBMITTING', generated=now(), resume_plan_sha256=sha256(HERE / 'resume_plan.json'),
                   manifest_sha256=plan['manifest_sha256'], jobs={}, commands={})

    def submit(stage, *arguments):
        command = ['sbatch', '--parsable', *map(str, arguments)]
        result = subprocess.check_output(command, text=True).strip()
        job = result.split(';')[0]
        if not job.isdecimal():
            raise RuntimeError(f'Unexpected sbatch response: {result!r}')
        receipt['jobs'][stage] = job
        receipt['commands'][stage] = command
        atomic_json(HERE / 'submission.json', receipt)
        print(stage, job, flush=True)
        return job

    def cpu_arguments(stage, dependency=None, array=None):
        options = [f'--job-name=q24_resume_{stage}',
                   f'--output={HERE}/logs/{stage}_%A_%a.out',
                   f'--error={HERE}/logs/{stage}_%A_%a.err']
        if dependency:
            options.append(f'--dependency=afterok:{dependency}')
        if array:
            options.append(f'--array={array}')
        return [*options, HERE / 'run_cpu.slurm', stage]

    try:
        preflight = submit('preflight', *cpu_arguments('preflight', array=f'0-{NSHARDS - 1}'))
        gate = submit('gate', *cpu_arguments('gate', dependency=preflight))
        gpu = submit('gpu', '--job-name=q24_B50_resume', f'--array=0-{NWORKERS - 1}',
                     f'--dependency=afterok:{gate}', '--open-mode=append',
                     f'--output={HERE}/logs/gpu_%A_%a.out', f'--error={HERE}/logs/gpu_%A_%a.err',
                     PRODUCTION / 'run_na100_ins.slurm')
        submit('final', *cpu_arguments('final', dependency=gpu))
        submit('performance', *cpu_arguments('performance'))
    except BaseException as error:
        receipt.update(status='PARTIAL_SUBMISSION_FAILED', error=repr(error), updated=now())
        atomic_json(HERE / 'submission.json', receipt)
        raise
    receipt.update(status='SUBMITTED_REQUIRES_RUNTIME_VALIDATION', updated=now(),
                   missing_task_count=len(plan['missing_tasks']),
                   completed_markers_pending_cpu_validation=len(plan['completed_marker_tasks']),
                   note='GPU workers resume one pending conf-flow at a time and requeue the SAME array element on success; no array concurrency cap')
    atomic_json(HERE / 'submission.json', receipt)


if __name__ == '__main__':
    main()
