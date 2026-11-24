"""Resume a partial submission without changing any input-gate frozen source."""
import argparse
import ast
from datetime import datetime
import json
import subprocess
from common import HERE, CONFS, load_request, sha256, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    request = load_request()
    old = json.loads((HERE / 'submission.json').read_text())
    assert old['status'] == 'PARTIAL_SUBMISSION_INSPECT_RECORDED_JOBS'
    assert set(old['jobs']) == {'gate'}, 'Do not duplicate a successfully submitted GPU array'
    gate = old['jobs']['gate']['id']
    assert gate.isdecimal()
    names = ('dispatch_conf.py', 'run_na100_ins_conf.slurm', 'submit_packed.py')
    for name in names:
        path = HERE / name
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        else:
            subprocess.run(['bash', '-n', str(path)], check=True)
    worker_ids = [w['worker'] for c in CONFS for w in request['workers'] if w['conf'] == c]
    assert sorted(worker_ids) == list(range(80)) and len(worker_ids) == 80
    if not args.submit:
        print('PLAN: reuse gate', gate, '; 50 configuration-level GPU jobs cover all 260 missing conf-flow tasks')
        return
    if (HERE / 'submission_packed.json').exists() or (HERE / 'packed_dispatch_request.json').exists():
        raise RuntimeError('Packed submission receipt exists; inspect before retrying')
    atomic_json(HERE / 'packed_dispatch_request.json', dict(
        schema='L32x64_B_Pz0_first50_packed_dispatch_v1',
        request_sha256=sha256(HERE / 'request.json'),
        files_sha256={n: sha256(HERE / n) for n in names},
        confs=list(CONFS), array='0-49', actual_GPU_job_count=50,
        original_worker_count=80, conf_flow_task_count=260,
        reason='account MaxSubmitJobs=200; pack by conf, do not cancel or throttle existing jobs'))
    record = dict(status='SUBMITTING', timestamp=datetime.now().astimezone().isoformat(),
                  original_submission_sha256=sha256(HERE / 'submission.json'),
                  packed_dispatch_sha256=sha256(HERE / 'packed_dispatch_request.json'),
                  jobs={'gate': old['jobs']['gate']}, array='0-49', new_conf_flow_count=260,
                  new_case_count=65520, resource=dict(request['resource'], array='0-49'))
    receipt = HERE / 'submission_packed.json'
    atomic_json(receipt, record)

    def submit(name, script, dependency):
        command = ['sbatch', '--parsable', f'--chdir={HERE}',
                   f'--output={HERE}/logs/{name}_%A_%a.out',
                   f'--error={HERE}/logs/{name}_%A_%a.err', f'--dependency=afterok:{dependency}', str(HERE / script)]
        answer = subprocess.check_output(command, text=True).strip()
        job = answer.split(';')[0]
        if not job.isdecimal():
            raise RuntimeError(answer)
        record['jobs'][name] = dict(id=job, command=command)
        atomic_json(receipt, record)
        print('SUBMITTED', name, job, flush=True)
        return job

    try:
        array = submit('production_conf', 'run_na100_ins_conf.slurm', gate)
        submit('audit', 'audit_cpu.slurm', array)
    except BaseException as error:
        record.update(status='PARTIAL_SUBMISSION_INSPECT_RECORDED_JOBS', error=str(error))
        atomic_json(receipt, record)
        raise
    record['status'] = 'SUBMITTED'
    atomic_json(receipt, record)


if __name__ == '__main__':
    main()
