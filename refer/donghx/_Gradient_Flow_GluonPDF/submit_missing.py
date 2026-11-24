"""Schedule only independently inventoried gaps using original frozen scripts."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def save(path, data):
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def submit(command):
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(';')[0]
    assert job.isdigit(), (result.stdout, result.stderr)
    print('SUBMITTED', job, ' '.join(command), flush=True)
    return {'job_id':job, 'command':command, 'scheduler_stderr':result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['l64','l96','w96'], required=True)
    parser.add_argument('--inventory-job', help='Queue behind the read-only CPU guard if still running')
    args = parser.parse_args()
    path = ROOT / f'submission_{args.kind}.json'
    report_path = ROOT / f'inventory_{args.kind}.json'
    if path.exists():
        raise RuntimeError(('Already has submission receipt: do not duplicate', str(path)))
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['status'] == 'PASS_INVENTORY' and report['kind'] == args.kind
        inventory_pending = False
    else:
        assert args.inventory_job, 'Inventory not complete: a CPU guard dependency is required'
        report_path = ROOT / f'plan_{args.kind}.json'
        report = json.loads(report_path.read_text())
        assert report['status']=='METADATA_ONLY_WAITING_CPU_INVENTORY' and report['kind']==args.kind
        assert report['cpu_job']==args.inventory_job
        inventory_pending = True
    workers = report['retry_workers']
    assert workers == sorted(set(workers)) and all(isinstance(i,int) and i>=0 for i in workers)
    if args.kind == 'l64':
        assert report['partition']=='na100-ins' and report['reservation']=='donghx_250'
        assert report['excluded_nodes']==['gpu025','gpu028']
    if args.kind == 'l96':
        assert report['partition']=='na800-sug' and report['reservation']=='donghx_223'
    # No concurrency limit: preserve the user's production scheduling policy.
    receipt = dict(kind=args.kind, state='PREPARED', created=datetime.now().astimezone().isoformat(),
        inventory_sha256=sha(report_path), inventory_cpu_job=report['cpu_job'],
        inventory_validation_pending_at_submission=inventory_pending,
        submitter_sha256=sha(__file__), source_script_sha256=sha(report['script']),
        audit_script_sha256=sha(report['full_audit_script']),
        retry_workers=workers, worker_count=len(workers), array_concurrency_cap=None,
        physics_or_kernel_changes=False, original_outputs_retained=True,
        partition=report['partition'], reservation=report.get('reservation'))
    save(path, receipt)
    environment = []
    if args.kind == 'w96':
        environment = [f'--chdir={report["root"]}',f'--export=ALL,WILSON_RUN_ROOT={report["root"]}']
    if workers:
        command = ['sbatch','--parsable',f'--job-name={args.kind}_gf_missing',
            '--array='+','.join(map(str,workers)),
            f'--output={ROOT}/logs/{args.kind}_production_%A_%a.out',
            f'--error={ROOT}/logs/{args.kind}_production_%A_%a.err']
        if inventory_pending:
            command.append('--dependency=afterok:'+args.inventory_job)
        command += environment + [report['script']]
        receipt['production'] = submit(command)
        receipt['state']='PRODUCTION_SUBMITTED'
        save(path, receipt)
    command = ['sbatch','--parsable',f'--job-name={args.kind}_gf_reaudit',
        f'--output={ROOT}/logs/{args.kind}_audit_%j.out',
        f'--error={ROOT}/logs/{args.kind}_audit_%j.err']
    if workers:
        command.append('--dependency=afterok:'+receipt['production']['job_id'])
    command += environment + [report['full_audit_script']] + report['full_audit_arguments']
    receipt['audit'] = submit(command)
    receipt['state']='SUBMITTED'
    save(path, receipt)


if __name__ == '__main__':
    main()
