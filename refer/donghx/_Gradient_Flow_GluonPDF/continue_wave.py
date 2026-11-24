"""Submit the next four-task wave after the preceding array terminates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import core as c

ROOT = Path(__file__).resolve().parent


def submit(arguments):
    # Retry only rejected submissions; accepted job IDs are written immediately.
    for attempt in range(3):
        process = subprocess.run(['sbatch', '--parsable', *arguments], text=True, capture_output=True)
        if process.returncode == 0:
            value = process.stdout.strip().split(';')[0]
            assert value.isdigit(), process.stdout
            return value
        if attempt < 2:
            time.sleep(10)
    raise RuntimeError(process.stderr)


def main(completed):
    assert os.environ.get('SLURM_JOB_ID')
    for name, digest in json.loads((ROOT/'source_lock.json').read_text())['sha256'].items():
        assert c.sha(ROOT/name) == digest, name
    if completed == 2:
        os.execv(sys.executable, [sys.executable, '-B', '-u', str(ROOT/'summary.py')])
    wave = completed+1
    receipt_path = ROOT/f'submission_wave{wave}.json'
    assert not receipt_path.exists(), 'never automatically resubmit an existing wave'
    array = f'{wave*4}-{wave*4+3}%1'
    receipt = dict(wave=wave, array=array, controller_job=os.environ['SLURM_JOB_ID'],
                   source_lock_sha256=c.sha(ROOT/'source_lock.json'), unix_time=time.time())
    try:
        gpu = submit(['--array='+array, str(ROOT/'validate.slurm')])
        receipt.update(gpu_array_job=gpu, status='GPU_SUBMITTED')
        c.write_json(receipt_path, receipt)
        following = submit(['--dependency=afterany:'+gpu, str(ROOT/'summary_cpu.slurm'), str(wave)])
        receipt.update(continuation_or_audit_job=following, status='SUBMITTED')
        c.write_json(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        receipt.update(status='SUBMISSION_FAILED', error=repr(error))
        c.write_json(receipt_path, receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-wave', type=int, choices=range(3), required=True)
    main(parser.parse_args().completed_wave)
