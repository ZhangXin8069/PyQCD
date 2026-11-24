"""Four consecutive flows per job; 100 jobs cover 50 conf x 8 flows."""
import argparse
import json
import os
import subprocess
import sys
from common import HERE, load_manifest, manifest_hash, atomic_json, require_allocation


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    request = json.loads((HERE/'request.json').read_text())
    if not 0 <= args.worker < len(request['workers']):
        raise ValueError(args.worker)
    manifest = load_manifest()
    tasks = request['workers'][args.worker]['tasks']
    assert len(tasks) == 4
    assert len({manifest['tasks'][t]['conf'] for t in tasks}) == 1
    for task in tasks:
        row = manifest['tasks'][task]
        print('FOUR_FLOW_PRODUCTION_START', args.worker, task, row['conf'],
              row['flow_time_a2'], flush=True)
        subprocess.run([sys.executable, '-B', '-u', str(HERE/'launch.py'),
                        '--task', str(task)], check=True)
    atomic_json(HERE/'status'/f'worker{args.worker}.json', dict(status='PASS',
        worker=args.worker, tasks=tasks, manifest_sha256=manifest_hash(),
        job=os.environ['SLURM_JOB_ID']))
    print('FOUR_FLOW_PRODUCTION_PASS', args.worker, tasks, flush=True)


if __name__ == '__main__':
    main()
