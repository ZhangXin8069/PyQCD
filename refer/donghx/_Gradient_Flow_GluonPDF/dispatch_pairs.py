"""Two conf-flow payloads per GPU job to respect AssocMaxSubmitJobLimit=200."""
import argparse
import json
import os
import subprocess
import sys
from common import HERE, sha256, manifest_hash, load_manifest, atomic_json, require_allocation


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.worker < 120:
        raise ValueError(args.worker)
    dispatch = json.loads((HERE/'dispatch_request.json').read_text())
    assert dispatch['manifest_sha256'] == manifest_hash()
    for name, digest in dispatch['files_sha256'].items():
        if sha256(HERE/name) != digest:
            raise RuntimeError(('submitted dispatcher changed', name))
    tasks = dispatch['workers'][args.worker]['tasks']
    manifest = load_manifest()
    for task in tasks:
        print('PAIR_PRODUCTION_START', args.worker, task, manifest['tasks'][task]['conf'],
              manifest['tasks'][task]['flow_time_a2'], flush=True)
        subprocess.run([sys.executable, '-B', '-u', str(HERE/'launch.py'), '--task', str(task)], check=True)
    atomic_json(HERE/'status'/f'worker{args.worker}.json', dict(status='PASS', worker=args.worker,
        tasks=tasks, manifest_sha256=manifest_hash(), job=os.environ['SLURM_JOB_ID']))
    print('PAIR_PRODUCTION_PASS', args.worker, tasks, flush=True)


if __name__ == '__main__':
    main()
