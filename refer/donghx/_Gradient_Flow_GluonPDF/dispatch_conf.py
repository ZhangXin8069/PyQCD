"""Pack the immutable 80-worker plan into 50 configuration-level allocations."""
import argparse
import json
import os
import sys
import launch
from common import HERE, CONFS, load_manifest, sha256, atomic_json, require_allocation


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-index', type=int, required=True)
    args = parser.parse_args()
    frozen = json.loads((HERE / 'packed_dispatch_request.json').read_text())
    assert frozen['request_sha256'] == sha256(HERE / 'request.json')
    for name, digest in frozen['files_sha256'].items():
        assert sha256(HERE / name) == digest, ('packed dispatch source changed', name)
    m = load_manifest()
    if not 0 <= args.conf_index < len(CONFS):
        raise ValueError(args.conf_index)
    conf = CONFS[args.conf_index]
    workers = [w for w in m['workers'] if w['conf'] == conf]
    assert len(workers) == (1 if args.conf_index < 20 else 2)
    state = dict(status='RUNNING', conf=conf, conf_index=args.conf_index,
                 workers=[w['worker'] for w in workers], job=os.environ['SLURM_JOB_ID'],
                 packed_dispatch_sha256=sha256(HERE / 'packed_dispatch_request.json'))
    path = HERE / 'status' / f'conf{args.conf_index}.json'
    try:
        for worker in workers:
            state['current_worker'] = worker['worker']
            atomic_json(path, state)
            sys.argv = [str(HERE / 'launch.py'), '--worker', str(worker['worker'])]
            launch.main()
        state['status'] = 'PASS'
        atomic_json(path, state)
        print('PZ0_CONFIGURATION_PACK_PASS', conf, state['workers'], flush=True)
    except BaseException as error:
        state.update(status='FAILED_OR_INTERRUPTED', error=str(error))
        atomic_json(path, state)
        raise


if __name__ == '__main__':
    main()
