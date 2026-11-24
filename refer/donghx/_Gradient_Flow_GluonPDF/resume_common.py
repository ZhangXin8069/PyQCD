"""Metadata-only recovery helpers. Numerical work belongs in Slurm jobs."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCTION = HERE.parent / 'quark_l24x72_B_50conf_matched_radius_20261104_v2'
PYTHON = '/public/home/donghx/AnaConda/bin/python'
NSHARDS = 8
NWORKERS = 24


def now():
    return datetime.now().astimezone().isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def verify_production():
    """Check frozen source and manifest, without importing numpy or a GPU stack."""
    request = read_json(PRODUCTION / 'request.json')
    for name, expected in request['files_sha256'].items():
        if sha256(PRODUCTION / name) != expected:
            raise RuntimeError(f'Frozen production source changed: {name}')
    for path, expected in request['upstream_sha256'].items():
        if sha256(path) != expected:
            raise RuntimeError(f'Frozen upstream source changed: {path}')
    if sha256(PRODUCTION / 'status/input_probe.json') != request['input_probe_sha256']:
        raise RuntimeError('Input-probe identity changed')
    manifest = read_json(PRODUCTION / 'manifest.json')
    if manifest['request_sha256'] != sha256(PRODUCTION / 'request.json'):
        raise RuntimeError('Request/manifest mismatch')
    if len(manifest['tasks']) != 350 or request['tasks'] != 350:
        raise RuntimeError('Unexpected task inventory')
    if request['resource']['workers'] != NWORKERS:
        raise RuntimeError('Worker partition changed')
    for task, row in enumerate(manifest['tasks']):
        if row['task'] != task:
            raise RuntimeError('Noncanonical task numbering')
    return request, manifest


def marker_snapshot(manifest, manifest_digest):
    completed = {}
    for row in manifest['tasks']:
        path = PRODUCTION / 'status' / f'task{row["task"]}.json'
        if not path.exists():
            continue
        marker = read_json(path)
        expected = (row['task'], row['conf'], row['flow_time_a2'], False, 252, manifest_digest)
        actual = tuple(marker[k] for k in
                       ('task', 'conf', 'flow_time_a2', 'smoke', 'case_count', 'manifest_sha256'))
        if marker['status'] != 'PASS' or actual != expected:
            raise RuntimeError(f'Invalid completion marker: {path}')
        completed[str(row['task'])] = sha256(path)
    return completed


def load_plan():
    plan = read_json(HERE / 'resume_plan.json')
    for name, expected in plan['resume_source_sha256'].items():
        if sha256(HERE / name) != expected:
            raise RuntimeError(f'Recovery source changed after planning: {name}')
    if sha256(PRODUCTION / 'request.json') != plan['production_request_sha256']:
        raise RuntimeError('Production request changed')
    if sha256(PRODUCTION / 'manifest.json') != plan['manifest_sha256']:
        raise RuntimeError('Production manifest changed')
    return plan


def require_allocation():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Published-array validation and performance analysis require a CPU allocation')
