import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MOVING = HERE / 'moving'
P0 = HERE / 'pz0'
CONFS = tuple(map(str, range(14000, 16000, 40)))
FLOWS = (.1, .2, .3, .4, .5, .6, .7, .8)
WINDOWS = (4, 5, 5, 6, 6, 7, 8, 8)
CURRENTS = (0, 16, 32, 63)
TSEPS = tuple(range(7, 13))
ALLOWED_NODES = ('gpu026', 'gpu027', 'gpu029')


def sha(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def allocation():
    assert os.environ.get('SLURM_JOB_ID'), 'compute allocation required'


def check_frozen():
    request = json.loads((HERE / 'request.json').read_text())
    for relative, digest in request['files_sha256'].items():
        assert sha(HERE / relative) == digest, relative
    return request


def load_module(name, root):
    spec = importlib.util.spec_from_file_location(name, root / 'common.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
