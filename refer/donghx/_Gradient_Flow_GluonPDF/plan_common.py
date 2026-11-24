"""Scheduler-only recovery plan; production numerical kernels stay immutable."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MOVING = ROOT/'quark_l32x64_B_expand100_P456_20261028'
P0 = HERE/'pz0'
P0_ORIGINAL = ROOT/'quark_l32x64_B_Pz0_fill50_8flow_20261029'
CONFS = tuple(map(str,range(12000,14000,40)))
FLOWS = (.1,.2,.3,.4,.5,.6,.7,.8)
WINDOWS = (4,5,5,6,6,7,8,8)
CURRENTS = (0,16,32,63)
TSEPS = tuple(range(7,13))
OLD_ARRAY = '3878917'
OLD_AUDIT = '3878918'
ALLOWED_NODES = ('gpu026','gpu027','gpu029')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)


def allocation():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required')


def check_frozen():
    request=json.loads((HERE/'request.json').read_text())
    for rel,digest in request['files_sha256'].items():
        assert sha(HERE/rel)==digest,('retry source changed',rel)
    for path,digest in request['upstream_sha256'].items():
        assert sha(path)==digest,('upstream frozen source changed',path)
    return request


def load_module(name,root):
    sys.path.insert(0,str(root/'reference'))
    spec=importlib.util.spec_from_file_location(name,root/'common.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
