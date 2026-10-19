"""Frozen A-scheme data/estimator/fit contract, deliberately separate from B."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE/'reference'))
ENSEMBLE='beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64'
RAW=ROOT/'quark_l32x64_flow_20261007/results'
CONFS=tuple(map(str,range(10000,11200,40)))
FLOWS=((.05,2),(.1,2),(.2,3),(.3,3),(.4,4),(.5,4),(.8,6),(1.,7))
CHANNELS=('unpolarized','helicity','transversity')
FLAVORS=('u','d','u_minus_d')
COMPONENTS=('even_real','odd_imag')
TSEPS=(5,6,7,8,9,10)
PZ=(3,4,5)
NT=64
DZ=21
NCONF=30
NBOOT=100
SEED=1115
A_FM=.0897
TMIN=7
Q_MIN=1e-3
CHI_MAX=1e3
CUTS=(1,2,3)
TMAXS=(8,9,10)
CANDIDATES=tuple((f'T7-{top}_L{cut}_R{cut}',tuple(range(7,top+1)),cut)
                 for cut in CUTS for top in TMAXS)
ESTIMATOR='source_mean_then_shared_configuration_bootstrap_ratio_of_complex_means_positive_z_vdv_v1'
SCHEME='A_legacy_unified_table'
TRANS_STATUS='pending_external_physics_confirmation'


def allocation():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Numerical/data processing requires a CPU allocation')


def tau_tag(value):return f'tau{value:.2f}'.replace('.','p')


def flow_tag(value):
    return f'thin_tf{value:.8f}_eps0.01000000_tw{dict(FLOWS)[value]}_Nv400_dz21'.replace('.','p')


def raw_path(conf,tau,tsep):
    return RAW/flow_tag(tau)/'Nev_200'/str(conf)/f'corr_projected_tsep{tsep}_dz0-20_Nev200.npz'


def ratio_path(tau,channel,flavor):
    return HERE/'ratios'/f'ratio_{tau_tag(tau)}_{channel}_{flavor}_Nboot100_seed1115.npz'


def fit_path(tau,channel,flavor):
    return HERE/'fits'/f'bare_ratio_AIC_{tau_tag(tau)}_{channel}_{flavor}.npz'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    os.replace(tmp,path)


def atomic_npz(path,**payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**payload)
    os.replace(tmp,path)


def stat_identity(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)


def code_check():
    lock=json.loads((HERE/'code_lock.json').read_text())
    for p,h in lock['files_sha256'].items():
        if sha256(HERE/p)!=h:raise RuntimeError(f'frozen source changed: {p}')
    return lock


def manifest():
    code_check()
    m=json.loads((HERE/'input_manifest.json').read_text())
    if m['confs']!=list(CONFS) or m['flows']!=[t for t,_ in FLOWS]:raise ValueError('analysis population changed')
    return m


def shared_indices():
    m=manifest();p=HERE/'bootstrap_indices_Nboot100_seed1115.npy'
    if sha256(p)!=m['bootstrap_indices_sha256']:raise ValueError('bootstrap indices changed')
    x=np.load(p,allow_pickle=False)
    if x.shape!=(100,30) or np.any(x<0) or np.any(x>=30):raise ValueError('bad shared indices')
    return x


def common_metadata(tau):
    return dict(ensemble=np.asarray(ENSEMBLE),confs=np.asarray(CONFS),nconf=np.asarray(NCONF),
        nboot=np.asarray(NBOOT),seed=np.asarray(SEED),boot_indices=shared_indices(),tau_a2=np.asarray(tau),
        a_fm=np.asarray(A_FM),flow_tag=np.asarray(flow_tag(tau)),epsilon=np.asarray(.01),nstep=np.asarray(round(tau/.01)),
        pz_values=np.asarray(PZ),delta_z_values=np.arange(DZ),tseps=np.asarray(TSEPS),
        weight_scheme=np.asarray(SCHEME),estimator_id=np.asarray(ESTIMATOR),
        analysis_scope=np.asarray('A-scheme finite-flow connected bare; positive-z only; unrenormalized/unmatched'),
        transversity_convention_status=np.asarray(TRANS_STATUS),
        input_manifest_sha256=np.asarray(sha256(HERE/'input_manifest.json')),
        code_lock_sha256=np.asarray(sha256(HERE/'code_lock.json')))
