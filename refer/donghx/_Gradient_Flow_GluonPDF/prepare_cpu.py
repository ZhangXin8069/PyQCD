#!/usr/bin/env python3
"""Freeze all input identities, shared indices and candidate definitions on CPU."""
import ast
from datetime import datetime
import json
import os
import numpy as np
from common import *
from raw_io import raw_headers


def main():
    allocation();code_check()
    if (HERE/'input_manifest.json').exists():raise RuntimeError('manifest already exists; audit before retry')
    prior=ROOT/'quark_l32x64_expand30_20261012/validation/final_audit.json'
    validation=json.loads(prior.read_text())
    accepted={(str(r['conf']),float(r['tau_a2'])) for r in validation['complete']}
    records=[];timings=[]
    for tau,_ in FLOWS:
        for conf in CONFS:
            if (conf,tau) not in accepted:raise RuntimeError(f'not in prior validation: {conf}/{tau}')
            timing=raw_path(conf,tau,7).parent/'timing_summary.json'
            metadata=json.loads(timing.read_text())
            if str(metadata['conf'])!=conf or metadata['flow_tag']!=flow_tag(tau):raise ValueError(timing)
            if metadata['c2_source_list']!=list(range(NT)) or metadata['current_list']!=list(range(NT)):
                raise ValueError(f'incomplete source/current coverage: {timing}')
            timings.append(dict(**stat_identity(timing),sha256=sha256(timing)))
            for tsep in TSEPS:
                p=raw_path(conf,tau,tsep);before=stat_identity(p);raw_headers(p)
                if stat_identity(p)!=before:raise ValueError('input changed during header check')
                records.append(dict(tau=tau,conf=conf,tsep=tsep,**before))
        print('CPU_HEADERS_PASS',tau,len(CONFS)*len(TSEPS),flush=True)
    indices=np.random.default_rng(SEED).integers(0,NCONF,size=(NBOOT,NCONF),dtype=np.int32)
    p=HERE/'bootstrap_indices_Nboot100_seed1115.npy'
    with p.open('xb') as f:np.save(f,indices,allow_pickle=False)
    atomic_json(HERE/'input_manifest.json',dict(status='INPUT_HEADERS_AND_PROVENANCE_PASS',
        timestamp=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        ensemble=ENSEMBLE,confs=list(CONFS),nconf=NCONF,flows=[t for t,_ in FLOWS],
        tseps=list(TSEPS),pz=list(PZ),nboot=NBOOT,seed=SEED,resample_size=NCONF,
        weight_scheme=SCHEME,estimator_id=ESTIMATOR,c2_mode='complex_nopol',
        raw_records=records,timing_records=timings,prior_validation=str(prior),prior_validation_sha256=sha256(prior),
        bootstrap_indices_sha256=sha256(p),
        candidates=[dict(label=l,tseps=list(t),cut=c) for l,t,c in CANDIDATES],
        fit_model='R(T,t_insert)=M',Tmin=7,Tmin_fm=7*A_FM,Q_min=Q_MIN,chi2_dof_max=CHI_MAX,
        covariance='frozen Schaefer-Strimmer correlation-to-identity; SVD eigenmode inverse threshold 1e-12',
        AIC='chi2+2*k+2*N_omit; k=1; windows averaged within each cut, usable cuts equal-weighted',
        selected_fallback='finite minimum precomputed candidate bootstrap statistical error, diagnostic only',
        validation_scope='headers/identity, not a fresh full raw-payload finite/CRC scan',
        transversity_status=TRANS_STATUS,code_lock_sha256=sha256(HERE/'code_lock.json')))
    print('MANIFEST_READY_1440_RAW_PRODUCTS',flush=True)


if __name__=='__main__':main()
