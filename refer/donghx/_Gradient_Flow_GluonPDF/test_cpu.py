#!/usr/bin/env python3
"""CPU gates for ZIP mapping, gamma/axes, shared bootstrap, AIC and a real case."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from common import *
from raw_io import member
from ratios_cpu import project_case,ratio_from_projected,SPECS
from fit_cpu import candidate_fit,weights,select,fallback
from conventions import validate_gamma_basis


def main():
    allocation();m=manifest();validate_gamma_basis();rng=np.random.default_rng(20261016)
    with tempfile.TemporaryDirectory(prefix='q64_A_test_') as temporary:
        path=Path(temporary)/'sample.npz'
        value=(rng.normal(size=(5,4,3))+1j*rng.normal(size=(5,4,3))).astype('complex128')
        # Generated ZIP64 local headers exercise the same path as >1GB raw files.
        with path.open('wb') as f:np.savez(f,data=value)
        got=member(path,'data')
        np.testing.assert_array_equal(got,value)
        assert isinstance(got,np.memmap)
        del got
    # Distinct momentum values catch the old advanced-indexing momentum collapse.
    raw2=np.arange(4*3*64*64).reshape(4,3,64,64)
    ts=np.arange(64);direct=np.array([np.mean([raw2[0,p,s,(s+7)%64] for s in ts]) for p in range(3)])
    np.testing.assert_array_equal(raw2[0][:,ts,(ts+7)%64].mean(axis=-1),direct)
    # Tiny toy arrays with full population axis compare explicit replicate loops.
    c3=rng.normal(size=(30,3,2,2,3,2,2,4))+1j*rng.normal(size=(30,3,2,2,3,2,2,4))
    c2=3+rng.normal(scale=.2,size=(30,3,2))+1j*rng.normal(scale=.02,size=(30,3,2))
    idx=shared_indices();ro,rb,_,_=ratio_from_projected(c3,c2,idx)
    full=np.concatenate((c3,(c3[:,:,0]-c3[:,:,1])[:,:,None]),axis=2)
    for b in (0,5,99):
        raw=full[idx[b]].mean(axis=0)/c2[idx[b]].mean(axis=0)[None,None,None,:,None,:,None]
        expect=np.stack((.5*(raw[:,:,0]+raw[:,:,1]),.5*(raw[:,:,0]-raw[:,:,1])),axis=2)
        np.testing.assert_allclose(rb[b],expect,rtol=3e-14,atol=2e-14)
    np.testing.assert_allclose(rb[:,:,2],rb[:,:,0]-rb[:,:,1],rtol=3e-14,atol=2e-14)
    # A correlated constant fit must reproduce the explicit linear projector.
    noise=rng.normal(scale=.2,size=(100,8));ys=1.5+noise+.2*noise[:,0,None]
    fit=candidate_fit(ys.mean(axis=0),ys)
    assert fit['sample_rank']>=1 and fit['effective_rank']>1
    np.testing.assert_allclose(weights([3.,5.]).sum(),1.)
    groups=[np.array([0,1]),np.array([2,3])]
    r=select(np.array([1.,2.,3.,4.]),np.array([1.,2.,3.,4.]),np.array([1.,2.,3.,4.]),
        np.array([True,False,False,False]),np.array([.3,.1,.2,.1]),groups)
    assert r['fallback'] and r['fallback_indices']==[3] and r['value']==2.5
    assert fallback(np.array([0,1,2]),np.array([.2,.1,.1]),np.array([0,3,2]))==2
    # Real-data smoke: small source-averaged products, independent one-point spin sum.
    path=raw_path(CONFS[0],.1,7);projected,c2real=project_case(path,7)
    assert projected.shape==(2,3,2,3,21,8) and c2real.shape==(3,)
    raw=member(path,'corr_3pt_u');checks=[]
    for channel,phase in ((0,1.),(1,-1j)):
        ip=2;sign=1;z=4;q=3;pol=int(SPECS.pol_indices[ip,channel])
        explicit=0j
        for source in range(64):
            for left in range(4):
                for right in range(4):
                    explicit+=raw[sign,pol,ip,z,left,right,source,(source+q)%64]*SPECS.gamma[ip,channel,left,right]/64
        got=projected[0,channel,sign,ip,z,q]
        np.testing.assert_allclose(got,phase*explicit,rtol=3e-12,atol=1e-15)
        checks.append(float(abs(got-phase*explicit)))
    del raw
    atomic_npz(HERE/'status/real_smoke_projection.npz',c3=projected,c2=c2real)
    atomic_json(HERE/'status/cpu_gate.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],
        code_lock_sha256=sha256(HERE/'code_lock.json'),input_manifest_sha256=sha256(HERE/'input_manifest.json'),
        checks=['stored ZIP64 NPY mmap vs normal NumPy load','gamma algebra and convention snapshot',
                'momentum-preserving C2 index oracle','shared bootstrap ratio-of-means explicit-loop oracle',
                'u-d exact paired subtraction','positive sample/design rank correlated constant',
                'within-cut AIC normalization and diagnostic minimum-error fallback',
                'real conf10000/tau0.1/tsep7 projection and explicit spin/source oracle'],
        real_smoke_absolute_differences=checks,Q_min=Q_MIN,chi2_dof_max=CHI_MAX,
        interpretation='implementation and real-case projection gate, not completed 30-conf ratio or accepted physics'))
    print('CPU_GATE_PASS_REAL_AND_SYNTHETIC',flush=True)


if __name__=='__main__':main()
