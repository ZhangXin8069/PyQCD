#!/usr/bin/env python3
"""Independent early CPU smoke while the full input scan is still running.

This report is not the admission/completion gate and cannot release downstream
jobs. It uses only one real raw case plus deterministic synthetic statistics.
"""
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
    allocation();code_check();validate_gamma_basis();rng=np.random.default_rng(20261016)
    with tempfile.TemporaryDirectory(prefix='q64_A_early_') as temporary:
        path=Path(temporary)/'probe.npz'
        value=(rng.normal(size=(3,4,5))+1j*rng.normal(size=(3,4,5)))
        np.savez(path,data=value)
        mapped=member(path,'data');np.testing.assert_array_equal(mapped,value);del mapped
    c3=rng.normal(size=(30,3,2,2,3,2,2,4))+1j*rng.normal(size=(30,3,2,2,3,2,2,4))
    c2=3+rng.normal(scale=.2,size=(30,3,2))+1j*rng.normal(scale=.02,size=(30,3,2))
    index=rng.integers(0,30,size=(100,30),dtype=np.int32)
    original,boot,_,_=ratio_from_projected(c3,c2,index)
    full=np.concatenate((c3,(c3[:,:,0]-c3[:,:,1])[:,:,None]),axis=2)
    for b in (0,5,99):
        raw=full[index[b]].mean(axis=0)/c2[index[b]].mean(axis=0)[None,None,None,:,None,:,None]
        expected=np.stack((.5*(raw[:,:,0]+raw[:,:,1]),.5*(raw[:,:,0]-raw[:,:,1])),axis=2)
        np.testing.assert_allclose(boot[b],expected,rtol=3e-14,atol=2e-14)
    fit=candidate_fit(np.ones(8)*1.5,1.5+rng.normal(scale=.2,size=(100,8)))
    assert fit['sample_rank']>=1 and fit['effective_rank']>1
    assert np.isfinite(fit['boot']).all()
    assert abs(weights([3,5]).sum()-1)<1e-14
    selection=select(np.array([1.,2.,3.,4.]),np.array([1.,2.,3.,4.]),np.array([1.,2.,3.,4.]),
        np.array([True,False,False,False]),np.array([.3,.1,.2,.1]),[np.array([0,1]),np.array([2,3])])
    assert selection['fallback'] and selection['fallback_indices']==[3] and selection['value']==2.5
    path=raw_path('10000',.1,7);projected,den=project_case(path,7)
    assert projected.shape==(2,3,2,3,21,8) and den.shape==(3,)
    raw=member(path,'corr_3pt_u');absolute=[]
    for ch,phase in ((0,1.),(1,-1j)):
        pol=int(SPECS.pol_indices[2,ch]);expected=0j
        for source in range(64):
            for left in range(4):
                for right in range(4):
                    expected+=raw[1,pol,2,4,left,right,source,(source+3)%64]*SPECS.gamma[2,ch,left,right]/64
        actual=projected[0,ch,1,2,4,3]
        np.testing.assert_allclose(actual,phase*expected,rtol=3e-12,atol=1e-15)
        absolute.append(float(abs(actual-phase*expected)))
    del raw
    atomic_json(HERE/'status/early_smoke.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],
        checks=['ZIP NPY mapping','explicit bootstrap ratio/VDV','correlated fit and cut/fallback selection',
                'real conf10000 tau0.1 tsep7; independent source/spin sum'],absolute_differences=absolute,
        source_sha256=sha256(__file__),admission_gate=False,scope='early implementation smoke only'))
    print('EARLY_CPU_REAL_AND_SYNTHETIC_PASS',flush=True)


if __name__=='__main__':main()
