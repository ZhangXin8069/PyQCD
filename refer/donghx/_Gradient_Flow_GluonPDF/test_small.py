#!/usr/bin/env python3
"""Small CPU-only algebra checks; no lattice inputs or physical contractions."""
import json
import os
from pathlib import Path
import sys
import numpy as np

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
from flow_common_l32x64 import (create_omega, spin_objects, first_order_weights,
    create_time_pair_omega, FlowConfig, FLOWED_VERTEX_SCHEMA)
from flowed_perambulator_factorization import RIGHT_FACTOR_SPECS, build_flowed_sequential
from spin2_contractions import Contractor, c3_reduced, c2_reduced, c2_reference
from memory_bounded import c2_momentum_batched, selected_h_streamed
from weights_and_terms import (weight_tables, table_for_times, iter_reduced_right,
                              current_block_h, signed_block_term, summed_flavors)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('run algebra/contraction tests on a CPU allocation')
    first, second, product, labels = weight_tables()
    np.testing.assert_allclose(second,create_omega(400).real,rtol=4e-15,atol=0)
    assert np.array_equal(np.diag(second),first)
    assert np.array_equal(table_for_times(1,1,second,product),second)
    assert np.array_equal(table_for_times(1,2,second,product),product)
    np.testing.assert_array_equal(first_order_weights(), first)
    np.testing.assert_allclose(create_time_pair_omega(1,1), second, rtol=4e-15)
    np.testing.assert_array_equal(create_time_pair_omega(1,2), product)
    np.testing.assert_allclose(create_time_pair_omega(0,64), second, rtol=4e-15)
    assert 'schemeB' in str(FlowConfig('10000', .5, .01, 4).vertex_root)
    assert 'schemeB' in FLOWED_VERTEX_SCHEMA
    # Each complete same-sector sum equals d^2; this does NOT by itself prove
    # the actual input noise distribution or all baryon shared-noise moments.
    from weights_and_terms import DIMENSIONS
    for r,d in enumerate(DIMENSIONS):
        idx = np.flatnonzero(labels==r)
        np.testing.assert_allclose(second[np.ix_(idx,idx)].sum(),d*d,rtol=2e-15)
    rng = np.random.default_rng(20261014)
    def z(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    ns,nv,ntime,nz,nmom = 3,4,2,3,2
    g,_,q = spin_objects(np)
    p = z((4,4,ns,ns)); a=z((ntime,4,4,ns,nv)); b=z((ntime,4,4,nv,ns))
    vs=z((nmom,ns,ns,ns));vr=z((nmom,ns,ns,ns))
    _,s,prod,_ = weight_tables((2,2),(2,6))
    raw=z((ntime,ntime,2,nz,nv,nv))
    ja=raw*s[None,None,None,None]
    jb=np.stack([np.stack([raw[ia,ib]*table_for_times(ia,ib,s,prod)
                          for ib in range(ntime)]) for ia in range(ntime)])
    j=(ja,jb);contract=Contractor()
    terms=np.zeros((2,ntime,ntime,6,2,4,nmom,nz,4,4),complex)
    max_original_error=0.
    for scheme in range(2):
        for ib in range(ntime):
            h=current_block_h(j[scheme][:,ib],a,contract)
            # Explicit combined-mode matrix product, independent of einsum
            # implementation used by the diagnostic's batch-H builder.
            for ia in range(ntime):
                gold=np.empty((2,nz,nv,4,4,ns),complex)
                for sign in range(2):
                    for dz in range(nz):
                        v=a[ia].reshape(4*4*ns,nv) @ j[scheme][ia,ib,sign,dz]
                        gold[sign,dz]=v.reshape(4,4,ns,nv).transpose(3,0,1,2)
                np.testing.assert_allclose(h[ia,0],gold,rtol=3e-14,atol=3e-14)
            for it,(spec,right,q2,basis) in enumerate(iter_reduced_right(
                    p,b[ib:ib+1],g,q,vs,vr,contract,np)):
                terms[scheme,:,ib,it]=signed_block_term(h,spec,right,q2,basis,contract)
                original=RIGHT_FACTOR_SPECS[it]
                arrays=dict(p=p,b=b[ib:ib+1],g=g,q=q,vs=vs,vr=vr)
                rr=contract(original.equation,*(arrays[k] for k in original.operands))
                for ia in range(ntime):
                    args=(h[ia],rr,q) if 'Pka' in original.final_equation else (h[ia],rr)
                    gold=original.coefficient*contract(original.final_equation,*args).transpose(1,0,2,3,4,5)
                    got=terms[scheme,ia,ib,it]
                    np.testing.assert_allclose(got,gold,rtol=4e-12,atol=8e-11)
                    max_original_error=max(max_original_error,float(np.max(abs(got-gold))))
        hf=build_flowed_sequential(j[scheme],a,contract)
        for subset in (np.arange(ntime),np.array([1])):
            streamed=selected_h_streamed(j[scheme],a,subset,contract,np)
            full=build_flowed_sequential(j[scheme][subset[:,None],subset[None,:]],a[subset],contract)
            np.testing.assert_allclose(streamed,full,rtol=2e-14,atol=2e-13)
        batched,oracle=c2_momentum_batched(p,vs,vr,g,q,contract,c2_reduced,c2_reference,np)
        np.testing.assert_allclose(batched,c2_reduced(p,vs,vr,g,q,contract),rtol=4e-12,atol=1e-10)
        np.testing.assert_allclose(oracle,c2_reference(p,vs,vr,g,q,contract),rtol=4e-12,atol=1e-10)
        u,d,rawterms=c3_reduced(hf,p,b,g,q,vs,vr,contract,return_terms=True)
        aggregate=summed_flavors(terms)[scheme]
        np.testing.assert_allclose(aggregate,np.stack([u,d]),rtol=4e-12,atol=1e-10)
        for it,spec in enumerate(__import__('weights_and_terms').SPECS):
            np.testing.assert_allclose(terms[scheme,:,:,it].sum(axis=(0,1)),
                spec[2]*rawterms[spec[0]],rtol=4e-12,atol=1e-10)
    np.testing.assert_array_equal(terms[0,np.arange(ntime),np.arange(ntime)],
                                  terms[1,np.arange(ntime),np.arange(ntime)])
    # Independent nested-window membership: non-consecutive/modulo time labels,
    # not positions in a sorted cache list. No multiplicity or time averaging.
    from common import distance
    times = [0, 1, 2, 3, 61, 62, 63]
    radii = np.asarray([distance(t, 0) for t in times])
    blocks = z((len(times), len(times), 2))
    shells = np.zeros((4, 2), complex)
    for ia in range(len(times)):
        for ib in range(len(times)):
            shells[max(radii[ia], radii[ib])] += blocks[ia, ib]
    for w in (1, 2, 3):
        idx = np.flatnonzero(radii <= w)
        gold = blocks[idx[:,None], idx[None,:]].sum(axis=(0,1))
        np.testing.assert_allclose(shells[:w+1].sum(axis=0), gold, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(c2_reduced(p,vs,vr,g,q,contract),
                              c2_reference(p,vs,vr,g,q,contract),rtol=3e-12,atol=1e-10)
    payload=dict(status='PASS',seed=20261014,checks=[
        'L64 weights equal frozen create_omega','explicit first-order vector',
        'sector sums equal d^2','explicit matrix H vs batched H',
        'each ordered block / each six-Wick sign vs frozen full-spin equations',
        'block sum vs production time-factorization for both schemes',
        'B original-time selector, explicit first weights, A/B cache-schema isolation',
        'nested windows via periodic-distance shell vs explicit ordered double sum',
        'same-time vertex/Wick equality','rank2 C2 vs full-spin C2'],
        maximum_original_wick_absolute_difference=max_original_error,
        scope='small deterministic CPU tensors only; not physical L32x64 results')
    (HERE/'status'/'small_tests.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
