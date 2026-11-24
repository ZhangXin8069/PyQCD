"""L24-specific raw zero-flow identities and APBC free-field heat-kernel gate.

Never call this native binding with n_steps=0: its fout and gout are then
uninitialized (interface_quda.cpp:5485--5648). K(0) is applied analytically.
"""
from dataclasses import replace
import os
import numpy as np
from common import (HERE,NT,NX,NV,NZ,load_manifest,config_for,require_allocation,
                    atomic_json,sha256)
from flow_common import generalized_vertex,transported_right,create_omega,spin_objects
from flow_backend import initialize_pyquda,raw_gauge_time
from generate_basis import evolve


def relative(a,b):
    return float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(a.ravel()),1e-300))


def main():
    require_allocation()
    manifest=load_manifest()
    row=manifest['tasks'][6]
    cfg=config_for(row)
    import cupy as cp
    from spin2_contractions import Contractor,c2_reduced,c2_reference,c3_reduced
    from flowed_perambulator_factorization import direct_time_summed_one_direction,build_flowed_sequential
    # Real unflowed columns: both +/-z lines compared to independent roll/link multiplication.
    zero=replace(cfg,flow_time=0.)
    v=cp.asarray(np.load(zero.eigen_path(0),mmap_mode='r')[:4])
    gauge=cp.asarray(raw_gauge_time(zero,0))
    j=generalized_vertex(v,v,gauge,'z',NZ,cp)
    errors=[]
    for si,sign in enumerate((-1,1)):
        for dz in (0,1,5,NZ-1):
            shifted=transported_right(v,gauge,'z',sign*dz,cp)
            gold=cp.einsum('nzyxc,mzyxc->nm',v.conj(),shifted,optimize=True)
            error=relative(cp.asnumpy(gold),cp.asnumpy(j[si,dz]))
            assert np.isfinite(error) and error<2e-12,(si,dz,error)
            errors.append(dict(sign=sign,dz=dz,relative_L2=error))
    # Full-spin C2 and direct double-time C3 with actual current blocks, small deterministic external tensors.
    rng=np.random.default_rng(20261104)
    def rand(shape):
        return cp.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape))
    g,_,q=spin_objects(cp);con=Contractor()
    p=rand((4,4,2,2));a=rand((1,4,4,2,4));b=rand((1,4,4,4,2))
    vs,vr=rand((1,2,2,2)),rand((1,2,2,2))
    j0=j[None,None,:,:3]
    h=build_flowed_sequential(j0,a,con)
    ur,dr=c3_reduced(h,p,b,g,q,vs,vr,con)
    ug,dg=direct_time_summed_one_direction(p,a,b,j0,g,q,vs,vr,Contractor('greedy'))
    ug=ug.transpose(1,0,2,3,4,5);dg=dg.transpose(1,0,2,3,4,5)
    contraction_errors=dict(C2=relative(cp.asnumpy(c2_reference(p,vs,vr,g,q,con)),cp.asnumpy(c2_reduced(p,vs,vr,g,q,con))),
                            C3_u=relative(cp.asnumpy(ug),cp.asnumpy(ur)),
                            C3_d=relative(cp.asnumpy(dg),cp.asnumpy(dr)))
    assert max(contraction_errors.values())<2e-10,contraction_errors
    del v,gauge,j,j0,h,p,a,b,vs,vr,ug,dg,ur,dr
    cp.get_default_memory_pool().free_all_blocks()
    # Exact lattice Laplacian eigenmode with anti-periodic temporal momentum.
    state=initialize_pyquda(cfg)
    li=state['latt_info']
    identity=np.broadcast_to(np.eye(3,dtype=complex),(4,NT,NX,NX,NX,3,3)).copy()
    unit=state['Gauge'](li,4,cp.asarray(li.evenodd(identity,multi=True)))
    unit.setAntiPeriodicT();state['dirac'].loadGauge(unit)
    pt=np.pi/NT;pz=2*np.pi*3/NX;py=2*np.pi/NX
    angle=(pt*np.arange(NT)[:,None,None,None]+pz*np.arange(NX)[None,:,None,None]
           +py*np.arange(NX)[None,None,:,None])
    lex=np.zeros((1,NT,NX,NX,NX,4,3),np.complex128)
    lex[0,...,0,0]=np.exp(1j*angle)
    got=evolve(state,lex)
    lam=4*(np.sin(pt/2)**2+np.sin(pz/2)**2+np.sin(py/2)**2)
    expected=lex*np.exp(-lam*row['flow_time_a2'])
    heat_error=relative(expected,got)
    assert np.isfinite(heat_error) and heat_error<2e-6,heat_error
    # Pure gauge identity is unchanged; strip the fermion center phase before checking.
    flowed=state['Gauge'](li,4);state['dirac'].saveSmearedGauge(flowed);flowed.setAntiPeriodicT()
    saved=li.lexico(cp.asnumpy(flowed.data),multi=True)
    gauge_error=float(np.max(np.abs(saved-identity)))
    assert gauge_error<2e-12,gauge_error
    atomic_json(HERE/'status/gpu_validation.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],
        manifest_sha256=sha256(HERE/'manifest.json'),zero_flow='analytic K(0); native n_steps=0 prohibited',
        actual_unflowed_vertex_checks=errors,small_tensor_GPU_regressions=contraction_errors,
        full4D_APBC_free_field_relative_L2=heat_error,free_gauge_max_abs=gauge_error,
        limits=['real full-size finite-flow C3 checked separately in smoke gate',
                'not an ensemble statistics or final six-direction physics validation']))
    print('L24_GPU_ZERO_AND_FREE_HEAT_PASS',heat_error,flush=True)


if __name__=='__main__':
    main()
