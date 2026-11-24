"""Full-4D APBC flow of original-time columns; retain selected outputs only."""
import argparse
from dataclasses import replace
import json
import os
import time
import numpy as np
from common import (HERE,NT,NX,NV,config_for,source_times,selected_currents,load_manifest,
    check_inputs,sha256,atomic_json,atomic_npz,fingerprint,require_allocation)
from flow_common import first_order_weights
from flow_backend import initialize_pyquda,save_flowed_gauge_once,read_flowed_gauge_index
from flow_safety import configure_initial_only,validate_initial_only


def evolve(state,lex):
    cp,li=state['cp'],state['latt_info']
    ns=lex.shape[0]
    src=state['Multi'](li,ns,cp.asarray(li.evenodd(lex,multi=True)))
    out=state['Multi'](li,ns)
    dr=state['dirac']
    validate_initial_only(dr.smear_param)
    state['perform'](out.data_ptrs,src.data_ptrs,dr.invert_param,dr.smear_param,dr.obs_param,ns)
    cp.cuda.get_current_stream().synchronize()
    result=li.lexico(cp.asnumpy(out.data),multi=True)
    del src,out
    return result


def step_halving_check(row,state):
    """Four actual columns; full 4D response, not just retained current slices."""
    cfg=config_for(row)
    lex=np.zeros((1,NT,NX,NX,NX,4,3),np.complex128)
    eig=np.load(cfg.eigen_path(0),mmap_mode='r')
    for k in range(4):
        lex[0,0,:,:,:,k,:]=eig[k]
    dr=state['dirac']
    try:
        configure_initial_only(dr.smear_param,row['nstep'],row['epsilon'])
        print('EPSILON_COARSE',row['task'],dr.smear_param.n_steps,
              dr.smear_param.epsilon,dr.smear_param.meas_interval,flush=True)
        coarse=evolve(state,lex)
        configure_initial_only(dr.smear_param,2*row['nstep'],row['epsilon']/2)
        print('EPSILON_FINE',row['task'],dr.smear_param.n_steps,
              dr.smear_param.epsilon,dr.smear_param.meas_interval,flush=True)
        fine=evolve(state,lex)
    finally:
        configure_initial_only(dr.smear_param,row['nstep'],row['epsilon'])
    rel=float(np.linalg.norm((fine-coarse).ravel())/max(np.linalg.norm(fine.ravel()),1e-300))
    if not np.isfinite(rel) or rel>2e-5:
        raise AssertionError(('full4D actual-column step-halving gate',rel))
    atomic_json(HERE/'status'/f'epsilon_task{row["task"]}.json',dict(status='PASS',relative_L2=rel,
        comparison='4 full-space-time actual eigenmodes at same flow time; epsilon vs epsilon/2',
        tolerance=2e-5,flow_time=row['flow_time_a2'],epsilon=row['epsilon'],
        nstep=row['nstep'],coarse_meas_interval=row['nstep']+1,
        fine_meas_interval=2*row['nstep']+1,
        manifest_sha256=sha256(HERE/'manifest.json'),job=os.environ['SLURM_JOB_ID']))
    print('EPSILON_PASS',row['task'],rel,flush=True)
    del lex,eig,coarse,fine
    state['cp'].get_default_memory_pool().free_all_blocks()


def generate(row,sigma,state,manifest):
    started=time.monotonic()
    cfg=config_for(row)
    targets=selected_currents(row,sigma)
    done=cfg.basis_done_path(sigma)
    if done.exists():
        meta=json.loads(done.read_text())
        assert meta['schema']==manifest['basis_schema']
        assert meta['manifest_sha256']==sha256(HERE/'manifest.json')
        assert meta['stored_t_current']==targets and meta['epsilon']==row['epsilon']
        assert meta['flow_time']==row['flow_time_a2'] and meta['n_steps']==row['nstep']
        for tj in targets:
            assert sha256(cfg.basis_path(sigma,tj))==meta['slice_sha256'][str(tj)]
        print('BASIS_VERIFIED_SKIP',row['task'],sigma,flush=True)
        return
    cp,li=state['cp'],state['latt_info']
    assert state['comm'].size==1 and state['t_start']==0 and state['t_stop']==NT
    eigen=np.load(cfg.eigen_path(sigma),mmap_mode='r')
    maps={}
    temps={}
    for tj in targets:
        path=cfg.basis_path(sigma,tj)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(f'.{path.name}.{os.getpid()}.partial.npy')
        maps[tj]=np.lib.format.open_memmap(temp,mode='w+',dtype='complex64',shape=(NV,NX,NX,NX,3))
        temps[tj]=temp
    norm=np.zeros((NV,NT),float)
    batch=int(os.environ.get('FLOW_MODE_BATCH','16'))
    if batch<4 or batch%4:
        raise ValueError('batch must be a positive multiple of 4')
    for start in range(0,NV,batch):
        count=min(batch,NV-start)
        lex=np.zeros(((count+3)//4,NT,NX,NX,NX,4,3),np.complex128)
        for k in range(count):
            lex[k//4,sigma,:,:,:,k%4,:]=eigen[start+k]
        full=evolve(state,lex)
        state['real_basis_flow_performed']=True
        for k in range(count):
            vector=full[k//4,:,:,:,:,k%4,:]
            norm[start+k]=np.sum(abs(vector)**2,axis=(1,2,3,4))
            for tj,mmap in maps.items():
                mmap[start+k]=vector[tj]
        del full,lex,vector
        cp.get_default_memory_pool().free_all_blocks()
        print('BASIS_FLOW_BATCH',row['task'],sigma,start,count,flush=True)
    if not np.isfinite(norm).all() or (norm.sum(axis=1)<=0).any():
        raise FloatingPointError('nonfinite/zero full4D basis norm')
    distance=np.asarray([min((t-sigma)%NT,(sigma-t)%NT) for t in range(NT)])
    tail=norm[:,distance>row['window']].sum(axis=1)
    frac=tail/norm.sum(axis=1)
    first=first_order_weights(NV)
    diagnostics=dict(max_per_mode_squared_norm_fraction=float(frac.max()),
        discarded_squared_norm_fraction=float(tail.sum()/norm.sum()),
        weighted_one_body_squared_norm_fraction=float(first@tail/(first@norm.sum(axis=1))))
    if frac.max()>manifest['per_column_squared_norm_gate']:
        atomic_json(done.with_name('failed_diagnostics.json'),diagnostics)
        raise AssertionError(('finite-window full4D tail gate',row['task'],sigma,diagnostics))
    for tj,mmap in maps.items():
        if not np.isfinite(mmap).all():
            raise FloatingPointError('nonfinite saved basis slice')
        mmap.flush()
        os.replace(temps[tj],cfg.basis_path(sigma,tj))
    slice_hash={str(tj):sha256(cfg.basis_path(sigma,tj)) for tj in targets}
    atomic_npz(done.with_name('norm_diagnostics.npz'),norm_by_mode_time=norm,
               first_order_weights=first,original_time=sigma,window=row['window'])
    atomic_json(done,dict(schema=manifest['basis_schema'],conf=row['conf'],original_time=sigma,
        flow_time=row['flow_time_a2'],epsilon=row['epsilon'],n_steps=row['nstep'],
        fermion_flow_dimension=4,temporal_window=row['window'],stored_t_current=targets,
        exact_temporal_support=False,nev=NV,dtype='complex64',slice_sha256=slice_hash,
        manifest_sha256=sha256(HERE/'manifest.json'),input_eigen_path=str(cfg.eigen_path(sigma)),
        window_diagnostics=diagnostics,seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID']))
    print('BASIS_PASS',row['task'],sigma,diagnostics,flush=True)


def main():
    require_allocation()
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--epsilon-only',action='store_true')
    args=parser.parse_args()
    manifest=load_manifest()
    row=manifest['tasks'][args.task]
    check_inputs(row)
    cfg=config_for(row)
    state=initialize_pyquda(cfg)
    if args.epsilon_only:
        if args.worker!=0:
            raise ValueError('epsilon-only requires worker0')
        step_halving_check(row,state)
        return
    if args.smoke and args.worker==0:
        step_halving_check(row,state)
    for sigma in source_times(row)[args.worker::4]:
        generate(row,sigma,state,manifest)
    # One owner publishes gauge from the final actual basis flow, with the same
    # thin-link history, epsilon, and nstep as both endpoints. No GPU-rank races.
    if args.worker==0:
        index=cfg.gauge_flow_root/'index.json'
        if not index.exists():
            if not state.get('real_basis_flow_performed'):
                eigen=np.load(cfg.eigen_path(0),mmap_mode='r')
                lex=np.zeros((1,NT,NX,NX,NX,4,3),np.complex128)
                for k in range(4):
                    lex[0,0,:,:,:,k,:]=eigen[k]
                evolve(state,lex)
            save_flowed_gauge_once(cfg,state)
        read_flowed_gauge_index(cfg)
    check_inputs(row)


if __name__=='__main__':
    main()
