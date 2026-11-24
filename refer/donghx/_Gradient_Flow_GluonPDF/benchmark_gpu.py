"""Paired, synchronized A100 timings on unchanged production C2/C3 equations."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
import opt_einsum as oe

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'quark_l24x72_B_conf101to120_matched_radius_20261114'
sys.path.insert(0,str(SOURCE))
from common import (load_manifest,check_inputs,config_for,case_times,support_times,
                    NE,NV,NZ,MOMENTA,output_dir,sha256,project_spin,read_case)
import cupy as cp
import gpu_worker as production
from spin2_contractions import Contractor,c2_reference,c2_reduced,c3_reduced,SPECS
from flow_common import spin_objects,seq_peram,baryon_vvv,phase_3pt
from insertion_kernels import one_right_time_h
from explicit_contract import ExplicitContractor

MODES=('current_optimal','explicit_optimal','explicit_dp','explicit_greedy')
REPEATS=3


def dump(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def sync(): cp.cuda.get_current_stream().synchronize()


def timed(function):
    sync();start=time.perf_counter();answer=function();sync()
    return answer,time.perf_counter()-start


def metric(reference,actual):
    reference=np.asarray(reference);actual=np.asarray(actual)
    assert reference.shape==actual.shape and reference.dtype==actual.dtype
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    norm=max(float(np.linalg.norm(reference.ravel())),1e-300)
    return dict(relative_L2=float(np.linalg.norm((actual-reference).ravel()))/norm,
        max_absolute=float(np.max(np.abs(actual-reference))),reference_L2=norm)


def checked(reference,actual,label):
    error=metric(reference,actual)
    assert error['relative_L2']<=2e-10,(label,error)
    return error


def make_contractor(mode):
    return Contractor('optimal') if mode=='current_optimal' else ExplicitContractor(mode.removeprefix('explicit_'))


class Profile:
    def __init__(self,con): self.con=con;self.records={}
    def __call__(self,eq,*arrays):
        answer,elapsed=timed(lambda:self.con(eq,*arrays))
        if eq not in self.records:
            self.records[eq]=dict(equation=eq,shapes=[list(x.shape) for x in arrays],calls=0,seconds=0.)
        self.records[eq]['calls']+=1;self.records[eq]['seconds']+=elapsed
        return answer


def c2_all(data,con,reference=False):
    func=c2_reference if reference else c2_reduced
    return np.concatenate([data['boundary']*cp.asnumpy(func(data['p'],vs,vr,data['g'],data['q'],con))
        for vs,vr in zip(data['vvs'],data['vvr'])],axis=1)


def c3_case(data,con):
    wick=np.zeros((6,2,4,len(MOMENTA),NZ,4,4),np.complex128)
    for ib in range(len(data['times'])):
        h=one_right_time_h(data['vertices'],data['a'],ib,con)
        for im,(vs,vr) in enumerate(zip(data['vvs'],data['vvr'])):
            _,_,terms=c3_reduced(h,data['p'],data['b'][ib:ib+1],data['g'],data['q'],vs,vr,con,return_terms=True)
            for it,spec in enumerate(SPECS):
                wick[it,:,:,im:im+1]+=data['boundary']*spec[2]*cp.asnumpy(terms[spec[0]])
            del terms
        del h
        # Same per-sigma memory-pool release as the current production driver.
        cp.get_default_memory_pool().free_all_blocks()
    return wick


def prepare(row,tj,sep,dt,manifest):
    started=time.perf_counter()
    check_inputs(row)
    cfg=config_for(row)
    source,sink,_,boundary=case_times(tj,sep,dt)
    times=support_times(tj,row['window'])
    src=np.load(cfg.peram_path(source),mmap_mode='r')
    snk=np.load(cfg.peram_path(sink),mmap_mode='r')
    a=seq_peram(cp.asarray(np.ascontiguousarray(snk[times,:,:,:NV,:NE])),cp)
    b=cp.asarray(np.ascontiguousarray(src[times,:,:,:NV,:NE]))
    p=cp.asarray(src[sink,:,:,:NE,:NE])
    vvs=[];vvr=[]
    for momentum in MOMENTA:
        phase=phase_3pt([momentum,0,0],xp=cp)[None]
        vvs.append(baryon_vvv(phase,np.load(cfg.eigen_path(sink),mmap_mode='r')[:NE],cp))
        vvr.append(baryon_vvv(phase,np.load(cfg.eigen_path(source),mmap_mode='r')[:NE],cp).conj())
    vertices=production.load_vertices(row,tj,times,manifest)
    g,_,q=spin_objects(cp)
    sync()
    record=dict(task=row['task'],conf=row['conf'],flow_time_a2=row['flow_time_a2'],
        window=row['window'],nsigma=len(times),source=source,sink=sink,t_current=tj,tsep=sep,dt=dt,
        boundary=boundary,NE=NE,NV=NV,NZ=NZ,momenta=list(MOMENTA),
        preparation_seconds=time.perf_counter()-started,
        tensors={k:dict(shape=list(x.shape),dtype=str(x.dtype),bytes=x.nbytes)
                 for k,x in dict(p=p,a=a,b=b,vertices=vertices).items()})
    return dict(p=p,a=a,b=b,vertices=vertices,g=g,q=q,vvs=vvs,vvr=vvr,times=times,boundary=boundary),record


def one_case(row,tj,sep,dt,manifest,index):
    data,record=prepare(row,tj,sep,dt,manifest)
    print('REAL_INPUTS_READY',index,json.dumps(record),flush=True)
    # The saved correlator is an independent completed production output.
    _,saved_wick,saved_c2,_,_=read_case(row,tj,sep,dt,verify_hash=True)
    baselines={};cons={};records={};samples={m:dict(c2_reduced=[],c2_reference=[],c3_case=[]) for m in MODES}
    for mode in MODES:
        con=make_contractor(mode);cons[mode]=con
        c2,seconds2=timed(lambda:c2_all(data,con))
        c2ref,secondsref=timed(lambda:c2_all(data,con,True))
        wick,seconds3=timed(lambda:c3_case(data,con))
        errors=dict(c2_vs_saved=checked(saved_c2,c2,(mode,'C2 saved')),
            c2_vs_full_spin=checked(c2ref,c2,(mode,'C2 full-spin')),
            signed_wick_vs_saved=checked(saved_wick,wick,(mode,'C3 saved')),
            each_wick={spec[0]:checked(saved_wick[it],wick[it],(mode,spec[0])) for it,spec in enumerate(SPECS)})
        flavor=np.stack((wick[:4].sum(axis=0),wick[4:].sum(axis=0)))
        savedflavor=np.stack((saved_wick[:4].sum(axis=0),saved_wick[4:].sum(axis=0)))
        errors['three_channel_projections']=checked(project_spin(savedflavor),project_spin(flavor),(mode,'projected'))
        records[mode]=dict(cold_seconds=dict(c2_reduced=seconds2,c2_reference=secondsref,c3_case=seconds3),correctness=errors)
        baselines[mode]=(c2,wick)
        print('COLD_AND_CORRECTNESS_PASS',index,mode,'C2',seconds2,'C3',seconds3,flush=True)
    # Paired rounds alternate the order; all modes have already warmed up once.
    execution_order=[]
    for repeat in range(REPEATS):
        order=list(MODES if repeat%2==0 else MODES[::-1])
        if index%2: order=order[::-1]
        execution_order.append(order)
        for mode in order:
            con=cons[mode]
            c2,s2=timed(lambda:c2_all(data,con))
            c2ref,sref=timed(lambda:c2_all(data,con,True))
            wick,s3=timed(lambda:c3_case(data,con))
            checked(baselines['current_optimal'][0],c2,(mode,'paired C2'))
            checked(baselines['current_optimal'][1],wick,(mode,'paired Wick'))
            samples[mode]['c2_reduced'].append(s2)
            samples[mode]['c2_reference'].append(sref)
            samples[mode]['c3_case'].append(s3)
            print('TIMING',index,repeat,mode,'C2',s2,'C2ref',sref,'C3',s3,flush=True)
    profiles={}
    # Profile a representative nonzero sigma/momentum separately from wall-time samples.
    for mode in MODES:
        prof=Profile(cons[mode])
        h=one_right_time_h(data['vertices'],data['a'],0,prof)
        c3_reduced(h,data['p'],data['b'][:1],data['g'],data['q'],data['vvs'][1],data['vvr'][1],prof,return_terms=True)
        c2_reduced(data['p'],data['vvs'][1],data['vvr'][1],data['g'],data['q'],prof)
        profiles[mode]=list(prof.records.values())
        del h
    for mode in MODES:
        records[mode]['warm_samples_seconds']=samples[mode]
        records[mode]['warm_median_seconds']={k:statistics.median(v) for k,v in samples[mode].items()}
        med=records[mode]['warm_median_seconds']
        med['c2_both']=med['c2_reduced']+med['c2_reference']
        med['c3_plus_c2_both']=med['c3_case']+med['c2_both']
        records[mode]['per_equation_profile']=profiles[mode]
        if isinstance(cons[mode],ExplicitContractor):
            dump(HERE/'results'/f'plans_case{index}_{mode}.json',cons[mode].metadata())
    for mode in MODES:
        records[mode]['speedup_vs_current_optimal']={k:records['current_optimal']['warm_median_seconds'][k]/value for k,value in records[mode]['warm_median_seconds'].items()}
    record.update(status='PASS',timing_scope='input/basis/vertex/VVV construction and disk I/O excluded; full C3 sigma/momentum loop includes CPU transfers and production per-sigma pool release; cold planning separate',
        repeats=REPEATS,warmup=1,execution_order=execution_order,modes=records,
        GPU_memory_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        GPU_memory_pool_active_bytes=cp.get_default_memory_pool().used_bytes())
    dump(HERE/'results'/f'case{index}.json',record)
    del data,cons,baselines
    cp.get_default_memory_pool().free_all_blocks()
    return record


def main():
    assert 'SLURM_JOB_ID' in os.environ
    assert json.loads((HERE/'results/cpu_tests.json').read_text())['status']=='PASS'
    lock=json.loads((HERE/'source_lock.json').read_text())
    for path,digest in lock['files_sha256'].items():
        assert sha256(path)==digest,('frozen source changed',path)
    cp.cuda.Device(0).use();cp.get_default_memory_pool().set_limit(size=72*1024**3)
    properties=cp.cuda.runtime.getDeviceProperties(0)
    device=properties['name'];device=device.decode() if isinstance(device,bytes) else str(device)
    context=dict(job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,device=device,
        cupy=cp.__version__,numpy=np.__version__,opt_einsum=oe.__version__,python=sys.version,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
        source_root=str(SOURCE),source_manifest_sha256=sha256(SOURCE/'manifest.json'),
        modes=list(MODES),generated=datetime.now().astimezone().isoformat())
    dump(HERE/'results/environment.json',context)
    print('ENVIRONMENT',json.dumps(context),flush=True)
    manifest=load_manifest()
    cases=[]
    for index,(task,tj,sep,dt) in enumerate(((0,0,7,3),(6,36,12,6))):
        cases.append(one_case(manifest['tasks'][task],tj,sep,dt,manifest,index))
    for path,digest in lock['files_sha256'].items(): assert sha256(path)==digest,path
    summary=dict(status='PASS',environment=context,cases=[{k:v for k,v in case.items() if k!='modes'} for case in cases],
        per_case_medians=[dict(case=index,modes={mode:dict(medians=case['modes'][mode]['warm_median_seconds'],speedups=case['modes'][mode]['speedup_vs_current_optimal']) for mode in MODES}) for index,case in enumerate(cases)],
        production_modified=False,source_integrity_before_and_after='PASS',
        limitations=['L24x72 conf30000 at two extreme production flow times only; L32 ensembles not directly GPU-tested',
            'no full-configuration end-to-end speedup claim; common input, VVV, vertex, basis generation and filesystem time excluded',
            'same GPU process with alternating paired order; one GPU allocation; no run-to-run hardware variability estimate',
            'GPU memory counters describe pool allocator, not a hardware peak'])
    dump(HERE/'results/summary.json',summary)
    print('GPU_EXPLICIT_DP_BENCHMARK_PASS',flush=True)

if __name__=='__main__':
    main()
