#!/usr/bin/env python3
"""Production-size paired CPU/GPU timed C2/C3 parity-reduction comparison.

Reads original perambulators, eigenvectors and all temporal vertex blocks.
One source (default 0), tsep=5, all dt=0..5, z=0..20, both line signs,
Pz=+3,+4,+5, nopol/pol15/pol25/pol35, u/d and all 16 current-spin entries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
from opt_einsum import contract_expression

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
from flow_common_l32x96 import FlowConfig,iter_vertex_blocks,spin_objects,seq_peram,baryon_vvv,phase_3pt
from flowed_perambulator_factorization import accumulate_triplet,build_flowed_sequential
from spin2_contractions import Contractor,c2_reference,c2_reduced,c3_reduced

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def compare(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise FloatingPointError('nonfinite')
    dif=np.abs(a-b)
    peak=float(np.max(np.abs(a)))
    result=dict(max_abs=float(dif.max()),relative_L2=float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(a.ravel()),1e-300)),
                relative_peak=float(dif.max()/max(peak,1e-300)))
    if max(result['relative_L2'],result['relative_peak'])>2e-10:raise AssertionError(result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=int,default=0)
    parser.add_argument('--tsep',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dt-list',default='0,1,2,3,4,5')
    parser.add_argument('--gpu-memory-gib',type=float,default=74.)
    parser.add_argument('--mom-batch',type=int,choices=[1,3],default=3)
    args=parser.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(out/'summary.json')
    import cupy as cp
    cp.cuda.Device(0).use();cp.get_default_memory_pool().set_limit(size=int(args.gpu_memory_gib*1024**3))
    def sync():cp.cuda.Device(0).synchronize()
    def timed(fn):
        sync();st=time.perf_counter();val=fn();sync();return val,time.perf_counter()-st
    contractor=Contractor('optimal')
    g,_,q=spin_objects(cp);config=FlowConfig.from_env('10000')
    assert config.nev_src==200 and config.nev_current==400 and config.delta_z==21
    source=args.source;sink=(source+args.tsep)%96
    boundary=-1 if source+args.tsep>=96 else 1
    dts=[int(x) for x in args.dt_list.split(',')];assert all(0<=dt<=args.tsep for dt in dts)
    sources={};setup={};vv={};nzlist=[3,4,5]
    def stamp(p):
        stat=p.stat();sources[str(p)]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    runstarted=time.perf_counter()
    props=cp.cuda.runtime.getDeviceProperties(0)
    info=dict(conf='10000',source=source,sink=sink,tsep=args.tsep,dt_list=dts,flow_time=config.flow_time,
        flow_epsilon=config.epsilon,temporal_window=config.temporal_window,Nev_src=200,Nev_current=400,
        dz=list(range(21)),line_signs=[-1,1],Pz_in_lattice_units=nzlist,pol=['nopol','pol15','pol25','pol35'],
        current_spins='all 4x4 retained',dtype='complex128 contractions; stored complex64 vertex, identical both methods',
        boundary=boundary,host=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),gpu=props['name'].decode(),
        cupy=cp.__version__,numpy=np.__version__,optimize='optimal',repeats=args.repeats,momentum_batch=args.mom_batch,
        warmup='one full execution per method and per shape; times exclude warmup/planning',
        timing='perf_counter with device synchronize before/after; alternate execution order',
        sources_complete=False)
    save_json(out/'run_settings.json',info);print('BEGIN '+json.dumps(info),flush=True)
    for t in [source,sink]:
        ep=config.eigen_path(t);stamp(ep);st=time.perf_counter()
        eig=cp.asarray(np.load(ep,mmap_mode='r')[:200]);vv[t]=[]
        for nz in nzlist:
            val=baryon_vvv(phase_3pt([nz,0,0],xp=cp),eig,xp=cp)[0]
            vv[t].append(val);sync();print(f'VVV time={t} nz={nz} elapsed={time.perf_counter()-st:.3f}',flush=True)
        vv[t]=cp.stack(vv[t]);del eig;cp.get_default_memory_pool().free_all_blocks()
        sync();setup[f'VVV_t{t}']=time.perf_counter()-st
    sv=vv[source].conj();tv=vv[sink]
    rawsource=np.load(config.peram_path(source),mmap_mode='r');rawsink=np.load(config.peram_path(sink),mmap_mode='r')
    stamp(config.peram_path(source));stamp(config.peram_path(sink))
    p,setup['peram_sink_H2D']=timed(lambda:cp.asarray(np.ascontiguousarray(rawsource[sink,:,:,:200,:200])))
    records=[]
    def measure_pair(label,fns,normalizer=lambda x:np.array(cp.asnumpy(x))):
        vals={};times={k:[] for k in fns};warm={}
        for k,fn in fns.items():
            a,warm[k]=timed(fn);vals[k]=normalizer(a);del a
            print(f'WARM {label} {k} {warm[k]:.6f}s',flush=True)
        checks=compare(vals['reference'],vals['spin2'])
        for rep in range(args.repeats):
            order=['reference','spin2'] if rep%2==0 else ['spin2','reference']
            for k in order:
                a,seconds=timed(fns[k]);times[k].append(seconds)
                compare(vals[k],normalizer(a));del a
                print(f'TIMING {label} rep={rep} {k} {seconds:.6f}s',flush=True)
        record=dict(label=label,correctness=checks,warmup_seconds=warm,timing_seconds=times,
            median_seconds={k:float(np.median(v)) for k,v in times.items()})
        record['speedup']=record['median_seconds']['reference']/record['median_seconds']['spin2']
        records.append(record);save_json(out/'progress.json',dict(records=records,setup_seconds=setup))
        print('RESULT '+json.dumps(record),flush=True)
        return vals
    def c2_call(fn):
        return cp.concatenate([fn(p,tv[i:i+args.mom_batch],sv[i:i+args.mom_batch],g,q,contractor)
                               for i in range(0,3,args.mom_batch)],axis=1)
    c2=measure_pair('C2',dict(reference=lambda:c2_call(c2_reference),spin2=lambda:c2_call(c2_reduced)))
    np.savez(out/'C2.npz',reference=boundary*c2['reference'],spin2=boundary*c2['spin2'])
    # Common setup is measured independently, once; both paths consume the same H.
    for dt in dts:
        current=(source+dt)%96;st=time.perf_counter();blocks=list(iter_vertex_blocks(config,current,['z']))
        taus=sorted({int(b[0]) for b in blocks});pos={t:i for i,t in enumerate(taus)}
        n=len(taus);assert n==2*config.temporal_window+1
        vertex=cp.empty((n,n,2,21,400,400),dtype=cp.complex64)
        for ta,tb,direction,path,metadata in blocks:
            stamp(path);stamp(path.with_suffix('.json'))
            vertex[pos[ta],pos[tb]]=cp.asarray(np.load(path,mmap_mode='r'))
        sync();setup[f'dt{dt}_vertex_read_H2D']=time.perf_counter()-st
        def make_a():
            raw=cp.asarray(np.ascontiguousarray(rawsink[taus,:,:,:400,:200]))
            return seq_peram(raw,cp)
        a,setup[f'dt{dt}_sequential_read_H2D']=timed(make_a)
        b,setup[f'dt{dt}_source_read_H2D']=timed(lambda:cp.asarray(np.ascontiguousarray(rawsource[taus,:,:,:400,:200])))
        h,setup[f'dt{dt}_common_flowed_H']=timed(lambda:build_flowed_sequential(vertex,a,contractor))
        del vertex,a;cp.get_default_memory_pool().free_all_blocks()
        print(f'INPUT_READY dt={dt} H_shape={h.shape} H_GiB={h.nbytes/1024**3:.3f} setup={time.perf_counter()-st:.3f}s',flush=True)
        def ref():
            u=cp.zeros((2,4,3,21,4,4),dtype=cp.complex128);d=cp.zeros_like(u)
            for i in range(0,3,args.mom_batch):
                ids=list(range(i,i+args.mom_batch))
                accumulate_triplet({'z':h},['z']*len(ids),p,b,g,q,tv[ids],sv[ids],u,d,ids,contractor)
            return u,d
        def reduced():
            parts=[c3_reduced(h,p,b,g,q,tv[i:i+args.mom_batch],sv[i:i+args.mom_batch],contractor)
                   for i in range(0,3,args.mom_batch)]
            return tuple(cp.concatenate([part[k] for part in parts],axis=2) for k in range(2))
        def hostpair(result):return np.stack([cp.asnumpy(a) for a in result])
        vals=measure_pair(f'C3_dt{dt}',dict(reference=ref,spin2=reduced),hostpair)
        # Check each flavor and each polarization separately as well as the aggregate.
        for flavor in range(2):
            for pol in range(4):compare(vals['reference'][flavor,:,pol],vals['spin2'][flavor,:,pol])
        np.savez(out/f'C3_dt{dt}.npz',reference=boundary*vals['reference'],spin2=boundary*vals['spin2'])
        del h,b,vals;cp.get_default_memory_pool().free_all_blocks()
    for path,metadata in sources.items():
        s=Path(path).stat();assert [s.st_size,s.st_mtime_ns]==[metadata['size'],metadata['mtime_ns']],path
    info['sources_complete']=True
    save_json(out/'input_files.json',sources)
    setup_total=sum(setup.values());r=sum(x['median_seconds']['reference'] for x in records);s=sum(x['median_seconds']['spin2'] for x in records)
    result=dict(status='PASS',settings=info,records=records,setup_seconds=setup,
        contraction_median_sum_reference=r,contraction_median_sum_spin2=s,contraction_speedup=r/s,
        one_pass_measured_common_setup_seconds=setup_total,
        common_setup_plus_contractions_speedup=(setup_total+r)/(setup_total+s),
        actual_benchmark_wall_seconds=time.perf_counter()-runstarted,
        limitations=['one source time and one conf; not an all-source production timing',
                     'common setup includes VVV construction/read and shared H, but no full-output write or MPI',
                     'source wrap sign identical; flow temporal sums include wrapping taus',
                     'no claim of speedup for gauge/fermion flow or Dirac inversions'],
        scripts_sha256={str(path.relative_to(HERE)):sha(path) for path in HERE.rglob('*.py')})
    save_json(out/'summary.json',result);print('DONE '+json.dumps(result),flush=True)

if __name__=='__main__':main()
