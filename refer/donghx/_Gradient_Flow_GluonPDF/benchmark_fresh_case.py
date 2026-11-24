"""Recreate production's distinct fresh C2/C3 Contractors per case after GPU warmup."""
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np

import benchmark_gpu as bench
from benchmark_gpu import cp,SOURCE,make_contractor,prepare,c3_case,checked,timed,dump
from common import load_manifest,read_case,sha256
from spin2_contractions import c2_reference,c2_reduced

HERE=Path(__file__).resolve().parent
MODES=('current_optimal','explicit_dp')


def c2_oracle(data,con):
    values=[]
    for vs,vr in zip(data['vvs'],data['vvr']):
        got=data['boundary']*cp.asnumpy(c2_reduced(data['p'],vs,vr,data['g'],data['q'],con))
        gold=data['boundary']*cp.asnumpy(c2_reference(data['p'],vs,vr,data['g'],data['q'],con))
        checked(gold,got,'fresh C2 full-spin oracle')
        values.append(got)
    return np.concatenate(values,axis=1)


def main():
    assert 'SLURM_JOB_ID' in os.environ
    assert json.loads((HERE/'results/summary.json').read_text())['status']=='PASS'
    lock=json.loads((HERE/'fresh_source_lock.json').read_text())
    for path,digest in lock['files_sha256'].items():assert sha256(path)==digest,path
    cp.cuda.Device(0).use();cp.get_default_memory_pool().set_limit(size=72*1024**3)
    manifest=load_manifest();cases=[]
    for index,(task,tj,sep,dt) in enumerate(((0,0,7,3),(6,36,12,6))):
        data,record=prepare(manifest['tasks'][task],tj,sep,dt,manifest)
        _,saved_wick,saved_c2,_,_=read_case(manifest['tasks'][task],tj,sep,dt)
        print('FRESH_INPUTS_READY',index,record['preparation_seconds'],flush=True)
        # Warm GPU kernels and allocators; these objects are deliberately discarded.
        for mode in MODES:
            checked(saved_c2,c2_oracle(data,make_contractor(mode)),(mode,'warm C2'))
            checked(saved_wick,c3_case(data,make_contractor(mode)),(mode,'warm C3'))
        samples={mode:[] for mode in MODES};plans={mode:[] for mode in MODES}
        for repeat in range(3):
            order=MODES if repeat%2==0 else MODES[::-1]
            if index%2:order=order[::-1]
            for mode in order:
                con2=make_contractor(mode)
                got2,s2=timed(lambda:c2_oracle(data,con2))
                con3=make_contractor(mode)
                got3,s3=timed(lambda:c3_case(data,con3))
                errors=dict(C2=checked(saved_c2,got2,(mode,'fresh C2')),
                    Wick=checked(saved_wick,got3,(mode,'fresh C3')),
                    each_wick=[checked(saved_wick[i],got3[i],(mode,'fresh Wick',i)) for i in range(6)])
                samples[mode].append(dict(repeat=repeat,c2_oracle_seconds=s2,c3_case_seconds=s3,
                    sum_seconds=s2+s3,correctness=errors))
                if mode=='explicit_dp':
                    plans[mode].append(dict(C2=con2.metadata(),C3=con3.metadata()))
                print('FRESH_TIMING',index,repeat,mode,s2,s3,s2+s3,flush=True)
        medians={mode:{key:statistics.median(r[key] for r in rows) for key in ('c2_oracle_seconds','c3_case_seconds','sum_seconds')} for mode,rows in samples.items()}
        speedups={key:medians['current_optimal'][key]/medians['explicit_dp'][key] for key in medians['current_optimal']}
        record.update(status='PASS',job=os.environ['SLURM_JOB_ID'],samples=samples,medians=medians,speedups=speedups,
            scope='warmed GPU kernels; new independent Contractor for C2 and C3 for EVERY production case; shape planning included; input/VVV/vertex excluded')
        dump(HERE/'results'/f'fresh_case{index}.json',record)
        dump(HERE/'results'/f'fresh_case{index}_dp_plans.json',plans['explicit_dp'])
        cases.append(record)
        del data,con2,con3
        cp.get_default_memory_pool().free_all_blocks()
    for path,digest in lock['files_sha256'].items():assert sha256(path)==digest,path
    dump(HERE/'results/fresh_summary.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        generated=datetime.now().astimezone().isoformat(),source_integrity_before_and_after='PASS',cases=cases,
        warning='This measures the contraction portion of fresh production cases; common data preparation and flow generation are outside timing'))
    print('FRESH_CASE_BENCHMARK_PASS',flush=True)

if __name__=='__main__':main()
