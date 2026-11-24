"""Parallel read-only numerical audit, whole configurations kept in one shard."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'quark_l32x96_B_expand70_matched_radius_20261109'
sys.path.insert(0,str(SOURCE))
import numpy as np
from common import (CURRENTS,TSEPS,MOMENTA,FLOWS64,SCHEME,load_manifest,load_request,
                    check_inputs,read_case,sha256,atomic_json,require_allocation)


def verify_lock():
    lock=json.loads((HERE/'audit_source_lock.json').read_text())
    for name,digest in lock['files_sha256'].items():
        assert sha256(HERE/name)==digest,name
    assert sha256(SOURCE/'manifest.json')==lock['manifest_sha256']
    assert sha256(SOURCE/'request.json')==lock['request_sha256']
    return lock


def worker(index):
    verify_lock()
    manifest=load_manifest();digest=sha256(SOURCE/'manifest.json')
    confs=set(manifest['confs'][index::10])
    rows=[row for row in manifest['tasks'] if row['conf'] in confs]
    references={};max_relative=0.;ledger=[];count=0
    for row in rows:
        check_inputs(row)
        marker=json.loads((SOURCE/'status'/f'task{row["task"]}.json').read_text())
        assert marker['status']=='PASS' and marker['case_count']==252 and marker['task']==row['task']
        assert marker['manifest_sha256']==digest
        for tj in CURRENTS:
            for sep in TSEPS:
                for dt in range(sep+1):
                    _,_,c2,_,receipt=read_case(row,tj,sep,dt,verify_hash=True)
                    key=(row['conf'],tj,sep,dt)
                    if key in references:
                        relative=float(np.linalg.norm(c2-references[key])/max(np.linalg.norm(references[key]),1e-300))
                        assert np.isfinite(relative) and relative<=2e-10,('C2 cross flow',key,relative)
                        max_relative=max(max_relative,relative)
                    else:
                        references[key]=c2
                    count+=1
        ledger.append(dict(task=row['task'],conf=row['conf'],tau96=row['flow_time_a2'],tau64=row['tau64'],status='PASS',case_count=252))
        print('ALL_CASES_VERIFIED',row['task'],count,flush=True)
    load_request();verify_lock()
    result=dict(status='PASS',worker=index,generated=datetime.now().astimezone().isoformat(),
        job=os.environ['SLURM_JOB_ID'],manifest_sha256=digest,tasks=len(rows),case_count=count,
        confs=sorted(confs,key=int),C2_cross_flow_relative_L2=max_relative,task_results=ledger,
        method='original check_inputs/read_case with hash and exact projection checks; C2 cross-flow check retained within each conf')
    atomic_json(HERE/'audit_shards'/f'worker{index:02d}.json',result)
    print('AUDIT_SHARD_PASS',index,count,flush=True)


def collect():
    verify_lock();manifest=load_manifest();digest=sha256(SOURCE/'manifest.json')
    ledger=[];shards=[]
    for index in range(10):
        d=json.loads((HERE/'audit_shards'/f'worker{index:02d}.json').read_text())
        assert d['status']=='PASS' and d['worker']==index and d['manifest_sha256']==digest
        expected_confs=manifest['confs'][index::10]
        assert d['confs']==sorted(expected_confs,key=int)
        expected_tasks=[r['task'] for r in manifest['tasks'] if r['conf'] in set(expected_confs)]
        assert [r['task'] for r in d['task_results']]==expected_tasks
        assert d['tasks']==len(expected_tasks) and d['case_count']==252*len(expected_tasks)
        ledger.extend(d['task_results']);shards.append(d)
    ledger.sort(key=lambda r:r['task'])
    assert [r['task'] for r in ledger]==list(range(140))
    assert all(r['status']=='PASS' and r['case_count']==252 for r in ledger)
    load_request();verify_lock()
    result=dict(status='COMPLETE_L32X96_B_MATCHED_RADIUS_C3_C2',generated=datetime.now().astimezone().isoformat(),
        job=os.environ['SLURM_JOB_ID'],manifest_sha256=digest,tasks=140,case_count=35280,nconf=20,
        momenta=list(MOMENTA),flows64=list(FLOWS64),weight_scheme=SCHEME,
        C2_cross_flow_relative_L2=max(d['C2_cross_flow_relative_L2'] for d in shards),task_results=ledger,
        scope='ordinary-field finite-flow connected B-scheme C3/C2; no Zchi subtraction or matching',
        remaining=['finite original-time support convergence in C3, not bounded solely by norm tail',
                   'shared higher noise moments/source-sink correlations remain a prescription limitation',
                   'transversity external relative sign confirmation','full six-direction estimator'],
        parallel_audit_shard_root=str(HERE/'audit_shards'),
        parallel_audit_jobs=[d['job'] for d in shards],
        method='all saved cases reread; original input fingerprints, arrays, hashes, Wick/projection identities and C2 cross-flow checks preserved')
    target=SOURCE/'summary/complete.json'
    if target.exists():
        raise RuntimeError('Final summary already exists; inspect concurrent validation before replacing')
    atomic_json(target,result)
    subprocess.run([sys.executable,'-B','-u',str(SOURCE/'finalize_cpu.py')],cwd=SOURCE,check=True)
    atomic_json(HERE/'reaudit_complete.json',dict(status='PASS',generated=result['generated'],job=result['job'],
        new20_summary=str(target),combined70_summary=str(SOURCE/'summary/combined70_complete.json'),
        all_cases=35280,C2_cross_flow_relative_L2=result['C2_cross_flow_relative_L2']))
    print('READBACK_AND_COMBINED70_PASS',flush=True)

if __name__=='__main__':
    require_allocation()
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--collect',action='store_true')
    args=parser.parse_args()
    if args.collect:collect()
    elif args.worker is not None and 0<=args.worker<10:worker(args.worker)
    else:raise ValueError(args)
