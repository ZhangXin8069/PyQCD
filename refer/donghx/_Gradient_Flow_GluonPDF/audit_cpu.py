"""Read every case/receipt and projection; compare unchanged C2 across flows."""
import argparse
from datetime import datetime
import json
import os
import numpy as np
from common import *


def main():
    require_allocation()
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',type=int)
    args=parser.parse_args()
    manifest=load_manifest()
    selected=manifest['tasks'] if args.task is None else [manifest['tasks'][args.task]]
    count=0
    references={}
    max_relative=0.
    ledger=[]
    for row in selected:
        check_inputs(row)
        marker=json.loads((HERE/'status'/f'task{row["task"]}.json').read_text())
        assert marker['status']=='PASS' and marker['case_count']==252
        assert marker['manifest_sha256']==sha256(HERE/'manifest.json')
        for tj in CURRENTS:
            for sep in TSEPS:
                for dt in range(sep+1):
                    _,_,c2,_,receipt=read_case(row,tj,sep,dt,verify_hash=True)
                    key=(row['conf'],tj,sep,dt)
                    if key in references:
                        diff=np.linalg.norm(c2-references[key])/max(np.linalg.norm(references[key]),1e-300)
                        if diff>2e-10:
                            raise AssertionError(('unflowed C2 changed across flow times',key,diff))
                        max_relative=max(max_relative,float(diff))
                    else:
                        references[key]=c2
                    count+=1
        ledger.append(dict(task=row['task'],conf=row['conf'],tau24=row['flow_time_a2'],
                           tau64=row['tau64'],status='PASS',case_count=252))
        print('ALL_CASES_VERIFIED',row['task'],count,flush=True)
    load_request()
    result=dict(status='COMPLETE_L24X72_B_MATCHED_RADIUS_C3_C2' if args.task is None else 'PASS_SINGLE_TASK',
        generated=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        manifest_sha256=sha256(HERE/'manifest.json'),tasks=len(selected),case_count=count,
        nconf=len({r['conf'] for r in selected}),momenta=list(MOMENTA),flows64=list(FLOWS64),
        weight_scheme=SCHEME,C2_cross_flow_relative_L2=max_relative,task_results=ledger,
        scope='ordinary-field finite-flow connected B-scheme C3/C2; no Zchi subtraction or matching',
        remaining=['finite original-time support convergence in C3, not bounded solely by norm tail',
                   'shared higher noise moments/source-sink correlations remain a prescription limitation',
                   'transversity external relative sign confirmation','full six-direction estimator'])
    target=HERE/'summary/complete.json' if args.task is None else HERE/'status'/f'reaudit_task{args.task}.json'
    atomic_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='task_results'},indent=2),flush=True)


if __name__=='__main__':
    main()
