"""Validate preserved products separately when original perambulators are gone."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'quark_l24x72_B_conf51to100_matched_radius_20261105'
NWORKERS=10


def main():
    assert 'SLURM_JOB_ID' in os.environ
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    parser.add_argument('--collect',action='store_true')
    args=parser.parse_args()
    if args.collect:
        records=[json.loads((HERE/'outputs_only_audit'/f'worker{i:02d}.json').read_text()) for i in range(NWORKERS)]
        assert all(r['status']=='PASS_SAVED_OUTPUTS_ONLY_INPUTS_NOT_REVALIDATED' for r in records)
        confs=[c for r in records for c in r['configurations']]
        assert sorted(confs,key=int)==list(map(str,range(20000,30000,200)))
        result=dict(status='PASS_SAVED_OUTPUTS_ONLY_INPUTS_NOT_REVALIDATED',
            generated=datetime.now().astimezone().isoformat(),source_root=str(SOURCE),nconf=len(confs),
            tasks=sum(r['tasks'] for r in records),case_count=sum(r['case_count'] for r in records),
            C2_cross_flow_relative_L2=max(r['C2_cross_flow_relative_L2'] for r in records),
            input_provenance_revalidation='INCOMPLETE: original perambulators disappeared; this result does not replace the original end-to-end audit',
            worker_results=records)
        assert result['tasks']==350 and result['case_count']==88200
        (HERE/'saved_outputs_complete.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='worker_results'},indent=2),flush=True)
        return
    assert args.worker is not None and 0<=args.worker<NWORKERS
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    from common import load_manifest,sha256,read_case,CURRENTS,TSEPS,atomic_json
    manifest=load_manifest()
    digest=sha256(SOURCE/'manifest.json')
    confs=manifest['confs'][args.worker::NWORKERS]
    count=0;tasks=0;max_relative=0.;missing=[]
    for conf in confs:
        refs={}
        selected=[r for r in manifest['tasks'] if r['conf']==conf]
        missing.extend(x['path'] for x in selected[0]['inputs'] if not Path(x['path']).exists())
        for row in selected:
            marker=json.loads((SOURCE/'status'/f'task{row["task"]}.json').read_text())
            assert marker['status']=='PASS' and marker['case_count']==252 and marker['manifest_sha256']==digest
            for tj in CURRENTS:
                for sep in TSEPS:
                    for dt in range(sep+1):
                        _,_,c2,_,_=read_case(row,tj,sep,dt,verify_hash=True)
                        key=(tj,sep,dt)
                        if key in refs:
                            relative=float(np.linalg.norm(c2-refs[key])/max(np.linalg.norm(refs[key]),1e-300))
                            assert relative<=2e-10,(conf,key,relative)
                            max_relative=max(max_relative,relative)
                        else:
                            refs[key]=c2
                        count+=1
            tasks+=1
            print('SAVED_OUTPUTS_VERIFIED',row['task'],conf,'cases',count,flush=True)
    result=dict(status='PASS_SAVED_OUTPUTS_ONLY_INPUTS_NOT_REVALIDATED',
        generated=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        worker=args.worker,configurations=confs,tasks=tasks,case_count=count,
        C2_cross_flow_relative_L2=max_relative,source_manifest_sha256=digest,
        missing_original_inputs=missing,input_provenance_revalidation='INCOMPLETE')
    atomic_json(HERE/'outputs_only_audit'/f'worker{args.worker:02d}.json',result)
    print('SAVED_OUTPUTS_WORKER_PASS',args.worker,tasks,count,flush=True)

if __name__=='__main__':
    main()
