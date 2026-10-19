#!/usr/bin/env python3
"""Full payload/hash/axes/finite audit after all GPU insertion tasks pass."""
import csv
import json
import os
import numpy as np
from common import (HERE,CURRENTS,TSEPS,SCHEME,load_manifest,check_inputs,
                    read_case,case_times,sha256,atomic_json)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('CPU allocation required')
    m=load_manifest()
    mh=sha256(HERE/'manifest.json')
    for task in (0,7):
        smoke=json.loads((HERE/'status'/f'smoke_task{task}.json').read_text())
        assert smoke['status']=='PASS' and smoke['manifest_sha256']==mh
    records=[]
    max_cross_flow=0.
    for ic,conf in enumerate(m['confs']):
        c2_reference={}
        for row in [r for r in m['tasks'] if r['conf']==conf]:
            check_inputs(row)
            marker=json.loads((HERE/'status'/f'task{row["task"]}.json').read_text())
            assert marker['status']=='PASS' and marker['manifest_sha256']==mh
            assert marker['case_count']==252 and not marker['smoke']
            count=reused=0
            for tj in CURRENTS:
                for sep in TSEPS:
                    for dt in range(sep+1):
                        c3,wick,c2,path,receipt=read_case(row,tj,sep,dt,verify_hash=True)
                        key=(tj,sep,dt)
                        if key not in c2_reference:
                            c2_reference[key]=c2
                        rel=float(np.linalg.norm(c2-c2_reference[key])/
                                  max(np.linalg.norm(c2_reference[key]),1e-300))
                        assert rel<=m['algebra_relative_l2_gate'],('C2 across flow',conf,key,rel)
                        max_cross_flow=max(max_cross_flow,rel)
                        count+=1
                        reused+=int(dt==sep//2)
            assert count==252 and reused==24
            records.append(dict(conf=conf,tau=row['flow_time_a2'],case_count=count,
                reused_midpoints=reused,new_cases=count-reused,status='COMPLETE',
                window=row['window'],task=row['task']))
            print('AUDITED_CONF_FLOW',row['task'],conf,row['flow_time_a2'],count,flush=True)
    assert sum(r['case_count'] for r in records)==40320
    assert sum(r['new_cases'] for r in records)==36480
    out=HERE/'summary'
    with (out/'configuration_progress.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    atomic_json(out/'complete.json',dict(status='COMPLETE_ALL_INSERTIONS_FOUR_CURRENTS_FINITE_WINDOW',
        configuration_count=20,flow_count=8,case_count=40320,new_case_count=36480,
        reused_midpoint_case_count=3840,flows=m['flows'],confs=m['confs'],weight_scheme=SCHEME,
        manifest_sha256=mh,job=os.environ['SLURM_JOB_ID'],records=records,
        max_C2_cross_flow_relative_L2=max_cross_flow,
        caveats=['four current anchors, not all64 source/current times',
                 'fixed widest finite window; no exact full-time proof',
                 'contact endpoints are stored, not approved for fits',
                 'raw connected C3/C2, no bootstrap/fit/Zchi/SFTE/LaMET']))
    print('B_INSERTIONS_FULL_AUDIT_PASS_40320_CASES',flush=True)


if __name__=='__main__':
    main()
