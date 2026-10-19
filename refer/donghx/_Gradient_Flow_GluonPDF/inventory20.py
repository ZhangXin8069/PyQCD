#!/usr/bin/env python3
"""Input/header inventory for the additional ten confs and existing products.

This is not a numerical scan; runtime validate_inputs checks all NPY headers.
"""
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE=Path(__file__).resolve().parent
SELECTED_NEW=[11000,11100,12300,12400,12500,12600,12700,12800,12900,13000]
sys.path.insert(0,str(HERE/'reference'))
from flow_common_l32x96 import FlowConfig,DEFAULT_SAVE_ROOT


def audit_new(conf):
    c=FlowConfig.from_env(str(conf));bad=[];counts={};samples=[]
    gauge=c.gauge_path();expected=96*32**3*4*3*3*16
    if not gauge.is_file() or gauge.stat().st_size!=expected:bad.append(str(gauge))
    for kind,base,pathfn,shape in [
        ('peram',c.peram_root/c.conf,c.peram_path,(96,4,4,400,400)),
        ('eigen',c.eigen_root/c.conf,c.eigen_path,(600,32,32,32,3))]:
        if not base.is_dir():
            bad.append(str(base));counts[kind]=0;continue
        files={entry.name:entry.stat().st_size for entry in os.scandir(base) if entry.is_file()}
        sample=pathfn(0)
        if sample.name not in files:
            bad.append(str(sample));counts[kind]=sum(pathfn(t).name in files for t in range(96));continue
        x=np.load(sample,mmap_mode='r',allow_pickle=False)
        if x.shape!=shape or x.dtype!=np.dtype('complex128'):bad.append(f'{sample}: {x.shape}/{x.dtype}')
        size=sample.stat().st_size
        samples.append(dict(path=str(sample),shape=list(x.shape),dtype=str(x.dtype),bytes=size))
        del x
        counts[kind]=0
        for t in range(96):
            path=pathfn(t)
            if files.get(path.name)!=size:bad.append(str(path))
            else:counts[kind]+=1
    return dict(conf=str(conf),status='PASS' if not bad else 'FAIL',counts=counts,
        gauge=str(gauge),gauge_bytes=gauge.stat().st_size if gauge.is_file() else None,samples=samples,errors=bad,
        flow_intermediates_present=c.root.is_dir(),
        scope='all filenames and byte sizes; t000 NPY shape/dtype; full headers checked at runtime')


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(audit_new,SELECTED_NEW))
    old=[];c=FlowConfig.from_env('10000')
    for conf in range(10000,11000,100):
        root=Path(DEFAULT_SAVE_ROOT)/c.tag/'Nev_200'/str(conf)
        outputs=[root/f'corr_projected_tsep{t}_dz0-20_Nev200.npz' for t in range(6,11)]
        old.append(dict(conf=str(conf),result_dir=str(root),
            files=[dict(path=str(p),exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else None) for p in outputs],
            scope='existing output filenames and sizes only; prior numerical checks not repeated'))
    report=dict(new_configurations=rows,existing_configurations=old,
        target_confs=list(map(str,list(range(10000,11000,100))+SELECTED_NEW)),flow_time=.5,
        selection='first ten available new input configurations after10900; empty11200..12200 skipped',
        skipped_missing_inputs=list(map(str,range(11200,12300,100))),
        common_tseps=[6,7,8,9,10],new_tseps=[5,6,7,8,9,10])
    out=HERE/'status'/'inventory20.json';out.write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:print(row['conf'],row['status'],row['counts'],flush=True)
    print('old10_present',all(f['exists'] for row in old for f in row['files']),flush=True)
    if not all(row['status']=='PASS' for row in rows):raise SystemExit(1)


if __name__=='__main__':main()
