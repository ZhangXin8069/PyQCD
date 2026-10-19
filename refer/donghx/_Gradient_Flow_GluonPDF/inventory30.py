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
SELECTED_NEW=list(range(13100,14100,100))
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
    previous=json.loads((HERE.parent/'quark_spin2_expansion20_20260922'/'status'/'inventory20.json').read_text())
    previous_confs=previous['target_confs']
    report=dict(new_configurations=rows,previous_configurations=previous_confs,
        target_confs=previous_confs+list(map(str,SELECTED_NEW)),
        selection='next ten complete input configurations after13000',
        flow_times=[0.05,0.10,0.20,0.50,0.80,1.00],
        common_tseps=[6,7,8,9,10],new_tseps=[5,6,7,8,9,10])
    out=HERE/'status'/'inventory30.json';out.write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:print(row['conf'],row['status'],row['counts'],flush=True)
    if not all(row['status']=='PASS' for row in rows):raise SystemExit(1)


if __name__=='__main__':main()
