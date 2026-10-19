#!/usr/bin/env python3
"""Read-only preflight; explicitly list missing configurations and vertex blocks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
from flow_common_l32x96 import FlowConfig,active_source_times,validate_flow_metadata,read_json,FLOWED_VERTEX_SCHEMA,FLOWED_BASIS_SCHEMA


def audit(conf):
    config=FlowConfig.from_env(conf);missing=[];errors=[];counts=dict(eigen=0,peram=0,basis_metadata=0,vertex=0)
    for kind,base in [('eigen',config.eigen_root),('peram',config.peram_root)]:
        if not (base/conf).is_dir():missing.append(str(base/conf))
    if missing:return dict(conf=conf,status='MISSING',missing=missing,counts=counts)
    for t in range(96):
        for kind,path,shape in [('eigen',config.eigen_path(t),(600,32,32,32,3)),
                                ('peram',config.peram_path(t),(96,4,4,400,400))]:
            if not path.is_file():missing.append(str(path));continue
            try:
                x=np.load(path,mmap_mode='r',allow_pickle=False)
                assert x.shape==shape and x.dtype==np.dtype('complex128'),(x.shape,x.dtype)
                del x;counts[kind]+=1
            except Exception as ex:errors.append(f'{path}: {ex}')
        path=config.basis_done_path(t)
        if not path.is_file():missing.append(str(path))
        else:
            try:
                meta=read_json(path);validate_flow_metadata(config,meta,expected_schema=FLOWED_BASIS_SCHEMA,path=path)
                assert meta['tau']==t and meta['discarded_norm_fraction']<=1e-6
                counts['basis_metadata']+=1
            except Exception as ex:errors.append(f'{path}: {ex}')
        taus=active_source_times(config,t)
        for ta in taus:
            for tb in taus:
                path=config.vertex_path(t,ta,tb,'z');done=path.with_suffix('.json')
                if not path.is_file() or not done.is_file():missing.append(str(path));continue
                try:
                    meta=read_json(done);validate_flow_metadata(config,meta,expected_schema=FLOWED_VERTEX_SCHEMA,path=done)
                    assert (meta['t_current'],meta['tau_a'],meta['tau_b'],meta['direction'])==(t,ta,tb,'z')
                    x=np.load(path,mmap_mode='r',allow_pickle=False)
                    assert x.shape==(2,21,400,400) and x.dtype==np.dtype('complex64')
                    del x;counts['vertex']+=1
                except Exception as ex:errors.append(f'{path}: {ex}')
    return dict(conf=conf,status='PASS' if not (missing or errors) else 'MISSING_OR_INVALID',counts=counts,
                missing=missing,errors=errors,flow_time=config.flow_time,flow_tag=config.tag,
                checks='file existence, NPY header and payload extent through mmap, metadata; no full data scan')


def main():
    p=argparse.ArgumentParser();p.add_argument('confs',nargs='+');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[]
    for conf in a.confs:
        r=audit(conf);rows.append(r);print(conf,r['status'],r['counts'],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
