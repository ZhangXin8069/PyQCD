"""Short real-gauge columns for paired private/shared QUDA comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import load_manifest,config_for,NT,NX,require_allocation,atomic_json
from flow_backend import initialize_pyquda
from generate_basis import evolve


def main():
    require_allocation()
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    row=load_manifest()['tasks'][6]
    cfg=config_for(row);state=initialize_pyquda(cfg)
    eig=np.load(cfg.eigen_path(0),mmap_mode='r')
    lex=np.zeros((1,NT,NX,NX,NX,4,3),np.complex128)
    for k in range(4):
        lex[0,0,:,:,:,k,:]=eig[k]
    result=evolve(state,lex)
    assert np.isfinite(result).all()
    with args.output.open('wb') as stream:
        np.save(stream,result,allow_pickle=False)
    atomic_json(args.output.with_suffix('.json'),state['quda_backend'])


if __name__=='__main__':
    main()
