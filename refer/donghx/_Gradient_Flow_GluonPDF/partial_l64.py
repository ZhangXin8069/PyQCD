"""Interim CPU diagnostic of completed L64; final all-copy audit remains required."""
import json
import os
from pathlib import Path
import core as c
import analysis as a

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    import numpy as np
    for name,digest in json.loads((ROOT/'source_lock.json').read_text())['sha256'].items():
        assert c.sha(ROOT/name)==digest,name
    req=json.loads((ROOT/'request.json').read_text())
    ens=req['ensembles'][0]
    a.self_test()
    values={r:[] for r in ('fundamental','adjoint')}
    selected={str(k):[] for k in (1,2,3,4)}
    for fi in range(4):
        paths=list((ROOT/'runs'/f'task{fi:02d}').glob('job*'))
        assert len(paths)==1
        out=paths[0]
        result=json.loads((out/'result.json').read_text())
        assert json.loads((out/'complete.json').read_text())['result_sha256']==c.sha(out/'result.json')
        assert result['status']=='PASS_MULTISTART_NUMERICS'
        for rep in values:
            copies=[]
            for si in range(4):
                p=out/f'start{si}'/'final_converged.npz'
                m=json.loads(p.with_suffix('.json').read_text())
                assert c.sha(p)==m['npz_sha256'] and m['theta']<=1e-12
                with np.load(p,allow_pickle=False) as d:
                    copies.append(d[rep].real.mean(axis=(0,1)))
            values[rep].append(copies)
        for k in selected:
            selected[k].append(result['selected_start_by_prefix'][k])
    fits=[]
    for rep,vs in values.items():
        vs=np.asarray(vs)
        for lo,hi in req['diagnostic']['windows_a']:
            for k,choice in selected.items():
                v=np.asarray([vs[fi,si] for fi,si in enumerate(choice)])
                result=a.fit(v,ens['taus'],0,lo,hi)
                fits.append(dict(representation=rep,lo=lo,hi=hi,best_of=int(k),choice=choice,**result))
    c.write_json(ROOT/'partial_l64.json',dict(status='INTERIM_DIAGNOSTIC_PENDING_FINAL_AUDIT',
        selected=selected,fits=fits,ncfg=1,ensemble_d_F=None,ensemble_d_A=None))
    print(json.dumps(dict(selected=selected,fits=[{k:v for k,v in row.items() if k in
        ('representation','lo','hi','best_of','d_diag','residual_rms','status')} for row in fits]),indent=2))
