#!/usr/bin/env python3
"""Validate saved products without refitting or modifying upstream data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    args=p.parse_args();out=args.out
    m=json.loads((out/'manifest.json').read_text())
    source=Path(m['input'])
    assert sha(source)==m['input_sha256']
    code_dir=Path(__file__).resolve().parent
    for name,expected in m['code_hashes'].items():
        assert sha(code_dir/name)==expected,name
    for path in out.rglob('*.npz'):
        with np.load(path,allow_pickle=False) as d:
            for key in d.files:assert not d[key].dtype.hasobject,(path,key)
    counts=[];statistics=[]
    with np.load(source,allow_pickle=False) as src,np.load(out/'lambda_fourier_quasi_matching.npz',allow_pickle=False) as allout:
        np.testing.assert_array_equal(allout['bootstrap_indices'],src['bootstrap_indices'])
        np.testing.assert_array_equal(allout['confs'],src['confs'])
        np.testing.assert_array_equal(allout['source_central'],src['params_selected'][...,0])
        np.testing.assert_array_equal(allout['source_bootstrap'],src['params_boot_selected'][...,0])
        assert allout['matched'].shape==(3,1001,201)
        h=np.concatenate([src['params_selected'][None,...,0],src['params_boot_selected'][...,0]])
        norm=np.moveaxis(h/h[:,:,0,None],1,0)
        np.testing.assert_array_equal(norm,allout['h_normalized_data'])
        for ip,n in enumerate(allout['pabs_values']):
            with np.load(out/f'per_momentum/Pz{n}.npz',allow_pickle=False) as d:
                np.testing.assert_array_equal(d['bootstrap_indices'],src['bootstrap_indices'])
                np.testing.assert_array_equal(d['h_normalized_data'],norm[ip])
                np.testing.assert_allclose(d['h_reconstructed'][:,0],1.)
                np.testing.assert_array_equal(d['matched'],allout['matched'][ip])
                mask=d['computed_mask'];v=d['matched'];x=d['x']
                assert np.isfinite(v[:,mask]).all()
                assert np.isnan(v[:,~mask]).all()
                np.testing.assert_allclose(v,v[:,::-1],atol=1e-8)
                expected=d['quasi_ms']-d['scheme_correction']-d['lamet_correction']
                np.testing.assert_allclose(v,expected)
                nb=len(d['tail_weights_boot'])
                assert nb==1000
                np.testing.assert_allclose(d['tail_weights'].sum(),1.)
                np.testing.assert_allclose(d['tail_weights_boot'].sum(axis=1),1.)
                status=d['tail_bootstrap_quality_accepted']|d['tail_replica_fallback_used']
                assert status.all()
                fb=int(d['tail_replica_fallback_used'].sum());counts.append(fb)
                # Recompute every recorded sensitivity from paired samples.
                region=(x>=.2-1e-10)&(x<=.8+1e-10)
                err=np.std(v[1:],axis=0,ddof=1)
                for check in (c for c in m['numerical_checks'] if c['n_z']==int(n)):
                    delta=d['variation_'+check['variation']]-v
                    np.testing.assert_allclose(check['max_shift_over_stat'],np.max(abs(delta[0,region])/err[region]))
                    np.testing.assert_allclose(check['max_paired_shift_error'],np.max(np.std(delta[1:,region],axis=0,ddof=1)))
                statistics.append(dict(n_z=int(n),finite_replicas=1000,tail_fallback=fb,
                    original_vs_boot_mean_max_shift=float(np.max(abs(v[0,mask]-v[1:,mask].mean(axis=0)))),
                    table=[dict(x=float(x[j]),central=float(v[0,j]),stat=float(err[j])) for j in [120,140,160,180]]))
    numerical=[c for c in m['numerical_checks'] if not c['variation'].startswith('join_')]
    tail=[c for c in m['numerical_checks'] if c['variation'].startswith('join_')]
    max_numeric=max(c['max_shift_over_stat'] for c in numerical)
    assert max_numeric<.1,'Numerical changes exceed 0.1 statistical sigma in 0.2<=x<=0.8'
    report=dict(status='PASS_saved_product_and_numerical_checks',
        checks=['source and code hashes','no object arrays','shared conf/bootstrap identities',
                'original central kept distinct','1000 finite replicas at each computed x',
                'strict NLO algebra','even parity','normalized h(0)=1','AIC coverage/fallback masks',
                'paired sensitivity metrics','numeric central changes <0.1 stat sigma in x=[0.2,0.8]'],
        tail_fallback_counts=counts,max_numeric_shift_over_stat=max_numeric,
        max_tail_splice_shift_over_stat=max(c['max_shift_over_stat'] for c in tail),
        statistics=statistics,physical_validation='NOT established by these numerical tests',
        uncertainty='statistical only in main bands; model spread and splice alternatives saved separately')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
