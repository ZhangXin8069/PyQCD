#!/usr/bin/env python3
"""Read-back checks for the no-z0 branch; never change source fit products."""
import argparse
import json
from pathlib import Path
import numpy as np
from gluon_absolute_normalization_v1 import kinematic_factor,local_spin2_coefficient
from lambda_fourier_quasi_matching_v2 import SOURCE,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=SOURCE.parent/'lambda_fourier_quasi_no_z0_v1')
    p.add_argument('--write-report',action='store_true')
    a=p.parse_args();m=json.loads((a.out/'manifest.json').read_text())
    assert digest(m['input'])==m['input_sha256']
    for path,sha in m['scripts'].items():assert digest(path)==sha,path
    checks=[];rows=[]
    with np.load(a.out/'no_z0_xg.npz',allow_pickle=False) as d,np.load(m['input'],allow_pickle=False) as src:
        np.testing.assert_array_equal(d['bootstrap_indices'],src['bootstrap_indices'])
        np.testing.assert_array_equal(d['confs'],src['confs'])
        checks.append('unchanged_configuration_bootstrap_identities')
        original=np.concatenate([src['params_selected'][None,...,0],src['params_boot_selected'][...,0]])
        np.testing.assert_array_equal(d['source_R'],original)
        factor=kinematic_factor(d['energy_lattice'],m['settings']['alpha_s'])
        np.testing.assert_allclose(d['kinematic_factor'],factor,rtol=1e-14)
        np.testing.assert_allclose(d['H'][:,:,1:],np.moveaxis(original[:,:,1:]*factor[:,:,None],1,0),rtol=1e-14)
        np.testing.assert_allclose(d['H'][:,:,0],(d['local_R']*factor).T,rtol=1e-14)
        np.testing.assert_allclose(d['local_moment'],d['H'][:,:,0].T/2,rtol=1e-14)
        assert not np.allclose(d['H'][:,:,0],1.)
        checks.append('absolute_scale_preserved_no_z0_division_no_unit_moment')
        np.testing.assert_allclose(d['local_coefficient'],local_spin2_coefficient(
            d['flow_times'],m['settings']['a_fm'],m['settings']['mu'],m['settings']['alpha_s']))
        assert d['matched'].shape==(3,1001,201)
        mask=d['computed_mask'];assert np.isfinite(d['matched'][:,:,mask]).all()
        assert np.isnan(d['matched'][:,:,~mask]).all()
        np.testing.assert_allclose(d['matched'][:,:,mask],d['matched'][:,:,::-1][:,:,mask],atol=1e-8)
        checks.append('all_1000_samples_finite_in_computed_x_domain_and_even')
        for ip,n in enumerate(d['pabs_values']):
            with np.load(a.out/'per_momentum'/f'Pz{n}.npz',allow_pickle=False) as f:
                np.testing.assert_array_equal(f['matched'],d['matched'][ip])
                np.testing.assert_allclose(f['local_matched'],f['local_bare']*(2-f['local_coefficient'])[None])
                row={'n_z':int(n)}
                for stage in ['energy','local','tail']:
                    np.testing.assert_allclose(f[stage+'_weights'].sum(),1.,atol=1e-12)
                    np.testing.assert_allclose(f[stage+'_weights_boot'].sum(axis=1),1.,atol=1e-12)
                    assert f[stage+'_selected_available'] and f[stage+'_bootstrap_selected_available'].all()
                    good=f[stage+'_bootstrap_quality_accepted'];fallback=f[stage+'_replica_fallback_used']
                    assert np.all(good^fallback)
                    row[stage+'_fallback_count']=int(fallback.sum())
                rows.append(row)
        checks.append('independent_quality_AIC_and_explicit_fallback_at_every_stage')
    numerical={'order16','lambda320','kernel96','xi100','xi400','y_halfstep','y_support4','y_support16','y_support24'}
    max_numerical=max(c['max_shift_over_stat'] for c in m['checks'] if c['variation'] in numerical)
    # Record, do not silently reclassify a large numerical shift as physics.
    report=dict(status='PASS' if max_numerical<.1 else 'NUMERICAL_REVIEW_REQUIRED',
        checks=checks,replica_counts=rows,max_numerical_shift_over_stat_x02_08=max_numerical,
        physics_validation=False,reason='finite-a, gg-only and model-dependent diagnostic')
    print(json.dumps(report,indent=2))
    if a.write_report:(a.out/'readback_validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
