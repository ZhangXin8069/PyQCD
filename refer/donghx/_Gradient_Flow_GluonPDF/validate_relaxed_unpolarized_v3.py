#!/usr/bin/env python3
"""Independent numerical audit of the v3 selected products; no refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2 as chi2_dist


def check_selection(f,beta,boot,chi,chi_boot,dof,numerical):
    # beta=(candidate,p,z,param), boot=(boot,candidate,p,z,param)
    np.testing.assert_allclose(np.einsum('wpz,wpzk->pzk',f['weights'],beta),f['params_selected'])
    np.testing.assert_allclose(np.einsum('bwpz,bwpzk->bpzk',f['weights_boot'],boot),f['params_boot_selected'])
    np.testing.assert_allclose(f['weights'].sum(axis=0),1.,atol=1e-12)
    np.testing.assert_allclose(f['weights_boot'].sum(axis=1),1.,atol=1e-12)
    np.testing.assert_allclose(f['stat_error'],f['params_boot_selected'].std(axis=0,ddof=1))
    err=boot[...,0].std(axis=0,ddof=1)
    np.testing.assert_allclose(f['candidate_statistical_error'],err)
    expected=(np.isfinite(boot).all(axis=-1)&numerical[None]&(dof[None]>0)
        &(chi2_dist.sf(chi_boot,np.maximum(dof,1)[None])>=1e-6)
        &(chi_boot/np.maximum(dof,1)[None]<=10))
    np.testing.assert_array_equal(f['candidate_boot_quality'],expected)
    np.testing.assert_array_equal(f['bootstrap_quality_accepted'],expected.any(axis=1))
    assert np.isfinite(f['params_boot_selected']).all()
    assert f['bootstrap_selected_available'].all()
    assert not np.any(f['replica_fallback_used']&f['bootstrap_quality_accepted'])
    for b,p,z in np.argwhere(f['replica_fallback_used']):
        assert not expected[b,:,p,z].any()
        pool=np.flatnonzero(numerical[:,p,z]&(dof[:,p,z]>0)&np.isfinite(boot[b,:,p,z]).all(axis=-1))
        assert len(pool)>0, 'Unexpected zero-dof fallback in actual data'
        order=np.lexsort((pool,chi_boot[b,pool,p,z],err[pool,p,z]))
        expected_index=pool[order[0]]
        assert expected_index==f['replica_fallback_candidate_index'][b,p,z]
        np.testing.assert_array_equal(f['params_boot_selected'][b,p,z],boot[b,expected_index,p,z])
    return dict(complete_replicas=int(f['bootstrap_selected_available'].sum()),
                quality_replicas=int(f['bootstrap_quality_accepted'].sum()),
                fallback_replicas=int(f['replica_fallback_used'].sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'unpolarized_relaxed_q1em6_chi10_minerr_v3')
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text())
    reference=None;reports=[]
    for record in manifest['bare_summaries']:
        with np.load(record['output'],allow_pickle=False) as f:
            current={k:f[k] for k in ['confs','bootstrap_indices']}
            if reference is None:reference=current
            else:
                for k in reference:np.testing.assert_array_equal(current[k],reference[k])
            assert float(f['q_min'])==1e-6 and float(f['chi2_dof_max'])==10
            np.testing.assert_allclose(f['params_boot_selected'],f['bare_params_boot_selected']*f['matching_factors'][None,...,None])
            result=check_selection(f,f['candidate_M'][...,None],np.moveaxis(f['candidate_M_boot'],1,0)[...,None],
                f['candidate_chi2'],np.moveaxis(f['candidate_chi2_boot'],1,0),f['candidate_dof'],f['candidate_numerical'])
            result['tau']=record['tau'];reports.append(result)
    path=root/'unpolarized_tau0_selected.npz'
    with np.load(path,allow_pickle=False) as f:
        for k in reference:np.testing.assert_array_equal(f[k],reference[k])
        flow=check_selection(f,f['candidate_beta'],f['candidate_beta_boot'],f['candidate_chi2'],
                            f['candidate_chi2_boot'],f['candidate_dof'],f['candidate_identifiable'])
        assert np.isfinite(f['params_selected']).all()
        flow['any_stage_replica_fallback_count_z1to15']=int(f['any_stage_replica_fallback'][:,:,1:16].sum())
        flow['max_any_stage_fraction_z1to15']=float(f['any_stage_replica_fallback_fraction'][:,1:16].max())
    for record in manifest['input_records']:
        s=Path(record['path']).stat()
        assert (s.st_size,s.st_mtime_ns)==(record['bytes'],record['mtime_ns'])
    for name,digest in manifest['code_hashes'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
    report=dict(status='pass',bare=reports,flow=flow,final_npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checks=['shared_indices','complete_finite_coverage','quality_gates','weight_normalization',
                        'weighted_estimator_reconstruction','std_all_selected_replicas','matching_scalar_identity',
                        'fallback_argmin_error_then_chi2_then_index','source_size_mtime','code_hashes'])
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
