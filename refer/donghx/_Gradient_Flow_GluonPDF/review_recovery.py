"""Check the completed numerical-only recovery, including the old noise-0 anchor."""
from pathlib import Path
import argparse
import json
import numpy as np
from .pipeline import verify, audit_raw, load_npz, write_json
from quark_ringed_kinetic_v1.core import digest
from quark_ringed_kinetic_v1.report_gpu import summarize


def review(original, recovered, output):
    m=verify(original)
    rmanifest=json.loads((recovered/'manifest.json').read_text())
    for name,sha in rmanifest['snapshots'].items():
        if digest(recovered/name)!=sha:raise ValueError('Changed recovery snapshot: '+name)
    plan=json.loads((recovered/'plan.json').read_text())
    numerical={'recovery_gcr_krylov','recovery_reliable_delta','recovery_corrections',
               'recovery_outer_precision','recovery_maxiter_per_correction','recovery_precondition_precision','recovery_null_precision'}
    if {k:v for k,v in plan.items() if k not in numerical}!=m['plan']:
        raise ValueError('Non-numerical recovery-plan change')
    raw=recovered/'measurement/kinetic_conf.npz'
    d=load_npz(raw)
    meta=json.loads(str(d['metadata_json']))
    code=meta['run_identity']['code_sha256']
    if code['core.py']!=m['measurement_code_sha256']['core.py']:
        raise ValueError('Kinetic estimator changed')
    for name,sha in code.items():
        if digest(recovered/'code/quark_ringed_kinetic_v1'/name)!=sha:raise ValueError('Changed recovery source')
    expected=dict(m,plan=plan,measurement_code_sha256=code)
    _,audit=audit_raw(raw,10800,expected)
    free=recovered/'free/kinetic_conf.npz'
    fa=summarize(free)
    if fa['true_residual_max']>1e-10 or fa['maxima_over_noise']['free_same_noise_max_relative_error']>5e-4:
        raise ValueError('Recovery free-field check failed')
    old=original/'measurements/conf10800/noise_00000.npz'
    new=recovered/'measurement/noise_00000.npz'
    a,b=load_npz(old),load_npz(new)
    ma,mb=[json.loads(str(x['metadata_json'])) for x in (a,b)]
    for key in ('gauge','seed','conf','noise_count','direct'):
        if ma['run_identity'][key]!=mb['run_identity'][key]:raise ValueError('Anchor uses different source/gauge')
    if ma['noise_id']!=0 or mb['noise_id']!=0:raise ValueError('Wrong anchor noise')
    if not float(a['true_residual'])<=1e-9:raise ValueError('Old anchor is not residual-accepted')
    da=a['kinetic_a4_noise'];db=b['kinetic_a4_noise']
    if not np.all(np.abs(da-db)<=1e-12+1e-7*np.abs(da)):
        raise ValueError('Same-noise solver observable mismatch')
    paths=[original/'manifest.json',recovered/'manifest.json',recovered/'plan.json',raw,free,old]
    paths += sorted((recovered/'measurement').glob('noise_*.npz'))
    paths += sorted((recovered/'free').glob('noise_*.npz'))
    paths += [recovered/'code/quark_ringed_kinetic_v1'/k for k in code]
    result={'schema':'quark_ringed_solver_recovery_review_v1','passed':True,'configuration':10800,
        'original_manifest_sha256':digest(original/'manifest.json'),'raw_path':str(raw.resolve()),
        'plan':plan,'measurement_code_sha256':code,'physics_parameters_unchanged':True,
        'free_field':fa,'recovery_audit':audit,'anchor_old':str(old),'anchor_new':str(new),
        'anchor_max_relative_difference':float(np.max(np.abs((db-da)/da))),
        'anchor_threshold':{'rtol':1e-7,'atol':1e-12},
        'evidence_sha256':{str(p.resolve()):digest(p) for p in paths},
        'scope':'numerical solver equivalence and full 32-noise completion; not full physics acceptance'}
    write_json(output,result)
    print(json.dumps({'passed':True,'max_residual':audit['true_residual_max'],
        'anchor_max_relative_difference':result['anchor_max_relative_difference']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original',type=Path);p.add_argument('recovered',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();review(a.original.resolve(),a.recovered.resolve(),a.output)
