"""Freeze the established measurement, then propagate the original 50 draws."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone
import numpy as np

from quark_ringed_kinetic_v1.core import digest, atomic_npz
from quark_ringed_kinetic_v1.collect import read_raw
from quark_ringed_kinetic_v1.report_gpu import summarize
from quark_ringed_kinetic_v1.measure import input_identity

PROJECT = Path('/public/group/lqcd/donghx/Gradient_Flow_GluonPDF')
REFERENCE = PROJECT/'quark_ringed_mass_pilot_v1/runs/critical_mass_20260918_ei1yb2qj'
BARE = {
    'sixflow_sumdiff': PROJECT/'quark_unpolarized_sixflow_bare_fit_v1/bare_matrix_elements_unpolarized_u_minus_d_sixflow_Nboot50.npz',
    'fourflow_ratio': PROJECT/'quark_unpolarized_ratio_fit_aic_v1/bare_matrix_elements_ratio_AIC_Nboot50.npz',
}
COEFFICIENT = Path('/public/home/donghx/Gradient_Flow/Blending_L32x96/small_flow_time/sfte_coefficients.py')
STATUS = 'paired_10conf_published_critical_valence_mass_diagnostic_NOT_final_PDF'


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: np.array(f[k]) for k in f.files}


def prepare():
    original = json.loads((REFERENCE/'manifest.json').read_text())
    for relative, sha in original['snapshots'].items():
        if digest(REFERENCE/relative) != sha:
            raise ValueError('Changed reference snapshot: '+relative)
    conf, plan, values, meta = read_raw(REFERENCE/'central/kinetic_conf.npz')
    audit = summarize(REFERENCE/'central/kinetic_conf.npz')
    if conf != 10000 or values.shape != (6, 32) or plan['mass'] != -.23539:
        raise ValueError('Wrong completed reference')
    review = json.loads((REFERENCE/'solver_validation.json').read_text())
    if not review['passed'] or not all(c['passed'] for c in review['checks']):
        raise ValueError('Solver review not passed')
    for c in review['checks']:
        for side in ('left', 'right'):
            if digest(c[side]) != c[side+'_sha256']:
                raise ValueError('Changed solver evidence')
    if digest(original['library']) != original['library_sha256']:
        raise ValueError('Changed library')
    if review['new_library_sha256'] != meta['runtime']['libquda_sha256']:
        raise ValueError('Backend review mismatch')
    main = load_npz(BARE['sixflow_sumdiff'])
    other = load_npz(BARE['fourflow_ratio'])
    if not np.array_equal(main['confs'], np.arange(10000,11000,100)):
        raise ValueError('Unexpected bare confs')
    for key in ('confs', 'bootstrap_indices'):
        np.testing.assert_array_equal(main[key], other[key])
    if main['bootstrap_indices'].shape != (50,10):
        raise ValueError('Expected original 50 draws of 10 configurations')
    if not np.array_equal(main['flow_times'], plan['flow_times']):
        raise ValueError('Flow mismatch')
    parent = PROJECT/'quark_ringed_ensemble_v1/runs'
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='critical_10conf_20260920_', dir=parent))
    (root/'code').mkdir()
    # Measurement and all dependencies are immutable copies of the tested run.
    shutil.copytree(REFERENCE/'code/quark_ringed_kinetic_v1', root/'code/quark_ringed_kinetic_v1')
    dst = root/'code/quark_ringed_ensemble_v1'
    dst.mkdir()
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f, dst/f.name)
    for name in ('run_conf.slurm','analyze.slurm'):
        shutil.copy2(Path(__file__).with_name(name), root/name)
    shutil.copy2(PROJECT/'quark_sixflow_ringed_sfte_v1.py', root/'code/quark_sixflow_ringed_sfte_v1.py')
    shutil.copy2(COEFFICIENT, root/'code/sfte_coefficients.py')
    shutil.copy2(REFERENCE/'solver_validation.json', root/'solver_validation.json')
    shutil.copy2(REFERENCE/'plans/central.json', root/'plan.json')
    write_json(root/'reference_audit.json', audit)
    reused = root/'reused/conf10000'
    reused.mkdir(parents=True)
    for f in sorted((REFERENCE/'central').glob('*.npz')):
        shutil.copy2(f, reused/f.name)
    (root/'logs').mkdir()
    (root/'measurements').mkdir()
    for c in main['confs'][1:]:
        cache = root/f'measurements/conf{c}/quda_cache'
        cache.mkdir(parents=True)
        shutil.copy2(REFERENCE/'warm_cache/tunecache.tsv', cache/'tunecache.tsv')
    raw = {str(c): str(root/(f'reused/conf{c}' if c == 10000 else f'measurements/conf{c}')/'kinetic_conf.npz') for c in main['confs']}
    manifest = {
        'schema': 'quark_ringed_10conf_campaign_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'plan': plan, 'confs': main['confs'].tolist(), 'new_confs': main['confs'][1:].tolist(),
        'noise_count':32, 'seed':9182026, 'raw_paths':raw,
        'reference_source':str(REFERENCE), 'reference_kinetic_sha256':digest(REFERENCE/'central/kinetic_conf.npz'),
        'measurement_code_sha256':meta['run_identity']['code_sha256'],
        'library':original['library'], 'library_sha256':original['library_sha256'],
        'gauge_preflight':{str(c):input_identity(plan,int(c),hash_gauge=False) for c in main['confs']},
        'bare':{k:{'path':str(v), 'sha256':digest(v)} for k,v in BARE.items()},
        'bootstrap_contract':{'Ncfg':10,'Nboot':50,'seed':int(main['seed']),
            'indices':'exact array in each pinned bare product; no new draws',
            'central':'inverse of ensemble mean of noise means; times original bare central',
            'error':'sample std(ddof=1) of 50 paired products',
            'covariance':'sample covariance(ddof=1) from shared bootstrap replicas',
            'noise':'configuration bootstrap of fixed 32-noise means; no extra nested resampling'},
        'physics_status':'published critical valence mass on fixed massive sea; retain diagnostic status and existing fit/window masks',
        'snapshots':{str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file() and 'measurements' not in f.parts},
    }
    write_json(root/'manifest.json',manifest)
    print(root,flush=True)


def verify(root):
    m=json.loads((root/'manifest.json').read_text())
    for name,sha in m['snapshots'].items():
        if digest(root/name) != sha: raise ValueError('Changed campaign snapshot: '+name)
    for b in m['bare'].values():
        if digest(b['path']) != b['sha256']: raise ValueError('Changed bare input')
    if digest(m['library']) != m['library_sha256']: raise ValueError('Changed backend')
    return m


def audit_raw(path, expected_conf, m):
    conf,plan,v,meta=read_raw(path)
    if conf != expected_conf or plan != m['plan'] or v.shape != (6,32):
        raise ValueError('Incompatible configuration/plan/count: '+str(path))
    ident=meta['run_identity']
    if ident['seed'] != m['seed'] or ident['direct'] or ident['code_sha256'] != m['measurement_code_sha256']:
        raise ValueError('Changed measurement/noise identity')
    if meta['runtime']['libquda_sha256'] != m['library_sha256']:
        raise ValueError('Changed measurement backend')
    if ident['gauge']['path'] != m['gauge_preflight'][str(conf)]['path']:
        raise ValueError('Wrong gauge input')
    for i in range(32):
        s=load_npz(path.parent/f'noise_{i:05d}.npz')
        sm=json.loads(str(s['metadata_json']))
        if sm['noise_id'] != i or sm['run_identity_sha256'] != meta['run_identity_sha256']:
            raise ValueError('Mismatched noise shard identity')
    a=summarize(path)
    if a['true_residual_max'] > m['plan']['residual_max'] or a['maxima_over_noise']['gauge_unitarity_determinant_errors']>1e-8:
        raise ValueError('Residual or gauge validation failed')
    return v,a


def paired(values, flow, indices, central, bare_boot):
    """values(flow,conf); never average inverse per-configuration energies."""
    if values.ndim != 2 or len(flow) != len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid kinetic configuration means')
    if indices.dtype.kind not in 'iu' or indices.ndim!=2 or indices.shape[1]!=values.shape[1] or indices.min()<0 or indices.max()>=values.shape[1]:
        raise ValueError('Invalid shared bootstrap indices')
    if bare_boot.shape != (len(flow),len(indices),*central.shape[1:]):
        raise ValueError('Invalid bare bootstrap axes')
    e=values.real.mean(axis=1)
    eb=values.real[:,indices].mean(axis=2)
    if np.any(e>=0) or np.any(eb>=0): raise ValueError('Wrong or unstable kinetic sign')
    numerator=-6/((4*np.pi)**2*np.asarray(flow)**2)
    z,zb=numerator/e,numerator[:,None]/eb
    rb=bare_boot*zb[:,:,None,None]
    return {'kinetic_central':e, 'kinetic_bootstrap':eb, 'Zchi':z, 'Zchi_bootstrap':zb,
            'Zchi_error':zb.std(axis=1,ddof=1), 'Zchi_covariance':np.cov(zb,ddof=1),
            'ringed_central':central*z[:,None,None], 'ringed_bootstrap':rb,
            'ringed_error':rb.std(axis=1,ddof=1)}


def analyze(root, recovery_review=None, output=None):
    m=verify(root)
    recovery=None
    if recovery_review is not None:
        recovery=json.loads(Path(recovery_review).read_text())
        if (recovery.get('schema')!='quark_ringed_solver_recovery_review_v1' or
            not recovery.get('passed') or recovery['original_manifest_sha256']!=digest(root/'manifest.json')):
            raise ValueError('Invalid solver-recovery review')
        if recovery['configuration']!=10800:
            raise ValueError('Recovery is scoped to conf10800 only')
        allowed={'recovery_gcr_krylov','recovery_reliable_delta','recovery_corrections',
                 'recovery_outer_precision','recovery_maxiter_per_correction','recovery_precondition_precision','recovery_null_precision'}
        if {k:v for k,v in recovery['plan'].items() if k not in allowed}!=m['plan']:
            raise ValueError('Recovery changed the measurement physics or original solver tolerance')
        for path,sha in recovery['evidence_sha256'].items():
            if digest(path)!=sha:raise ValueError('Changed recovery evidence')
        m['raw_paths']['10800']=recovery['raw_path']
    missing=[c for c in m['confs'] if not Path(m['raw_paths'][str(c)]).is_file()]
    if missing: raise ValueError('Incomplete ensemble, no analysis produced: '+str(missing))
    measurements,audits=[],[]
    for c in m['confs']:
        expected=m
        if c==10800 and recovery is not None:
            expected=dict(m,plan=recovery['plan'],measurement_code_sha256=recovery['measurement_code_sha256'])
        v,a=audit_raw(Path(m['raw_paths'][str(c)]),c,expected)
        measurements.append(v);audits.append(a)
    noise=np.stack(measurements,axis=1)  # flow,conf,noise
    means=noise.mean(axis=2)
    out=Path(output) if output is not None else root/'analysis'
    out.mkdir(parents=True,exist_ok=False)
    # Import the frozen coefficient explicitly: the older collector imports
    # an external module of the same name, so do not rely on module caching.
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_sfte',root/'code/sfte_coefficients.py')
    sfte=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sfte;spec.loader.exec_module(sfte)
    summary={'status':STATUS,'configurations':m['confs'],'Nnoise_per_conf':32,'Nboot':50,
             'measurement_audits':audits,'methods':{},'formal_physics_acceptance':False}
    if recovery is not None:
        summary['solver_recovery_review']={'path':str(Path(recovery_review).resolve()),'sha256':digest(recovery_review)}
    for kind,source in m['bare'].items():
        b=load_npz(source['path']);flow=b['flow_times'];z=b['delta_z_values'];p=b['momenta_integer_pz']
        np.testing.assert_array_equal(b['confs'],m['confs'])
        selection=[]
        for t in flow:
            matches=np.flatnonzero(np.isclose(t,m['plan']['flow_times'],atol=1e-12,rtol=0))
            if len(matches)!=1: raise ValueError('Missing/duplicate flow label')
            selection.append(matches.item())
        if kind=='sixflow_sumdiff':
            central,boot,valid=b['M_filled'],b['M_boot_filled'],b['fit_valid']
        else:
            central,boot,valid=b['M'],b['M_boot'],np.zeros_like(b['M'],dtype=bool)
        indices=b['bootstrap_indices']
        if indices.shape!=(50,10) or not np.isfinite(central).all() or not np.isfinite(boot).all():
            raise ValueError('Changed original bare bootstrap contract')
        d=paired(means[selection],flow,indices,central,boot)
        imag=means[selection].imag
        imean=imag.mean(axis=1); ierror=imag[:,indices].mean(axis=2).std(axis=1,ddof=1)
        imag_pass=np.abs(imean)<=5*ierror+1e-14
        factor=[];windows=[]
        for t in flow:
            f,_,_,_=sfte.gf_to_msbar_factor(z,flow_time_over_a2=t,a_fm=.0775,mu_gev=2.,alpha_s=.30)
            f[z==0]=np.nan;factor.append(f)
            windows.append(sfte.small_flow_diagnostics(z,p,nx=32,flow_time_over_a2=t,a_fm=.0775,
                lambda_qcd_gev=.3,min_flow_radius_over_a=2.,min_z_over_flow_radius=1.,max_sqrt_t_pz=.5,max_t_lambda2=.1))
        factor=np.array(factor)
        sb=d['ringed_bootstrap']*factor[:,None,None,:]
        mask=np.array([w['small_flow_valid_mask_pz_z'] for w in windows])
        dest=out/kind;dest.mkdir()
        meta={'status':STATUS,'formula':'Z_b=-6/[(4pi)^2 tau^2 mean_conf_b(mean_noise E)]; ringed_b=Z_b*M_b',
              'input_bare':source,'raw_input_hashes':{str(a['conf']):a['sha256'] for a in audits},
              'bootstrap_contract':m['bootstrap_contract'],'mass_prescription':m['plan']['critical_mass_reference'],
              'imaginary_check_pass_by_flow':imag_pass.tolist(),'mu_GeV':2.,'alpha_s':.30,
              'statistical_pairing_complete':True,'formal_physics_acceptance':False,
              'unresolved':['mass/action source-artifact and boundary-improvement review','mass-prescription systematic','flow-window/continuum control','full six-direction estimator'],
              'additional_noise_variance_added':False,'LaMET_applied':False}
        if recovery is not None:meta['solver_recovery_review']=summary['solver_recovery_review']
        d.update(schema=np.array('quark_ringed_paired_10conf_v1'),status=np.array(STATUS),
            metadata_json=np.array(json.dumps(meta)),flow_times=flow,momenta_integer_pz=p,delta_z_values=z,
            confs=b['confs'],bootstrap_indices=indices,seed=b['seed'],channel=b['channel'],flavor=b['flavor'],
            estimator_id=b['estimator_id'],a_fm=np.array(.0775),
            bare_central=central,bare_bootstrap=boot,bare_error=boot.std(axis=1,ddof=1),
            fit_valid=valid,kinetic_noise=noise[selection],kinetic_config=means[selection],
            kinetic_imaginary_mean=imean,kinetic_imaginary_bootstrap_error=ierror,kinetic_imaginary_check_pass=imag_pass,
            normalization_noise_ids=np.arange(32),factor_without_Zchi=factor,
            sfte_central=d['ringed_central']*factor[:,None,:],sfte_bootstrap=sb,sfte_error=sb.std(axis=1,ddof=1),
            small_flow_window_mask=mask,diagnostic_quality_mask=mask & valid & imag_pass[:,None,None],
            axes_central=np.array(['flow','pz','z']),axes_bootstrap=np.array(['flow','bootstrap','pz','z']),
            axes_kinetic_noise=np.array(['flow','conf','noise']),formal_result_valid=np.zeros_like(valid))
        for k in ('central_fallback_used','replica_fallback_used','estimate_available','fixed_tmin','fixed_tmin_fm','q_min','correlated_chi2_dof_max'):
            if k in b:d['source_'+k]=b[k]
        atomic_npz(dest/'paired_matrix_elements.npz',**d)
        meta['output_sha256']=digest(dest/'paired_matrix_elements.npz')
        write_json(dest/'manifest.json',meta)
        plots(dest,d,kind)
        summary['methods'][kind]={'flows':flow.tolist(),'Zchi':d['Zchi'].tolist(),
            'Zchi_error':d['Zchi_error'].tolist(),'imaginary_check_pass':imag_pass.tolist(),
            'output':str(dest/'paired_matrix_elements.npz'),'output_sha256':meta['output_sha256']}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary['methods'],indent=2),flush=True)


def plots(out,d,kind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    flow,z,p=d['flow_times'],d['delta_z_values'],d['momenta_integer_pz']
    colors=plt.get_cmap('viridis')(np.linspace(.08,.9,len(flow)))
    shifts=np.linspace(-.20,.20,len(flow))
    handles=[Line2D([],[],color=c,marker='o',ls='none',label=rf'$t/a^2={t:g}$') for c,t in zip(colors,flow)]
    for name,stages in [('ringed_before_after',('bare','ringed')),('gradient_flow_matching_before_after',('ringed','sfte'))]:
        with PdfPages(out/(name+'.pdf')) as pdf:
            for zmax in (15,20):
                fig,axs=plt.subplots(2,3,figsize=(15,9))
                for row,stage in enumerate(stages):
                    for ip,ax in enumerate(axs[row]):
                        for i,c in enumerate(colors):
                            y,e=d[stage+'_central'][i,ip],d[stage+'_error'][i,ip]
                            good=d['fit_valid'][i,ip].copy()
                            if stage=='sfte':good &= d['small_flow_window_mask'][i,ip]
                            for ok in (True,False):
                                s=(good==ok)&np.isfinite(y)&np.isfinite(e)&(z<=zmax)
                                ax.errorbar(z[s]+shifts[i],y[s],yerr=e[s],fmt='o',ls='none',ms=3.5,capsize=1.5,
                                    color=c,mfc=c if ok else 'white',alpha=.95 if ok else .65)
                        ax.set(xlim=(-.5,zmax+.5),xlabel=r'$z/a$',ylabel=stage if ip==0 else '',title=rf'$n_z={p[ip]}$')
                        ax.axhline(0,color='.6',ls='--',lw=.6);ax.grid(alpha=.15)
                fig.suptitle('Quark u-d unpolarized | '+kind+' | 10 conf paired normalization',y=.99)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=len(flow),frameon=False)
                fig.text(.5,.015,'50 original paired gauge-bootstrap replicas; published critical valence mass.\nOpen: fit fallback/unweighted diagnostic and/or outside SFTE window. Physics systematics not included.',ha='center',fontsize=9)
                fig.subplots_adjust(top=.86,bottom=.11,hspace=.29,wspace=.28)
                pdf.savefig(fig);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','verify','check-conf','analyze','status'))
    p.add_argument('run',nargs='?',type=Path)
    p.add_argument('--conf',type=int)
    p.add_argument('--recovery-review',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare();return
    if a.run is None:p.error('Need run directory')
    root=a.run.resolve()
    if a.action=='analyze':analyze(root,a.recovery_review,a.output);return
    m=verify(root)
    if a.action=='check-conf':
        _,audit=audit_raw(Path(m['raw_paths'][str(a.conf)]),a.conf,m)
        print(json.dumps(audit,indent=2));return
    if a.action=='status':
        for c in m['confs']:
            path=Path(m['raw_paths'][str(c)])
            print(c,'complete' if path.is_file() else 'incomplete',len(list(path.parent.glob('noise_*.npz'))))
    else:print('Frozen inputs and backend verified',flush=True)


if __name__=='__main__':main()
