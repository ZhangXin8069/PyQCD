"""CPU-only local-normalization audit, paired transformation, and plots."""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from normalization import local_normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = {
    'moving': (ROOT/'quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2',
               tuple(i/10 for i in range(1,9)), (3,4,5,6)),
    'Pz0': (ROOT/'quark_l32x64_B_Pz0_ratio20_boot100_T7_Q1em3_chi1e3_20261021',
            tuple(i/10 for i in range(2,9)), (0,)),
}
CHANNELS = ('unpolarized','helicity','transversity')
FLAVORS = ('u','d','u_minus_d')
COMPONENTS = ('even_real','odd_imag')
CONFS = tuple(map(str,range(10000,10800,40)))
INDEX_NAME = 'bootstrap_indices_Nboot100_seed1115.npy'
INDEX_SHA = '86c034214702c29af6671956f4ff23c7749c4e34af103a9173d03da455b6056b'
SCHEME = 'same_time_second_cross_time_independent_product'
LOCAL_SCHEME = 'same_channel_same_flavor_same_P_same_tau_even_real_z0_v1'
NBOOT = 100


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    def safe(value):
        if isinstance(value,float) and not math.isfinite(value):
            return None
        if isinstance(value,dict):
            return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [safe(v) for v in value]
        return value
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(safe(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)


def write_npz(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**data)
    os.replace(tmp,path)


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tag(tau):
    return f'tau{tau:.2f}'.replace('.','p')


def source_path(branch,tau,channel,flavor):
    return INPUTS[branch][0]/'fits'/f'bare_ratio_AIC_{tag(tau)}_{channel}_{flavor}.npz'


def output_path(branch,tau,channel,flavor):
    return HERE/'normalized'/branch/f'localnorm_{tag(tau)}_{channel}_{flavor}.npz'


def check_lock():
    lock=json.loads((HERE/'code_lock.json').read_text())
    for name,expected in lock['files_sha256'].items():
        if sha(HERE/name)!=expected:
            raise RuntimeError(f'Frozen local code changed: {name}')
    for name,expected in lock['upstream_json_sha256'].items():
        if sha(name)!=expected:
            raise RuntimeError(f'Frozen input provenance changed: {name}')


def load_verified(branch,tau,channel,flavor,expected,indices):
    path=source_path(branch,tau,channel,flavor)
    digest=sha(path)
    if digest!=expected[str(path)]:
        raise RuntimeError(f'Input hash does not match upstream complete.json: {path}')
    with np.load(path,allow_pickle=False) as archive:
        d={key:archive[key] for key in archive.files}
    pz=INPUTS[branch][2]
    shape=(2,len(pz),21)
    for key in ('M_selected','selected_available','quality_accepted','fallback_used',
                'central_fallback_used','exact_symmetry_zero'):
        if d[key].shape!=shape:
            raise ValueError((path,key,d[key].shape,shape))
    for key in ('M_boot_selected','fallback_used_boot'):
        if d[key].shape!=(2,NBOOT,len(pz),21):
            raise ValueError((path,key,d[key].shape))
    np.testing.assert_array_equal(d['boot_indices'],indices)
    np.testing.assert_array_equal(d['confs'],CONFS)
    np.testing.assert_array_equal(d['component_labels'],COMPONENTS)
    np.testing.assert_array_equal(d['pz_values'],pz)
    np.testing.assert_array_equal(d['delta_z_values'],np.arange(21))
    assert str(d['axes_matrix'])=='component,p,z'
    assert str(d['axes_boot'])=='component,boot,p,z'
    assert str(d['channel'])==channel and str(d['flavor_combo'])==flavor
    assert str(d['weight_scheme'])==SCHEME and int(d['weight_applied_count'])==1
    assert str(d['bootstrap_indices_sha256'])==INDEX_SHA
    assert int(d['nconf'])==20 and int(d['nboot'])==NBOOT
    assert float(d['tau_a2'])==tau and float(d['a_fm'])==.0897
    assert int(d['fit_Tmin'])==7 and float(d['q_min'])==1e-3
    assert float(d['chi2_dof_max'])==1e3
    # The projected imaginary odd local component is an algebraic zero. It is
    # never a denominator; fail instead of silently ignoring an incompatible file.
    np.testing.assert_array_equal(d['M_selected'][1,:,0],0)
    np.testing.assert_array_equal(d['M_boot_selected'][1,:,:,0],0)
    return d,digest


def transform(branch,tau,channel,flavor,d,digest):
    r=local_normalize(d['M_selected'],d['M_boot_selected'])
    denominator=d['M_selected'][0,:,0]
    db=d['M_boot_selected'][0,:,:,0]
    dfb=d['fallback_used_boot'][0,:,:,0]
    central_fb=d['central_fallback_used'] | d['central_fallback_used'][0,:,0][None,:,None]
    fb=d['fallback_used_boot'] | dfb[None,:,:,None]
    fallback=central_fb | fb.any(axis=1)
    quality=d['quality_accepted'] & d['quality_accepted'][0,:,0][None,:,None]
    # No statistical-quality claim comes from normalizing a variable by itself.
    anchor=np.zeros_like(r['selected_available'])
    anchor[0,:,0]=True
    exactzero=d['exact_symmetry_zero'].copy()
    error=np.std(db,axis=0,ddof=1)
    sign_flips=np.sum(db*denominator[None,:]<0,axis=0)
    invalid=np.sum(~r['denominator_valid_boot'],axis=0)
    quantiles=np.percentile(db,[2.5,16,50,84,97.5],axis=0)
    interval_crosses_zero=(quantiles[0]<=0)&(quantiles[-1]>=0)
    snr=np.divide(np.abs(denominator),error,out=np.full_like(error,np.nan),where=error>0)
    # SNR<3 and the empirical central-95% interval are diagnostics, not masks.
    # Do not use these heuristics to remove otherwise finite replicas or points.
    unstable=(sign_flips>0)|(invalid>0)|interval_crosses_zero|(snr<3)
    strict=(r['selected_available'] & quality & ~fallback
            & ~unstable[None,:,None] & ~anchor & ~exactzero)
    status=np.full(r['selected_available'].shape,'inherited_fit_or_denominator_diagnostic',dtype='U80')
    status[strict]='upstream_quality_no_fallback_denominator_no_warning'
    status[anchor]='local_normalization_constraint_not_a_fit'
    status[exactzero]='exact_symmetry_zero_not_a_fit'
    status[~r['selected_available']]='missing_finite_central_or_complete_bootstrap'
    metadata_keys=('ensemble','confs','nconf','nboot','seed','boot_indices','tau_a2','a_fm',
                   'flow_tag','epsilon','nstep','temporal_window','pz_values','signed_pz_values',
                   'delta_z_values','tseps','current_anchors','weight_scheme','weight_applied_count',
                   'component_labels','bootstrap_indices_sha256','transversity_convention_status',
                   'fit_Tmin','fit_Tmin_fm','q_min','chi2_dof_max','fit_model','aic_policy',
                   'estimator_id','sampling_scope','denominator_scope','momentum_average_stage')
    payload={key:d[key] for key in metadata_keys}
    payload.update(r)
    payload.update(channel=np.asarray(channel),flavor_combo=np.asarray(flavor),
        axes_matrix=np.asarray('component,p,z'),axes_boot=np.asarray('component,boot,p,z'),
        axes_denominator=np.asarray('p'),axes_denominator_boot=np.asarray('boot,p'),
        normalization_scheme=np.asarray(LOCAL_SCHEME),
        normalization_formula=np.asarray('H_c,b(P,z,tau)=M_c,b(P,z,tau)/M_even_real,b(P,z=0,tau)'),
        local_anchor_constraint=anchor,exact_symmetry_zero=exactzero,
        upstream_numerator_quality_accepted=d['quality_accepted'],
        upstream_denominator_quality_accepted=d['quality_accepted'][0,:,0],
        joint_upstream_quality_accepted=quality,
        upstream_numerator_fallback_used=d['fallback_used'],
        upstream_denominator_fallback_used=d['fallback_used'][0,:,0],
        central_fallback_used=central_fb,fallback_used=fallback,fallback_used_boot=fb,
        fallback_fraction=fb.mean(axis=1),strict_quality_nonconstraint=strict,status=status,
        denominator_statistical_error=error,denominator_sign_flip_count=sign_flips,
        denominator_invalid_bootstrap_count=invalid,
        denominator_quantile_probabilities=np.asarray([.025,.16,.5,.84,.975]),
        denominator_quantiles=quantiles,denominator_central_over_bootstrap_sigma=snr,
        denominator_empirical95_crosses_zero=interval_crosses_zero,
        denominator_diagnostic_warning=unstable,
        denominator_warning_policy=np.asarray('any sign flip, invalid replica, empirical 95% crosses zero, or |central|/std<3; diagnostic only; no replica rejection'),
        bare_M_selected=d['M_selected'],bare_M_boot_selected=d['M_boot_selected'],
        upstream_window_systematic=d['window_systematic'],upstream_cut_systematic=d['cut_systematic'],
        fit_systematic_status=np.asarray('not_propagated_through_nonlinear_local_ratio; original numerator scales only, not normalized errors'),
        bootstrap_error_definition=np.asarray('std(H_local_boot,axis=1,ddof=1); all 100 paired samples; no sqrt(Nboot) divisor'),
        central_estimator=np.asarray('ratio of original-ensemble fitted numerator and local fitted denominator; not mean of bootstrap ratios'),
        input_fit=np.asarray(str(source_path(branch,tau,channel,flavor))),
        input_fit_sha256=np.asarray(digest),input_branch=np.asarray(branch),
        input_complete_sha256=np.asarray(sha(INPUTS[branch][0]/'complete.json')),
        code_lock_sha256=np.asarray(sha(HERE/'code_lock.json')),
        pipeline_version=np.asarray('L32x64_B_paired_local_normalization_v1'),
        physical_result_status=np.asarray('finite_flow_connected_local_ratio_diagnostic; Zchi_cancels; NOT_ringed_absolute_MSbar_or_PDF'),
        line_mass_subtraction_applied=np.asarray(False),SFTE_applied=np.asarray(False),
        LaMET_matching_applied=np.asarray(False),local_charge_restored=np.asarray(False))
    path=output_path(branch,tau,channel,flavor)
    write_npz(path,payload)
    with np.load(path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['boot_indices'],d['boot_indices'])
        np.testing.assert_array_equal(saved['H_local_boot'],r['H_local_boot'])
        np.testing.assert_array_equal(saved['H_local'][0,:,0][r['denominator_valid']],1)
        # Independent scalar oracle at fixed replicas and separations, all momenta.
        checks=0
        for co in range(2):
            for b in (0,1,50,99):
                for ip in range(len(denominator)):
                    for z in (0,1,5,20):
                        div=float(d['M_boot_selected'][0,b,ip,0])
                        val=float(d['M_boot_selected'][co,b,ip,z])
                        expected=val/div if np.isfinite(div) and div!=0 and np.isfinite(val) else np.nan
                        np.testing.assert_allclose(saved['H_local_boot'][co,b,ip,z],expected,equal_nan=True)
                        checks+=1
        if r['selected_available'].all():
            np.testing.assert_allclose(saved['H_local_boot']*db[None,:,:,None],
                                       d['M_boot_selected'],rtol=2e-14,atol=1e-14)
    denom_rows=[]
    for ip,pz in enumerate(d['pz_values']):
        denom_rows.append(dict(branch=branch,tau=tau,channel=channel,flavor=flavor,Pz=int(pz),
            local_even_real=float(denominator[ip]),statistical_error=float(error[ip]),
            central_over_bootstrap_sigma=float(snr[ip]),
            min_abs_bootstrap=float(np.min(np.abs(db[:,ip]))),
            bootstrap_q025=float(quantiles[0,ip]),bootstrap_median=float(quantiles[2,ip]),
            bootstrap_q975=float(quantiles[-1,ip]),sign_flip_count=int(sign_flips[ip]),
            invalid_bootstrap_count=int(invalid[ip]),diagnostic_warning=bool(unstable[ip]),
            upstream_quality_accepted=bool(d['quality_accepted'][0,ip,0]),
            upstream_fallback_used=bool(d['fallback_used'][0,ip,0]),
            upstream_fallback_fraction=float(d['fallback_fraction'][0,ip,0]),
            input_fit=str(source_path(branch,tau,channel,flavor)),input_sha256=digest))
    point_rows=[]
    for co,label in enumerate(COMPONENTS):
        for ip,pz in enumerate(d['pz_values']):
            for z in range(21):
                point_rows.append(dict(branch=branch,tau=tau,channel=channel,flavor=flavor,
                    component=label,Pz=int(pz),z_a=z,H_local=float(r['H_local'][co,ip,z]),
                    statistical_error=float(r['statistical_error'][co,ip,z]),
                    selected_available=bool(r['selected_available'][co,ip,z]),
                    strict_quality_nonconstraint=bool(strict[co,ip,z]),
                    joint_upstream_quality_accepted=bool(quality[co,ip,z]),
                    fallback_used=bool(fallback[co,ip,z]),
                    denominator_warning=bool(unstable[ip]),
                    local_anchor_constraint=bool(anchor[co,ip,z]),
                    exact_symmetry_zero=bool(exactzero[co,ip,z]),status=str(status[co,ip,z]),
                    normalized_file=str(path)))
    return path,denom_rows,point_rows,checks


def make_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':11,
                         'savefig.dpi':130,'pdf.fonttype':42})
    colors=plt.get_cmap('tab10').colors
    paths=[]
    for branch,(_,flows,pz) in INPUTS.items():
        folder=HERE/'plots'/branch
        folder.mkdir(parents=True,exist_ok=True)
        for channel in CHANNELS:
            for flavor in FLAVORS:
                fig,axes=plt.subplots(len(pz),2,figsize=(13,3.0*len(pz)+1.5),squeeze=False,
                                      sharex=True,sharey=False)
                for it,tau in enumerate(flows):
                    with np.load(output_path(branch,tau,channel,flavor),allow_pickle=False) as d:
                        x=np.arange(21)+(it-(len(flows)-1)/2)*.065
                        for ip,p in enumerate(pz):
                            for co,label in enumerate(COMPONENTS):
                                ax=axes[ip,co]
                                good=d['selected_available'][co,ip]
                                constraint=d['local_anchor_constraint'][co,ip]|d['exact_symmetry_zero'][co,ip]
                                strict=d['strict_quality_nonconstraint'][co,ip]
                                for mask,filled in ((good&~constraint&strict,True),(good&~constraint&~strict,False)):
                                    if mask.any():
                                        ax.errorbar(x[mask],d['H_local'][co,ip,mask],
                                            yerr=d['statistical_error'][co,ip,mask],fmt='o',ls='none',
                                            ms=3.0,capsize=1.8,elinewidth=.8,color=colors[it],
                                            mfc=colors[it] if filled else 'none',alpha=.88)
                                mask=good&constraint
                                ax.plot(x[mask],d['H_local'][co,ip,mask],marker='x',ls='none',ms=4,color=colors[it])
                                ax.axhline(0,color='.7',lw=.6,zorder=0)
                                ax.set_xlim(-.5,20.5)
                                ax.set_title(rf'$P_z/(2\pi/L)={p}$; '+label.replace('_',' '))
                                ax.set_ylabel(r'$h_\Gamma(z,P,\tau)/h_\Gamma(0,P,\tau)$')
                                ax.grid(alpha=.15)
                for ax in axes[-1]:
                    ax.set_xlabel(r'$z/a$ (flow times shifted horizontally only)')
                handles=[Line2D([],[],color=colors[it],marker='o',ls='none',ms=4,
                                label=rf'$\tau={tau:.1f}$') for it,tau in enumerate(flows)]
                handles.extend([
                    Line2D([],[],color='black',marker='o',mfc='none',ls='none',label='inherited fit / denominator diagnostic'),
                    Line2D([],[],color='black',marker='o',ls='none',label='upstream quality, no fallback/warning'),
                    Line2D([],[],color='black',marker='x',ls='none',label='constraint / exact odd zero (not a fit)')])
                fig.suptitle(f'L32x64 B | {channel} | {flavor.replace("_minus_"," - ")} | same-P local normalization',y=.985,fontsize=13)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.957),ncol=4,
                           frameon=False,fontsize=9,columnspacing=1.3)
                fig.text(.5,.013,'20 conf | 100 paired bootstrap | statistical errors only | local charge removed; line mass NOT subtracted',
                         ha='center',fontsize=9)
                fig.subplots_adjust(top=1-1.25/(3*len(pz)+1.5),bottom=.07 if len(pz)>1 else .16,
                                    left=.09,right=.98,hspace=.36,wspace=.25)
                path=folder/f'localnorm_{channel}_{flavor}_allflow.pdf'
                fig.savefig(path)
                plt.close(fig)
                paths.append(path)
    return paths


def denominator_plots(rows):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=HERE/'plots'
    paths=[]
    pz=(0,3,4,5,6)
    colors=plt.get_cmap('tab10').colors
    for channel in CHANNELS:
        fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=False)
        for ax,flavor in zip(axes,FLAVORS):
            for ip,p in enumerate(pz):
                selected=[r for r in rows if r['channel']==channel and r['flavor']==flavor and r['Pz']==p]
                for diagnostic in (False,True):
                    rr=[r for r in selected if (r['diagnostic_warning'] or r['upstream_fallback_used']
                                               or not r['upstream_quality_accepted'])==diagnostic]
                    if rr:
                        ax.errorbar([r['tau']+(ip-2)*.004 for r in rr],
                            [r['local_even_real'] for r in rr],yerr=[r['statistical_error'] for r in rr],
                            fmt='o',ls='none',ms=4,capsize=2,color=colors[ip],mfc='none' if diagnostic else colors[ip])
            ax.axhline(0,color='.6',lw=.7)
            ax.set_xlabel(r'$\tau=t_f/a^2$ (momenta offset horizontally)')
            ax.set_title(flavor.replace('_minus_',' - '))
            ax.set_ylabel(r'local even-real $h_\Gamma(0,P,\tau)$')
            ax.grid(alpha=.15)
        handles=[Line2D([],[],marker='o',ls='none',color=colors[ip],label=f'Pz={p}') for ip,p in enumerate(pz)]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=5,frameon=False)
        fig.suptitle(f'L32x64 B local denominators | {channel} | 20 conf / 100 paired bootstrap',y=.985)
        fig.text(.5,.015,'Open: inherited fit fallback/incomplete quality or denominator warning. No sign changes or absolute-value correction imposed.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.79,bottom=.17,left=.07,right=.98,wspace=.34)
        path=folder/f'local_denominators_{channel}.pdf'
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('All numerical processing and plotting require a CPU allocation')
    started=time.monotonic()
    check_lock()
    if (HERE/'complete.json').exists():
        raise RuntimeError('Complete product exists; audit before explicitly creating a successor')
    index_path=INPUTS['moving'][0]/INDEX_NAME
    assert sha(index_path)==INDEX_SHA
    indices=np.load(index_path,allow_pickle=False)
    assert indices.shape==(100,20) and indices.dtype==np.int32
    assert (indices>=0).all() and (indices<20).all()
    shutil.copyfile(index_path,HERE/INDEX_NAME)
    inputs={}
    summaries={}
    for branch,(root,flows,pz) in INPUTS.items():
        assert sha(root/INDEX_NAME)==INDEX_SHA
        receipt=json.loads((root/'complete.json').read_text())
        assert receipt['status'].startswith('COMPLETE_B_')
        assert receipt['nconf']==20 and receipt['nboot']==100
        assert receipt['missing_finite_points']==0
        inputs[branch]={str(Path(a['path'])):a['sha256'] for a in receipt['artifacts']}
        summaries[branch]=dict(root=str(root),complete_sha256=sha(root/'complete.json'),
                               flows=flows,pz=pz,status=receipt['status'])
    manifest=dict(generated=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
                  normalization_scheme=LOCAL_SCHEME,inputs=summaries,confs=CONFS,nconf=20,nboot=100,
                  bootstrap_indices_sha256=INDEX_SHA,seed=1115,resample_size=20,
                  central_estimator='ratio of original-ensemble fitted values',
                  bootstrap_estimator='ratio of paired fitted values inside each existing replica; do not resample fitted points',
                  denominator='same channel, same flavor, same P, same tau, even_real at z=0',
                  error='std of 100 paired normalized replicas, ddof=1; statistical only',
                  charge_restoration=False,line_mass_subtraction=False,matching=False,
                  transversity_status='pending_external_physics_confirmation')
    write_json(HERE/'input_manifest.json',manifest)
    denoms=[]
    points=[]
    artifacts=[]
    oracle_checks=0
    for branch,(_,flows,pz) in INPUTS.items():
        for tau in flows:
            for channel in CHANNELS:
                for flavor in FLAVORS:
                    d,digest=load_verified(branch,tau,channel,flavor,inputs[branch],indices)
                    path,dr,pr,count=transform(branch,tau,channel,flavor,d,digest)
                    denoms.extend(dr)
                    points.extend(pr)
                    oracle_checks+=count
                    artifacts.append(dict(path=str(path),sha256=sha(path),input=str(source_path(branch,tau,channel,flavor)),input_sha256=digest))
            print('NORMALIZED',branch,tau,'products',len(artifacts),flush=True)
    write_csv(HERE/'local_denominator_audit.csv',denoms)
    write_csv(HERE/'localnorm_matrix_elements.csv',points)
    figures=make_plots()+denominator_plots(denoms)
    for path in figures:
        artifacts.append(dict(path=str(path),sha256=sha(path)))
    for name in ('local_denominator_audit.csv','localnorm_matrix_elements.csv','input_manifest.json',INDEX_NAME):
        artifacts.append(dict(path=str(HERE/name),sha256=sha(HERE/name)))
    signflip=[r for r in denoms if r['sign_flip_count']]
    warning=[r for r in denoms if r['diagnostic_warning']]
    summary=dict(status='COMPLETE_B_PAIRED_SAME_P_LOCAL_NORMALIZATION_DIAGNOSTICS',
        generated=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        node=os.environ.get('SLURMD_NODENAME'),nconf=20,nboot=100,
        normalization_scheme=LOCAL_SCHEME,inputs=summaries,
        normalized_products=135,plots=len(figures),points=len(points),local_denominators=len(denoms),
        finite_complete_bootstrap_points=sum(r['selected_available'] for r in points),
        missing_finite_points=sum(not r['selected_available'] for r in points),
        inherited_fallback_points=sum(r['fallback_used'] for r in points),
        joint_upstream_quality_points=sum(r['joint_upstream_quality_accepted'] for r in points),
        strict_quality_nonconstraint_points=sum(r['strict_quality_nonconstraint'] for r in points),
        local_anchor_constraints=sum(r['local_anchor_constraint'] for r in points),
        exact_odd_zero_constraints=sum(r['exact_symmetry_zero'] for r in points),
        denominator_sign_flip_cases=len(signflip),denominator_warning_cases=len(warning),
        denominator_invalid_cases=sum(r['invalid_bootstrap_count']>0 for r in denoms),
        denominator_fit_fallback_cases=sum(r['upstream_fallback_used'] for r in denoms),
        minimum_local_denominator_central_over_bootstrap_sigma=min(r['central_over_bootstrap_sigma'] for r in denoms),
        sign_flip_details=signflip,denominator_warning_details=warning,
        independent_scalar_oracle_checks=oracle_checks,
        bootstrap_indices_sha256=INDEX_SHA,code_lock_sha256=sha(HERE/'code_lock.json'),
        tests='7 numerical unit tests in same CPU job; source hashes, indices, axes, per-replica scalar oracles and inverse reconstruction checked',
        error_scope='paired bootstrap statistical only; fit-window and cut systematics require dedicated paired treatment',
        physical_status='local-normalized finite-flow connected matrix-element diagnostics; NOT absolute ringed/MSbar/lightcone PDF',
        transversity_status='pending_external_physics_confirmation',
        seconds=time.monotonic()-started,artifacts=artifacts)
    assert len([a for a in artifacts if a['path'].endswith('.npz')])==135
    assert len(points)==14742 and len(denoms)==351
    # Final pass detects accidental modifications while this CPU job ran.
    check_lock()
    for artifact in artifacts:
        assert sha(artifact['path'])==artifact['sha256']
    write_json(HERE/'complete.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('artifacts','inputs')},indent=2),flush=True)


if __name__=='__main__':
    main()
