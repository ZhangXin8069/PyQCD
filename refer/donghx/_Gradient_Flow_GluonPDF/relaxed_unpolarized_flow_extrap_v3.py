#!/usr/bin/env python3
"""Reselect saved unpolarized candidates and refit tau>=1, q>=1e-6, chi2/dof<=10.

Preserves N406/Nboot1000 and all old products; selected fallback != quality.
The saved matching is deterministic per (tau,z), hence candidate reselection
commutes with matching. No C3/C2 regeneration or second-level resampling.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

from replica_selection_minerror_v3 import select_grid
from extrapolate_unpolarized_tau_ge1_correlated_v2 import extrapolate,FIT_TAUS
from plot_unpolarized_multiflow_fixed_ylim_v1 import DEFAULT_ROOT,LABEL,load_and_audit,product_path,array_digest

BASE=Path(__file__).resolve().parent


def reselect_tau(root,tau,op,args):
    path=product_path(root,tau)
    with np.load(path,allow_pickle=False) as f:
        assert list(f['axes_candidate_M_boot'])==['operator','candidate','bootstrap','momentum_abs','z']
        assert int(f['fixed_tmin'])==8 and float(f['fixed_tmin_fm'])>.6
        fields=['candidate_M','candidate_M_boot','candidate_chi2','candidate_chi2_boot',
                'candidate_dof','candidate_fit_succeeded','candidate_covariance_rank']
        a={k:f[k][op] for k in fields}
        cut=f['candidate_cut']; omit=f['candidate_n_omitted_difference_points']
        meta={k:f[k] for k in ['confs','bootstrap_indices','seed','nconf','nboot','a_fm',
              'pabs_values','z_values','candidate_tmin','candidate_tmax_summed','covariance_method','svdcut']}
        factors=f['matching_factors'][op]
    # These candidate solutions were saved before old quality/mixture masking.
    # Retain their numerical/rank guards; do not use the old candidate_eligible.
    numerical=a['candidate_fit_succeeded']&(a['candidate_covariance_rank']>=1)
    numerical &= (a['candidate_dof']==0)|(a['candidate_covariance_rank']>1)
    result=select_grid(a['candidate_M'][...,None],np.moveaxis(a['candidate_M_boot'],1,0)[...,None],
        a['candidate_chi2'],np.moveaxis(a['candidate_chi2_boot'],1,0),a['candidate_dof'],
        2.+2*omit,cut,numerical,qmin=args.qmin,chi2max=args.chi2max,allow_zero_dof=True)
    result.update(meta)
    result.update(matching_factors=factors,flow_tau_t_over_a2=np.array(tau),operator=np.array(LABEL),
        candidate_cut=cut,candidate_n_omitted_difference_points=omit,
        candidate_numerical=numerical,**a)
    result['bare_params_selected']=result['params_selected']/factors[...,None]
    result['bare_params_boot_selected']=result['params_boot_selected']/factors[None,...,None]
    result['input_fit_arrays_sha256']=np.array(array_digest(a))
    result['q_min']=np.array(args.qmin);result['chi2_dof_max']=np.array(args.chi2max)
    result['axes_params_selected']=np.array(['momentum_abs','z','parameter'])
    result['axes_params_boot_selected']=np.array(['bootstrap','momentum_abs','z','parameter'])
    result['candidate_error_definition']=np.array('std(candidate_M_boot finite replicas,ddof=1); parameter M')
    result['bootstrap_min_quality_fraction_gate']=np.array('not applied; quality vs selected coverage recorded separately')
    result['selection_policy']=np.array('independent central/replica gates; AIC within cut; equal available cuts; global min-error fallback only if none pass')
    return result


def fit_flow(products,args):
    rs=[products[float(t)] for t in FIT_TAUS]
    data=dict(y=np.stack([r['params_selected'][...,0] for r in rs]),
        boot=np.stack([r['params_boot_selected'][...,0] for r in rs],axis=1),
        usable=np.stack([r['selected_available']&r['bootstrap_selected_available'].all(axis=0) for r in rs]))
    # Recompute every full covariance and linear fit from the new selected samples.
    old=extrapolate(data,args)
    windows=old['windows']
    penalty=np.array([4+2*(len(FIT_TAUS)-(hi-lo)) for lo,hi in windows])
    chi_boot=old['candidate_aic_boot']-penalty[None,:,None,None]
    new=select_grid(old['candidate_beta'],old['candidate_beta_boot'],old['candidate_chi2'],chi_boot,
        old['candidate_dof'],penalty,np.zeros(len(windows),int),old['candidate_identifiable'],
        qmin=args.qmin,chi2max=args.chi2max,allow_zero_dof=False)
    # Keep numerical candidates; discard the old central-only AIC aggregation.
    for k,v in old.items():
        if k.startswith('candidate_') and k not in new:
            new[k]=v
    new['windows']=windows
    new['candidate_chi2_boot']=chi_boot
    new['input_central']=data['y'];new['input_bootstrap']=data['boot'];new['input_available']=data['usable']
    new['input_quality']=np.stack([r['quality_accepted'] for r in rs])
    new['input_central_fallback']=np.stack([r['central_fallback_used'] for r in rs])
    new['input_replica_fallback']=np.stack([r['replica_fallback_used'] for r in rs],axis=1)
    # Propagate whether a positively weighted flow window uses fallback input.
    upstream_c=np.zeros_like(new['quality_accepted'])
    upstream_b=np.zeros_like(new['bootstrap_quality_accepted'])
    for iw,(lo,hi) in enumerate(windows):
        upstream_c|=(new['weights'][iw]>0)&new['input_central_fallback'][lo:hi].any(axis=0)
        upstream_b|=(new['weights_boot'][:,iw]>0)&new['input_replica_fallback'][:,lo:hi].any(axis=1)
    new['upstream_central_fallback_used']=upstream_c
    new['upstream_replica_fallback_used']=upstream_b
    new['any_stage_replica_fallback']=upstream_b|new['replica_fallback_used']
    new['any_stage_replica_fallback_fraction']=new['any_stage_replica_fallback'].mean(axis=0)
    new['strict_quality_chain_central']=new['quality_accepted']&~upstream_c
    new['flow_times']=FIT_TAUS
    new['pabs_values']=rs[0]['pabs_values'];new['z_values']=rs[0]['z_values']
    for k in ['confs','bootstrap_indices','seed','nconf','nboot','a_fm']:
        new[k]=rs[0][k]
    new['q_min']=np.array(args.qmin);new['chi2_dof_max']=np.array(args.chi2max)
    new['svdcut_relative_correlation_eigenvalue']=np.array(args.svdcut)
    new['physical_result_status']=np.array('single_a_linear_intercept_with_labelled_fallback_not_controlled_SFTE_or_physical_PDF')
    return new


def draw(products,r,args,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    colors=['#1f77b4','#d62728','#228833']
    z=r['z_values'];vis=(z>=1)&(z<=15)
    note=('Bars: std of all 1000 selected bootstrap samples. Open: fallback used in the chain; X: central flow-fit fallback.\n'
          'Q >= 1e-6, chi2/dof <= 10; single-a linear intercept, not a controlled continuum/small-flow limit. z=0 excluded.')
    def series(ax,ip,x):
        available=r['selected_available'][ip]&r['bootstrap_selected_available'][:,ip].all(axis=0)&vis
        fallback=r['upstream_central_fallback_used'][ip]|(r['any_stage_replica_fallback_fraction'][ip]>0)
        central=r['central_fallback_used'][ip]
        for mask,marker,face in [(available&central,'X',colors[ip]),
                (available&~central&fallback,'o','white'),(available&~central&~fallback,'o',colors[ip])]:
            ax.errorbar(x[mask],r['params_selected'][ip,mask,0],yerr=r['stat_error'][ip,mask,0],
                fmt=marker,ls='none',color=colors[ip],mfc=face,ms=4.5,capsize=2)
    matrix=out/'unpolarized_tau0_matrix_elements.pdf'
    with PdfPages(matrix) as pdf:
        fig,axes=plt.subplots(1,3,figsize=(16,5.6),sharey=True)
        for ip,ax in enumerate(axes):
            series(ax,ip,z)
            ax.set(xlim=(-.5,15.5),ylim=(args.ymin,args.ymax),xlabel=r'$z/a$',title=rf'$n_z={r["pabs_values"][ip]}$')
            ax.axhline(0,c='.6',ls='--',lw=.6);ax.grid(alpha=.2)
        axes[0].set_ylabel(r'$M_0$ from GF-matched $(T_U-S_U)_{\rm even}$')
        fig.suptitle('Unpolarized gluon: relaxed replica selection + correlated linear/AIC flow extrapolation\n'
                      +r'$t_f/a^2\geq1$; Nconf=406, Nboot=1000; $a=0.0775$ fm',fontsize=13)
        fig.text(.5,.025,note,ha='center',fontsize=9)
        fig.subplots_adjust(left=.06,right=.985,top=.78,bottom=.22,wspace=.10)
        pdf.savefig(fig);fig.savefig(out/'unpolarized_tau0_vs_z.png',dpi=190);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6))
        for ip,pz in enumerate(r['pabs_values']):series(ax,ip,z*2*np.pi*pz/32+(ip-1)*.025)
        ax.set(xlim=(-.15,15.1),ylim=(args.ymin,args.ymax),xlabel=r'$\lambda=zP_z$',ylabel=r'$M_0$')
        ax.grid(alpha=.2);ax.axhline(0,c='.5',ls='--',lw=.6)
        ax.legend(handles=[Line2D([],[],color=c,marker='o',ls='none',label=f'n_z={p}') for c,p in zip(colors,r['pabs_values'])],frameon=False)
        fig.suptitle('Unpolarized zero-flow linear intercept: relaxed replica selection')
        fig.text(.5,.025,note,ha='center',fontsize=8)
        fig.subplots_adjust(left=.1,right=.98,top=.89,bottom=.22)
        pdf.savefig(fig);fig.savefig(out/'unpolarized_tau0_vs_zPz.png',dpi=190);plt.close(fig)
    flow=out/'unpolarized_flow_extrapolation_fits.pdf'
    xx=np.linspace(0,3.8,200)
    with PdfPages(flow) as pdf:
        for iz in range(1,16):
            fig,axes=plt.subplots(1,3,figsize=(16,5.3),sharey=True)
            for ip,ax in enumerate(axes):
                for tau,d in products.items():
                    y,e=d['params_selected'][ip,iz,0],d['stat_error'][ip,iz,0]
                    if not np.isfinite(y+e):continue
                    fb=d['central_fallback_used'][ip,iz] or d['replica_fallback_fraction'][ip,iz]>0
                    used=False
                    if tau>=1:
                        it=np.flatnonzero(np.isclose(FIT_TAUS,tau)).item()
                        used=any(r['weights'][iw,ip,iz]>0 and lo<=it<hi for iw,(lo,hi) in enumerate(r['windows']))
                    ax.errorbar(tau,y,yerr=e,fmt='X' if d['central_fallback_used'][ip,iz] else 'o',ls='none',
                        color=colors[ip],mfc=colors[ip] if used and not fb else 'white',alpha=1 if used else .3,ms=4,capsize=2)
                b=r['params_selected'][ip,iz];bb=r['params_boot_selected'][:,ip,iz]
                if np.isfinite(b).all() and np.isfinite(bb).all():
                    pred=b[0]+b[1]*xx;err=np.std(bb[:,0,None]+bb[:,1,None]*xx,axis=0,ddof=1)
                    ax.plot(xx,pred,color=colors[ip]);ax.fill_between(xx,pred-err,pred+err,color=colors[ip],alpha=.15)
                    ax.errorbar(0,b[0],yerr=r['stat_error'][ip,iz,0],fmt='s',color='black',capsize=3)
                ax.set(xlim=(-.12,3.92),ylim=(args.ymin,args.ymax),xlabel=r'$t_f/a^2$',title=rf'$n_z={r["pabs_values"][ip]}$')
                ax.axvline(1,c='.6',ls=':',lw=.7);ax.grid(alpha=.2)
                ax.text(.97,.03,f'central windows: {r["qualified_count"][ip,iz]}; flow replica fallback: '
                        f'{100*r["replica_fallback_fraction"][ip,iz]:.1f}%',transform=ax.transAxes,ha='right',fontsize=8)
            axes[0].set_ylabel('GF-matched matrix element')
            fig.suptitle(rf'Unpolarized $(T_U-S_U)_{{\rm even}}$, $z/a={iz}$: relaxed linear/AIC extrapolation')
            fig.text(.5,.035,'Light: unused flow time. Open: replica fallback in bare input; X: central bare fallback. '
                     'Black square: selected t=0 intercept; band: selected-bootstrap stat.',ha='center',fontsize=9)
            fig.subplots_adjust(left=.06,right=.985,top=.80,bottom=.20,wspace=.10)
            pdf.savefig(fig)
            if iz==4:fig.savefig(out/'unpolarized_flow_fit_z4.png',dpi=190)
            plt.close(fig)
    return [str(matrix.resolve()),str(flow.resolve())]


def validate_selection(r):
    available=r['selected_available'];ba=r['bootstrap_selected_available']
    np.testing.assert_allclose(r['weights'].sum(axis=0)[available],1,atol=1e-12)
    np.testing.assert_allclose(r['weights_boot'].sum(axis=1)[ba],1,atol=1e-12)
    assert not np.any(r['central_fallback_used']&r['quality_accepted'])
    assert not np.any(r['replica_fallback_used']&r['bootstrap_quality_accepted'])
    complete=ba.all(axis=0)
    np.testing.assert_allclose(r['stat_error'][complete],np.std(r['params_boot_selected'][:,complete],axis=0,ddof=1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--out',type=Path,default=BASE/'unpolarized_relaxed_q1em6_chi10_minerr_v3')
    p.add_argument('--qmin',type=float,default=1e-6);p.add_argument('--chi2max',type=float,default=10.)
    p.add_argument('--svdcut',type=float,default=1e-10)
    p.add_argument('--ymin',type=float,default=-2.5);p.add_argument('--ymax',type=float,default=1.)
    args=p.parse_args()
    if not (0<=args.qmin<1 and args.chi2max>0 and 0<args.svdcut<1 and args.ymin<args.ymax):p.error('invalid parameters')
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Use a fresh output directory')
    original,records,settings,index_hash=load_and_audit(args.input_root)
    args.out.mkdir(parents=True,exist_ok=True);prod_dir=args.out/'reselected_matched';prod_dir.mkdir()
    products={};summaries=[]
    for tau,row in original.items():
        r=reselect_tau(args.input_root,tau,row['operator_index'],args)
        validate_selection(r)
        filename=prod_dir/f'unpolarized_tau{tau:.3f}_selected.npz'.replace('.','p',1)
        np.savez_compressed(filename,**r)
        products[tau]={k:r[k] for k in ['params_selected','params_boot_selected','stat_error',
            'selected_available','bootstrap_selected_available','quality_accepted',
            'central_fallback_used','replica_fallback_used','replica_fallback_fraction',
            'pabs_values','z_values','confs','bootstrap_indices','seed','nconf','nboot','a_fm']}
        s=dict(tau=tau,output=str(filename.resolve()),input_fit_arrays_sha256=str(r['input_fit_arrays_sha256']),
            central_quality=int(r['quality_accepted'].sum()),central_fallback=int(r['central_fallback_used'].sum()),
            replica_quality=int(r['bootstrap_quality_accepted'].sum()),replica_fallback=int(r['replica_fallback_used'].sum()),
            selected_replicas=int(r['bootstrap_selected_available'].sum()),total_replicas=int(r['bootstrap_selected_available'].size))
        summaries.append(s);print(json.dumps(s),flush=True)
    flow=fit_flow(products,args);validate_selection(flow)
    dest=args.out/'unpolarized_tau0_selected.npz';np.savez_compressed(dest,**flow)
    paths=draw(products,flow,args,args.out)
    rows=[]
    for ip,pz in enumerate(flow['pabs_values']):
        for iz,z in enumerate(flow['z_values']):
            rows.append(dict(n_z=int(pz),z_over_a=int(z),M0=float(flow['params_selected'][ip,iz,0]),
                statistical_error=float(flow['stat_error'][ip,iz,0]),selected_available=bool(flow['selected_available'][ip,iz]),
                flow_central_quality=bool(flow['quality_accepted'][ip,iz]),flow_central_fallback=bool(flow['central_fallback_used'][ip,iz]),
                flow_replica_fallback_fraction=float(flow['replica_fallback_fraction'][ip,iz]),
                any_stage_replica_fallback_fraction=float(flow['any_stage_replica_fallback_fraction'][ip,iz]),
                upstream_central_fallback=bool(flow['upstream_central_fallback_used'][ip,iz])))
    with (args.out/'extrapolated_matrix_elements.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for rec in records:
        s=Path(rec['path']).stat();assert (s.st_size,s.st_mtime_ns)==(rec['bytes'],rec['mtime_ns'])
    manifest=dict(status='completed_single_a_selection_sensitivity_not_physical_PDF',generated_utc=datetime.now(timezone.utc).isoformat(),
        operator=LABEL,flow_times=FIT_TAUS.tolist(),nconf=406,nboot=1000,seed=1115,common_index_sha256=index_hash,
        quality_gates=dict(q_min=args.qmin,chi2_dof_max=args.chi2max),matching=settings,
        bare_selection='independent original/replica gates across registered candidates; AIC within cut; equal passing cuts; no old 90% cut gate',
        fallback='minimum precomputed candidate bootstrap std(ddof=1); tie raw chi2 then index; never relabel as quality',
        central_estimator='original data selection, not bootstrap mean',
        covariance='bare: inherited Schaefer-Strimmer fits; flow: full covariance of new selected shared replicas, fixed across replicas',
        svdcut_flow=args.svdcut,flow_windows='21 original contiguous >=3-point windows within tau>=1; selected fallback input permitted and propagated',
        score='chi2+2*k+2*N_omitted; group-wise bare AIC then equal-cut mixture; one group for flow windows',
        zero_dof='bare last-resort diagnostic only; forbidden in two-parameter flow fit',
        errors='std of all 1000 selected replicas, ddof=1; window spread separate; min-error fallback can bias/shrink errors',
        no_second_level_resampling=True,bare_candidate_refit=False,input_records=records,bare_summaries=summaries,
        flow_summary=dict(central_quality=int(flow['quality_accepted'].sum()),central_fallback=int(flow['central_fallback_used'].sum()),
            replica_quality=int(flow['bootstrap_quality_accepted'].sum()),replica_fallback=int(flow['replica_fallback_used'].sum()),
            selected_replicas=int(flow['bootstrap_selected_available'].sum()),total_replicas=int(flow['bootstrap_selected_available'].size),
            available_z1to15=flow['selected_available'][:,1:16].sum(axis=1).tolist()),
        outputs=paths,ylim=[args.ymin,args.ymax],z0='formal diagnostic only; excluded from plots',
        code_hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in
                     ['relaxed_unpolarized_flow_extrap_v3.py','replica_selection_minerror_v3.py','extrapolate_unpolarized_tau_ge1_correlated_v2.py']})
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['flow_summary'],indent=2),flush=True)


if __name__=='__main__':main()
