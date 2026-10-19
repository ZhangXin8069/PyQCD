#!/usr/bin/env python3
"""Single-a diagnostic: correlated linear/AIC flow extrapolation, TU-SU only.

Keeps the frozen N406/Nboot1000 configuration bootstrap; no upstream fits,
matching coefficients, or legacy extrapolation files are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from plot_unpolarized_multiflow_fixed_ylim_v1 import (
    DEFAULT_ROOT, LABEL, TAUS, load_and_audit, product_path, array_digest,
)

BASE = Path(__file__).resolve().parent
FIT_TAUS = np.array([1.,1.4,1.8,2.2,2.6,3.,3.4,3.8])


def correlated_linear(x, y, boot, svdcut=1e-10):
    """GLS from full bootstrap covariance; relative correlation eigenmode cut.

    This is explicit eigenmode truncation, NOT gvar's positive SVD-floor rule.
    The covariance is frozen across replicas. Bootstrap spread is not divided
    by sqrt(Nboot). Exactly singular input is never made full rank.
    """
    x,y,boot = np.asarray(x,float),np.asarray(y,float),np.asarray(boot,float)
    n=len(x)
    if n<3 or y.shape!=(n,) or boot.ndim!=2 or boot.shape[1]!=n or boot.shape[0]<3:
        raise ValueError('need >=3 flow points and matched (boot,flow) arrays')
    if not all(np.isfinite(a).all() for a in (x,y,boot)):
        raise ValueError('nonfinite fit input')
    cov=np.cov(boot,rowvar=False,ddof=1)
    sigma=np.sqrt(np.diag(cov))
    if np.any(sigma<=0):
        raise ValueError('zero variance is not statistical information')
    corr=cov/sigma[:,None]/sigma[None,:]
    corr=(corr+corr.T)/2
    eig,vec=np.linalg.eigh(corr)
    if eig[0]<-1e-10*eig[-1]:
        raise ValueError('covariance not positive semidefinite')
    raw_rank=int(np.count_nonzero(eig>1e-12*eig[-1]))
    keep=eig>svdcut*eig[-1]
    rank=int(keep.sum())
    if raw_rank<2 or rank<=2:
        raise ValueError('statistically_unidentifiable: insufficient covariance rank/dof')
    white=(vec[:,keep].T / np.sqrt(eig[keep])[:,None])/sigma[None,:]
    X=np.column_stack([np.ones(n),x])
    WX=white@X
    scaled=WX/np.linalg.norm(WX,axis=0)
    if np.linalg.matrix_rank(scaled,tol=1e-12)<2:
        raise ValueError('statistically_unidentifiable: design rank')
    coef=np.linalg.lstsq(WX,white,rcond=None)[0]
    beta=coef@y
    beta_boot=boot@coef.T
    resid=white@(y-X@beta)
    resid_boot=(boot-beta_boot@X.T)@white.T
    chi=float(resid@resid)
    chi_boot=np.einsum('bi,bi->b',resid_boot,resid_boot)
    dof=rank-2
    cond=float(eig[-1]/eig[0]) if eig[0]>0 else float('inf')
    return dict(beta=beta,beta_boot=beta_boot,chi2=chi,chi2_boot=chi_boot,
        dof=dof,q=float(chi2_distribution.sf(chi,dof)),coef=coef,
        covariance=cov,correlation_eigenvalues=eig,raw_rank=raw_rank,retained_rank=rank,
        correlation_condition=cond,retained_correlation_condition=float(eig[keep][-1]/eig[keep][0]),
        scaled_whitened_design_condition=float(np.linalg.cond(scaled)))


def aic_weights(score):
    score=np.asarray(score,float)
    if not np.isfinite(score).all():
        raise ValueError('AIC weights require finite candidate scores')
    w=np.exp(-.5*(score-score.min(axis=-1,keepdims=True)))
    return w/w.sum(axis=-1,keepdims=True)


def average_candidates(beta,beta_boot,score,score_boot):
    """Original central estimator plus fully paired, replica-dependent AIC."""
    w=aic_weights(score)
    wb=aic_weights(score_boot)
    central=np.einsum('w,wk->k',w,beta)
    replicas=np.einsum('bw,bwk->bk',wb,beta_boot)
    fixed=np.einsum('w,bwk->bk',w,beta_boot)
    spread=np.sqrt(np.einsum('w,wk->k',w,(beta-central)**2))
    return central,replicas,w,wb,spread,fixed


def load_data(root):
    products,records,settings,index_hash=load_and_audit(root)
    rows,boots=[],[]
    confs=indices=None
    for tau in FIT_TAUS:
        row=products[float(tau)]
        op=row['operator_index']
        with np.load(product_path(root,tau),allow_pickle=False) as f:
            if list(f['axes_M_boot'])!=['bootstrap','operator','momentum_abs','z']:
                raise ValueError('Unexpected bootstrap axes')
            bb=f['M_boot_filled'][:,op]
            if bb.shape!=(1000,3,25):
                raise ValueError('Unexpected bootstrap matrix shape')
            frac=f['replica_fallback_fraction'][op]
            if confs is None:
                confs=f['confs']; indices=f['bootstrap_indices']
        rows.append(dict(value=row['M_filled'][op],error=row['filled_statistical_error'][op],
            usable=row['estimate_available'][op]&row['fit_accepted'][op]&~row['central_fallback_used'][op],
            replica_fallback_fraction=frac))
        boots.append(bb)
    return dict(products=products,records=records,settings=settings,index_hash=index_hash,
        y=np.stack([r['value'] for r in rows]),err=np.stack([r['error'] for r in rows]),
        usable=np.stack([r['usable'] for r in rows]),boot=np.stack(boots,axis=1),
        upstream_replica_fallback_fraction=np.stack([r['replica_fallback_fraction'] for r in rows]),
        confs=confs,indices=indices,z=np.arange(25),pz=np.array([3,4,5]))


def extrapolate(data,args):
    x=FIT_TAUS
    windows=[(i,j) for i in range(len(x)-2) for j in range(i+3,len(x)+1)]
    nw,nb,npz,nz=len(windows),len(data['boot']),3,25
    shape=(nw,npz,nz)
    c={key:np.full(shape,np.nan) for key in ['chi2','q','aic','dof','raw_rank','retained_rank',
        'correlation_condition','retained_correlation_condition','scaled_whitened_design_condition']}
    c['beta']=np.full((*shape,2),np.nan)
    c['beta_boot']=np.full((nb,*shape,2),np.nan)
    c['aic_boot']=np.full((nb,*shape),np.nan)
    c['covariance']=np.full((*shape,8,8),np.nan)
    c['correlation_eigenvalues']=np.full((*shape,8),np.nan)
    c['identifiable']=np.zeros(shape,bool)
    c['quality_pass']=np.zeros(shape,bool)
    c['reason']=np.full(shape,'input_quality_or_missing',dtype='U100')
    result={key:np.full((npz,nz,2),np.nan) for key in ['beta','stat_error','window_spread',
        'total_diagnostic_error','all_windows_beta_diagnostic','all_windows_error_diagnostic']}
    result['beta_boot']=np.full((nb,npz,nz,2),np.nan)
    result['fixed_weight_beta_boot']=np.full((nb,npz,nz,2),np.nan)
    result['weights']=np.zeros(shape)
    result['weights_boot']=np.zeros((nb,*shape))
    result['valid']=np.zeros((npz,nz),bool)
    result['qualified_count']=np.zeros((npz,nz),int)
    result['window_systematic_estimable']=np.zeros((npz,nz),bool)
    result['status']=np.full((npz,nz),'no_identifiable_window',dtype='U64')
    for ip in range(npz):
        for iz in range(nz):
            for iw,(lo,hi) in enumerate(windows):
                # Registered windows are contiguous on the original time grid.
                # A failed point invalidates its window; never compress holes.
                if not np.all(data['usable'][lo:hi,ip,iz]):
                    continue
                try:
                    fit=correlated_linear(x[lo:hi],data['y'][lo:hi,ip,iz],
                                          data['boot'][:,lo:hi,ip,iz],args.svdcut)
                except (ValueError,np.linalg.LinAlgError) as exc:
                    c['reason'][iw,ip,iz]=str(exc)[:100]
                    continue
                for key in ['chi2','q','dof','raw_rank','retained_rank','correlation_condition',
                            'retained_correlation_condition','scaled_whitened_design_condition']:
                    c[key][iw,ip,iz]=fit[key]
                c['beta'][iw,ip,iz]=fit['beta']
                c['beta_boot'][:,iw,ip,iz]=fit['beta_boot']
                penalty=4+2*(len(x)-(hi-lo))
                c['aic'][iw,ip,iz]=fit['chi2']+penalty
                c['aic_boot'][:,iw,ip,iz]=fit['chi2_boot']+penalty
                c['covariance'][iw,ip,iz,:hi-lo,:hi-lo]=fit['covariance']
                c['correlation_eigenvalues'][iw,ip,iz,:hi-lo]=fit['correlation_eigenvalues']
                c['identifiable'][iw,ip,iz]=True
                quality=fit['q']>=args.qmin and fit['chi2']/fit['dof']<=args.chi2max
                c['quality_pass'][iw,ip,iz]=quality
                c['reason'][iw,ip,iz]='quality_pass' if quality else 'flow_fit_quality_fail'
            numeric=np.flatnonzero(c['identifiable'][:,ip,iz])
            selected=np.flatnonzero(c['quality_pass'][:,ip,iz])
            if not len(numeric):
                continue
            def average(sel):
                return average_candidates(c['beta'][sel,ip,iz],c['beta_boot'][:,:,ip,iz][:,sel],
                    c['aic'][sel,ip,iz],c['aic_boot'][:,:,ip,iz][:,sel])
            avg,bavg,*_=average(numeric)
            result['all_windows_beta_diagnostic'][ip,iz]=avg
            result['all_windows_error_diagnostic'][ip,iz]=np.std(bavg,axis=0,ddof=1)
            if not len(selected):
                result['status'][ip,iz]='all_flow_windows_fail_quality'
                continue
            avg,bavg,w,wb,spread,fixed=average(selected)
            result['beta'][ip,iz]=avg
            result['beta_boot'][:,ip,iz]=bavg
            result['fixed_weight_beta_boot'][:,ip,iz]=fixed
            result['stat_error'][ip,iz]=np.std(bavg,axis=0,ddof=1)
            result['window_spread'][ip,iz]=spread
            result['total_diagnostic_error'][ip,iz]=np.hypot(result['stat_error'][ip,iz],spread)
            result['weights'][selected,ip,iz]=w
            result['weights_boot'][:,selected,ip,iz]=wb
            result['valid'][ip,iz]=True
            result['qualified_count'][ip,iz]=len(selected)
            result['window_systematic_estimable'][ip,iz]=len(selected)>1
            result['status'][ip,iz]=('aic_correlated_linear' if len(selected)>1 else 'single_window_diagnostic')
    result['windows']=np.array(windows)
    result.update({'candidate_'+k:v for k,v in c.items()})
    return result


def plot(data,r,out,args,ylim=(-1.5,1.)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    colors=['#1f77b4','#d62728','#228833']
    z=data['z']; visible=(z>=1)&(z<=15)
    main=out/'unpolarized_tau0_matrix_elements.pdf'
    def series(ax,ip,x,sel,shift=0):
        for multiple in [True,False]:
            mask=sel&r['valid'][ip]&(r['window_systematic_estimable'][ip]==multiple)
            ax.errorbar(x[mask]+shift,r['beta'][ip,mask,0],
                yerr=r['total_diagnostic_error'][ip,mask,0],fmt='none',ecolor=colors[ip],alpha=.25,elinewidth=2.8)
            ax.errorbar(x[mask]+shift,r['beta'][ip,mask,0],yerr=r['stat_error'][ip,mask,0],
                fmt='o',ls='none',c=colors[ip],mfc=colors[ip] if multiple else 'white',ms=4,capsize=2)
        failed=sel&~r['valid'][ip]&np.isfinite(r['all_windows_beta_diagnostic'][ip,:,0])
        ax.errorbar(x[failed]+shift,r['all_windows_beta_diagnostic'][ip,failed,0],
            yerr=r['all_windows_error_diagnostic'][ip,failed,0],fmt='x',ls='none',color=colors[ip],alpha=.5)
    footer=('Inner bars: paired-bootstrap stat.; pale outer: stat + AIC-window spread in quadrature (diagnostic).\n'
            'Open: one qualified window; x: all windows fail quality. z=0 excluded. Single-a ansatz intercept, not a continuum PDF.')
    with PdfPages(main) as pdf:
        fig,axes=plt.subplots(1,3,figsize=(16,5.4),sharey=True)
        for ip,ax in enumerate(axes):
            series(ax,ip,z,visible)
            ax.set(xlim=(-.5,15.5),ylim=ylim,xlabel=r'$z/a$',title=rf'$n_z={data["pz"][ip]}$')
            ax.axhline(0,c='.5',lw=.7,ls='--'); ax.grid(alpha=.2)
        axes[0].set_ylabel(r'$M_0$ from GF-matched $(T_U-S_U)_{\rm even}$')
        fig.suptitle(r'Unpolarized gluon: correlated linear/AIC $t_f\to0$ test, $t_f/a^2\geq1$'
                     +'\nNconf=406, Nboot=1000; a=0.0775 fm; no continuum extrapolation',fontsize=13)
        fig.text(.5,.035,footer,ha='center',fontsize=9)
        fig.subplots_adjust(left=.065,right=.985,bottom=.21,top=.78,wspace=.1)
        pdf.savefig(fig);fig.savefig(out/'unpolarized_tau0_vs_z.png',dpi=190);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6))
        for ip,pz in enumerate(data['pz']):
            series(ax,ip,z*2*np.pi*pz/32,visible,shift=(ip-1)*.025)
        ax.set(xlim=(-.15,15.1),ylim=ylim,xlabel=r'$\lambda=zP_z$',ylabel=r'$M_0$')
        ax.axhline(0,c='.5',lw=.7,ls='--');ax.grid(alpha=.2)
        ax.legend(handles=[Line2D([],[],color=c,marker='o',ls='none',label=rf'$n_z={p}$')
                           for c,p in zip(colors,data['pz'])],frameon=False)
        fig.suptitle(r'Unpolarized $(T_U-S_U)_{\rm even}$: $t_f=0$ intercept vs. $zP_z$',fontsize=13)
        fig.text(.5,.025,footer,ha='center',fontsize=8)
        fig.subplots_adjust(bottom=.22,top=.89,left=.1,right=.98)
        pdf.savefig(fig);fig.savefig(out/'unpolarized_tau0_vs_zPz.png',dpi=190);plt.close(fig)
    # One z per page, with all momenta: directly inspect the extrapolation.
    flowpdf=out/'unpolarized_flow_extrapolation_fits.pdf'
    xx=np.linspace(0,3.8,200)
    with PdfPages(flowpdf) as pdf:
        for iz in range(1,16):
            fig,axes=plt.subplots(1,3,figsize=(16,5.1),sharey=True)
            for ip,ax in enumerate(axes):
                for tau,row in data['products'].items():
                    io=row['operator_index']; y=row['M_filled'][io,ip,iz];e=row['filled_statistical_error'][io,ip,iz]
                    good=bool(row['estimate_available'][io,ip,iz]&row['fit_accepted'][io,ip,iz]&~row['central_fallback_used'][io,ip,iz])
                    if not np.isfinite(y+e):continue
                    used=False
                    if tau>=1 and r['valid'][ip,iz]:
                        it=np.flatnonzero(np.isclose(FIT_TAUS,tau)).item()
                        used=any(r['weights'][iw,ip,iz]>0 and lo<=it<hi for iw,(lo,hi) in enumerate(r['windows']))
                    ax.errorbar(tau,y,yerr=e,fmt='o' if good else 'x',ls='none',color=colors[ip],
                        mfc=colors[ip] if used else 'white',alpha=1 if used else .30,ms=4,capsize=2)
                if r['valid'][ip,iz]:
                    b=r['beta'][ip,iz];bb=r['beta_boot'][:,ip,iz]
                    pred=b[0]+b[1]*xx
                    predboot=bb[:,0,None]+bb[:,1,None]*xx
                    sig=np.std(predboot,axis=0,ddof=1)
                    ax.fill_between(xx,pred-sig,pred+sig,color=colors[ip],alpha=.15)
                    ax.plot(xx,pred,color=colors[ip],lw=1.2)
                    ax.errorbar(0,b[0],yerr=r['stat_error'][ip,iz,0],fmt='s',c='black',ms=5,capsize=3)
                ax.axvline(1,c='.6',ls=':',lw=.8);ax.axhline(0,c='.6',lw=.5)
                ax.set(xlim=(-.12,3.92),ylim=ylim,xlabel=r'$t_f/a^2$',title=rf'$n_z={data["pz"][ip]}$')
                ax.grid(alpha=.2)
                ax.text(.97,.03,f'{r["qualified_count"][ip,iz]} qualified windows',transform=ax.transAxes,ha='right',fontsize=9)
            axes[0].set_ylabel('GF-matched matrix element')
            fig.suptitle(rf'Unpolarized $(T_U-S_U)_{{\rm even}}$, $z/a={iz}$: correlated linear / AIC extrapolation',fontsize=13)
            fig.text(.5,.03,'Filled: enters qualified fit windows; open/light: unused; x: upstream diagnostic. '
                     'Black square: t=0 intercept; band: bootstrap statistical error. Single-a test.',ha='center',fontsize=9)
            fig.subplots_adjust(left=.065,right=.985,bottom=.19,top=.81,wspace=.1)
            pdf.savefig(fig)
            if iz==4:fig.savefig(out/'unpolarized_flow_fit_z4.png',dpi=190)
            plt.close(fig)
    return [str(main.resolve()),str(flowpdf.resolve())]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--out',type=Path,default=BASE/'bare_all_operators_matching_v1/unpolarized_flow_extrap_tau_ge1_correlated_v2')
    p.add_argument('--svdcut',type=float,default=1e-10)
    p.add_argument('--qmin',type=float,default=.01)
    p.add_argument('--chi2max',type=float,default=3.)
    args=p.parse_args()
    if not 0<args.svdcut<1 or not 0<=args.qmin<1 or not args.chi2max>0:
        p.error('Invalid fixed statistical thresholds')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a fresh output directory; legacy results are preserved')
    data=load_data(args.input_root)
    print('Audited shared indices and loaded eight matched bootstrap products.',flush=True)
    r=extrapolate(data,args)
    args.out.mkdir(parents=True,exist_ok=True)
    physics_mask=(data['z']>0)[None,:]&r['valid']
    payload={**r,'schema':np.array('gluon_unpolarized_flow_extrap_correlated_v2'),
        'operator':np.array(LABEL),'flow_times':FIT_TAUS,'z_values':data['z'],'pabs_values':data['pz'],
        'bootstrap_indices':data['indices'],'confs':data['confs'],'nconf':np.array(406),'nboot':np.array(1000),
        'input_central':data['y'],'input_bootstrap':data['boot'],'input_usable':data['usable'],
        'input_replica_fallback_fraction':data['upstream_replica_fallback_fraction'],
        'plot_eligible_nonlocal':physics_mask,'z0_formal_diagnostic_only':data['z']==0,
        'axes_beta':np.array(['momentum_abs','z','parameter']),
        'axes_beta_boot':np.array(['bootstrap','momentum_abs','z','parameter']),
        'parameter_names':np.array(['M0','slope_per_t_over_a2']),
        'physical_result_status':np.array('single_a_model_intercept_not_controlled_small_flow_or_continuum_limit'),
        'svdcut_relative_correlation_eigenvalue':np.array(args.svdcut),
        'qmin':np.array(args.qmin),'chi2_dof_max':np.array(args.chi2max)}
    np.savez_compressed(args.out/'unpolarized_tau0_correlated_aic.npz',**payload)
    plots=plot(data,r,args.out,args)
    summary={'status':'completed_single_spacing_extrapolation_diagnostic','operator':LABEL,
        'generated_utc':datetime.now(timezone.utc).isoformat(),'input_records':data['records'],
        'fitted_flow_times':FIT_TAUS.tolist(),'plot_only_small_flow_times':TAUS[TAUS<1].tolist(),
        'input_fit_arrays_sha256':array_digest({k:data[k] for k in ['y','boot','usable']}),
        'matching_parameters':data['settings'],'common_index_sha256':data['index_hash'],
        'nconf':406,'nboot':1000,'bootstrap':'original shared configuration indices; no resampling here',
        'fit':'M=M0+c*(t/a^2); all registered contiguous windows with >=3 points; no holes',
        'candidate_windows':[[float(FIT_TAUS[i]),float(FIT_TAUS[j-1])] for i,j in r['windows']],
        'covariance':'full across flow times, estimated from 1000 filled replicas, ddof=1; frozen in bootstrap',
        'svdcut':args.svdcut,'svd_semantics':'relative correlation eigenmode truncation; no shrinkage or eigenvalue floor',
        'quality_gates':{'Q_min':args.qmin,'chi2_dof_max':args.chi2max},
        'candidate_selection':'central quality gates fixed for every replica; no replica-specific acceptance changes',
        'AIC':'chi2+2*k+2*N_omitted, k=2; project window score, not unmodified AIC on identical data',
        'AIC_weights':'central and each bootstrap separately; fixed central-weight bootstrap saved as cross-check',
        'central_estimator':'AIC-weighted original central values, not bootstrap mean',
        'statistical_error':'std of all 1000 paired AIC bootstrap intercepts, ddof=1',
        'window_spread':'sqrt(sum_w weight_w*(M0_w-M0)^2); not estimable with only one window',
        'outer_plot_error':'sqrt(stat^2+window_spread^2), heuristic diagnostic; not full systematic uncertainty',
        'valid_counts_by_momentum_all_z':r['valid'].sum(axis=1).tolist(),
        'valid_counts_by_momentum_z1to15':r['valid'][:,1:16].sum(axis=1).tolist(),
        'status_counts':{str(k):int(v) for k,v in zip(*np.unique(r['status'],return_counts=True))},
        'not_done':['continuum_limit','a2_over_t_removal','nonlinear_flow_model_systematic','matching_order_scale_systematic',
                    'light_cone_matching','singlet_mixing','full_six_direction_average'],
        'z0':'stored as formal diagnostic only; omitted from nonlocal final plots',
        'plots':plots,'script':str(Path(__file__).resolve()),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','valid_counts_by_momentum_z1to15','status_counts','plots']},indent=2))


if __name__=='__main__':
    main()
