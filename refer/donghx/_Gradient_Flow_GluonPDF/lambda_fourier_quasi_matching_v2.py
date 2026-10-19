#!/usr/bin/env python3
"""Paired-bootstrap lambda reconstruction, Fourier and NLO gg matching test.

Uses the N406/1000 relaxed zero-flow product. The finite-distance data are
locally normalized replica by replica. Tail shapes follow arXiv:2510.26425v2
Eq.(9), with registered fixed shape parameters and correlated amplitude fits.
The MSbar->hybrid bridge is a perturbative proxy; absolute gluon normalization
and a measured zero-momentum hybrid denominator are not supplied by this run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from itertools import product
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import chi2 as chi2_dist

from replica_selection_minerror_v3 import select_cell
from match_unpolarized_quasi_from_extrap_v2 import build_kernel,physical_pz,HBARC,CA
from match_unpolarized_quasi_from_extrap_v1 import interp_row

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'unpolarized_relaxed_q1em6_chi10_minerr_v3/unpolarized_tau0_selected.npz'
OUT=SOURCE.parent/'lambda_fourier_quasi_matching_v2'
POWERS=(.5,1.,2.)
DECAY_LENGTHS=(4.,8.,16.)
FIT_STARTS=(4.,5.,6.)
ANCHOR=8.


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def shape(lam,power,length):
    lam=np.asarray(lam,float)
    if np.any(lam<=0):raise ValueError('tail shape is only defined at positive lambda')
    return (lam/ANCHOR)**(-power)*np.exp(-(lam-ANCHOR)/length)


def amplitude_fit(phi,y,boot,svdcut=1e-10):
    """One free amplitude, full covariance; correlation-eigenmode truncation."""
    n=len(phi)
    if n<3 or not all(np.isfinite(a).all() for a in (phi,y,boot)):
        raise ValueError('need >=3 finite points')
    cov=np.cov(boot,rowvar=False,ddof=1);sd=np.sqrt(np.diag(cov))
    if np.any(sd<=0):raise ValueError('zero variance')
    corr=cov/sd[:,None]/sd[None,:];eig,vec=np.linalg.eigh((corr+corr.T)/2)
    if eig[0]<-1e-10*eig[-1]:raise ValueError('negative covariance eigenvalue')
    raw_rank=int((eig>1e-12*eig[-1]).sum());keep=eig>svdcut*eig[-1]
    rank=int(keep.sum())
    if raw_rank<1 or rank<=1:raise ValueError('insufficient covariance rank')
    W=vec[:,keep].T/np.sqrt(eig[keep])[:,None]/sd[None,:]
    wp=W@phi
    if not np.isfinite(wp@wp) or wp@wp<=0:raise ValueError('unidentifiable amplitude')
    coef=(wp@W)/(wp@wp);A=float(coef@y);Ab=boot@coef
    res=W@(y-A*phi);rb=(boot-Ab[:,None]*phi)@W.T
    chi=float(res@res);chib=np.einsum('bi,bi->b',rb,rb)
    return dict(A=A,Ab=Ab,chi=chi,chib=chib,dof=rank-1,raw_rank=raw_rank,rank=rank,
                covariance=cov,eigenvalues=eig,coef=coef,q=float(chi2_dist.sf(chi,rank-1)))


def fit_tail(lam,z,h,args):
    """h[0] is original central; h[1:] are the unchanged paired replicas."""
    registry=np.array(list(product(FIT_STARTS,POWERS,DECAY_LENGTHS)))
    nc=len(registry);nb=len(h)-1;nz=len(z)
    amp=np.full(nc,np.nan);ab=np.full((nb,nc),np.nan)
    chi=np.full(nc,np.nan);chib=np.full((nb,nc),np.nan);dof=np.zeros(nc,int)
    numerical=np.zeros(nc,bool);counts=np.zeros(nc,int)
    cov=np.full((nc,nz,nz),np.nan);eig=np.full((nc,nz),np.nan)
    ranks=np.zeros((nc,2),int);masks=np.zeros((nc,nz),bool)
    reasons=np.full(nc,'not_evaluated',dtype='U96')
    universe=(lam>=min(FIT_STARTS))&(lam<=12)&(z<=16)
    for j,(lo,power,length) in enumerate(registry):
        use=(lam>=lo)&(lam<=12)&(z<=16);masks[j]=use;counts[j]=use.sum()
        try:fit=amplitude_fit(shape(lam[use],power,length),h[0,use],h[1:,use],args.svdcut)
        except (ValueError,np.linalg.LinAlgError) as e:reasons[j]=str(e);continue
        amp[j]=fit['A'];ab[:,j]=fit['Ab'];chi[j]=fit['chi'];chib[:,j]=fit['chib'];dof[j]=fit['dof']
        numerical[j]=True;reasons[j]='numerically_identifiable'
        ranks[j]=[fit['raw_rank'],fit['rank']]
        ids=np.flatnonzero(use);cov[j][np.ix_(ids,ids)]=fit['covariance'];eig[j,:len(ids)]=fit['eigenvalues']
    penalty=2+2*(int(universe.sum())-counts)
    r=select_cell(amp[:,None],ab[...,None],chi,chib,dof,penalty,np.zeros(nc,int),numerical,
                  qmin=args.qmin,chi2max=args.chi2max,allow_zero_dof=False)
    if not r['selected_available'] or not r['bootstrap_selected_available'].all():
        raise ValueError('Tail family does not cover all replicas; see numerical candidates')
    r.update(registry=registry,amplitudes=amp,amplitudes_boot=ab,chi2=chi,chi2_boot=chib,
             dof=dof,numerical=numerical,fit_masks=masks,ranks=ranks,covariance=cov,
             correlation_eigenvalues=eig,candidate_reason=reasons,penalty=penalty)
    return r


def quadrature_nodes(lam,stop,join,pz,mu,zs,order=8):
    # Extra near-zero intervals resolve the integrable logarithm in C0.
    bounds=np.r_[np.arange(0,stop,.5),stop,lam[lam<stop],join-1,join,
                 np.array([1e-7,1e-5,1e-3,.01,.1]),zs*pz/HBARC]
    bounds=np.unique(bounds[(bounds>=0)&(bounds<=stop)])
    q,w=leggauss(order);lo=bounds[:-1,None];hi=bounds[1:,None]
    return (((hi-lo)*q/2+(hi+lo)/2).ravel(),((hi-lo)*w/2).ravel())


def reconstruction_basis(nodes,lam,registry,join=8.):
    """C1 smooth blend over [join-1,join]; h(0) remains the input value."""
    t=np.clip(nodes-(join-1),0.,1.);switch=t*t*(3-2*t)
    idx,wl,wr,ok=interp_row(lam,nodes)
    if np.any((1-switch>1e-14)&~ok):raise ValueError('measured interpolation domain too short')
    D=np.zeros((len(nodes),len(lam)))
    D[np.arange(len(nodes)),idx]=(1-switch)*wl
    D[np.arange(len(nodes)),idx+1]+=(1-switch)*wr
    T=np.zeros((len(nodes),len(registry)))
    use=switch>0
    for j,(_,power,length) in enumerate(registry):T[use,j]=switch[use]*shape(nodes[use],power,length)
    return D,T


def coefficients(r):
    return np.vstack([r['weights']*np.nan_to_num(r['amplitudes']),
                      r['weights_boot']*np.nan_to_num(r['amplitudes_boot'])])


def fourier_operators(lam,registry,pz,x,args,join=8.,stop=160.,order=8):
    nodes,w=quadrature_nodes(lam,stop,join,pz,args.mu,args.zs_fm,order)
    D,T=reconstruction_basis(nodes,lam,registry,join)
    F=np.cos(np.outer(x,nodes))*w[None,:]/np.pi
    zeff=np.minimum(nodes/pz,args.zs_fm/HBARC)
    c0=(5/6)*np.log(zeff**2*args.mu**2*np.exp(2*np.euler_gamma)/4)+1.5
    return F@D,F@T,(F*c0[None,:])@D,(F*c0[None,:])@T


def transform(h,r,lam,pz,x,y,args,join=8.,stop=160.,order=8,xi_max=50.,kernel_order=48):
    D,T,DC,TC=fourier_operators(lam,r['registry'],pz,x,args,join,stop,order)
    Dy,Ty,DCy,TCy=fourier_operators(lam,r['registry'],pz,y,args,join,stop,order)
    B,mask=build_kernel(y,x,pz,args.mu,args.zs_fm,xi_max,kernel_order,args.xgap)
    g=args.alpha_s*CA/(2*np.pi);coeff=coefficients(r)
    q0=h@D.T+coeff@T.T
    q0y=h@Dy.T+coeff@Ty.T
    scheme_corr=g*(h@DC.T+coeff@TC.T)
    lamet_corr=g*(q0y@B.T)
    qhy=q0-scheme_corr
    matched=qhy-lamet_corr
    matched[:,~mask]=np.nan;lamet_corr[:,~mask]=np.nan
    # Central model spread before averaging, with the same interpolation part.
    common=h[0]@(D-g*DC-g*(B@Dy)).T
    basis=T-g*TC-g*(B@Ty)
    candidate=common[None,:]+np.nan_to_num(r['amplitudes'])[:,None]*basis.T
    spread=np.sqrt(np.sum(r['weights'][:,None]*(candidate-matched[0])**2,axis=0))
    spread[~mask]=np.nan
    return dict(quasi_ms=q0,quasi_hybrid=qhy,matched=matched,scheme_correction=scheme_corr,
                lamet_correction=lamet_corr,quasi_ms_grid=q0y,
                quasi_hybrid_grid=q0y-g*(h@DCy.T+coeff@TCy.T),kernel_matrix=B,
                computed_mask=mask,model_spread=spread,central_candidate_matched=candidate)


def summaries(values):
    return values[0],np.std(values[1:],axis=0,ddof=1)


def make_plots(out,pabs,lam_data,hdata,tail,primary,variants,lambda_plot,hplot,x,y):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors=['#2166ac','#b2182b','#228833']
    with PdfPages(out/'lambda_tail_all_Pz.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for ip,(ax,n) in enumerate(zip(axes,pabs)):
            m,e=summaries(hdata[ip]);v,s=summaries(hplot[ip])
            ax.errorbar(lam_data[ip],m,yerr=e,fmt='o',ls='none',ms=3,capsize=2,color=colors[ip],alpha=.7)
            ax.plot(lambda_plot,v,color=colors[ip]);ax.fill_between(lambda_plot,v-s,v+s,color=colors[ip],alpha=.18)
            ax.axvspan(4,12,color='.5',alpha=.07);ax.axvline(8,color='.5',ls='--',lw=.8)
            ax.set(xlim=(0,25),ylim=(-1,1.7),xlabel=r'$\lambda=zP_z$',title=rf'$n_z={n}$')
            ax.grid(alpha=.2)
        axes[0].set_ylabel(r'$M_0(z)/M_0(0)$')
        fig.suptitle('Correlated amplitude fits: registered power-exponential tail models')
        fig.text(.5,.015,'Gray: candidate fit range (also z/a<=16); dashed: blend ends at lambda=8. Band: 1000 paired-bootstrap stat.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.94));pdf.savefig(fig);fig.savefig(out/'lambda_tail_all_Pz.png',dpi=180);plt.close(fig)
    for kind in ['before_after','matched_all_momenta','tail_sensitivity','quasi_full_support']:
        fig,axes=plt.subplots(1,3,figsize=(16,4.9)) if kind!='matched_all_momenta' else plt.subplots(figsize=(8,5))
        if kind=='matched_all_momenta':axs=[axes]*3
        else:axs=axes
        for ip,(ax,n) in enumerate(zip(axs,pabs)):
            if kind=='before_after':series=[(primary[ip]['quasi_hybrid'],'quasi (hybrid proxy)','#4477aa'),(primary[ip]['matched'],'gg NLO matched','#bb3322')];xx=x
            elif kind=='tail_sensitivity':series=[(variants[ip][f'join_{j}'],f'join={j}',c) for j,c in zip([7,8,9],colors)];xx=x
            elif kind=='quasi_full_support':series=[(primary[ip]['quasi_ms_grid'],'MSbar quasi','#4477aa'),(primary[ip]['quasi_hybrid_grid'],'hybrid proxy','#bb3322')];xx=y
            else:series=[(primary[ip]['matched'],rf'$n_z={n}$',colors[ip])];xx=x
            for values,label,c in series:
                m,e=summaries(values);mask=np.isfinite(m)&((xx>=.02-1e-9) if kind!='quasi_full_support' else True)
                ax.plot(xx[mask],m[mask],color=c,label=label);ax.fill_between(xx[mask],m[mask]-e[mask],m[mask]+e[mask],color=c,alpha=.17)
            ax.set(xlim=(-3,3) if kind=='quasi_full_support' else (0,1),xlabel='$x$')
            if kind!='matched_all_momenta':ax.set_title(rf'$n_z={n}$')
            ax.axhline(0,c='.6',lw=.6);ax.grid(alpha=.2)
            if kind not in ['quasi_full_support','matched_all_momenta']:
                ax.axvspan(0,.2,color='.6',alpha=.1);ax.axvspan(.8,1,color='.6',alpha=.1)
        axs[0].set_ylabel('Locally normalized Fourier shape')
        for ax in set(axs):ax.legend(frameon=False,fontsize=8)
        fig.suptitle(dict(before_after='Fourier quasi distribution and NLO gg inverse matching',
            matched_all_momenta='NLO gg matched shapes: momentum comparison',
            tail_sensitivity='Lambda-tail splice sensitivity (same paired samples)',
            quasi_full_support='Quasi support retained beyond physical [-1,1]')[kind])
        fig.text(.5,.018,'mu=2 GeV, alpha_s=0.25; perturbative MSbar-to-hybrid proxy; singlet mixing omitted. Shape diagnostic; absolute normalization unverified.',ha='center',fontsize=8 if kind!='matched_all_momenta' else 6.5)
        fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(out/f'{kind}.pdf');fig.savefig(out/f'{kind}.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=SOURCE);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--mu',type=float,default=2.);p.add_argument('--alpha-s',type=float,default=.25)
    p.add_argument('--zs-fm',type=float,default=.25);p.add_argument('--qmin',type=float,default=1e-6)
    p.add_argument('--chi2max',type=float,default=10.);p.add_argument('--svdcut',type=float,default=1e-10)
    p.add_argument('--dx',type=float,default=.01);p.add_argument('--xgap',type=float,default=.02)
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Use a new output directory')
    meta=json.loads((args.input.parent/'manifest.json').read_text())
    assert meta['operator']=='unpolarized_TU_minus_SU_even'
    if args.mu!=meta['matching']['mu_GeV'] or args.alpha_s!=meta['matching']['alpha_s']:
        raise ValueError('GF conversion parameters differ; recompute upstream before changing mu/alpha')
    source_hash=digest(args.input)
    with np.load(args.input,allow_pickle=False) as d:
        central=d['params_selected'][...,0];boot=d['params_boot_selected'][...,0]
        z=d['z_values'];pabs=d['pabs_values'];afm=float(d['a_fm'])
        indices=d['bootstrap_indices'];confs=d['confs'];seed=int(d['seed'])
        upstream_fallback=d['any_stage_replica_fallback_fraction']
    if boot.shape!=(1000,3,25) or indices.shape!=(1000,406):raise ValueError('Unexpected bootstrap/coordinate axes')
    allh=np.concatenate([central[None],boot],axis=0)
    if not np.isfinite(allh).all() or np.any(allh[:,:,0]==0):raise ValueError('Nonfinite input/zero local denominator')
    den=allh[:,:,0].copy();hnorm=allh/den[:,:,None]
    x=np.linspace(-1,1,int(round(2/args.dx))+1);y=np.linspace(-3,3,int(round(6/args.dx))+1)
    lp=np.linspace(0,40,801);lam_data=np.array([2*np.pi*n*z/32 for n in pabs])
    hdata=np.moveaxis(hnorm,1,0);hplot=[];tails=[];primary=[];variants=[];checks=[];records=[]
    args.out.mkdir(parents=True);per=args.out/'per_momentum';per.mkdir()
    for ip,n in enumerate(pabs):
        h=hdata[ip];lam=lam_data[ip];P=float(physical_pz(n,afm))
        tail=fit_tail(lam,z,h,args);tails.append(tail)
        D,T=reconstruction_basis(lp,lam,tail['registry']);hp=h@D.T+coefficients(tail)@T.T;hplot.append(hp)
        print(f'n_z={n}: tail central candidates={tail["qualified_count"]}, replica fallback={int(tail["replica_fallback_used"].sum())}/1000',flush=True)
        r=transform(h,tail,lam,P,x,y,args);primary.append(r)
        var={'join_8':r['matched']}
        # One registered tail/window family. Variations keep all replica identities.
        for name,kw,yy in [('join_7',dict(join=7.),y),('join_9',dict(join=9.),y),
            ('lambda_max_320',dict(stop=320.),y),('fourier_order_16',dict(order=16),y),
            ('kernel_order_96',dict(kernel_order=96),y),('xi_max_100',dict(xi_max=100.),y),
            ('quasi_grid_half_step',{},np.linspace(-3,3,2*(len(y)-1)+1)),
            ('quasi_support_4',{},np.linspace(-4,4,int(round(8/args.dx))+1))]:
            rr=transform(h,tail,lam,P,x,yy,args,**kw);var[name]=rr['matched']
            delta=rr['matched']-r['matched'];region=(x>=.2-1e-10)&(x<=.8+1e-10)
            error=np.std(r['matched'][1:],axis=0,ddof=1)
            checks.append(dict(n_z=int(n),variation=name,max_central_shift=float(np.max(np.abs(delta[0,region]))),
                max_paired_shift_error=float(np.max(np.std(delta[1:,region],axis=0,ddof=1))),
                max_shift_over_stat=float(np.max(np.abs(delta[0,region])/np.maximum(error[region],1e-15)))))
        variants.append(var)
        np.testing.assert_allclose(tail['weights'].sum(),1.)
        np.testing.assert_allclose(tail['weights_boot'].sum(axis=1),1.)
        assert np.isfinite(r['matched'][:,r['computed_mask']]).all()
        np.testing.assert_allclose(r['quasi_ms'],r['quasi_ms'][:,::-1],atol=1e-12)
        np.testing.assert_allclose(r['matched'][:,r['computed_mask']],r['matched'][:,::-1][:,r['computed_mask']],atol=1e-8)
        rec=dict(n_z=int(n),Pz_GeV=P,central_tail_quality=bool(tail['quality_accepted']),
            tail_quality_candidates=int(tail['qualified_count']),tail_replica_fallback=int(tail['replica_fallback_used'].sum()),
            min_raw_rank=int(tail['ranks'][tail['numerical'],0].min()),
            removed_covariance_modes=int(np.sum(tail['ranks'][:,0]-tail['ranks'][:,1])),
            local_denominator_central=float(den[0,ip]),local_denominator_quantiles=np.quantile(den[1:,ip],[0,.01,.5,.99,1]).tolist(),
            denominator_sign_flips=int(np.sum(np.sign(den[1:,ip])!=np.sign(den[0,ip]))),
            finite_matched_replicas=1000,tail_model_parameters='fixed model grid, only amplitude fitted')
        records.append(rec)
        np.savez_compressed(per/f'Pz{n}.npz',**r,**{'tail_'+k:v for k,v in tail.items()},
            **{'variation_'+k:v for k,v in var.items()},x=x,quasi_grid=y,lambda_data=lam,lambda_plot=lp,
            h_normalized_data=h,h_reconstructed=hp,bootstrap_indices=indices,confs=confs,
            source_local_denominator=den[:,ip],upstream_fallback_fraction=upstream_fallback[ip],
            axes_curves=np.array(['central_then_bootstrap','x']),pz_GeV=P,
            plot_normalization=np.array('F[M0(z)/M0(0)], full-line convention; no absolute gluon momentum fraction'))
        print(json.dumps(rec),flush=True)
    hplot=np.array(hplot)
    make_plots(args.out,pabs,lam_data,hdata,tails,primary,variants,lp,hplot,x,y)
    payload=dict(x=x,quasi_grid=y,lambda_plot=lp,lambda_data=lam_data,h_normalized_data=hdata,
        h_reconstructed=hplot,pabs_values=pabs,pz_GeV=physical_pz(pabs,afm),bootstrap_indices=indices,confs=confs,
        nboot=1000,nconf=406,seed=seed,source_central=central,source_bootstrap=boot,
        axes_curves=np.array(['momentum','central_then_bootstrap','x']),
        axes_coordinate=np.array(['momentum','central_then_bootstrap','lambda']),
        absolute_gluon_normalization=np.array('not determined; locally normalized diagnostic only'))
    for key in ['quasi_ms','quasi_hybrid','matched','model_spread','computed_mask','scheme_correction','lamet_correction']:
        payload[key]=np.stack([r[key] for r in primary])
    np.savez_compressed(args.out/'lambda_fourier_quasi_matching.npz',**payload)
    assert digest(args.input)==source_hash
    report=dict(status='complete_exploratory_shape_pipeline',created_utc=datetime.now(timezone.utc).isoformat(),
        input=str(args.input.resolve()),input_sha256=source_hash,operator=meta['operator'],nconf=406,nboot=1000,seed=seed,
        input_flow_times=meta['flow_times'],input_quality_gates=meta['quality_gates'],a_fm=afm,
        source_fallback_propagation='same selected source replicas; upstream fallback fractions saved',
        local_normalization='M0(z)/M0(0) for original central and each replica separately; z=0 has inherited formal GF conversion',
        normalization_limit='no independently measured gluon momentum fraction or measured zero-momentum hybrid denominator',
        tail_formula='A*(lambda/8)^(-p)*exp[-(lambda-8)/lambda0]',powers=POWERS,decay_lengths=DECAY_LENGTHS,
        tail_windows=dict(lambda_min=FIT_STARTS,lambda_max=12,z_over_a_max=16),tail_parameters='p/lambda0 fixed shape hypotheses; one GLS amplitude each',
        tail_selection='replica-wise AIC chi2+2+2*N_omitted; q>=1e-6,chi2/dof<=10; labelled min candidate std fallback',
        tail_covariance='full from shared normalized bootstrap, ddof=1; fixed across fits; relative correlation eigenmode cut',svdcut=args.svdcut,
        splice='smooth cubic blend lambda in [7,8], model only above 8; sensitivities ends7/9',
        fourier='Q(x)=1/pi integral_0^infinity d_lambda cos(x*lambda) h(lambda); Gauss-Legendre with log-resolving near-zero intervals',
        lambda_max=160,lambda_max_check=320,quasi_y_support=[-3,3],quasi_support_check=[-4,4],dx=args.dx,
        mu_GeV=args.mu,alpha_s=args.alpha_s,zs_fm=args.zs_fm,
        matching='strict NLO: F[h]-g*F[c0*h]-g*K_hybrid*F[h], g=alpha_s*CA/(2*pi)',
        bridge='c0=(5/6)*log(min(|z|,zs)^2*mu^2*exp(2gammaE)/4)+3/2; perturbative leading-twist proxy',
        kernel_source='arXiv:2510.26425v2 Eqs.(14),(15); parton momentum |y|Pz in logs and hybrid Si term',
        x_mask=f'|x|>={args.xgap}',singlet_mixing='omitted as previously requested',
        uncertainty='original central estimate; std of 1000 paired selected replicas(ddof=1); tail-model spread saved separately',
        physical_status='shape sensitivity study; local/scheme normalization provisional; single a, no controlled SFTE, continuum, target-mass or large-P limit',
        moments=records,numerical_checks=checks,tests=dict(shared_indices=True,parity=True,all_replicas_finite_on_mask=True,source_unchanged=True),
        code_hashes={f:digest(HERE/f) for f in ['lambda_fourier_quasi_matching_v2.py','replica_selection_minerror_v3.py',
                       'match_unpolarized_quasi_from_extrap_v1.py','match_unpolarized_quasi_from_extrap_v2.py']})
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],out=str(args.out),moments=records,numerical_checks=checks),indent=2),flush=True)


if __name__=='__main__':main()
