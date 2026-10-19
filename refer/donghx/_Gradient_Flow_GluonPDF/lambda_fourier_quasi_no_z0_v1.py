#!/usr/bin/env python3
"""Absolute-amplitude, no-M(0)-division gluon PDF diagnostic; separate branch.

See NO_Z0_NORMALIZATION.md for derivation and unresolved physics. In particular
the local spin-2 projection, short-distance completion, gg-only matching and
fixed-a flow limit do NOT constitute a controlled physical PDF extraction.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gluon_absolute_normalization_v1 import (
    kinematic_factor,local_spin2_coefficient,fit_energies,fit_local_flow)
from lambda_fourier_quasi_matching_v2 import (
    SOURCE,digest,fit_tail,coefficients,quadrature_nodes,reconstruction_basis)
from match_unpolarized_quasi_from_extrap_v2 import build_kernel,physical_pz,HBARC,CA

HERE=Path(__file__).resolve().parent
C2=HERE.parent/'ratio_bootstrap_all_operators_v1/results_v1/c3_c2_bootstrap_allops_tau1p000_N406_Nboot1000.npz'
CONF_MANIFEST=HERE.parent/'ratio_bootstrap_latest_v1/manifests/common_all14_plusz_P345.tsv'
TAUS=np.array([1.,1.4,1.8,2.2,2.6,3.,3.4,3.8])


def selected_rows(r):
    return np.concatenate([r['params_selected'][None],r['params_boot_selected']],axis=0)


def c0_coefficient(z_gev,mu,zs_fm):
    zeff=np.minimum(np.asarray(z_gev),zs_fm/HBARC)
    return (5/6)*np.log(zeff**2*mu**2*np.exp(2*np.euler_gamma)/4)+1.5


def coordinate_basis(nodes,lam,registry,pz,args,join=8.,cut=1):
    """Local value is a local operator, NOT the bilocal MSbar limit.

    For 0<lambda<lambda_c, quadratic completion of the *hybrid proxy*.
    A separate order-zero basis is used inside the NLO convolution.
    Vary cut=1 -> 2; this is a model, not a measurement of the Pz=0 denominator.
    """
    D,T=reconstruction_basis(nodes,lam,registry,join)
    c=c0_coefficient(nodes/pz,args.mu,args.zs_fm)
    DC=c[:,None]*D;TC=c[:,None]*T
    inside=nodes<lam[cut];u=(nodes[inside]/lam[cut])**2
    D[inside]=0;T[inside]=0;DC[inside]=0;TC[inside]=0
    D[inside,0]=1-u;D[inside,cut]=u
    DC[inside,cut]=u*c0_coefficient(lam[cut]/pz,args.mu,args.zs_fm)
    return D,T,DC,TC


def transform(h,tail,lam,pz,x,y,args,join=8.,cut=1,stop=160.,order=8,
              kernel_order=48,xi_max=50.):
    nodes,w=quadrature_nodes(lam,stop,join,pz,args.mu,args.zs_fm,order)
    D,T,DC,TC=coordinate_basis(nodes,lam,tail['registry'],pz,args,join,cut)
    F=np.cos(np.outer(np.r_[x,y],nodes))*w[None,:]/np.pi
    fd,ft,fc,ftc=F@D,F@T,F@DC,F@TC
    coeff=coefficients(tail);n=len(x)
    tree=h@fd.T+coeff@ft.T
    correction=args.alpha_s*CA/(2*np.pi)*(h@fc.T+coeff@ftc.T)
    B,mask=build_kernel(y,x,pz,args.mu,args.zs_fm,xi_max,kernel_order,args.xgap)
    lamet=args.alpha_s*CA/(2*np.pi)*(tree[:,n:]@B.T)
    matched=tree[:,:n]-correction[:,:n]-lamet
    matched[:,~mask]=np.nan;lamet[:,~mask]=np.nan
    return dict(matched=matched,quasi_tree=tree[:,:n],
        quasi_hybrid=tree[:,:n]-correction[:,:n],scheme_correction=correction[:,:n],
        lamet_correction=lamet,quasi_tree_grid=tree[:,n:],
        quasi_hybrid_grid=tree[:,n:]-correction[:,n:],kernel_matrix=B,computed_mask=mask)


def stats(a):
    return a[0],np.std(a[1:],axis=0,ddof=1)


def plot_results(out,pabs,x,h,lam,tails,results,variants,local_values,local_fit,E,t,c2,args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2166ac','#b2182b','#228833']
    def save(fig,name):
        fig.tight_layout(rect=(0,.06,1,.92))
        fig.savefig(out/(name+'.pdf'));fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    def line(ax,a,label,color,xx=x):
        v,s=stats(a);ok=np.isfinite(v)&(xx>=args.xgap-1e-10)
        ax.plot(xx[ok],v[ok],label=label,color=color,lw=1.4)
        ax.fill_between(xx[ok],v[ok]-s[ok],v[ok]+s[ok],color=color,alpha=.18)
    for name in ['xg_all_Pz','xg_comparison','short_distance_sensitivity']:
        fig,axes=plt.subplots(1,3,figsize=(15,4.9))
        for ip,(ax,n) in enumerate(zip(axes,pabs)):
            if name=='xg_comparison':
                line(ax,results[ip]['quasi_hybrid'],'quasi: hybrid proxy','#777777')
            line(ax,results[ip]['matched'],'gg NLO matched',colors[ip])
            if name=='short_distance_sensitivity':
                for key,label,col in [('cut2','short-distance cutoff 2a','#dd9933'),
                                      ('local_unexpanded','local inverse 1/C','#8844aa')]:
                    v,_=stats(variants[ip][key]);ok=np.isfinite(v)&(x>=args.xgap-1e-10)
                    ax.plot(x[ok],v[ok],ls='--',label=label,color=col)
            ax.axhline(0,color='.6',lw=.6);ax.set(xlim=(0,1),xlabel='$x$',title=rf'$n_z={n},\ P_z={physical_pz(n,args.a_fm):.2f}\ {{\rm GeV}}$')
            ax.axvspan(0,.2,color='.5',alpha=.07);ax.axvspan(.8,1,color='.5',alpha=.07)
            ax.grid(alpha=.16);ax.legend(fontsize=9)
        axes[0].set_ylabel(r'$xg(x,\mu=2\ {\rm GeV})$ (diagnostic)')
        fig.suptitle('No division by M(0): absolute-amplitude test, NOT a controlled physical PDF')
        fig.text(.5,.015,'Bands: 1000 paired-bootstrap statistics only. Single a; gg only; model-dependent short-distance and lambda completion.',ha='center',fontsize=9)
        save(fig,name)
    fig,ax=plt.subplots(figsize=(8,5.5))
    for ip,n in enumerate(pabs):line(ax,results[ip]['matched'],rf'$n_z={n}$',colors[ip])
    ax.set(xlim=(0,1),xlabel='$x$',ylabel=r'$xg(x,2\ {\rm GeV})$ (diagnostic)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('No M(0) division | gg-only, finite-a diagnostic')
    fig.text(.5,.015,'Statistical bands do not include matching / flow-window / short-distance / tail-model systematics.',ha='center',fontsize=8)
    save(fig,'xg_overlay')
    fig,axes=plt.subplots(1,3,figsize=(15,4.9));lp=np.linspace(1e-6,25,601)
    for ip,(ax,n) in enumerate(zip(axes,pabs)):
        D,T,DC,TC=coordinate_basis(lp,lam[ip],tails[ip]['registry'],physical_pz(n,args.a_fm),args)
        curve=h[ip]@(D-args.alpha_s*CA/(2*np.pi)*DC).T+coefficients(tails[ip])@(T-args.alpha_s*CA/(2*np.pi)*TC).T
        m,e=stats(h[ip]);v,s=stats(curve)
        ax.errorbar(lam[ip],m,yerr=e,fmt='o',ls='none',ms=3,capsize=2,color=colors[ip],label='local point + bilocal MS data')
        ax.plot(lp,v,color='.2',label='hybrid proxy completion');ax.fill_between(lp,v-s,v+s,color='.5',alpha=.2)
        ax.set(xlim=(0,20),xlabel=r'$\lambda=zP_z$',title=rf'$n_z={n}$');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel(r'$H=-2R/[g^2(aE)]$; $H(0)=2\langle x\rangle_g$')
    fig.suptitle('Local and bilocal quantities are distinct: no forced H(0)=1')
    save(fig,'lambda_no_z0')
    fig,axes=plt.subplots(1,3,figsize=(15,4.9))
    for ip,(ax,n) in enumerate(zip(axes,pabs)):
        # Local values are converted with each paired energy.
        fac=kinematic_factor(E[:,ip],args.alpha_s)/2
        m,e=stats(local_values[:,:,ip]*fac[:,None]);ax.errorbar(TAUS,m,yerr=e,fmt='o',ls='none',capsize=2,color=colors[ip])
        pars=selected_rows(local_fit[ip])*fac[:,None];tt=np.linspace(0,4,101)
        v,s=stats(pars[:,0,None]+pars[:,1,None]*tt)
        ax.plot(tt,v,color=colors[ip]);ax.fill_between(tt,v-s,v+s,color=colors[ip],alpha=.18)
        ax.errorbar([0],[v[0]],yerr=[s[0]],fmt='s',color='black',capsize=3)
        ax.set(xlabel=r'$t_f/a^2$',title=rf'$n_z={n}$');ax.grid(alpha=.2)
    axes[0].set_ylabel(r'$\langle x\rangle_g$ (local spin-2 projection diagnostic)')
    fig.suptitle('Local matching followed by a separate correlated flow extrapolation')
    save(fig,'local_matching_flow')
    fig,axes=plt.subplots(1,3,figsize=(15,4.9))
    # Effective energies shown only where every paired logarithm is defined.
    for ip,(ax,n) in enumerate(zip(axes,pabs)):
        valid=np.all((c2[:,1:,ip]>0)&(c2[:,:-1,ip]>0),axis=0)
        eff=np.log(c2[:,:-1,ip][:,valid]/c2[:,1:,ip][:,valid]);m,e=stats(eff)
        ax.errorbar(t[:-1][valid]+.5,m,yerr=e,fmt='o',ls='none',capsize=2,color=colors[ip])
        mean,err=stats(E[:,ip]);ax.axhline(mean,color='black');ax.axhspan(mean-err,mean+err,color='.5',alpha=.2)
        ax.set(xlabel=r'$t/a$',title=rf'$n_z={n}$');ax.grid(alpha=.2)
    axes[0].set_ylabel(r'$aE_{\rm eff}$');fig.suptitle('Same-configuration C2 energy: correlated one-state window/AIC fit')
    save(fig,'energy_audit')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,default=SOURCE.parent/'lambda_fourier_quasi_no_z0_v1')
    p.add_argument('--mu',type=float,default=2.);p.add_argument('--alpha-s',type=float,default=.25)
    p.add_argument('--zs-fm',type=float,default=.25);p.add_argument('--qmin',type=float,default=1e-6)
    p.add_argument('--chi2max',type=float,default=10.);p.add_argument('--svdcut',type=float,default=1e-10)
    p.add_argument('--xgap',type=float,default=.02)
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('New branch refuses to overwrite output')
    meta=json.loads((args.input.parent/'manifest.json').read_text())
    if meta['operator']!='unpolarized_TU_minus_SU_even':raise ValueError('operator mismatch')
    if args.mu!=meta['matching']['mu_GeV'] or args.alpha_s!=meta['matching']['alpha_s']:raise ValueError('upstream coupling/scale mismatch')
    with np.load(args.input,allow_pickle=False) as f:
        R=np.concatenate([f['params_selected'][None,...,0],f['params_boot_selected'][...,0]])
        z=f['z_values'];pabs=f['pabs_values'];args.a_fm=float(f['a_fm'])
        indices=f['bootstrap_indices'];confs=f['confs'];seed=int(f['seed'])
        upstream_fallback=f['any_stage_replica_fallback_fraction']
    if R.shape!=(1001,3,25) or not np.isfinite(R).all():raise ValueError('unexpected source')
    conf_table=np.genfromtxt(CONF_MANIFEST,names=True,dtype=int,delimiter='\t')
    np.testing.assert_array_equal(conf_table['conf_id'].astype(str),confs.astype(str))
    with np.load(C2,allow_pickle=False) as f:
        np.testing.assert_array_equal(indices,f['bootstrap_indices'])
        if str(f['manifest_sha256'])!=digest(CONF_MANIFEST):raise ValueError('C2 configuration manifest hash')
        t=f['tsep'];c2=np.concatenate([f['c2_original'][None,0].real,f['c2_bootstrap'][:,0].real])
        c2_contract=str(f['contract_json'])
    args.out.mkdir(parents=True);per=args.out/'per_momentum';per.mkdir()
    print('Fitting paired C2 energies ...',flush=True)
    energy,energy_fits=fit_energies(t,c2,args)
    E=np.stack([selected_rows(r)[:,0] for r in energy],axis=1)
    fac=kinematic_factor(E,args.alpha_s)
    bare=[];sources=[]
    for tau in TAUS:
        src=args.input.parent/'reselected_matched'/('unpolarized_tau'+f'{tau:.3f}'.replace('.','p')+'_selected.npz')
        with np.load(src,allow_pickle=False) as f:
            np.testing.assert_array_equal(indices,f['bootstrap_indices']);np.testing.assert_array_equal(confs,f['confs'])
            np.testing.assert_array_equal(pabs,f['pabs_values'])
            bare.append(np.concatenate([f['bare_params_selected'][None,:,0,0],f['bare_params_boot_selected'][:,:,0,0]]))
        sources.append({'path':str(src),'sha256':digest(src)})
    bare=np.stack(bare,axis=1)
    C=local_spin2_coefficient(TAUS,args.a_fm,args.mu,args.alpha_s)
    loc=bare*(2-C)[None,:,None];loc_alt=bare/C[None,:,None]
    local=[fit_local_flow(TAUS,loc[:,:,ip],args) for ip in range(3)]
    local_alt=[fit_local_flow(TAUS,loc_alt[:,:,ip],args) for ip in range(3)]
    Rlocal=np.stack([selected_rows(r)[:,0] for r in local],axis=1)
    Ralt=np.stack([selected_rows(r)[:,0] for r in local_alt],axis=1)
    H=R*fac[:,:,None];H_oldlocal=H.copy();H[:,:,0]=Rlocal*fac
    H_alt=H.copy();H_alt[:,:,0]=Ralt*fac
    h=np.moveaxis(H,1,0);ha=np.moveaxis(H_alt,1,0)
    x=np.linspace(-1,1,201);y=np.linspace(-3,3,601)
    lam=np.array([2*np.pi*n*z/32 for n in pabs])
    results=[];tails=[];variants=[];records=[];checks=[]
    for ip,n in enumerate(pabs):
        print(f'n_z={n}: E={E[0,ip]:.7f} +/- {np.std(E[1:,ip],ddof=1):.7f}; local <x>={H[0,ip,0]/2:.7f}',flush=True)
        tail=fit_tail(lam[ip],z,h[ip],args);tails.append(tail)
        P=float(physical_pz(n,args.a_fm));r=transform(h[ip],tail,lam[ip],P,x,y,args);results.append(r)
        var={}
        spec=[('join7',h[ip],dict(join=7.),y),('join9',h[ip],dict(join=9.),y),
              ('cut2',h[ip],dict(cut=2),y),('local_unexpanded',ha[ip],{},y),
              ('old_endpoint_local',H_oldlocal[:,ip],{},y),
              ('order16',h[ip],dict(order=16),y),('lambda320',h[ip],dict(stop=320.),y),
              ('kernel96',h[ip],dict(kernel_order=96),y),('xi100',h[ip],dict(xi_max=100.),y),
              ('y_halfstep',h[ip],{},np.linspace(-3,3,1201)),
              ('y_support4',h[ip],{},np.linspace(-4,4,801))]
        region=(x>=.2-1e-10)&(x<=.8+1e-10);err=np.std(r['matched'][1:],axis=0,ddof=1)
        for key,hh,kw,yy in spec:
            rr=transform(hh,tail,lam[ip],P,x,yy,args,**kw);var[key]=rr['matched']
            delta=rr['matched']-r['matched']
            checks.append(dict(n_z=int(n),variation=key,
                max_central_shift=float(np.max(abs(delta[0,region]))),
                max_shift_over_stat=float(np.max(abs(delta[0,region])/err[region])),
                max_paired_shift_error=float(np.max(np.std(delta[1:,region],axis=0,ddof=1)))))
        variants.append(var)
        rec=dict(n_z=int(n),Pz_GeV=P,energy_lattice=float(E[0,ip]),energy_stat=float(np.std(E[1:,ip],ddof=1)),
                 energy_replica_fallback=int(energy[ip]['replica_fallback_used'].sum()),
                 local_R=float(Rlocal[0,ip]),local_xg_moment=float(H[0,ip,0]/2),
                 local_xg_moment_stat=float(np.std(H[1:,ip,0]/2,ddof=1)),
                 local_replica_fallback=int(local[ip]['replica_fallback_used'].sum()),
                 local_central_quality=bool(local[ip]['quality_accepted']),
                 local_unexpanded_moment=float(H_alt[0,ip,0]/2),
                 tail_replica_fallback=int(tail['replica_fallback_used'].sum()),
                 tail_central_quality=bool(tail['quality_accepted']),tail_candidates=int(tail['qualified_count']),
                 finite_matched_replicas=1000)
        records.append(rec);print(json.dumps(rec),flush=True)
        ep=dict(energy[ip])
        ec=np.full((len(energy_fits[ip]),len(t),len(t)),np.nan)
        for j,(window,fit) in enumerate(zip(ep['windows'],energy_fits[ip])):
            ids=np.flatnonzero((t>=window[0])&(t<=window[1]));ec[j][np.ix_(ids,ids)]=fit['covariance']
        ep['covariance']=ec
        np.savez_compressed(per/f'Pz{n}.npz',**r,**{'tail_'+k:v for k,v in tail.items()},
            **{'energy_'+k:v for k,v in ep.items()},**{'local_'+k:v for k,v in local[ip].items()},
            **{'local_unexpanded_'+k:v for k,v in local_alt[ip].items()},
            **{'variation_'+k:v for k,v in var.items()},x=x,quasi_grid=y,lambda_data=lam[ip],H=h[ip],
            E_lattice=E[:,ip],kinematic_factor=fac[:,ip],local_bare=bare[:,:,ip],
            local_matched=loc[:,:,ip],local_coefficient=C,flow_times=TAUS,
            bootstrap_indices=indices,confs=confs,upstream_fallback_fraction=upstream_fallback[ip])
    plot_results(args.out,pabs,x,h,lam,tails,results,variants,loc,local,E,t,c2,args)
    payload=dict(x=x,pabs_values=pabs,z_values=z,lambda_data=lam,H=h,
        matched=np.stack([r['matched'] for r in results]),quasi_hybrid=np.stack([r['quasi_hybrid'] for r in results]),
        computed_mask=results[0]['computed_mask'],energy_lattice=E,kinematic_factor=fac,
        local_coefficient=C,flow_times=TAUS,local_moment=H[:,:,0]/2,
        local_moment_unexpanded=H_alt[:,:,0]/2,source_R=R,
        local_R=Rlocal,bootstrap_indices=indices,confs=confs,seed=seed,
        axes_curves=np.array(['pabs','central_then_bootstrap','x']),c2_tsep=t,c2=c2)
    np.savez_compressed(args.out/'no_z0_xg.npz',**payload)
    for r in results:
        assert np.isfinite(r['matched'][:,r['computed_mask']]).all()
        np.testing.assert_allclose(r['matched'][:,r['computed_mask']],r['matched'][:,::-1][:,r['computed_mask']],atol=1e-8)
    np.testing.assert_allclose(h[:,:,1:],np.moveaxis(R[:,:,1:]*fac[:,:,None],1,0))
    metadata=dict(schema='no_z0_absolute_gluon_diagnostic_v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        input=str(args.input),input_sha256=digest(args.input),source_flows=sources,
        scripts={str(q):digest(q) for q in [Path(__file__),HERE/'gluon_absolute_normalization_v1.py',
            HERE/'lambda_fourier_quasi_matching_v2.py',HERE/'match_unpolarized_quasi_from_extrap_v2.py',
            HERE/'replica_selection_minerror_v3.py',HERE/'extrapolate_unpolarized_tau_ge1_correlated_v2.py']},
        C2_path=str(C2),C2_contract=json.loads(c2_contract),configuration_manifest_sha256=digest(CONF_MANIFEST),
        Nconf=len(confs),Nboot=len(indices),settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        no_division_by_local_point=True,no_moment_rescaling=True,
        normalization='H=-2 R_MS[gF*gF]/(4 pi alpha_s aE); full-line 1/pi cosine transform gives xg for x>0',
        local_matching='1-C_A alpha_s/(4pi)*[(11/3)*(ln(2 mu^2 t)+gamma_E)+7/3]; spin-2 forward projection; gg diagonal only',
        local_status='continuum spin-2 forward-projection assumption, not exact finite-a TU-SU tensor identity',
        scheme='leading-twist MSbar-to-hybrid perturbative proxy; short-distance quadratic completion, not measured Pz=0 denominator',
        physical_status='exploratory absolute-amplitude diagnostic, NOT controlled physical PDF',
        covariance='full fixed bootstrap covariance ddof1; correlation eigenmode cutoff; original-data central; paired 1000 replicas',
        energy_model='forward one-state, tmin 8..12 tmax 13..15 >=4 times; thermal effects omitted at T96',
        flow_model='contiguous >=3 of eight tau>=1 points; correlated linear, per-replica quality/AIC, flagged min-error fallback',
        tail_model='27 fixed power/exponential shapes, one amplitude; registered lambda windows, inherited v2 family',
        limitations=['single lattice spacing; flow radius sqrt(8tau)/z is not uniformly small',
            'quark-singlet term omitted, not a quenched theory','local finite-a spin-2 projection not independently measured',
            'C2 one-state excited-state systematic not resolved','alpha_s fixed, higher orders and scale error not in bands',
            'tail and short-distance model scans separate, not included in statistical bands',
            'x<0.02 not evaluated; endpoints unreliable; no positivity clipping; no enforced integral'],
        records=records,checks=checks)
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    (args.out/'validation.json').write_text(json.dumps(dict(paired_indices=True,nonlocal_input_unchanged=True,
        all_3000_replica_curves_finite_on_computed_domain=True,even_parity=True,
        local_not_unit_normalized=True,physical_validation=False),indent=2)+'\n')
    print('FINISHED '+str(args.out),flush=True)


if __name__=='__main__':main()
