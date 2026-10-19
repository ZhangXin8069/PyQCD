#!/usr/bin/env python3
"""Gluon-only NLO test with an explicit perturbative MSbar -> hybrid bridge.

arXiv:2510.26425v2 Eqs.(14),(15) and its short-distance C0; numerical strategy
from Matching_norm.py, but with gluon coefficients. Read quasi_matching_README.md.
"""
import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sici
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from match_unpolarized_quasi_from_extrap_v1 import gluon_ratio_kernel, interp_row, fourier_matrix

BASE = Path(__file__).resolve().parent/'bare_all_operators_matching_v1'
OPERATOR = 'unpolarized_TU_minus_SU_even'
CA, HBARC = 3., .1973269804


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def physical_pz(n, a):
    return 2*np.pi*np.asarray(n)*HBARC/(32*a)


def kernel(xi, parton_pz, mu, zs):
    r = gluon_ratio_kernel(xi, np.broadcast_to(parton_pz, xi.shape), mu)
    if zs is not None:
        d = 1-xi
        r += (5/6)*(-1/np.abs(d)+2*sici(d*parton_pz*zs/HBARC)[0]/(np.pi*d))
    return r


def quadrature(limit, order, extra=()):
    bounds = [-limit, limit, -10., -5., -2., -1., 0., 1., 2., 5., 10.] + list(extra)
    for d in (1e-6, 1e-4, .001, .01, .05, .1, .25, .5): bounds += [-d, d, 1-d, 1+d]
    bounds = np.unique([v for v in bounds if -limit <= v <= limit])
    q, w = leggauss(order)
    l, r = bounds[:-1,None], bounds[1:,None]
    return ((r-l)*q/2+(r+l)/2).ravel(), ((r-l)*w/2).ravel()


def build_kernel(ygrid, xgrid, pz_gev, mu, zs, limit=50., order=48, gap=.02):
    """Kf=int dxi [R(xi,|x/xi|P)f(x/xi)/|xi|-R(xi,|x|P)f(x)].

    Plus subtraction at xi=1: counterterm parameters evaluated at y=x.
    First term zero outside declared QUASI support, not physical [-1,1].
    """
    B = np.zeros((len(xgrid), len(ygrid))); mask = np.abs(xgrid) >= gap-1e-12
    for ix, x in enumerate(xgrid):
        if not mask[ix]: continue
        xi, w = quadrature(limit, order, (x/ygrid[0], x/ygrid[-1]))
        y = x/xi
        idx, wl, wr, ok = interp_row(ygrid, y)
        coeff = w[ok]*kernel(xi[ok], np.abs(y[ok])*abs(pz_gev), mu, zs)/np.abs(xi[ok])
        np.add.at(B[ix], idx[ok], coeff*wl[ok]); np.add.at(B[ix], idx[ok]+1, coeff*wr[ok])
        idx0, wl0, wr0, ok0 = interp_row(ygrid, np.array([x]))
        if not ok0[0]: raise ValueError('output outside quasi support')
        sub = -np.sum(w*kernel(xi, np.full_like(xi, abs(x*pz_gev)), mu, zs))
        B[ix,idx0[0]] += sub*wl0[0]; B[ix,idx0[0]+1] += sub*wr0[0]
    if not np.isfinite(B).all(): raise ValueError('nonfinite kernel')
    return B, mask


def c0_coefficient(z, a, mu, zs):
    """C0=1+g*c0, g=alpha_s CA/(2pi). z=0 is a separate lattice contact convention."""
    zf = np.asarray(z, float)*a
    if zs is not None: zf = np.minimum(zf, zs)
    c = np.zeros_like(zf); ok = zf>0
    c[ok] = (5/6)*np.log((zf[ok]/HBARC)**2*mu**2*np.exp(2*np.euler_gamma)/4)+1.5
    return c


def meanerr(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(a,axis=axis), np.nanstd(a,axis=axis,ddof=1)


def pipeline(h, Fy, Fx, B, c0, alpha, mask):
    """Additional layers strictly O(alpha): do not apply K to corrected h."""
    g = alpha*CA/(2*np.pi)
    q0 = h@Fx.T
    scheme_corr = g*h@(Fx*c0[None,:]).T
    lamet_corr = g*h@(B@Fy).T
    qscheme = q0-scheme_corr
    f = qscheme-lamet_corr
    f[:,~mask] = np.nan; lamet_corr[:,~mask] = np.nan
    return q0, qscheme, f, scheme_corr, lamet_corr


def draw(out, x, pabs, quasi, matched, cuts, cut_samples, scheme):
    for sensitivity in (False, True):
        fig, axs = plt.subplots(1,len(pabs),figsize=(14,4.2),squeeze=False)
        for ip,n in enumerate(pabs):
            ax = axs[0,ip]
            if not np.isfinite(matched[ip]).any():
                ax.text(.5,.5,'Not computed:\nmissing short-distance M0 points',ha='center',va='center',transform=ax.transAxes)
            else:
                series = [(cut_samples[ic,ip], f'zmax/a={cut}', 'o', None) for ic,cut in enumerate(cuts)] if sensitivity else [
                    (quasi[ip],f'{scheme} quasi','o','#759cba'),(matched[ip],'NLO gg matched','s','#ac280e')]
                for k,(v,label,marker,color) in enumerate(series):
                    y,e = meanerr(v)
                    idx = np.flatnonzero((x>=.02-1e-10)&np.isfinite(y))[::5]
                    ax.errorbar(x[idx]+.0015*(k-.5),y[idx],yerr=e[idx],fmt=marker,ls='none',ms=3,capsize=2,alpha=.8,color=color,label=label)
                ax.legend(fontsize=7,frameon=False)
            ax.set(xlim=(0,1),xlabel='$x$',title=rf'$n_z={n}$')
            ax.axhline(0,lw=.6,c='.5'); ax.grid(alpha=.2)
            ax.axvspan(0,.1,color='.8',alpha=.2); ax.axvspan(.9,1,color='.8',alpha=.2)
        axs[0,0].set_ylabel('locally normalized transform (diagnostic)')
        fig.suptitle('Finite-z cutoff sensitivity (no tail fit)' if sensitivity else r'$(T_U-S_U)_{even}$: perturbative scheme bridge; no mixing')
        fig.tight_layout()
        tag = 'unpolarized_zmax_sensitivity' if sensitivity else 'unpolarized_before_after_all_Pz'
        fig.savefig(out/f'{tag}.pdf'); fig.savefig(out/f'{tag}.png',dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=BASE/'flow_extrap_tau_ge1/matched_small_flow_extrap_tau_ge1.npz')
    p.add_argument('--out',type=Path,default=BASE/'quasi_matching_tau_ge1_v2')
    p.add_argument('--mu',type=float); p.add_argument('--alpha-s',type=float)
    p.add_argument('--scheme',choices=('hybrid','ratio'),default='hybrid'); p.add_argument('--zs-fm',type=float,default=.25)
    p.add_argument('--quasi-xmax',type=float,default=3.); p.add_argument('--dx',type=float,default=.005)
    p.add_argument('--zmax',type=int,default=24); p.add_argument('--xi-max',type=float,default=50.)
    p.add_argument('--quadrature-order',type=int,default=48); p.add_argument('--xkernel-min',type=float,default=.02)
    a = p.parse_args()
    if a.dx<=0 or a.quasi_xmax<=1 or a.zs_fm<=0 or a.xi_max<=10 or a.quadrature_order<8: p.error('invalid numerical parameters')
    with np.load(a.input,allow_pickle=False) as d:
        io = list(d['operator_labels']).index(OPERATOR); z=d['z_values'].copy(); pabs=d['pabs_values'].copy()
        raw=d['m0_boot'][:,io].copy(); central=d['m0'][io].copy(); indices=d['bootstrap_indices'].copy()
        ncfg,nb,afm=int(d['nconf']),int(d['nboot']),float(d['a_fm'])
        imu,ialpha=float(d['matching_mu_GeV']),float(d['matching_alpha_s'])
    mu=imu if a.mu is None else a.mu; alpha=ialpha if a.alpha_s is None else a.alpha_s
    if not np.isclose(mu,imu) or not np.isclose(alpha,ialpha): p.error('rerun GF conversion at the new mu/alpha first: source parameters differ')
    if raw.shape!=(nb,len(pabs),len(z)) or indices.shape!=(nb,ncfg): raise ValueError('bootstrap axes mismatch')
    if not np.array_equal(z,np.arange(len(z))) or a.zmax not in z or a.zmax<2: raise ValueError('invalid z grid/cutoff')
    a.out.mkdir(parents=True,exist_ok=True)
    x=np.linspace(-1,1,int(round(2/a.dx))+1)
    y=np.linspace(-a.quasi_xmax,a.quasi_xmax,int(round(2*a.quasi_xmax/a.dx))+1)
    cuts=np.unique([v for v in (12,16,20,a.zmax) if v<=a.zmax])
    arrays=[np.full((len(pabs),nb,len(x)),np.nan) for _ in range(5)]
    q0,qs,matched,sc,kc=arrays
    cutoff=np.full((len(cuts),len(pabs),nb,len(x)),np.nan)
    norm=np.full_like(raw,np.nan); original=np.full((len(pabs),len(x)),np.nan)
    records=[]; checks=[]; zs=a.zs_fm if a.scheme=='hybrid' else None
    mask=np.abs(x)>=a.xkernel_min-1e-12
    for ip,n in enumerate(pabs):
        P=float(physical_pz(n,afm)); use=z<=a.zmax
        missing=z[use&~np.isfinite(raw[:,ip]).all(axis=0)]
        rec=dict(n_z=int(n),Pz_GeV=P,missing_z=missing.tolist(),nboot=nb)
        if len(missing) or not np.isfinite(central[ip,use]).all():
            rec['status']='not_computed_missing_input'; records.append(rec); continue
        den=raw[:,ip,0]
        if np.any(den==0) or not np.isfinite(den).all(): raise ValueError('invalid normalization denominator')
        norm[:,ip]=raw[:,ip]/den[:,None]
        rec['local_denominator_quantiles']=np.quantile(den,[0,.01,.5,.99,1]).tolist()
        rec['local_denominator_sign_flip_count']=int(np.sum(np.sign(den)!=np.sign(central[ip,0])))
        rec['normalization_status']='heavy_tail_risk_sign_crossing' if rec['local_denominator_sign_flip_count'] else 'no_sign_crossing_observed'
        print(f'kernel n_z={n}, Pz={P:.9f} GeV, {a.scheme}',flush=True)
        B,mask=build_kernel(y,x,P,mu,zs,a.xi_max,a.quadrature_order,a.xkernel_min)
        zz=z[use]; h=norm[:,ip,use]
        Fy=fourier_matrix(zz,n,afm,y); Fx=fourier_matrix(zz,n,afm,x); c=c0_coefficient(zz,afm,mu,zs)
        vals=pipeline(h,Fy,Fx,B,c,alpha,mask)
        for arr,v in zip(arrays,vals): arr[ip]=v
        original[ip]=pipeline((central[ip,use]/central[ip,0])[None,:],Fy,Fx,B,c,alpha,mask)[2][0]
        for ic,cut in enumerate(cuts):
            uz=z<=cut; zz0=z[uz]
            cutoff[ic,ip]=pipeline(norm[:,ip,uz],fourier_matrix(zz0,n,afm,y),fourier_matrix(zz0,n,afm,x),B,c0_coefficient(zz0,afm,mu,zs),alpha,mask)[2]
        for name,lim,order,yy in (
            ('quadrature_order_doubled',a.xi_max,a.quadrature_order*2,y),
            ('xi_max_doubled',a.xi_max*2,a.quadrature_order,y),
            ('input_grid_spacing_halved',a.xi_max,a.quadrature_order,np.linspace(y[0],y[-1],2*(len(y)-1)+1)),
            ('quasi_support_extended_4',a.xi_max,a.quadrature_order,np.linspace(-4.,4.,int(round(8/a.dx))+1))):
            BB,_=build_kernel(yy,x,P,mu,zs,lim,order,a.xkernel_min)
            ff=pipeline(h,fourier_matrix(zz,n,afm,yy),Fx,BB,c,alpha,mask)[2]
            dm,de=meanerr(ff-matched[ip]); _,se=meanerr(matched[ip]); region=(x>=.1)&(x<=.9)
            checks.append(dict(n_z=int(n),check=name,max_absolute_central_shift=float(np.max(np.abs(dm[region]))),
                max_paired_shift_error=float(np.max(de[region])),max_shift_over_stat_error=float(np.max(np.abs(dm[region])/se[region]))))
        rec.update(status='exploratory_NLO_matched',finite_bootstrap_per_computed_x_min=int(np.min(np.isfinite(matched[ip][:,mask]).sum(axis=0))),lambda_max=float(2*np.pi*n*a.zmax/32))
        records.append(rec)
    meta=dict(schema='gluon_quasi_matching_perturbative_bridge_v2',created_utc=datetime.now(timezone.utc).isoformat(),
        input=str(a.input.resolve()),input_sha256=digest(a.input),script_sha256=digest(__file__),kernel_source_sha256=digest(Path(__file__).with_name('match_unpolarized_quasi_from_extrap_v1.py')),
        operator=OPERATOR,Nconf=ncfg,Nboot=nb,a_fm=afm,Lz=32,mu_GeV=mu,alpha_s=alpha,C_A=CA,scheme=a.scheme,zs_fm=zs,
        normalization='per-replica M0(z)/M0(0); NOT an independently measured gluon momentum fraction',
        bridge='C0=1+g*((5/6)*log(z_eff^2*mu^2*exp(2gamma_E)/4)+3/2); z_eff=min(|z|,zs) for hybrid; c0(0)=0 local-grid convention',
        bridge_status='perturbative leading-twist proxy, NOT measured zero-momentum hybrid denominator',
        formula='F[h]-g*F[c0*h]-g*K_scheme*F[h]; g=alpha_s*CA/(2pi); no exact inverse or new GF factors',
        kernel='arXiv:2510.26425v2 Eqs.(14),(15)',mixing='quark-singlet contribution omitted by user; not quenched QCD',
        source_status='GF-to-MSbar quasi diagnostic then tau>=1 extrapolation; fixed input parameters',
        source_flow_statistics='diagonal-error linear flow fits and fixed central AIC weights inherited, not re-fitted',
        fourier='rectangular finite-z cosine transform, no tail model or missing-z filling',zmax_a=a.zmax,zmax_scan=cuts.tolist(),
        quasi_y_support=[float(y[0]),float(y[-1])],dx=float(y[1]-y[0]),xi_max=a.xi_max,quadrature_order=a.quadrature_order,xkernel_min=a.xkernel_min,
        bootstrap='same source indices; no resampling or discarded replicas; mean/std(ddof=1) over axis boot',
        physical_status='unverified diagnostic; no controlled local normalization, flow/coordinate limits, nonperturbative scheme bridge, mixing or continuum/large-P limit',
        moments=records,numerical_checks=checks)
    qm,qe=meanerr(qs,1); fm,fe=meanerr(matched,1)
    np.savez_compressed(a.out/'unpolarized_quasi_matching.npz',x=x,quasi_input_x=y,pabs_values=pabs,pz_GeV=physical_pz(pabs,afm),
        z_values=z,source_m0=central,source_m0_boot=raw,local_normalized_boot=norm,bootstrap_indices=indices,
        quasi_ms_normalized_boot=q0,quasi_boot=qs,matched_boot=matched,scheme_correction_boot=sc,lamet_correction_boot=kc,
        quasi_mean=qm,quasi_error=qe,matched_mean=fm,matched_error=fe,matched_original_central=original,computed_x_mask=mask,
        zmax_values=cuts,zmax_matched_boot=cutoff,axes_samples=['pabs','boot','x'],axes_coordinate=['boot','pabs','z'],nconf=ncfg,nboot=nb,contract_json=json.dumps(meta))
    draw(a.out,x,pabs,qs,matched,cuts,cutoff,a.scheme)
    (a.out/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(output=str(a.out),momenta=records,checks=checks),indent=2))


if __name__=='__main__': main()
