#!/usr/bin/env python3
"""Diagnostic comparison and small-flow extrapolation after component matching.

This consumes the existing ratio_v3 products (N231), which contain the
component-resolved jackknife means/errors.  Since those products do not retain
the individual jackknife replicas, the flow-time fit uses uncorrelated
jackknife errors and is explicitly marked diagnostic.  A production result
should rerun the ratio/bootstrap stage with ``calc_ratio_matched.py``.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from .gluon_flow_to_quasi import coefficients, GEV_FM
except ImportError:
    from gluon_flow_to_quasi import coefficients, GEV_FM

TAUS = (0.1,0.2,0.3,0.4,0.5,0.6,1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8)
LABELS = ("unpolarized", "helicity_TH", "helicity_TH_plus_SH")

def tag(t): return f"tau{t:.3f}".replace(".", "p")

def load_points(root, taus, tsep, a_fm, mu, alpha_s):
    before=[]; after=[]; eb=[]; ea=[]; used=[]; pabs=zvals=None
    for tau in taus:
        p=Path(root)/f"ratio_v3_{tag(tau)}_N231.npz"
        if not p.exists(): continue
        with np.load(p,allow_pickle=False) as d:
            r=d["ratio_jackknife_mean"]
            er=d["ratio_jackknife_error_real"]; ei=d["ratio_jackknife_error_imag"]
            ti=list(d["tsep_values"]).index(tsep); ins=min(tsep//2, r.shape[6]-1)
            # axes: channel, orientation, component, direction, z, tsep, insertion, p
            # raw combination is component 0; independent pieces are 1 and 2.
            ub=r[0,2,0,0,:,ti,ins,:].real; ue=er[0,2,0,0,:,ti,ins,:]
            # The ratio-v3 combined helicity component is the stored
            # (T_H+S_H)_odd.  The independently retained Mtiti component is
            # the T_H diagnostic; keep both instead of duplicating combined.
            hth=r[1,3,1,0,:,ti,ins,:].imag; ehth=ei[1,3,1,0,:,ti,ins,:]
            hplus=r[1,3,0,0,:,ti,ins,:].imag; ehplus=ei[1,3,0,0,:,ti,ins,:]
            mt_u=r[0,2,1,0,:,ti,ins,:].real; emt_u=er[0,2,1,0,:,ti,ins,:]
            ms_u=r[0,2,2,0,:,ti,ins,:].real; ems_u=er[0,2,2,0,:,ti,ins,:]
            mt_h=r[1,3,1,0,:,ti,ins,:].imag; emt_h=ei[1,3,1,0,:,ti,ins,:]
            ms_h=r[1,3,2,0,:,ti,ins,:].imag; ems_h=ei[1,3,2,0,:,ti,ins,:]
            cpar, cperp, dm, _=coefficients(tau,a_fm,mu,alpha_s)
            z=np.asarray(d["z_values"],float)*a_fm*GEV_FM
            line=np.exp(-dm*z)[:,None]
            fu=1/(cperp*cperp); fh=1/(cpar*cperp)
            um=line*fu*ub; hm=line*fh*hth; hpm=line*fh*hplus
            ume=line*fu*ue; hme=line*fh*ehth; hpme=line*fh*ehplus
            before.append(np.stack([ub,hth,hplus],axis=0))
            eb.append(np.stack([ue,ehth,ehplus],axis=0))
            after.append(np.stack([um,hm,hpm],axis=0)); ea.append(np.stack([ume,hme,hpme],axis=0))
            used.append(tau); pabs=np.asarray(d["pabs_list"]); zvals=np.asarray(d["z_values"])
    if not used: raise FileNotFoundError(f"no ratio_v3 products under {root}")
    return np.asarray(used),np.asarray(before),np.asarray(after),np.asarray(eb),np.asarray(ea),pabs,zvals

def aic_extrap(tau,y,e):
    n,nc,nz,np_=y.shape; windows=[(i,j) for i in range(n) for j in range(i+2,n)]
    m=np.full((len(windows),nc,nz,np_),np.nan); a=np.full_like(m,np.nan)
    for iw,(i,j) in enumerate(windows):
      x=tau[i:j+1]; X=np.c_[np.ones(len(x)),x]
      for c in range(nc):
       for z in range(nz):
        for p in range(np_):
         ok=np.isfinite(y[i:j+1,c,z,p])&np.isfinite(e[i:j+1,c,z,p])&(e[i:j+1,c,z,p]>0)
         if ok.sum()<3: continue
         xx=X[ok]; yy=y[i:j+1,c,z,p][ok]; ss=e[i:j+1,c,z,p][ok]
         W=np.diag(1/ss**2); b=np.linalg.solve(xx.T@W@xx,xx.T@W@yy)
         m[iw,c,z,p]=b[0]; a[iw,c,z,p]=np.sum(((yy-xx@b)/ss)**2)+4
    w=np.zeros_like(a); good=np.isfinite(a)
    amin=np.full(a.shape[1:],np.nan); anygood=np.any(good,axis=0)
    if np.any(anygood): amin[anygood]=np.min(np.where(good,a,np.inf),axis=0)[anygood]
    for iw in range(len(windows)):
      q=good[iw]; wi=w[iw]; ai=a[iw]; wi[q]=np.exp(-.5*(ai[q]-amin[q])); w[iw]=wi
    den=np.nansum(w,axis=0); w=np.divide(w,den,where=den>0,out=np.zeros_like(w)); m0=np.nansum(w*m,axis=0)
    return m0,w,np.asarray(windows)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--ratio-root',type=Path,default=Path('/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/results_v3')); p.add_argument('--out',type=Path,default=Path('matching/diagnostic_v1')); p.add_argument('--tsep',type=int,default=8); p.add_argument('--tau-min',type=float,default=0.1); p.add_argument('--tau-max',type=float,default=3.8); p.add_argument('--mu',type=float,default=2.0); p.add_argument('--alpha-s',type=float,default=.25); p.add_argument('--a-fm',type=float,default=.0775); a=p.parse_args()
 taus=np.asarray([t for t in TAUS if a.tau_min<=t<=a.tau_max]); used,bef,aft,eb,ea,pabs,z=load_points(a.ratio_root,taus,a.tsep,a.a_fm,a.mu,a.alpha_s); a.out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.out/'flow_matching_points.npz',schema='gluon_flow_matching_diagnostic_v1',flow_times=used,before=bef,before_error=eb,after=aft,after_error=ea,pabs_values=pabs,z_values=z,tsep=a.tsep,mu_GeV=a.mu,alpha_s=a.alpha_s)
 for name,vals,errs,sub in [('before',bef,eb,'matching_before'),('after',aft,ea,'matching_after')]:
  for c,label in enumerate(LABELS):
   fig,axs=plt.subplots(1,len(pabs),figsize=(13,3.5),sharey=True); axs=np.atleast_1d(axs)
   for ip,ax in enumerate(axs):
    for iz,zz in enumerate(z): ax.errorbar(used,vals[:,c,iz,ip],yerr=errs[:,c,iz,ip],fmt='o',ms=2.5,alpha=.65)
    ax.set_title(f'$P_z={pabs[ip]}$'); ax.set_xlabel(r'$t/a^2$'); ax.grid(alpha=.2)
   axs[0].set_ylabel(label+' (diagnostic)'); fig.tight_layout(); fig.savefig(a.out/f'{sub}_{label}.png',dpi=180); plt.close(fig)
  m0,w,win=aic_extrap(used,vals,errs); np.savez_compressed(a.out/f'{sub}_aic_extrap.npz',m0=m0,window_weights=w,windows=win,pabs_values=pabs,z_values=z,flow_times=used)
 # Direct before/after extrapolated comparison versus zPz.
 mb,wb,_=aic_extrap(used,bef,eb); ma,wa,_=aic_extrap(used,aft,ea)
 fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=False)
 for c,ax in enumerate(axs):
  for ip,pz in enumerate(pabs):
   x=z*pz*2*np.pi/32.0
   ax.plot(x,mb[c,:,ip],'-o',ms=3,label=f'before $P_z={pz}$'); ax.plot(x,ma[c,:,ip],'--s',ms=3,label=f'after $P_z={pz}$')
  ax.set_title(LABELS[c]); ax.set_xlabel(r'$\nu=zP_z=2\pi(z/a)n_z/L_z$'); ax.grid(alpha=.2); ax.legend(fontsize=7,frameon=False)
 axs[0].set_ylabel('AIC $t\to0$ diagnostic matrix element'); fig.tight_layout(); fig.savefig(a.out/'before_after_aic_extrap_vs_zpz.png',dpi=220); fig.savefig(a.out/'before_after_aic_extrap_vs_zpz.pdf'); plt.close(fig)
 # Fixed-flow comparison at the closest available tau to 0.5.
 ik=int(np.argmin(abs(used-0.5))); fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=False)
 for c,ax in enumerate(axs):
  for ip,pz in enumerate(pabs):
   x=z*pz*2*np.pi/32.0
   ax.errorbar(x,bef[ik,c,:,ip],yerr=eb[ik,c,:,ip],fmt='o',ms=3,capsize=2,label=f'GF bare $P_z={pz}$')
   ax.errorbar(x,aft[ik,c,:,ip],yerr=ea[ik,c,:,ip],fmt='s',ms=3,capsize=2,label=f'flow matched $P_z={pz}$')
  ax.set_title(LABELS[c]+f', $t/a^2={used[ik]:.1f}$'); ax.set_xlabel('$zP_z$'); ax.grid(alpha=.2); ax.legend(fontsize=7,frameon=False)
 axs[0].set_ylabel('matrix element (diagnostic)'); fig.tight_layout(); fig.savefig(a.out/'before_after_tau0p500_vs_zpz.png',dpi=220); fig.savefig(a.out/'before_after_tau0p500_vs_zpz.pdf'); plt.close(fig)
 (a.out/'manifest.json').write_text(json.dumps({'schema':'gluon_flow_matching_diagnostic_v1','flow_times_used':used.tolist(),'tsep':a.tsep,'tau_fit_min':a.tau_min,'tau_fit_max':a.tau_max,'mu_GeV':a.mu,'alpha_s':a.alpha_s,'note':'ratio_v3 jackknife means/errors; no replica covariance; diagnostic only'},indent=2)+'\n')
 print(a.out/'before_after_aic_extrap_vs_zpz.pdf')
if __name__=='__main__': main()
