"""CPU-only length slopes and a dependence. Outputs are scheme-dependent diagnostics."""
import json
from pathlib import Path
import numpy as np
import core as c
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis'

def jk_error(r):return np.sqrt((len(r)-1)*np.mean(abs(r-r.mean(axis=0))**2,axis=0))
def jackknife(x,block):
 assert len(x)%block==0
 grouped=x.reshape(len(x)//block,block,*x.shape[1:]).mean(axis=1)
 return (grouped.sum(axis=0)-grouped)/(len(grouped)-1)

def gls(y,rep,x):
 # Fixed jackknife covariance and correlation SVD map reused in every replica.
 n=len(rep);d=rep-rep.mean(axis=0);cov=(n-1)/n*d.T@d
 scale=np.sqrt(np.diag(cov));assert np.all(scale>0)
 corr=cov/scale[:,None]/scale[None,:];e,v=np.linalg.eigh(corr)
 sample_rank=int(np.sum(e>e[-1]*1e-12)); keep=e>e[-1]*1e-8;effective=int(keep.sum())
 design=np.column_stack([np.ones(len(x)),-x])
 whiten=(v[:,keep]/np.sqrt(e[keep])).T/scale[None,:]
 wx=whiten@design;model_rank=int(np.linalg.matrix_rank(wx));dof=effective-2
 assert model_rank==2 and dof>0, ('unidentifiable',sample_rank,effective,model_rank,dof)
 mapping=np.linalg.pinv(wx)@whiten
 p=mapping@y;pr=rep@mapping.T;residual=whiten@(y-design@p)
 return p,pr,dict(sample_rank=sample_rank,effective_rank=effective,model_rank=model_rank,dof=dof,chi2=float(residual@residual),correlation_eigenvalues=e.tolist(),svdcut=1e-8)

def analyze(audit):
 assert audit['status']=='PASS' and audit['completed']==300
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from scipy.stats import chi2
 req=json.loads((ROOT/'request.json').read_text()); fits=[]; figs,axes=plt.subplots(2,3,figsize=(15,8))
 for col,(name,ens) in enumerate(req['ensembles'].items()):
  entries=sorted([x for x in audit['complete'] if x['ensemble']==name],key=lambda x:x['task'])
  arrays={r:[] for r in req['representations']}
  for entry in entries:
   with np.load(Path(entry['run'])/'final_converged.npz',allow_pickle=False) as d:
    for r in arrays:arrays[r].append(d[r].copy())
  raw={r:np.stack(x) for r,x in arrays.items()};nmax=ens['line_nmax'];a=ens['a_fm'];length=np.arange(nmax+1);z=length*a
  saved=dict(conf_ids=np.array([x['conf_id'] for x in entries]),lengths_a=length,z_fm=z,**raw)
  for ri,(r,all_directions) in enumerate(raw.items()):
   # Reverse paths conjugate exactly; this is a declared real projection, not abs(W).
   x=all_directions.real.mean(axis=(1,2)); mean=x.mean(axis=0)
   saved[r+'_real_mean']=mean
   for block in [1,2,5]:
    rep=jackknife(x,block);nb=len(rep)
    saved[r+'_jk_block%d'%block]=rep
    valid=(mean>0)&np.all(rep>0,axis=0)
    log=np.full_like(mean,np.nan);log_rep=np.full_like(rep,np.nan)
    log[valid]=np.log(mean[valid]);log_rep[:,valid]=np.log(rep[:,valid])
    slope=-(log[1:]-log[:-1])/a;sr=-(log_rep[:,1:]-log_rep[:,:-1])/a
    saved[r+'_mu_eff_block%d'%block]=slope;saved[r+'_mu_eff_error_block%d'%block]=jk_error(sr)
    if block==1:
     axes[0,col].errorbar(z[valid],log[valid],yerr=jk_error(log_rep[:,valid]),fmt='o-',label=r)
     ok=np.isfinite(slope)&np.isfinite(sr).all(axis=0)
     axes[1,col].errorbar((z[1:]+z[:-1])[ok]/2,slope[ok],yerr=jk_error(sr[:,ok]),fmt='o-',label=r)
    for lo,hi in [(.2,.6),(.3,.7),(.4,.8)]:
     idx=np.flatnonzero((z>=lo-1e-12)&(z<=hi+1e-12)&(length>=2))
     row=dict(ensemble=name,representation=r,a_fm=a,block_size=block,Ncfg=100,Nblocks=nb,z_window_fm=[lo,hi],lengths_a=length[idx].tolist(),status='unvalidated')
     try:
      assert len(idx)>=3 and np.all(valid[idx]), 'Too few lengths or nonpositive central/jackknife mean'
      p,pr,diag=gls(log[idx],log_rep[:,idx],length[idx])
      err=jk_error(pr);row.update(diag,status='FIT_DIAGNOSTIC',intercept=float(p[0]),a_mu_effective=float(p[1]),a_mu_error=float(err[1]),mu_fm_inverse=float(p[1]/a),mu_error_fm_inverse=float(err[1]/a),Q=float(chi2.sf(diag['chi2'],diag['dof'])))
      key='%s_b%d_z%03d_%03d'%(r,block,round(100*lo),round(100*hi));saved[key+'_parameters_jk']=pr
     except (AssertionError,np.linalg.LinAlgError) as e:row.update(status='statistically_unidentifiable',reason=str(e))
     fits.append(row)
  np.savez_compressed(OUT/(name+'_statistics.npz'),**saved)
  axes[0,col].set(title=name,xlabel='z [fm]',ylabel='ln Re <W>');axes[1,col].set(xlabel='z midpoint [fm]',ylabel='effective line mass [1/fm]')
  for ax in axes[:,col]:ax.grid(alpha=.25);ax.legend()
 figs.tight_layout();figs.savefig(OUT/'wilson_line_length_slopes.pdf');figs.savefig(OUT/'wilson_line_length_slopes.png',dpi=160);plt.close(figs)
 # Three independently generated ensembles: diagonal covariance across a.
 comparisons=[]
 for r in req['representations']:
  for block in [1,2,5]:
   for window in [[.2,.6],[.3,.7],[.4,.8]]:
    rows=[x for x in fits if x['representation']==r and x['block_size']==block and x['z_window_fm']==window]
    entry=dict(representation=r,block_size=block,z_window_fm=window,model='mu_eff(a) = c/a + m0',interpretation='diagnostic only; finite gauge/smearing/length terms and running not isolated',status='unvalidated')
    good=[x for x in rows if x['status']=='FIT_DIAGNOSTIC']
    if len(good)==3 and all(x['mu_error_fm_inverse']>0 for x in good):
     y=np.array([x['mu_fm_inverse'] for x in good]);sig=np.array([x['mu_error_fm_inverse'] for x in good]);xx=np.array([1/x['a_fm'] for x in good]);design=np.column_stack([xx,np.ones(3)]);w=design/sig[:,None];cov=np.linalg.inv(w.T@w);p=cov@w.T@(y/sig);chi=float(np.sum(((y-design@p)/sig)**2))
     entry.update(status='SCHEME_DEPENDENT_DIAGNOSTIC',c=float(p[0]),c_stat_error=float(np.sqrt(cov[0,0])),m0_fm_inverse=float(p[1]),parameter_covariance=cov.tolist(),chi2=chi,dof=1,Q=float(chi2.sf(chi,1)))
    else:entry['status']='statistically_unidentifiable'
    comparisons.append(entry)
 c.write_json(OUT/'length_window_fits.json',fits);c.write_json(OUT/'spacing_diagnostics.json',comparisons)
 c.write_json(OUT/'coefficient_status.json',dict(status='MEASURED_AND_DIAGNOSTIC_FITS',Ncfg_per_ensemble=100,gauge='4D Landau, original HYP start only',flow_time_scan=False,gradient_flow_double_ratio_used=False,universal_linear_divergence_isolated=False,final_d_approved=False,scale_uncertainty_included=False,hyp_alpha_parameters_confirmed=False,blocking=[1,2,5],caveat='Block stability is a diagnostic, not a proof of negligible autocorrelation. Same 5-step HYP label does not certify equal alpha parameters. Fits cannot determine a unique universal UV coefficient without further scheme and finite-part controls.'))
 c.write_json(OUT/'complete.json',dict(status='PASS_DIAGNOSTIC_ANALYSIS',source_lock_sha256=c.sha(ROOT/'source_lock.json'),fits=len(fits),spacing_comparisons=len(comparisons)))
