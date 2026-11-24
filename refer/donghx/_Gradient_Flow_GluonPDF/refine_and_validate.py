"""CPU-only independent saved-statistics verification and readable detail plots."""
import csv,io,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statistics_helpers import read,sha,write
ROOT=Path(__file__).resolve().parent
REP=['fundamental','adjoint']

def save(fig,p):
 fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(p.with_suffix('.png'),dpi=170,bbox_inches='tight')

def main():
 assert os.environ.get('SLURM_JOB_ID');cfg=read(ROOT/'config.json');complete=read(ROOT/'complete.json')
 assert complete['status']=='PASS'
 for name,digest in complete['artifacts_sha256'].items():assert sha(ROOT/name)==digest,name
 reports=[]
 with PdfPages(ROOT/'figures'/'fixed_z_details.pdf') as pages:
  for alias,e in cfg['ensembles'].items():
   out=ROOT/alias
   with np.load(out/'statistics.npz',allow_pickle=False) as d:
    raw=d['raw_real_direction_sign_average'];N=len(raw);mean=raw.mean(axis=0);selected=[0,N//2,N-1]
    assert N==e['Ncfg']
    manual=[]
    for i in selected:manual.append(np.log(np.concatenate([raw[:i],raw[i+1:]],axis=0).mean(axis=0)))
    assert np.allclose(np.stack(manual),d['ln_mean_W_jackknife'][selected],rtol=1e-12,atol=1e-12)
    assert np.allclose(np.log(mean),d['ln_mean_W'],rtol=0,atol=1e-14)
    jk=d['ln_mean_W_jackknife'];err=np.std(jk,axis=0,ddof=0)*np.sqrt(N-1)
    assert np.allclose(err,d['ln_mean_W_error'],rtol=1e-12,atol=1e-15)
    assert np.all(d['valid']) and np.all(np.diff(d['ln_mean_W'][:,:,1:],axis=0)>0)
    sqrt=d['sqrt_t_f_fm'];xx=d['inv_sqrt_t_f_fm_inverse'];z=d['z_fm'];y=d['ln_mean_W'];sy=d['ln_mean_W_error']
    # Read original NPZ for first/middle/last configs and confirm projections.
    for ci in selected:
     for fi in range(4):
      path=Path(e['items'][ci]['run'])/('flow%d'%fi)/'final_converged.npz'
      with np.load(path,allow_pickle=False) as original:
       for ri,r in enumerate(REP):assert np.array_equal(original[r].real.mean(axis=(0,1))[:13],raw[ci,fi,ri])
    rows=list(csv.DictReader((out/'flow_slopes.csv').open()));lookup={(r['representation'],int(r['n'])):r for r in rows if r['window']=='all4'}
    # Independent direct inverse covariance GLS for one representative length.
    for ri,rep in enumerate(REP):
     length=6;replicas=jk[:,:,ri,length];cov=(N-1)*np.cov(replicas,rowvar=False,bias=True);X=np.column_stack([xx,np.ones(4)])
     eigen=np.linalg.eigvalsh(cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov))))
     r=lookup[(rep,length)]
     if eigen[0]>eigen[-1]*1e-10:
      inv=np.linalg.inv(cov);p=np.linalg.solve(X.T@inv@X,X.T@inv@y[:,ri,length])
      assert np.allclose(p,[float(r['b_fm']),float(r['intercept'])],rtol=1e-5,atol=1e-7)
    for ri,rep in enumerate(REP):
     for view in ['sqrt_t_f','inv_sqrt_t_f']:
      fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
      for ax,length in zip(axes.flat,cfg['selected_lengths_a']):
       px=sqrt if view=='sqrt_t_f' else xx;r=lookup[(rep,length)];b=float(r['b_fm']);c=float(r['intercept']);q=float(r['Q'])
       ax.errorbar(px,y[:,ri,length],yerr=sy[:,ri,length],fmt='o',color='black',capsize=3)
       gx=np.linspace(px.min(),px.max(),160);ax.plot(gx,b*(1/gx if view=='sqrt_t_f' else gx)+c,'--' if q<.01 else '-',color='tab:blue')
       label='b = %.6f +/- %.2g fm\nchi2/dof = %.1f/%s\nQ = %.2g'%(b,float(r['b_error_fm']),float(r['chi2']),r['dof'],q)
       ax.text(.04,.96,label,transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
       ax.set(title='n=%d, z=%.3f fm'%(length,z[length]),xlabel=r'$\sqrt{t_f}$ [fm]' if view=='sqrt_t_f' else r'$1/\sqrt{t_f}$ [fm$^{-1}$]',ylabel=r'$\ln\,\mathrm{Re}\langle W\rangle$');ax.grid(alpha=.2);ax.margins(y=.3)
      fig.suptitle('%s | %s | N=%d; b is slope vs 1/sqrt(t_f)'%(alias,rep,N));save(fig,out/'figures'/('%s_fixed_z_%s'%(rep,view)))
      if view=='inv_sqrt_t_f':pages.savefig(fig)
      plt.close(fig)
     # Raw residuals, errors are point errors (not covariance of fitted residual).
     fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
     for ax,length in zip(axes.flat,cfg['selected_lengths_a']):
      r=lookup[(rep,length)];res=y[:,ri,length]-float(r['b_fm'])*xx-float(r['intercept'])
      ax.errorbar(sqrt,res,yerr=sy[:,ri,length],fmt='o',color='tab:red',capsize=3);ax.axhline(0,c='black',lw=1)
      ax.set(title='n=%d, z=%.3f fm'%(length,z[length]),xlabel=r'$\sqrt{t_f}$ [fm]',ylabel=r'$\ln W - (b/\sqrt{t_f}+c)$');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.grid(alpha=.2)
     fig.suptitle('%s | %s: residuals to all-four inverse-sqrt fit\nBars show original point errors; fit uses full correlated covariance'%(alias,rep));save(fig,out/'figures'/('%s_fit_residuals'%rep));plt.close(fig)
    reports.append(dict(ensemble=alias,status='PASS',checks=['manual delete-one mean/log for 3 configs','independent jackknife std prefactor','original NPZ projections for 3conf x4flow x2rep','positive log inputs','monotonic means','independent GLS inverse-covariance cross-check where full rank']))
 write(ROOT/'independent_validation.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],ensembles=reports,visual_refinement='scientific error formatting; residual panels added'))
 artifacts={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.suffix in ['.pdf','.png','.csv','.npz']}
 write(ROOT/'delivery.json',dict(status='PASS_VALIDATED_PLOTS',job=os.environ['SLURM_JOB_ID'],report_sha256=sha(ROOT/'report.json'),validation_sha256=sha(ROOT/'independent_validation.json'),artifacts_sha256=artifacts))
 print('PASS_INDEPENDENT_VALIDATION_AND_PLOT_REFINEMENT',flush=True)
if __name__=='__main__':main()
