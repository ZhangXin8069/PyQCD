"""Model-independent adjacent-flow secants, paired JK errors, and readable overview."""
import csv,hashlib,json,os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error(j):return np.sqrt((len(j)-1)/len(j)*np.sum((j-j.mean(axis=0))**2,axis=0))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def save(fig,out,name):
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
def main():
 assert os.environ.get('SLURM_JOB_ID')
 lock=read(ROOT/'local_slopes_lock.json');assert sha(ROOT/'local_slopes.py')==lock['script_sha256']
 assert read(ROOT/'validation.json')['status']=='PASS'
 config=read(ROOT/'config.json');files=[];summary=[]
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#333333']
 for ens in config['ensembles']:
  out=ROOT/ens['name'];figdir=out/'figures';rows=[];reps=['fundamental','adjoint']
  with np.load(out/'paired_means_and_jackknife.npz',allow_pickle=False) as z:
   y,j,er,x=z['log_mean'],z['log_jk'],z['log_error'],z['x'];kinds=json.loads(str(z['kinds_json']));taus=z['taus']
  results=read(out/'fit_results.json');fits=results['flow_slopes']
  xm=(x[:-1]+x[1:])/2;dx=np.diff(x)
  sec=np.diff(y,axis=0)/dx[:,None,None];sj=np.diff(j,axis=1)/dx[None,:,None,None];se=error(sj)
  for ri,rep in enumerate(reps):
   for ki,g in enumerate(kinds):
    L=g['n'] if g['kind']=='line' else 2*(g['n']+g['T'])
    for fi in range(3):
     valid=bool(np.isfinite(sec[fi,ri,ki]) and np.isfinite(se[fi,ri,ki]))
     rows.append(dict(ensemble=ens['name'],Ncfg=ens['Ncfg'],representation=rep,**g,
      tau_left=float(taus[fi]),tau_right=float(taus[fi+1]),x_mid=float(xm[fi]),valid=valid,
      local_b=float(sec[fi,ri,ki]) if valid else None,local_b_error=float(se[fi,ri,ki]) if valid else None,
      local_d_path=float(-np.sqrt(8)*sec[fi,ri,ki]/L) if valid else None,
      local_d_path_error=float(np.sqrt(8)*se[fi,ri,ki]/L) if valid else None))
  with (out/'local_flow_slopes.csv').open('w',newline='') as stream:
   writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
  np.savez_compressed(out/'local_flow_slope_jackknife.npz',local_b=sec,local_b_jk=sj,local_b_error=se,
      x_mid=xm,taus=taus,conf_ids=[v['conf_id'] for v in ens['items']])
  fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
  for ri,rep in enumerate(reps):
   for col,kind in enumerate(['line','loop']):
    ax=axes[ri,col];t=0 if kind=='line' else ens['main_T']
    for n,color in zip(config['plot_lengths'],colors):
     ki=kinds.index(dict(kind=kind,T=t,n=n));v=np.isfinite(sec[:,ri,ki])&np.isfinite(se[:,ri,ki]);L=n if kind=='line' else 2*(n+t)
     ax.errorbar(xm[v],-np.sqrt(8)*sec[v,ri,ki]/L,yerr=np.sqrt(8)*se[v,ri,ki]/L,fmt='o-',ms=4,color=color,label=f'n={n}')
    ax.set_title(f'{rep.capitalize()} | '+('Landau line' if kind=='line' else f'loop, T/a={t}'))
    ax.set_ylabel(r'$d_{\rm path}^{\rm local}=-\sqrt{8}\,\Delta\ln\langle W\rangle/(L_{\rm path}\Delta x)$');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3)
    if ri==1:ax.set_xlabel(r'$x_{\rm mid}$, $x=1/\sqrt{\tau}$')
  fig.suptitle(f"{ens['name']} | adjacent-flow slopes, no straight-line model | paired Ncfg={ens['Ncfg']}")
  fig.tight_layout(rect=[0,0,1,.95]);save(fig,figdir,'06_local_flow_slopes')
  fig,axes=plt.subplots(2,2,figsize=(12,8.5),sharex=True)
  for ri,rep in enumerate(reps):
   for col,kind in enumerate(['line','loop']):
    ax=axes[ri,col];t=0 if kind=='line' else ens['main_T'];grid=np.linspace(min(x)-.01,max(x)+.01,150)
    for n,color in zip(config['plot_lengths'],colors):
     ki=kinds.index(dict(kind=kind,T=t,n=n));r=next(v for v in fits if v['representation']==rep and v['kind']==kind and v['T']==t and v['n']==n and v['flow_window']=='all4');v=np.isfinite(y[:,ri,ki])&np.isfinite(er[:,ri,ki])
     ax.errorbar(x[v],y[v,ri,ki],yerr=er[v,ri,ki],fmt='o',ms=4,color=color,label=f'n={n}')
     if r['b'] is not None:ax.plot(grid,r['b']*grid+r['intercept'],color=color,lw=1,ls='--' if r['Q']<.01 else '-')
    ax.set_title(f'{rep.capitalize()} | '+('Landau Wilson line' if kind=='line' else f'Wilson loop, T/a={t}'))
    ax.set_ylabel(r'$\ln\langle\mathrm{Re}\,W\rangle$');ax.grid(alpha=.2)
    ax.legend(fontsize=8,ncol=3,loc='lower left',framealpha=.95)
    ax.text(.98,.98,'Dashed fits: Q < 0.01',transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    if ri==1:ax.set_xlabel(r'$x=1/\sqrt{\tau}$, $\tau=t_f/a^2$')
  fig.suptitle(f"{ens['name']} | paired Ncfg={ens['Ncfg']} | four matched flow points | log of ensemble mean")
  fig.tight_layout(rect=[0,0,1,.96]);save(fig,figdir,'07_lnW_overview')
  # Mean-value theorem labels are interval secants, not point derivatives at the midpoint.
  valid=int(np.sum(np.isfinite(sec)&np.isfinite(se)))
  summary.append(dict(ensemble=ens['name'],valid_secants=valid,total_secants=int(sec.size),
    interpretation='interval secants plotted at x midpoint for display; no claim of exact midpoint derivative'))
  files.extend([out/'local_flow_slopes.csv',out/'local_flow_slope_jackknife.npz'])
  files.extend(figdir/(name+suffix) for name in ['06_local_flow_slopes','07_lnW_overview'] for suffix in ['.pdf','.png'])
 write(ROOT/'local_slopes_complete.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],checked_at=datetime.now().astimezone().isoformat(),
  ensembles=summary,artifacts={str(p.relative_to(ROOT)):sha(p) for p in files}))
 print('LOCAL_SLOPES_DONE',summary,flush=True)
if __name__=='__main__':main()
