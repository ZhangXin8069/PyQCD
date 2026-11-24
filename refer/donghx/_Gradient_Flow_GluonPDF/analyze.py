"""CPU-only full-ensemble flowed Wilson-line plots and finite-flow slope diagnostics."""
import io,json,os,traceback
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statistics_helpers import sha,read,write,table,log_stats,jk_error,gls,tests
ROOT=Path(__file__).resolve().parent
REP=['fundamental','adjoint'];LABEL=['Fundamental','Adjoint']


def source_check():
 for name,digest in read(ROOT/'source_lock.json')['sha256'].items():assert sha(ROOT/name)==digest,name

def load_ensemble(e,out):
 assert sha(e['merged_audit'])==e['merged_audit_sha256']
 raw=np.empty((e['Ncfg'],4,2,13));evidence=[];roots={};maxq=0.;mixed=Counter()
 for ci,item in enumerate(e['items']):
  run=Path(item['run']);parent=run.parents[2]
  if parent not in roots:
   assert sha(parent/'source_lock.json')==item['source_lock_sha256']
   for name,digest in read(parent/'source_lock.json')['sha256'].items():assert sha(parent/name)==digest,(parent,name)
   req=read(parent/'request.json');ens=req['ensemble']
   assert np.allclose(ens['taus'],e['taus'],rtol=0,atol=1e-14) and ens['a_fm']==e['a_fm']
   assert req['selection']==dict(starts=[0],name='original',random_starts=False,best_of=False,warm_start_between_flows=False)
   roots[parent]=req
  assert sha(parent/'source_lock.json')==item['source_lock_sha256']
  assert sha(run/'result.json')==item['result_sha256']
  mark=read(run/'complete.json');result=read(run/'result.json')
  assert mark['result_sha256']==item['result_sha256'] and mark['source_lock_sha256']==item['source_lock_sha256']
  assert result['status']=='PASS_CUDA_START0_MEASUREMENT' and str(result['conf_id'])==str(item['conf_id']) and len(result['results'])==4
  mixed[str(parent.name)]+=1
  for fi,row in enumerate(result['results']):
   assert row['status']=='PASS' and row['start_index']==0 and abs(row['tau']-e['taus'][fi])<1e-14
   flow=run/('flow%d'%fi);meta=read(flow/'final_converged.json');start=read(flow/'start_verified.json')
   assert meta==row['measurement'] and meta['theta']<=1e-12
   assert start['start_index']==0 and not start['random_transform_applied']
   assert start['start_data_sha256']==start['unfixed_flow_sha256']
   payload=(flow/'final_converged.npz').read_bytes();digest=__import__('hashlib').sha256(payload).hexdigest()
   assert digest==meta['npz_sha256'];maxq=max(maxq,meta['theta'])
   with np.load(io.BytesIO(payload),allow_pickle=False) as saved:
    f,a=saved['fundamental'],saved['adjoint'];assert f.shape==a.shape and f.shape[:2]==(3,2) and f.shape[2]>=13
    assert f.dtype==np.complex128 and a.dtype==np.float64 and np.isfinite(f).all() and np.isfinite(a).all()
    assert np.max(abs(f[:,1]-f[:,0].conj()))<2e-10 and np.max(abs(a[:,1]-a[:,0]))<2e-10
    for ri,v in enumerate([f,a]):raw[ci,fi,ri]=v.real.mean(axis=(0,1))[:13]
   evidence.append(dict(conf_id=item['conf_id'],flow_index=fi,path=str(flow/'final_converged.npz'),sha256=digest,theta=meta['theta']))
  if (ci+1)%200==0:print('READ',e['geometry'],ci+1,'/',e['Ncfg'],flush=True)
 assert np.max(abs(raw[:,:,:,0]-1))<1e-14
 table(out/'source_measurements.csv',evidence)
 write(out/'input_audit.json',dict(status='PASS',Ncfg=len(raw),flow_points=len(evidence),max_theta=maxq,source_counts=mixed,solver_policy_mixed=e['solver_policy_mixed']))
 return raw


def savefig(fig,path):
 fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight')


def analyze(alias,e,cfg):
 out=ROOT/alias;out.mkdir(exist_ok=False);(out/'figures').mkdir()
 raw=load_ensemble(e,out);mean,y,jy,valid=log_stats(raw);err=jk_error(jy)
 a=e['a_fm'];taus=np.array(e['taus']);tf=taus*a*a;sqrt_t=np.sqrt(tf);x=1/sqrt_t;n=np.arange(13);z=n*a
 np.savez_compressed(out/'statistics.npz',conf_ids=np.array([i['conf_id'] for i in e['items']]),raw_real_direction_sign_average=raw,taus=taus,t_f_fm2=tf,sqrt_t_f_fm=sqrt_t,inv_sqrt_t_f_fm_inverse=x,z_fm=z,lengths_a=n,mean_W=mean,ln_mean_W=y,ln_mean_W_jackknife=jy,ln_mean_W_error=err,valid=valid,representations=REP)
 points=[];fits=[];fit_lookup={};local=[];pj_saved={}
 for ri,rep in enumerate(REP):
  for length in range(1,13):
   for fi in range(4):points.append(dict(ensemble=alias,Ncfg=e['Ncfg'],representation=rep,n=length,z_fm=z[length],tau=taus[fi],t_f_fm2=tf[fi],sqrt_t_f_fm=sqrt_t[fi],inv_sqrt_t_f_fm_inverse=x[fi],mean_W=mean[fi,ri,length],lnW=y[fi,ri,length],lnW_error=err[fi,ri,length],valid=bool(valid[fi,ri,length])))
   for window,indices in cfg['windows'].items():
    ix=np.array(indices);yy=y[ix,ri,length];jj=jy[:,ix,ri,length]
    fit,pj=gls(yy,jj,np.column_stack([x[ix],np.ones(len(ix))]));row=dict(ensemble=alias,representation=rep,n=length,z_fm=z[length],window=window,indices=indices,model='ln<W> = b / sqrt(t_f) + c',b_units='fm',status=fit['status'],interpretation='finite-flow fixed-length diagnostic; not final d')
    if fit['status']=='identifiable':
     b,c=fit['params'];be,ce=fit['errors'];row.update(b_fm=b,b_error_fm=be,intercept=c,intercept_error=ce,d_path=-np.sqrt(8)*b/z[length],d_path_error=np.sqrt(8)*be/z[length],**{k:v for k,v in fit.items() if k not in ['params','errors','status']})
     pj_saved['%s_n%d_%s'%(rep,length,window)]=pj
    fits.append(row);fit_lookup[(ri,length,window)]=row
   for interval in range(3):
    dx=x[interval+1]-x[interval];slope=(y[interval+1,ri,length]-y[interval,ri,length])/dx
    sj=(jy[:,interval+1,ri,length]-jy[:,interval,ri,length])/dx
    local.append(dict(ensemble=alias,representation=rep,n=length,z_fm=z[length],interval=interval,flow_indices=[interval,interval+1],sqrt_t_f_midpoint_fm=float((sqrt_t[interval]+sqrt_t[interval+1])/2),b_local_fm=slope,b_local_error_fm=jk_error(sj),d_path_local=-np.sqrt(8)*slope/z[length],d_path_local_error=np.sqrt(8)*jk_error(sj)/z[length]))
 table(out/'lnW_points.csv',points);table(out/'flow_slopes.csv',fits);table(out/'adjacent_flow_slopes.csv',local)
 np.savez_compressed(out/'fit_jackknife.npz',**pj_saved)
 chosen=cfg['selected_lengths_a'];colors=plt.cm.viridis(np.linspace(.08,.92,len(chosen)))
 for view in ['sqrt_t_f','inv_sqrt_t_f']:
  fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
  for ri,ax in enumerate(axes):
   for color,length in zip(colors,chosen):
    xx=sqrt_t if view=='sqrt_t_f' else x;order=np.argsort(xx);ok=valid[:,ri,length]
    ax.errorbar(xx[ok],y[ok,ri,length],yerr=err[ok,ri,length],fmt='o',ms=4,capsize=2,color=color,label='n=%d, z=%.3f fm'%(length,z[length]))
    ax.plot(xx[order],y[order,ri,length],color=color,alpha=.35,lw=.8)
    f=fit_lookup[(ri,length,'all4')]
    if f['status']=='identifiable':
     gx=np.linspace(xx.min(),xx.max(),160);inv=1/gx if view=='sqrt_t_f' else gx
     ax.plot(gx,f['b_fm']*inv+f['intercept'],color=color,ls='--' if f['Q']<.01 else '-',lw=1.2,alpha=.8)
   ax.set_title(LABEL[ri]);ax.set_xlabel(r'$\sqrt{t_f}\;[\mathrm{fm}]$' if view=='sqrt_t_f' else r'$1/\sqrt{t_f}\;[\mathrm{fm}^{-1}]$');ax.set_ylabel(r'$\ln\,\mathrm{Re}\langle W_R(z,t_f)\rangle$');ax.grid(alpha=.2);ax.legend(fontsize=8)
  fig.suptitle('%s (%s), Ncfg=%d, Landau start0\nPoints: data; curves: inverse-sqrt diagnostic (dashed if Q<0.01)'%(alias,e['geometry'],e['Ncfg']),fontsize=11)
  savefig(fig,out/'figures'/('lnW_vs_'+view));plt.close(fig)
 # One fixed length per panel: expanded scale makes fit curvature/residuals visible.
 for ri,rep in enumerate(REP):
  for view in ['sqrt_t_f','inv_sqrt_t_f']:
   fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
   for ax,length in zip(axes.flat,chosen):
    xx=sqrt_t if view=='sqrt_t_f' else x;ax.errorbar(xx,y[:,ri,length],yerr=err[:,ri,length],fmt='o',color='black',capsize=3)
    f=fit_lookup[(ri,length,'all4')]
    if f['status']=='identifiable':
     gx=np.linspace(xx.min(),xx.max(),200);inv=1/gx if view=='sqrt_t_f' else gx
     ax.plot(gx,f['b_fm']*inv+f['intercept'],'--' if f['Q']<.01 else '-',color='tab:blue')
     annotation=r'$b=%.4f\pm%.4f$ fm'%(f['b_fm'],f['b_error_fm'])+'\n'+r'$\chi^2/\mathrm{dof}=%.1f/%d$'%(f['chi2'],f['dof'])+'\nQ=%.2g'%f['Q']
     ax.text(.04,.96,annotation,transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    ax.set_title('n=%d, z=%.3f fm'%(length,z[length]));ax.set_xlabel(r'$\sqrt{t_f}$ [fm]' if view=='sqrt_t_f' else r'$1/\sqrt{t_f}$ [fm$^{-1}$]');ax.set_ylabel(r'$\ln\,\mathrm{Re}\langle W\rangle$');ax.grid(alpha=.2);ax.margins(y=.3)
   fig.suptitle('%s: %s, Ncfg=%d; b is slope versus 1/sqrt(t_f)'%(alias,LABEL[ri],e['Ncfg']),fontsize=12);savefig(fig,out/'figures'/('%s_fixed_z_%s'%(rep,view)));plt.close(fig)
 # Finite difference slopes use no straight-line assumption.
 fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
 for ri,ax in enumerate(axes):
  for color,length in zip(colors,chosen):
   rows=[v for v in local if v['representation']==REP[ri] and v['n']==length]
   ax.errorbar([v['sqrt_t_f_midpoint_fm'] for v in rows],[v['d_path_local'] for v in rows],yerr=[v['d_path_local_error'] for v in rows],fmt='o-',capsize=2,color=color,label='n=%d'%length)
  ax.set(title=LABEL[ri],xlabel=r'Adjacent interval midpoint of $\sqrt{t_f}$ [fm]',ylabel=r'$d_{\rm path}^{\rm local}=-\sqrt{8}\,\Delta\ln W/(z\,\Delta(1/\sqrt{t_f}))$');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
 fig.suptitle('%s: adjacent-flow diagnostic; no final UV coefficient claimed'%alias);savefig(fig,out/'figures'/'local_flow_slopes');plt.close(fig)
 flagged=[v for v in fits if v['window']=='all4' and v['status']=='identifiable' and v['Q']<.01]
 summary=dict(ensemble=alias,Ncfg=e['Ncfg'],valid_log_points=int(valid[:,:,1:].sum()),expected_log_points=96,fit_status_counts=Counter(v['status'] for v in fits),all4_model_tension_Q_lt_001=len(flagged),all4_negative_slopes=sum(v.get('b_fm',0)<0 for v in fits if v['window']=='all4'),all4_total=24,selected_examples=[fit_lookup[(ri,6,'all4')] for ri in range(2)],final_d_approved=False)
 write(out/'summary.json',summary);print('DONE',alias,'model_tension',len(flagged),flush=True)
 return dict(e=e,mean=mean,y=y,jy=jy,err=err,valid=valid,sqrt_t=sqrt_t,x=x,z=z,lookup=fit_lookup,summary=summary)


def combined(all_results,cfg):
 for view in ['sqrt_t_f','inv_sqrt_t_f']:
  fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
  colors=plt.cm.viridis(np.linspace(.08,.92,len(cfg['selected_lengths_a'])))
  for col,(alias,d) in enumerate(all_results.items()):
   for ri in range(2):
    ax=axes[ri,col];xx=d['sqrt_t'] if view=='sqrt_t_f' else d['x']
    for color,length in zip(colors,cfg['selected_lengths_a']):
     ax.errorbar(xx,d['y'][:,ri,length],yerr=d['err'][:,ri,length],fmt='o',ms=3,capsize=2,color=color,label='n=%d (%.3f fm)'%(length,d['z'][length]))
     f=d['lookup'][(ri,length,'all4')]
     if f['status']=='identifiable':
      gx=np.linspace(xx.min(),xx.max(),150);inv=1/gx if view=='sqrt_t_f' else gx
      ax.plot(gx,f['b_fm']*inv+f['intercept'],color=color,ls='--' if f['Q']<.01 else '-',lw=1)
    ax.set_title('%s | %s | N=%d'%(alias,LABEL[ri],d['e']['Ncfg']),fontsize=10);ax.set_xlabel(r'$\sqrt{t_f}$ [fm]' if view=='sqrt_t_f' else r'$1/\sqrt{t_f}$ [fm$^{-1}$]');ax.set_ylabel(r'$\ln\,\mathrm{Re}\langle W\rangle$');ax.legend(fontsize=7);ax.grid(alpha=.2)
  fig.suptitle('Flowed Wilson lines: Landau original starts; inverse-sqrt fit dashed if Q<0.01\nLengths fixed within each panel; same n across ensembles does not imply the same physical z',fontsize=12)
  savefig(fig,ROOT/'figures'/('three_ensemble_lnW_vs_'+view));plt.close(fig)
 with PdfPages(ROOT/'figures'/'fixed_z_details.pdf') as pdf:
  for alias,d in all_results.items():
   for ri in range(2):
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,length in zip(axes.flat,cfg['selected_lengths_a']):
     ax.errorbar(d['x'],d['y'][:,ri,length],yerr=d['err'][:,ri,length],fmt='o',capsize=3,color='black')
     f=d['lookup'][(ri,length,'all4')]
     if f['status']=='identifiable':
      gx=np.linspace(d['x'].min(),d['x'].max(),100);ax.plot(gx,f['b_fm']*gx+f['intercept'],'--' if f['Q']<.01 else '-',color='tab:blue')
      ax.text(.04,.96,'b=%.5f +/- %.5f fm\nchi2/dof=%.1f/%d, Q=%.2g'%(f['b_fm'],f['b_error_fm'],f['chi2'],f['dof'],f['Q']),transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
     ax.set(title='n=%d, z=%.3f fm'%(length,d['z'][length]),xlabel=r'$1/\sqrt{t_f}$ [fm$^{-1}$]',ylabel=r'$\ln\,\mathrm{Re}\langle W\rangle$');ax.grid(alpha=.2);ax.margins(y=.3)
    fig.suptitle('%s | %s | Ncfg=%d'%(alias,LABEL[ri],d['e']['Ncfg']));pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':
 assert os.environ.get('SLURM_JOB_ID'),'Numerical work requires CPU allocation'
 try:
  source_check();tests();cfg=read(ROOT/'config.json');assert sha(cfg['input_final_audit'])==cfg['input_final_audit_sha256']
  results={alias:analyze(alias,e,cfg) for alias,e in cfg['ensembles'].items()};combined(results,cfg)
  source_check();write(ROOT/'report.json',dict(status='PASS_PLOTS_AND_FINITE_FLOW_DIAGNOSTICS',job=os.environ['SLURM_JOB_ID'],finished_at=datetime.now().astimezone().isoformat(),ensembles=[d['summary'] for d in results.values()],statistics=cfg['statistics'],final_d_approved=False,solver_policy_mixed=True,autocorrelation_controlled=False))
  artifacts={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.suffix in ['.pdf','.png','.csv','.npz']}
  write(ROOT/'complete.json',dict(status='PASS',report_sha256=sha(ROOT/'report.json'),source_lock_sha256=sha(ROOT/'source_lock.json'),artifacts_sha256=artifacts))
 except BaseException as error:
  write(ROOT/'failed.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
