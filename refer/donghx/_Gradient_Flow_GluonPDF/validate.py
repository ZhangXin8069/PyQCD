"""Independent artifact/statistical checks after analysis finishes."""
import hashlib,json,os
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def err(v):return np.sqrt((len(v)-1)/len(v)*np.sum((v-v.mean(axis=0))**2,axis=0))
def main():
 assert os.environ.get('SLURM_JOB_ID')
 lock=read(ROOT/'validation_lock.json')
 assert sha(ROOT/'validate.py')==lock['validate_sha256']
 assert sha(ROOT/'source_lock.json')==lock['analysis_source_lock_sha256']
 complete=read(ROOT/'complete.json')
 assert complete['status']=='PASS_NUMERICAL_ANALYSIS'
 for n,h in complete['artifacts'].items():assert sha(ROOT/n)==h,n
 report=read(ROOT/'report.json');config=read(ROOT/'config.json')
 checked=0;differences=[];rows=[]
 for ens in config['ensembles']:
  out=ROOT/ens['name'];results=read(out/'fit_results.json');fits=results['flow_slopes'];coeff=results['length_coefficients']
  with np.load(out/'paired_means_and_jackknife.npz',allow_pickle=False) as z:
   mean,y,jk=z['mean'],z['log_mean'],z['log_jk'];kinds=json.loads(str(z['kinds_json']));x=z['x'];confs=z['conf_ids']
   assert confs.tolist()==[v['conf_id'] for v in ens['items']]
   mask=np.isfinite(y);assert np.all(mean[mask]>0);assert np.allclose(y[mask],np.log(mean[mask]),rtol=0,atol=1e-14)
   assert len(jk)==ens['Ncfg'] and len(x)==4
   with np.load(out/'fit_jackknife_samples.npz',allow_pickle=False) as samples:
    for r in fits:
     if r['data_identifiability']!='identifiable':continue
     ri=['fundamental','adjoint'].index(r['representation']);ki=kinds.index({k:r[k] for k in ['kind','T','n']});ix=r['flow_indices'];win=r['flow_window']
     p=samples[f'b_{ri}_{ki}_{win}'];assert abs(err(p)-r['b_error'])<1e-9*max(1,r['b_error'])
     j=jk[:,ix,ri,ki];d=j-j.mean(axis=0);cov=(len(j)-1)/len(j)*d.T@d;std=np.sqrt(np.diag(cov));cor=cov/np.outer(std,std)
     ev,U=np.linalg.eigh(cor);use=ev>ev[-1]*1e-10
     precision=((U[:,use]/ev[use])@U[:,use].T)/np.outer(std,std)
     X=np.column_stack([x[ix],np.ones(len(ix))]);q=np.linalg.solve(X.T@precision@X,X.T@precision@y[ix,ri,ki])
     assert abs(q[0]-r['b'])<2e-5*max(1,abs(r['b'])),(ens['name'],r,q)
     length=r['n'] if r['kind']=='line' else 2*(r['n']+r['T'])
     assert abs(-np.sqrt(8)*r['b']/length-r['d_per_path_length'])<1e-13
     checked+=1
    for r in coeff:
     if r['status']!='identifiable':continue
     ri=['fundamental','adjoint'].index(r['representation']);p=samples[f"kappa_{ri}_{r['kind']}_T{r['T']}_n{r['length_min']}to{r['length_max']}_{r['flow_window']}"]
     assert abs(err(p)-r['kappa_error'])<1e-9*max(1,r['kappa_error'])
     assert abs(r['d']+np.sqrt(8)*r['kappa'])<1e-13
     assert abs(r['d_error']-np.sqrt(8)*r['kappa_error'])<1e-13
    for r in results['paired_differences']:
     ri=['fundamental','adjoint'].index(r['representation']);lo,hi,win=r['length_min'],r['length_max'],r['flow_window']
     delta=samples[f'kappa_{ri}_loop_T{r["T"]}_n{lo}to{hi}_{win}']-samples[f'kappa_{ri}_line_T0_n{lo}to{hi}_{win}']
     assert abs(err(delta)-r['paired_error'])<1e-9*max(1,r['paired_error'])
   # Main-geometry, fixed-n raw slope examples for a compact report.
   for rep in ['fundamental','adjoint']:
    for kind,T in [('line',0),('loop',ens['main_T'])]:
     r=next(v for v in fits if v['representation']==rep and v['kind']==kind and v['T']==T and v['n']==6 and v['flow_window']=='all4')
     rows.append(dict(ensemble=ens['name'],representation=rep,kind=kind,T=T,n=6,b=r.get('b'),b_error=r.get('b_error'),Q=r.get('Q')))
  differences.extend([r for r in results['paired_differences'] if r['T']==ens['main_T'] and r['length_min']==4 and r['length_max']==8 and r['flow_window']=='all4'])
 validation=dict(status='PASS',checked_at=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
   checked_usable_flow_fits=checked,checks=['all published artifact SHA256','paired conf order','log of positive ensemble mean',
    'independent precision-matrix GLS reconstruction','saved JK slope errors','sqrt8 and loop geometry factors','paired difference errors'],
   raw_slope_examples=rows,coefficient_examples=differences,analysis_report_sha256=sha(ROOT/'report.json'))
 (ROOT/'validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n')
 text='# Wilson loop / Wilson line 斜率比较结果\n\n'
 text+='使用L32×64的1698个、L32×96的869个配对完整组态；两表示分别分析。\n'
 text+='横轴为无量纲1/sqrt(tau)，纵轴为log(Re ensemble mean)。每个几何独立相关拟合四个共同flow点，误差为配对delete-one jackknife。\n\n'
 text+='## 固定n=6的原始flow斜率示例\n\n|ensemble|表示|对象|T/a|b ± JK误差|Q|\n|---|---|---|---:|---:|---:|\n'
 for r in rows:
  text+=f"|{r['ensemble']}|{r['representation']}|{r['kind']}|{r['T']}|{r['b']:.6f} ± {r['b_error']:.6f}|{r['Q']:.3g}|\n" if r['b'] is not None else f"|{r['ensemble']}|{r['representation']}|{r['kind']}|{r['T']}|不可辨识|—|\n"
 text+='\nLoop的b包括两条空间边与两条时间边，不与单线b直接相等。\n\n'
 text+='## n=4..8长度窗的几何修正系数示例\n\n'
 text+='拟合b_line=kappa*n+c，b_loop=2*kappa*n+c_T；kappa=−d/sqrt(8)。下面仅是有限flow/有限长度诊断，并非最终渐近系数。\n\n'
 text+='|ensemble|表示|loop T/a|kappa_line|kappa_loop|差值（配对误差）|长度拟合Q(line/loop)|\n|---|---|---:|---:|---:|---:|---:|\n'
 for r in differences:
  text+=f"|{r['ensemble']}|{r['representation']}|{r['T']}|{r['kappa_line']:.6f} ± {r['kappa_line_error']:.6f}|{r['kappa_loop']:.6f} ± {r['kappa_loop_error']:.6f}|{r['delta_kappa_loop_minus_line']:.6f} ± {r['paired_error']:.6f}|{r['line_Q']:.3g}/{r['loop_Q']:.3g}|\n"
 text+='\n完整窗口和其他固定T均见CSV，未按Q筛选系数。Q<0.01只表示当前简单直线模型有张力；图中flow直线用虚线标识。\n'
 text+='原始loop的epsilon与line不同；小圈流化历史对照见各ensemble/small_loop_flow_history_comparison.csv。本轮未控制自相关、epsilon→0、tau→0、a→0和T→∞极限，不能据此宣称最终系数相同或不同。\n'
 (ROOT/'RESULTS.md').write_text(text)
 print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__':main()
