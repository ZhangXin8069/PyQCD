"""Independent CPU audit of frozen HYP measurements and full coverage."""
import argparse, json, traceback
from pathlib import Path
import numpy as np
import core as c
ROOT=Path(__file__).resolve().parent

def inspect(task,req,pilot=False):
 item=req['items'][task];ens=req['ensembles'][item['ensemble']];contract=req['contract']
 marks=list((ROOT/'runs'/('task%03d'%task)).glob('job*/complete.json'))
 assert len(marks)==1, ('complete count',len(marks))
 run=marks[0].parent;load=lambda name:json.loads((run/name).read_text())
 marker=load('complete.json'); result=load('result.json'); q=load('final_converged.json')
 assert marker['status']==result['status']=='PASS_HYP_LANDAU_START0'
 assert marker['task']==result['task']==task and result['conf_id']==item['conf_id'] and result['ensemble']==item['ensemble']
 assert marker['source_lock_sha256']==result['source_lock_sha256']==c.sha(ROOT/'source_lock.json')
 assert c.sha(run/'result.json')==marker['result_sha256']
 assert q==result['measurement'] and c.sha(run/'final_converged.npz')==q['npz_sha256']
 for phase in ['primary','tight']:
  assert result[phase]==load(phase+'_phase.json') and result[phase]['converged']
 assert q['theta']<=contract['theta_tight']
 assert max(q['unitarity_max'],q['determinant_max'])<contract['su3_atol']
 assert c.identity(item['gauge'])==item['identity']
 assert c.sha(item['header'])==item['header_sha256'] and c.sha(item['checksum_xml'])==item['checksum_sha256']
 provenance=load('hyp_provenance.json');assert provenance['hyp_label']==ens['hyp_label']=='4D 10times_new'
 assert provenance['ensemble']==item['ensemble'] and provenance['hyp_root']==ens['hyp_root'] and provenance['lattice_xyzt']==ens['lattice_xyzt']
 checksum=load('input_checksum.json');assert checksum['passed']
 assert checksum['expected']==checksum['actual']==[int(item['checksum'][k],16) for k in ['suma','sumb']]
 start=load('start_verified.json');assert start['input']==item and start['start_index']==0
 assert start['raw_data_sha256']==start['start_data_sha256']
 assert not start['random_transform_applied'] and not start['gradient_flow'] and not start['extra_smearing'] and not start['reunitarization']
 env=load('environment.json');assert env['host'].split('.')[0] in req['resources']['allowed_nodes']
 assert env['partition']=='na100-ins' and env['reservation']=='donghx_250' and env['pyquda']=='0.10.54'
 with np.load(run/'final_converged.npz',allow_pickle=False) as data,np.load(run/'hyp_unfixed.npz',allow_pickle=False) as raw:
  f,a=data['fundamental'],data['adjoint']; shape=(3,2,ens['line_nmax']+1)
  assert f.shape==a.shape==shape and f.dtype==np.complex128 and a.dtype==np.float64
  assert np.isfinite(f).all() and np.isfinite(a).all()
  assert np.max(abs(f[:,:,0]-1))<1e-14 and np.max(abs(a[:,:,0]-1))<1e-14
  assert np.max(abs(f[:,1]-f[:,0].conj()))<contract['line_reversal_atol']
  assert np.max(abs(a[:,1]-a[:,0]))<contract['line_reversal_atol']
  delta=float(np.max(abs(data['closed_loops']-raw['closed_loops'])))
  assert delta<contract['closed_loop_atol'] and abs(delta-q['closed_loop_max_change'])<1e-15
  assert np.array_equal(data['lengths_a'],np.arange(shape[-1]))
  assert np.array_equal(data['directions'],['x','y','z']) and np.array_equal(data['signs'],[1,-1])
 if pilot:assert load('reader_crosscheck.json')['status']=='PASS'
 if item['selection_index']==0:
  tiny=load('tiny_contract.json');assert tiny['unit_and_pure_gauge_paths']=='PASS'
  for test in tiny['algorithms'].values():assert test['theta']<1.1e-12 and test['loop_error']<contract['closed_loop_atol']
 return dict(task=task,ensemble=item['ensemble'],conf_id=item['conf_id'],run=str(run),theta=q['theta'],loop_error=delta,elapsed_seconds=result['elapsed_seconds'])

def audit(stage):
 req=json.loads((ROOT/'request.json').read_text())
 for name,digest in json.loads((ROOT/'source_lock.json').read_text())['sha256'].items():assert c.sha(ROOT/name)==digest, ('source changed',name)
 tasks=req['pilot_tasks'] if stage=='pilot' else list(range(len(req['items'])))
 complete=[]; errors=[]
 for t in tasks:
  try:complete.append(inspect(t,req,t in req['pilot_tasks']))
  except Exception as e: errors.append(dict(task=t,ensemble=req['items'][t]['ensemble'],conf_id=req['items'][t]['conf_id'],error=repr(e)))
 report=dict(status='PASS' if not errors else 'INCOMPLETE',stage=stage,expected=len(tasks),completed=len(complete),complete=complete,errors=errors,source_lock_sha256=c.sha(ROOT/'source_lock.json'),per_ensemble={name:sum(x['ensemble']==name for x in complete) for name in req['ensembles']})
 c.write_json(ROOT/('validation/pilot_gate.json' if stage=='pilot' else 'completion_audit.json'),report)
 print(json.dumps({k:v for k,v in report.items() if k not in ['complete','errors']},indent=2),flush=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','final']);a=p.parse_args()
 report=audit(a.stage)
 if report['status']!='PASS':raise SystemExit(1)
 if a.stage=='final':
  from analysis import analyze
  analyze(report)
