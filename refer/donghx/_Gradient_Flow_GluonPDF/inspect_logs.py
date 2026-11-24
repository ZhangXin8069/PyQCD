from pathlib import Path
from datetime import datetime
import ast,json,statistics
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'gluon_flow_gpu025_dense_tau_20261113_v2';OUT=Path(__file__).resolve().parent

def stat(v):
 if not v:return dict(n=0)
 return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),min=min(v),max=max(v))
report=dict(checked_at=datetime.now().astimezone().isoformat(),ensembles=[])
for label in ['L32x64','L32x96']:
 batch_rows={}
 for p in sorted((ROOT/'runs'/label/'batches').glob('batch*/job*/progress.json')):
  try:
   for r in json.loads(p.read_text())['results']:batch_rows[r['conf_id']]=r
  except Exception as e:print('READ_ERROR',p,repr(e))
 rows=[];failures=[];tau_rows={}
 for p in sorted((ROOT/'runs'/label/'production').glob('conf*/job*/result.json')):
  r=json.loads(p.read_text());flow=0.;ope=0.;by=[]
  for point in r['results']:
   md=json.loads((Path(point['path'])/'metadata.json').read_text());timing=md['timing'];flow+=timing['flow_seconds'];ope+=timing['ope_seconds']
   tau_rows.setdefault(str(point['tau']),[]).append(timing)
   by.append(dict(tau=point['tau'],**timing))
  b=batch_rows.get(r['conf_id'],{})
  rows.append(dict(conf_id=r['conf_id'],worker_seconds=r['elapsed_seconds'],process_seconds=b.get('seconds'),
       flow_seconds=flow,ope_seconds=ope,other_worker_seconds=r['elapsed_seconds']-flow-ope,
       startup_and_exit_seconds=b['seconds']-r['elapsed_seconds'] if b else None,points=by,run=str(p.parent)))
 for p in (ROOT/'runs'/label/'production').glob('conf*/job*/failed.json'):failures.append(dict(path=str(p),content=json.loads(p.read_text())))
 summary=dict(ensemble=label,production_started=len(list((ROOT/'runs'/label/'production').glob('conf*/job*'))),
   production_complete=len(rows),failed=failures,returncode_errors=[r for r in batch_rows.values() if r['returncode']!=0],
   timings={k:stat([r[k] for r in rows if r[k] is not None]) for k in ['process_seconds','worker_seconds','flow_seconds','ope_seconds','other_worker_seconds','startup_and_exit_seconds']},
   per_tau={tau:{k:stat([x[k] for x in ts]) for k in ['flow_seconds','ope_seconds']} for tau,ts in tau_rows.items()},rows=rows)
 report['ensembles'].append(summary)
 print(label,'started',summary['production_started'],'complete',len(rows),'failed',len(failures))
 print(json.dumps(summary['timings'],indent=2))
 if rows:print('SAMPLE_RUN',rows[len(rows)//2]['run'])
(OUT/'log_timing_snapshot.json').write_text(json.dumps(report,indent=2)+'\n')
