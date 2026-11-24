"""Bounded workers; each conf uses a fresh Python/QUDA process."""
import argparse, fcntl, json, os, subprocess, sys
from pathlib import Path
import core as c
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','production']);a=p.parse_args()
 req=json.loads((ROOT/'request.json').read_text()); idx=int(os.environ['SLURM_ARRAY_TASK_ID'])
 if a.stage=='production':
  gate=json.loads((ROOT/'validation/pilot_gate.json').read_text())
  assert gate['status']=='PASS' and gate['source_lock_sha256']==c.sha(ROOT/'source_lock.json')
  tasks=req['worker_tasks'][idx]
 else:tasks=[req['pilot_tasks'][idx]]
 failures=[]
 for task in tasks:
  parent=ROOT/'runs'/('task%03d'%task);parent.mkdir(parents=True,exist_ok=True)
  with (parent/'claim.lock').open('w') as claim:
   fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
   assert not list(parent.glob('job*/complete.json')),('unexpected duplicate',task)
   env=os.environ.copy()
   env['QUDA_RESOURCE_PATH']=str(ROOT/'cache'/('job'+os.environ['SLURM_JOB_ID'])/('task%03d'%task)/'quda')
   env['CUPY_CACHE_DIR']=str(Path(env['QUDA_RESOURCE_PATH']).parent/'cupy')
   env['QUDA_PROFILE_OUTPUT_BASE']='profile'
   for key in ['QUDA_RESOURCE_PATH','CUPY_CACHE_DIR']:Path(env[key]).mkdir(parents=True,exist_ok=True)
   print('CONF_START',task,flush=True)
   code=subprocess.call([sys.executable,'-B','-u',str(ROOT/'worker.py'),'--task',str(task)],env=env)
   print('CONF_END',task,'exit',code,flush=True)
   if code:failures.append(task)
 c.write_json(ROOT/'runs'/('batch_%s_%s_%s.json'%(a.stage,os.environ['SLURM_ARRAY_JOB_ID'],idx)),dict(tasks=tasks,failed_tasks=failures,status='PASS' if not failures else 'INCOMPLETE'))
 sys.exit(bool(failures))
