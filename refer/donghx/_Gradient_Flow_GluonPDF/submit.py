"""Persist every submission receipt; real pilot -> CPU numerical gate -> GPU production."""
from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
receipt=ROOT/'submission.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d):
 tmp=receipt.with_name(receipt.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(receipt)
for name,d in json.loads((ROOT/'source_lock.json').read_text())['sha256'].items():assert sha(ROOT/name)==d,name
req=json.loads((ROOT/'request.json').read_text())
state=json.loads(receipt.read_text()) if receipt.exists() else dict(created_at=datetime.now().astimezone().isoformat(),jobs={},
 source_lock_sha256=sha(ROOT/'source_lock.json'),diagnostic_jobs=[3993224,3993237,3993249],
 diagnostic_pass=str(ROOT/'validation/diagnostic_3993249/complete.json'))
assert state['source_lock_sha256']==sha(ROOT/'source_lock.json')
assert json.loads(Path(state['diagnostic_pass']).read_text())['status']=='PASS'
def submit(key,script,dependency=None):
 if key in state['jobs']:return state['jobs'][key]['job_id']
 cmd=['sbatch','--parsable']
 if dependency:cmd+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
 cmd.append(str(ROOT/script));job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
 assert job.isdigit(),job
 state['jobs'][key]=dict(job_id=job,script=str(ROOT/script),dependency=dependency,submitted_at=datetime.now().astimezone().isoformat())
 save(state);print(key,job,dependency,flush=True);return job
pilots={};arrays={}
for ens in req['ensembles']:pilots[ens]=submit(ens+'_pilot',ens+'_pilot_gpu.slurm')
for ens in req['ensembles']:
 gate=submit(ens+'_gate',ens+'_gate_cpu.slurm','afterany:'+pilots[ens])
 arrays[ens]=submit(ens+'_production',ens+'_production_gpu.slurm','afterok:'+gate)
submit('final_audit','final_audit_cpu.slurm','afterany:'+':'.join(list(pilots.values())+list(arrays.values())))
state['status']='SUBMITTED';state['requested_configurations']=2587;state['requested_flow_points']=12935;save(state)
