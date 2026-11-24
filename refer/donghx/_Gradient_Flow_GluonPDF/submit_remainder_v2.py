"""Resume partial submission; reservation+exclude chooses one of three nodes."""
import ast
import subprocess
from plan_common import *
check_frozen()
receipt=json.loads((HERE/'submission.json').read_text())
assert set(receipt['jobs'])=={'gate'} and receipt['jobs']['gate']['job_id']=='3883880'
assert not (HERE/'scheduler_v2_submission.json').exists()
for rel in ('dispatch_v2.py','submit_remainder_v2.py'):
    ast.parse((HERE/rel).read_text(),filename=rel)
subprocess.run(['bash','-n',str(HERE/'run_na100_ins_v2.slurm')],check=True)
atomic_json(HERE/'scheduler_v2_lock.json',dict(files_sha256={rel:sha(HERE/rel)
    for rel in ('dispatch_v2.py','submit_remainder_v2.py','run_na100_ins_v2.slurm')},
    original_request_sha256=sha(HERE/'request.json')))
command=['sbatch','--parsable','--dependency=afterok:3883880',
    f'--output={HERE}/logs/production_v2_%A_%a.out',f'--error={HERE}/logs/production_v2_%A_%a.err',
    str(HERE/'run_na100_ins_v2.slurm')]
r=subprocess.run(command,check=True,capture_output=True,text=True)
gpu=r.stdout.strip().split(';')[0];assert gpu.isdigit()
record=dict(state='GPU_SUBMITTED',production_job=gpu,production_command=command,
    scheduler_v2_lock_sha256=sha(HERE/'scheduler_v2_lock.json'),
    reason='Slurm nodelist is a required-node list, not candidate list; use existing reservation plus exclude for one-node jobs',
    eligible_nodes=list(ALLOWED_NODES),excluded_nodes=['gpu025','gpu028'])
atomic_json(HERE/'scheduler_v2_submission.json',record)
receipt['jobs']['production']=dict(job_id=gpu,command=command);atomic_json(HERE/'submission.json',receipt)
command=['sbatch','--parsable',f'--dependency=afterok:{gpu}','--job-name=q64_P0456_audit',
    f'--output={HERE}/logs/audit_%A_%a.out',f'--error={HERE}/logs/audit_%A_%a.err',str(HERE/'cpu.slurm'),'audit']
r=subprocess.run(command,check=True,capture_output=True,text=True)
audit=r.stdout.strip().split(';')[0];assert audit.isdigit()
record.update(state='SUBMITTED',audit_job=audit,audit_command=command)
atomic_json(HERE/'scheduler_v2_submission.json',record)
receipt['jobs']['audit']=dict(job_id=audit,command=command);receipt['state']='SUBMITTED'
receipt['scheduler_v2_sha256']=sha(HERE/'scheduler_v2_submission.json');atomic_json(HERE/'submission.json',receipt)
print('SUBMITTED_PRODUCTION',gpu,'AUDIT',audit,flush=True)
