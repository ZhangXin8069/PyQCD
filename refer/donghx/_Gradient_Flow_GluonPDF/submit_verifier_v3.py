"""Retry completed/failed verification only; never alter queued source locks."""
import ast
import subprocess
from analysis_common import *
code_check()
assert not (HERE/'verification_v3_submission.json').exists()
for rel in ('verify_cpu_v3.py','submit_verifier_v3.py'):
    ast.parse((HERE/rel).read_text(),filename=rel)
subprocess.run(['bash','-n',str(HERE/'verify_v3_cpu.slurm')],check=True)
atomic_json(HERE/'verification_v3_lock.json',dict(files_sha256={rel:sha(HERE/rel)
    for rel in ('verify_cpu_v3.py','verify_v3_cpu.slurm','submit_verifier_v3.py')}))
command=['sbatch','--parsable',f'--output={HERE}/logs/verify_v3_%j.out',
    f'--error={HERE}/logs/verify_v3_%j.err',str(HERE/'verify_v3_cpu.slurm')]
r=subprocess.run(command,check=True,capture_output=True,text=True)
job=r.stdout.strip().split(';')[0];assert job.isdigit()
atomic_json(HERE/'verification_v3_submission.json',dict(job_id=job,command=command,
    replaced_failed_job='3883783',reason='Compute nodes lack pdfinfo/PyPDF2; use producer-specific structural page check, representative independent visual check separate',
    verification_v3_lock_sha256=sha(HERE/'verification_v3_lock.json')))
print('VERIFIER_V3_SUBMITTED',job)
