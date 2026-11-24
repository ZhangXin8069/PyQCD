"""Replace only the queued PDF verifier; preserve submitted physics/plot code."""
import ast
import subprocess
from analysis_common import *

code_check()
for rel in ('verify_cpu_v2.py', 'replace_verifier.py'):
    ast.parse((HERE / rel).read_text(), filename=rel)
subprocess.run(['bash', '-n', str(HERE / 'verify_v2_cpu.slurm')], check=True)
assert not (HERE / 'verification_v2_submission.json').exists()
atomic_json(HERE / 'verification_v2_lock.json', dict(files_sha256={
    rel:sha(HERE/rel) for rel in ('verify_cpu_v2.py','verify_v2_cpu.slurm','replace_verifier.py')}))
result = subprocess.run(['squeue','-h','-j','3883762','-o','%T'], check=True, capture_output=True, text=True)
assert result.stdout.strip() == 'PENDING', result.stdout
subprocess.run(['scancel','3883762'], check=True)
command = ['sbatch','--parsable','--dependency=afterok:3883761',
           f'--output={HERE}/logs/verify_v2_%j.out',f'--error={HERE}/logs/verify_v2_%j.err',
           str(HERE / 'verify_v2_cpu.slurm')]
result = subprocess.run(command, check=True, capture_output=True, text=True)
job = result.stdout.strip().split(';')[0]; assert job.isdigit()
atomic_json(HERE / 'verification_v2_submission.json', dict(job_id=job, replaced_queued_job='3883762',
    reason='CPU environment provides pdfinfo, not PyPDF2; no physics/ratio/plot sources changed',command=command,
    verification_v2_lock_sha256=sha(HERE / 'verification_v2_lock.json')))
print('VERIFIER_V2_SUBMITTED', job)
