"""Read-only Slurm snapshot and cleanup-target guard; no job mutations."""
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
label = sys.argv[1] if len(sys.argv) > 1 else 'before'
if label not in {'before', 'after'}:
    raise SystemExit('Only before/after snapshot labels are allowed.')
plan = json.loads((HERE/'cleanup_plan.json').read_text())
moves = {x['path'] for x in plan['archive_moves']}
text = subprocess.check_output(
    ['squeue', '-r', '-u', 'donghx', '-h', '-o', '%i|%j|%T|%R|%Z'], text=True)
rows = []
representatives = {}
for line in text.splitlines():
    job_id, name, state, reason, directory = line.split('|', 4)
    row = dict(job_id=job_id, name=name, state=state, reason_or_nodes=reason, workdir=directory)
    rows.append(row)
    base = job_id.split('_')[0]
    representatives.setdefault(base, row)
details, conflicts = [], []
for base, row in sorted(representatives.items()):
    result = subprocess.run(['scontrol', 'show', 'job', row['job_id'], '-o'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f'Cannot inspect active job {row["job_id"]}: {result.stderr}')
    fields = dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)', result.stdout.strip()))
    wanted = {key: fields.get(key) for key in
              ('JobId', 'JobName', 'JobState', 'ArrayJobId', 'Command', 'WorkDir', 'StdOut', 'StdErr')}
    details.append(wanted)
    for key in ('Command', 'WorkDir', 'StdOut', 'StdErr'):
        value = wanted.get(key)
        if not value:
            continue
        # Command paths in this snapshot contain no whitespace; retain only
        # paths, not arbitrary scheduler/environment fields or credentials.
        path = Path(value.split()[0])
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in moves:
            conflicts.append(dict(job=base, field=key, target=str(relative)))
save = dict(generated=datetime.now().astimezone().isoformat(), root=str(ROOT),
            rows=rows, representative_job_details=details, conflicts=conflicts,
            note='Read-only scheduler state; pending/finished jobs can change naturally during cleanup.')
(HERE/f'active_jobs_{label}.json').write_text(json.dumps(save, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(snapshot=label, active_job_families=len(details), tasks=len(rows),
                      conflicts=conflicts, jobs=details), ensure_ascii=False, indent=2))
if conflicts:
    raise SystemExit('STOP: archive target is referenced by an active job.')
