#!/usr/bin/env python3
"""Record user-authorized parallel production; never fabricate a passed gate."""
import json
import os
from datetime import datetime
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
conf,regression=sys.argv[1:]
report_path=root/'status'/f'job_{regression}.json'
state='pending_full_regression'
if report_path.exists():
    report=json.loads(report_path.read_text())
    if report.get('status')=='PASS' and report.get('verify'):
        state='regression_report_present_PASS'
job=os.environ['SLURM_JOB_ID']
receipt=dict(conf=conf,job=job,regression_job=regression,dependency=None,
    authorization='2026-09-22 user explicitly requested production before regression passes',
    validation_state_at_start=state,started=datetime.now().isoformat(),
    result_policy='retain isolated products; full-regression validation remains separate',
    input_and_output_validation='enabled')
out=root/'status'/f'parallel_start_{job}.json'
out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
print('PARALLEL_PRODUCTION_AUTHORIZED '+json.dumps(receipt,ensure_ascii=False),flush=True)
