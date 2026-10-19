#!/usr/bin/env python3
"""Refuse production unless the full conf10000 spin2 regression passed."""
from pathlib import Path
import json
import sys
root=Path(__file__).resolve().parent
job=sys.argv[1]
report=json.loads((root/'status'/f'job_{job}.json').read_text())
assert report['status']=='PASS' and report['verify'] and report['mpi_size']==4
assert report['checks']==15552,report['checks']
assert all(r['relative_L2']<2e-10 for r in report['records'])
synthetic=json.loads((root/'status'/'synthetic_driver.json').read_text())
assert synthetic['status']=='PASS'
print('FULL_CONF10000_REGRESSION_PASS',job)
