"""Small prerequisite CPU inventory; no gauge/eigenvector tensor calculations."""
import json
import os
from datetime import datetime
from common import *

require_allocation()
cfg=config_for(rows()[0])
report=dict(date=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
    schedule=[{k:r[k] for k in ('tau64','flow_time_a2','epsilon','nstep','window','radius_fm')} for r in rows()[:7]],
    configs=[],headers={})
for conf in CONFS:
    r=next(r for r in rows() if r['conf']==conf)
    cfg=config_for(r)
    for path in (cfg.gauge_path(),cfg.eigen_path(0),cfg.peram_path(0)):
        if not path.is_file():
            raise FileNotFoundError(path)
    report['configs'].append(conf)
for name,path in [('eigen',cfg.eigen_path(0)),('peram',cfg.peram_path(0))]:
    report['headers'][name]=fingerprint(path)
report['lime_payload_offset_and_bytes']=config_for(rows()[0]).gauge_payload()
atomic_json(HERE/'inspection.json',report)
print(json.dumps(report,indent=2),flush=True)
