#!/usr/bin/env python3
"""Small stdlib staging request; CPU gate freezes the full file inventory."""
from datetime import datetime
import json
from common import HERE,PRIOR,CONFS,FLOWS,SCHEMA,SCHEME,sha256,atomic_json


def main():
    if (HERE/'request.json').exists():
        raise RuntimeError('request already frozen')
    prior_manifest=PRIOR/'manifest.json'
    prior_complete=PRIOR/'summary/complete.json'
    m=json.loads(prior_manifest.read_text())
    c=json.loads(prior_complete.read_text())
    assert m['confs']==list(CONFS) and m['flows']==list(FLOWS)
    assert c['case_count']==3840 and c['manifest_sha256']==sha256(prior_manifest)
    files=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.slurm'))+sorted((HERE/'reference').glob('*.py'))
    atomic_json(HERE/'request.json',dict(schema=SCHEMA,weight_scheme=SCHEME,
        generated=datetime.now().astimezone().isoformat(),confs=list(CONFS),flows=list(FLOWS),
        expected_case_count=40320,new_case_count=36480,reused_midpoint_case_count=3840,
        prior_manifest=str(prior_manifest),prior_manifest_sha256=sha256(prior_manifest),
        prior_complete=str(prior_complete),prior_complete_sha256=sha256(prior_complete),
        files_sha256={str(p.relative_to(HERE)):sha256(p) for p in files}))
    print('REQUEST_FROZEN; full metadata inventory and all tests run in CPU gate',flush=True)


if __name__=='__main__':
    main()
