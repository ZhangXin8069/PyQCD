#!/usr/bin/env python3
import ast
import os
import subprocess
import sys
from common import (HERE, CONFS, FLOWS, CURRENTS, TSEPS, SCHEME, load_manifest,
                    check_inputs, midpoint_receipt, sha256, atomic_json, load_request)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('CPU allocation required')
    request=load_request()
    for name in request['files_sha256']:
        if name.endswith('.py'):
            ast.parse((HERE/name).read_text(),filename=name)
    for script in ('test_small.py','test_insertions.py'):
        subprocess.run([sys.executable,'-B','-u',str(HERE/script)],check=True)
    if not (HERE/'manifest.json').exists():
        subprocess.run([sys.executable,'-B','-u',str(HERE/'prepare.py')],check=True)
    m = load_manifest()
    assert m['confs'] == list(CONFS) and m['flows'] == list(FLOWS)
    assert len(m['tasks']) == 160 and m['expected_case_count'] == 40320
    assert m['weight_scheme'] == SCHEME and m['weight_applied_count'] == 1
    for row in m['tasks']:
        check_inputs(row)
        for tj in CURRENTS:
            for sep in TSEPS:
                midpoint_receipt(row,tj,sep)
    atomic_json(HERE/'status/cpu_gate.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],
        manifest_sha256=sha256(HERE/'manifest.json'),configuration_count=20,flow_count=8,
        cases=40320,new_cases=36480,checks=['all external input headers and frozen provenance',
            'read-only B midpoint reuse; no A substitution',
            'original six-Wick/weights/parity tests and new insertion/time-map tests']))
    print('B_INSERTIONS_CPU_GATE_PASS',flush=True)


if __name__ == '__main__':
    main()
