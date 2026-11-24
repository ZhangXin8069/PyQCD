"""Bind the newly audited 30 configurations to the completed previous 70."""
from datetime import datetime
import json
import os
from pathlib import Path
from common import HERE,CONFS,load_request,load_manifest,require_allocation,sha256,atomic_json


def main():
    require_allocation()
    request=load_request()
    manifest=load_manifest()
    summary=json.loads((HERE/'summary/complete.json').read_text())
    assert summary['status']=='COMPLETE_L32X96_B_MATCHED_RADIUS_C3_C2'
    assert summary['nconf']==30 and summary['tasks']==210 and summary['case_count']==52920
    assert summary['manifest_sha256']==sha256(HERE/'manifest.json')
    assert all(r['status']=='PASS' and r['case_count']==252 for r in summary['task_results'])
    assert manifest['confs']==list(CONFS)
    prior_path=Path(request['previous70_completion'])
    prior=json.loads(prior_path.read_text())
    assert prior['combined_nconf']==70
    assert prior['status']=='COMPLETE_PREVIOUS50_PLUS_EXPANSION20'
    assert sha256(prior_path)==request['upstream_sha256'][str(prior_path)]
    combined=(HERE/'combined_confs100.txt').read_text().splitlines()
    assert combined==request['previous_confs']+list(CONFS) and len(set(combined))==100
    load_request()
    result=dict(status='COMPLETE_PREVIOUS70_PLUS_EXPANSION30',combined_nconf=100,
        combined_conf_flow_tasks=700,combined_case_count=176400,
        generated=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        new_nconf=30,new_cases=52920,new_summary=str(HERE/'summary/complete.json'),
        new_summary_sha256=sha256(HERE/'summary/complete.json'),
        previous70_completion=str(prior_path),previous70_completion_sha256=sha256(prior_path),
        combined_configuration_list=str(HERE/'combined_confs100.txt'),
        combined_configuration_list_sha256=sha256(HERE/'combined_confs100.txt'),
        request_sha256=sha256(HERE/'request.json'),
        scope='new30 full readback plus immutable previous70 completion receipts; no remeasurement of previous70')
    atomic_json(HERE/'summary/combined100_complete.json',result)
    print('COMBINED100_COMPLETE',flush=True)


if __name__=='__main__':main()
