"""Check immutable launcher, original campaign identity and existing gates."""
import hashlib
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert 'SLURM_JOB_ID' in os.environ
    lock=json.loads((HERE/'source_lock.json').read_text())
    for path,digest in lock['files_sha256'].items():assert sha(path)==digest,path
    original=Path(lock['campaign_root']);sys.path.insert(0,str(original))
    from common import load_manifest,load_request
    load_request();manifest=load_manifest()
    digest=sha(original/'manifest.json')
    worker=int(os.environ['SLURM_ARRAY_TASK_ID']);assert worker in (22,23)
    for path in [original/'status/cpu_gate.json',original/'status/epsilon_gate.json',
                 original.parent/'quark_l24x72_B_resume_20261117/preflight_result.json']:
        gate=json.loads(path.read_text())
        assert gate['status']=='PASS' and gate['manifest_sha256']==digest,path
    assert len(manifest['tasks'])==560
    assert os.environ.get('SLURMD_NODENAME')=='gpu025'
    print('GPU025_HANDOVER_IDENTITY_PASS',worker,flush=True)

if __name__=='__main__':main()
