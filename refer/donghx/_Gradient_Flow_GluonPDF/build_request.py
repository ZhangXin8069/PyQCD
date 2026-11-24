"""Metadata-only source freeze. No gauge, basis or correlator payload reads."""
import ast
from datetime import datetime
import json
import subprocess
from plan_common import *


def main():
    assert not (HERE/'request.json').exists(),'already frozen; use a new successor'
    request=json.loads((MOVING/'request.json').read_text())
    manifest=json.loads((MOVING/'manifest.json').read_text())
    assert manifest['confs']==list(CONFS) and manifest['flows']==list(FLOWS)
    for rel,digest in manifest['files_sha256'].items():
        assert sha(MOVING/rel)==digest,rel
    assert len(request['workers'])==100
    assert {t for w in request['workers'] for t in w['tasks']}==set(range(400))
    for rel in ('gpu_worker.py','launch.py','test_cpu.py'):
        assert sha(P0/rel)==sha(P0_ORIGINAL/rel),rel
    for p in (P0/'reference').glob('*.py'):
        assert sha(p)==sha(P0_ORIGINAL/'reference'/p.name),p.name
    paths=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.slurm'))+sorted(P0.glob('*.py'))+sorted((P0/'reference').glob('*.py'))
    for path in paths:
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        else:subprocess.run(['bash','-n',str(path)],check=True)
    upstream=[MOVING/'request.json',MOVING/'manifest.json',MOVING/'common.py',MOVING/'launch.py',
              MOVING/'gpu_worker.py',MOVING/'generate_basis.py',MOVING/'audit_cpu.py',
              P0_ORIGINAL/'gpu_worker.py',P0_ORIGINAL/'launch.py',P0_ORIGINAL/'test_cpu.py']
    upstream += list((P0_ORIGINAL/'reference').glob('*.py'))
    p0paths=sorted(P0.glob('*.py'))+sorted((P0/'reference').glob('*.py'))
    atomic_json(P0/'request.json',dict(schema='L32x64_B_Pz0_conf51to100_8flow_request_v1',
        generated=datetime.now().astimezone().isoformat(),confs=list(CONFS),flows=list(FLOWS),
        windows=list(WINDOWS),momenta_z=[0],weight_scheme=manifest['weight_scheme'],
        workers=request['workers'],new_conf_flow_count=400,new_case_count=100800,
        files_sha256={str(p.relative_to(P0)):sha(p) for p in p0paths},
        upstream_sha256={str(p):sha(p) for p in upstream},
        persistent_basis_cache=True,persistent_vertex_cache=False,persistent_VVV_cache=False,
        basis_policy='reuse same moving basis only after its task passes; no standalone P0 basis regeneration'))
    paths+=[P0/'request.json',HERE/'README.md']
    atomic_json(HERE/'request.json',dict(schema='L32x64_B_conf51to100_P0456_retry_v1',
        generated=datetime.now().astimezone().isoformat(),confs=list(CONFS),flows=list(FLOWS),windows=list(WINDOWS),
        currents=list(CURRENTS),tseps=list(TSEPS),momenta=[0,4,-4,5,-5,6,-6],
        epsilon=.01,nstep=[10,20,30,40,50,60,70,80],weight_scheme=manifest['weight_scheme'],
        files_sha256={str(p.relative_to(HERE)):sha(p) for p in paths},
        upstream_sha256={str(p):sha(p) for p in upstream},old_array=OLD_ARRAY,old_audit=OLD_AUDIT,
        resource=dict(partition='na100-ins',reservation='donghx_250',eligible_nodes=list(ALLOWED_NODES),
            exclude_nodes=['gpu025','gpu028'],gpus_per_job=4,mpi_ranks_per_job=4,mem='240G',time='3-00:00:00',
            array='0-99',array_concurrency_cap=None,flows_per_job=4),
        workflow='CPU full existing-case/input audit -> missing moving flows + same-basis P0 -> full CPU audit',
        numerical_kernel_changes=False,new_GPU_smoke=False))
    print('FROZEN_CONF51_100_P0456',len(CONFS),'conf, 8 flows, 100 packed workers',flush=True)


if __name__=='__main__':main()
