#!/usr/bin/env python3
"""Serial original vs serial/4-rank spin2: boundaries, output axes and MPI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
from flow_common_l32x96 import FlowConfig,flow_metadata,FLOWED_BASIS_SCHEMA,FLOWED_VERTEX_SCHEMA


def main():
    root=Path(tempfile.mkdtemp(prefix='gf_spin2_driver_',dir='/tmp'))
    rng=np.random.default_rng(20260922)
    def rand(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    def js(p,payload):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload))
    config=FlowConfig('synthetic',.01,.01,1,nev_current=3,nev_src=2,delta_z=1,
        work_root=root/'work',peram_root=root/'peram',gauge_root=root/'Nosmear')
    for tau in range(96):
        js(config.basis_done_path(tau),dict(schema=FLOWED_BASIS_SCHEMA,**flow_metadata(config),tau=tau,discarded_norm_fraction=0.))
        path=config.peram_path(tau);path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,rand((96,4,4,3,2)))
    for current in [0,1]:
        taus=sorted([(current+i)%96 for i in [-1,0,1]])
        for a in taus:
            for b in taus:
                for direction in ['z','y','x']:
                    path=config.vertex_path(current,a,b,direction);path.parent.mkdir(parents=True,exist_ok=True)
                    np.save(path,rand((2,1,3,3)).astype(np.complex64))
                    js(path.with_suffix('.json'),dict(schema=FLOWED_VERTEX_SCHEMA,**flow_metadata(config),
                        t_current=current,tau_a=a,tau_b=b,direction=direction))
    jobs=[dict(label='Pp3_'+d,mom=m,dir=d) for d,m in [('z',[3,0,0]),('y',[0,3,0]),('x',[0,0,3])]]
    digest=hashlib.sha256(json.dumps(jobs,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:12]
    vpath=root/'vvv'/'synthetic'/f'vvv_Nev2_Nm3_{digest}.npy';vpath.parent.mkdir(parents=True,exist_ok=True)
    np.save(vpath,rand((96,3,2,2,2)))
    js(vpath.with_suffix('.json'),dict(shape=[96,3,2,2,2],mom_labels=[j['label'] for j in jobs]))
    env=os.environ.copy()
    env.update(BACKEND='numpy',REQUIRE_MPI_SIZE='1',FLOW_TIME='.01',FLOW_EPSILON='.01',FLOW_TEMPORAL_WINDOW='1',
        FLOW_WORK_ROOT=str(root/'work'),PERAM_ROOT=str(root/'peram'),GAUGE_ROOT=str(root/'Nosmear'),
        NEV_CURRENT='3',NEV_SRC='2',NEV_VVV='2',DELTA_Z='1',FLOW_REQUIRE_MAX_DISCARDED_NORM='0',
        TSEP_LIST='5',PABS_LIST='3',DIRS='z,y,x',SIGNS='+',MOM_FILTER='Pp3_z,Pp3_y,Pp3_x',
        CURRENT_LIST='0,1',C2_SOURCE_LIST='0,1,91,92,93,94,95',C2_MAX_DT='5',
        C3_MAX_SINK_GROUPS='0',C3_MAX_TASKS_PER_GROUP='0',SPIN2_VERIFY='1',SPIN2_MOM_BATCH='3',
        SPIN2_VVV_ROOT=str(root/'vvv'),VVV_PATH=str(vpath),SPIN2_STATUS_ROOT=str(root/'status'),
        STAGE_VVV_LOCAL='0',STAGE_PERAM_LOCAL='0',FLOWED_PERAM_STORAGE='host',CONTRACT_OPTIMIZE='greedy',
        PROFILE_WICK_TIMING='0',PROFILE_SYNC_GPU_TIMING='0',VALIDATE_OUTPUT='1',
        PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    outputs=[]
    for name,ranks,entry in [('reference',1,HERE/'reference'/'calc_flowed_c2c3_factorized_l32x96.py'),
                             ('spin2_serial',1,HERE/'run_spin2.py'),('spin2_mpi4',4,HERE/'run_spin2.py')]:
        case=root/name;env.update(SAVE_ROOT=str(case/'save'),RESULT_TMP_ROOT=str(case/'tmp'),
            INPUT_STAGE_ROOT=str(case/'stage'),FLOWED_PERAM_TMP_ROOT=str(case/'flowed'),REQUIRE_MPI_SIZE=str(ranks))
        cmd=[sys.executable,str(entry),'synthetic']
        if ranks>1:cmd=['mpirun','-n',str(ranks)]+cmd
        with (root/(name+'.log')).open('w') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        outputs.append(case/'save'/config.tag/'Nev_2'/'synthetic'/'corr_projected_tsep5_dz0-0_Nev2.npz')
    checks=[]
    with np.load(outputs[0]) as ref:
        for path in outputs[1:]:
            with np.load(path) as got:
                assert set(ref.files)==set(got.files)
                for key in ['corr_2pt','corr_3pt_u','corr_3pt_d']:
                    x=ref[key];y=got[key]
                    np.testing.assert_allclose(x,y,rtol=2e-10,atol=1e-8)
                    checks.append(dict(case=str(path),key=key,shape=list(x.shape),relative_L2=float(np.linalg.norm(x-y)/np.linalg.norm(x))))
                for key in ['mom','mom_labels','mom_dirs','pol_labels','axes_corr_2pt','axes_corr_3pt','boundary_condition']:
                    np.testing.assert_array_equal(got[key],ref[key])
    report=dict(status='PASS',temporary_root=str(root),checks=checks,
        includes='one/four MPI ranks; nonwrapping and wrapping C2/C3; x/y/z distinct vertices; exact saved NPZ axes')
    out=HERE/'status'/'synthetic_driver.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print('SYNTHETIC_DRIVER_PASS '+str(out),flush=True)


if __name__=='__main__':main()
