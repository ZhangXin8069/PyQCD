#!/usr/bin/env python3
"""L32x64 raw LIME -> four-dimensional flow -> quark C2/C3."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
from flow_common_l32x64 import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('conf',choices=[str(x) for x in range(10800,11200,40)])
    a=p.parse_args();conf=a.conf;job=os.environ['SLURM_JOB_ID']
    logdir=HERE/'logs'/f'{conf}_{job}';logdir.mkdir(parents=True,exist_ok=True)
    status=HERE/'status'/f'full_chain_{conf}_{job}.json'
    flow_time=float(os.environ['FLOW_TIME']); epsilon=float(os.environ.get('FLOW_EPSILON','0.01'))
    window=int(os.environ['FLOW_TEMPORAL_WINDOW'])
    tseps=[int(x) for x in os.environ.get('TSEP_LIST','5,6,7,8,9,10').split(',') if x]
    receipt=dict(conf=conf,job=job,ensemble='L32x64',lattice_spacing_fm=0.0897,
        flow_time_units='a^2',
        flow_time=flow_time,epsilon=epsilon,window=window,tseps=tseps,
        mom_labels=['Pp3_z','Pp4_z','Pp5_z'],flow_work_root=os.environ['FLOW_WORK_ROOT'],
        save_root=os.environ['SAVE_ROOT'],stages=[],state='RUNNING')
    def clean_vertex_cache():
        if os.environ.get('CLEAN_VERTEX_CACHE') != '1':
            return
        from flow_common_l32x64 import FlowConfig
        vertex_root=FlowConfig.from_env(conf).vertex_root
        if vertex_root.is_symlink():
            raise RuntimeError(f'refusing symlink vertex cache: {vertex_root}')
        if vertex_root.is_dir():
            shutil.rmtree(vertex_root)
        receipt['vertex_cache_removed']=not vertex_root.exists()
        print('VERTEX_CACHE_REMOVED',conf,vertex_root,flush=True)
    active=[]
    def interrupted(signum,frame):
        for child in active:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise RuntimeError(f'received signal {signum}; partial inputs retained for resume')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def run(stage,command,env=None):
        start=time.monotonic();receipt['current_stage']=stage;atomic_json(status,receipt)
        print('STAGE_BEGIN',conf,stage,flush=True)
        with (logdir/(stage+'.log')).open('a') as output:
            child=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(child);ret=child.wait();active.remove(child)
        receipt['stages'].append(dict(stage=stage,returncode=ret,seconds=time.monotonic()-start))
        atomic_json(status,receipt)
        print('STAGE_END',conf,stage,ret,flush=True)
        if ret:raise RuntimeError(f'{stage} failed; see {logdir/(stage+".log")}')
    python=sys.executable;generator=HERE/'reference'/'generate_flowed_vertex_l32x64.py'
    try:
        run('all_raw_input_headers',[python,str(HERE/'reference'/'validate_production_conf_l32x64.py'),conf,'--inputs'])
        run('pyquda_preflight',[python,str(generator),conf,'--stage','preflight'])
        devices=[x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x.strip()]
        if len(devices)!=4 or len(set(devices))!=4:
            raise RuntimeError(f'expected exactly four allocated visible GPUs, got {devices}')
        receipt['allocated_cuda_devices']=devices
        receipt['current_stage']='basis_4gpu';atomic_json(status,receipt)
        handles=[];workers=[];start=time.monotonic()
        print('STAGE_BEGIN',conf,'basis_4gpu',flush=True)
        for worker,device in enumerate(devices):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
            cache=Path(os.environ['WORKER_CACHE_ROOT'])/f'gpu{worker}'
            (cache/'quda').mkdir(parents=True,exist_ok=True);(cache/'cupy').mkdir(exist_ok=True)
            env.update(QUDA_RESOURCE_PATH=str(cache/'quda'),CUPY_CACHE_DIR=str(cache/'cupy'),
                GRID_SIZE='1,1,1,1',FLOW_GAUGE_OWNER_TAU='0',
                QUDA_PROFILE_OUTPUT_BASE=str(cache/'quda_profile'))
            taus=','.join(map(str,range(worker,64,4)))
            output=(logdir/f'basis_gpu{worker}.log').open('a');handles.append(output)
            child=subprocess.Popen(['mpirun','--bind-to','none','-n','1',python,'-u',str(generator),
                conf,'--stage','basis','--taus',taus],env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append(child);active.append(child)
        pending=set(range(4));codes={}
        while pending:
            for i in list(pending):
                ret=workers[i].poll()
                if ret is None:continue
                pending.remove(i);codes[i]=ret
                if ret:
                    for child in workers:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                    raise RuntimeError(f'basis worker {i} failed ({ret}); see basis_gpu{i}.log')
            if pending:time.sleep(2)
        for child in workers:active.remove(child)
        for output in handles:output.close()
        receipt['stages'].append(dict(stage='basis_4gpu',returncode=0,worker_codes=codes,seconds=time.monotonic()-start))
        atomic_json(status,receipt);print('STAGE_END',conf,'basis_4gpu',0,flush=True)
        run('vertex_z',['mpirun','-n','4',python,'-u',str(generator),conf,'--stage','vertex','--currents','0-63','--directions','z'])
        run('vvv_and_contraction_input_validation',['mpirun','-n','4',python,'-u',
            str(HERE/'validate_and_prepare.py'),conf,'--validate-input-only'])
        run('spin2_c2c3',['mpirun','-n','4',python,'-u',str(HERE/'run_spin2.py'),conf])
        run('validate_outputs',[python,str(HERE/'reference'/'validate_production_conf_l32x64.py'),conf,'--outputs'])
        if os.environ.get('CLEAN_FLOW_CACHE') == '1':
            from flow_common_l32x64 import FlowConfig
            cache_root=FlowConfig.from_env(conf).root
            if cache_root.is_dir():
                shutil.rmtree(cache_root)
        clean_vertex_cache()
        receipt['state']='COMPLETED'
        atomic_json(status,receipt);print('FULL_CHAIN_COMPLETED',conf,job,flush=True)
    except BaseException as ex:
        for child in active:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        try:
            clean_vertex_cache()
        except Exception as cleanup_error:
            receipt['vertex_cleanup_error']=str(cleanup_error)
        receipt['state']='FAILED_OR_INTERRUPTED';receipt['error']=str(ex);atomic_json(status,receipt)
        raise


if __name__=='__main__':main()
