"""Freeze exact required input headers/stats and the physical flow schedule."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
from functools import lru_cache
import json
import os
from common import (HERE,NT,NX,NV,NE,NZ,CONFS,CURRENTS,TSEPS,MOMENTA,CHANNELS,SCHEME,SCHEMA,
    BASIS_SCHEMA,PROFILE,A_FM,A64_FM,EPSILON,FLOWS64,load_request,require_allocation,
    rows,case_times,fingerprint,config_for,source_times,sha256,atomic_json)
from flow_common import ENSEMBLE


def main():
    require_allocation()
    request=load_request()
    if (HERE/'manifest.json').exists():
        raise RuntimeError('Immutable manifest already exists; use an audited successor')
    probe=json.loads((HERE/'status/input_probe.json').read_text())
    assert probe['status']=='PASS' and [r['conf'] for r in probe['configurations']]==list(CONFS)
    endpoints=sorted({t for tj in CURRENTS for sep in TSEPS for dt in range(sep+1)
                      for t in case_times(tj,sep,dt)[:2]})
    cached=lru_cache(maxsize=None)(fingerprint)
    tasks=[];errors=[];all_rows=rows()
    for conf in CONFS:
        selected=[r for r in all_rows if r['conf']==conf]
        cfg=config_for(selected[0])
        eigen_times=sorted(set(endpoints)|{s for r in selected for s in source_times(r)})
        files=[cfg.eigen_path(t) for t in eigen_times]+[cfg.peram_path(t) for t in endpoints]+[cfg.gauge_path()]
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                inputs=list(pool.map(cached,files))
            for f in inputs:
                if '/eigensystem/' in f['path']:
                    assert f['shape']==[NV,NX,NX,NX,3] and f['dtype']=='complex128',f
                if '/perambulators/' in f['path']:
                    assert f['shape']==[NT,4,4,NV,NV] and f['dtype'] in ('complex64','complex128'),f
            cfg.gauge_payload()
        except Exception as error:
            errors.append(dict(conf=conf,error=repr(error)))
            continue
        for row in selected:
            row.update(inputs=inputs,endpoint_times=endpoints,source_times=source_times(row),basis_generation=True)
            tasks.append(row)
        print('INPUT_HEADERS_PASS',conf,len(inputs),flush=True)
    if errors:
        atomic_json(HERE/'status/input_errors.json',dict(status='FAIL',errors=errors))
        raise RuntimeError(f'{len(errors)} configurations have missing/incompatible input')
    assert len(tasks)==350 and [r['task'] for r in tasks]==list(range(350))
    load_request()
    basis_bytes=sum(4*(2*r['window']+1)*NV*NX**3*3*8 for r in tasks)
    cases_per_task=len(CURRENTS)*sum(t+1 for t in TSEPS)
    m=dict(schema=SCHEMA,profile=PROFILE,date=datetime.now().astimezone().isoformat(),
        ensemble=ENSEMBLE,lattice_size_xyzt=[NX,NX,NX,NT],confs=list(CONFS),a_fm=A_FM,
        matched_a64_fm=A64_FM,flows64=list(FLOWS64),flows=[r['flow_time_a2'] for r in tasks[:7]],
        epsilon=EPSILON,currents=list(CURRENTS),tseps=list(TSEPS),momenta_z=list(MOMENTA),
        channels=list(CHANNELS),tasks=tasks,weight_scheme=SCHEME,weight_applied_count=1,
        expected_case_count=len(tasks)*cases_per_task,combined_configuration_target=50,
        source_sampling='four fixed currents; source=(tJ-dt)%72; dt=0..T; not all72 currents',
        temporal_window_policy='ceil(L64_window*a64/a24); full4D evolved, finite original-time support',
        per_column_squared_norm_gate=1e-6,algebra_relative_l2_gate=2e-10,
        basis_schema=BASIS_SCHEMA,basis_dtype='complex64',vertex_dtype='complex64',correlator_dtype='complex128',
        nev_current=NV,nev_external=NE,delta_z=NZ,basis_cache_estimated_bytes=basis_bytes,
        linear_mass_subtraction=False,Zchi_applied=False,SFTE_applied=False,LaMET_applied=False,
        persistent_vertex_cache=False,persistent_VVV_cache=False,array_concurrency_cap=None,
        request_sha256=sha256(HERE/'request.json'),transversity_status='pending_external_physics_confirmation',
        matched_scope='same physical flow radii only; momentum and tsep integers are not equal physical scales across ensembles')
    atomic_json(HERE/'manifest.json',m)
    with (HERE/'flow_schedule.csv').open('w',newline='') as stream:
        fields=['tau64','flow_time_a2','tf_fm2','radius_fm','epsilon','nstep','window64','window']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in tasks[:7])
    atomic_json(HERE/'status/cpu_gate.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],
        manifest_sha256=sha256(HERE/'manifest.json'),conf_count=50,tasks=350,
        input_files_verified=cached.cache_info().currsize,expected_cases=len(tasks)*cases_per_task,
        basis_cache_estimated_bytes=basis_bytes))
    print('CPU_GATE_PASS',50,350,'basis_TiB',basis_bytes/1024**4,flush=True)


if __name__=='__main__':
    main()
