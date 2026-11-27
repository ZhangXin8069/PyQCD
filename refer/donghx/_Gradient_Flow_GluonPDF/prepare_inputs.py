"""Validate external input files and freeze a two-radius production manifest."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import common


def main():
    common.require_allocation()
    root = common.HERE
    destination = root / 'manifest.json'
    if destination.exists():
        raise RuntimeError('Refusing to overwrite the frozen manifest')
    request = json.loads((root / 'request.json').read_text())
    for name, expected in request['files_sha256'].items():
        assert common.sha256(root / name) == expected, name
    for name, expected in request['upstream_sha256'].items():
        assert common.sha256(name) == expected, name
    assert list(common.CONFS) == request['confs'] and list(common.MOMENTA) == request['momenta']
    if request['ensemble'] == 'C24':
        assert common.sha256(common.QUDA_LIBRARY) == request['backend_library_sha256']
        common.load_request()
    rows = []
    for conf in common.CONFS:
        for schedule in request['schedule']:
            fields = dict(schedule, task=len(rows), conf=conf)
            fields['tag'] = common.config_for(fields).tag
            rows.append(fields)
    assert len(rows) == 100
    if hasattr(common, 'rows'):
        production_rows = common.rows()
        for expected, actual in zip(rows, production_rows):
            for name in expected:
                if isinstance(expected[name], float):
                    assert abs(expected[name] - actual[name]) < 1e-14, (name, expected, actual)
                else:
                    assert expected[name] == actual[name], (name, expected, actual)
    endpoints = sorted({time for current in common.CURRENTS for sep in common.TSEPS
                        for dt in range(sep + 1) for time in common.case_times(current, sep, dt)[:2]})
    ready = {}
    blocked = []
    for offset, conf in enumerate(common.CONFS):
        selected = rows[2*offset:2*offset+2]
        config = common.config_for(selected[-1])
        source_times = set(endpoints) | {time for row in selected for time in common.source_times(row)}
        eigen = {config.eigen_path(time) for time in source_times}
        peram = {config.peram_path(time) for time in endpoints}
        shape_eigen, shape_peram = {
            'C24': ([650,24,24,24,3], [72,4,4,650,650]),
            'E32': ([400,32,32,32,3], [64,4,4,400,400]),
            'F32': ([600,32,32,32,3], [96,4,4,400,400]),
        }[request['ensemble']]
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                records = list(pool.map(common.fingerprint, sorted(eigen | peram | {config.gauge_path()})))
            for record in records:
                path = record['path']
                if '/eigensystem/' in path:
                    assert record.get('shape') == shape_eigen, record
                if '/perambulators/' in path:
                    assert record.get('shape') == shape_peram, record
                if path.endswith('.npy'):
                    assert record['dtype'] in ('complex64', 'complex128'), record
            config.gauge_payload()
        except (OSError, ValueError, AssertionError, RuntimeError) as error:
            blocked.append(dict(conf=conf, tasks=[r['task'] for r in selected], error=repr(error)))
            print('INPUT_BLOCKED', conf, repr(error), flush=True)
            continue
        ready[conf] = records
        for row in selected:
            if request['ensemble'] != 'E32':
                row['inputs'] = records
                row.update(endpoint_times=endpoints, source_times=common.source_times(row), basis_generation=True)
            if request['ensemble'] == 'C24':
                row['quda_library_sha256'] = request['backend_library_sha256']
        print('INPUT_HEADERS_PASS', conf, len(records), flush=True)
    common.atomic_json(root / 'status/input_report.json', dict(ready=list(ready), blocked=blocked))
    if not ready:
        raise RuntimeError('No group has a complete input inventory')
    momenta = list(common.MOMENTA)
    info = dict(schema=common.SCHEMA, profile=getattr(common, 'PROFILE', ''),
                date=datetime.now().astimezone().isoformat(), confs=list(common.CONFS),
                a_fm=request['a_fm'], matched_a64_fm=.0897, flows64=[r['tau64'] for r in request['schedule']],
                flows=[r['flow_time_a2'] for r in request['schedule']], epsilon=rows[0]['epsilon'],
                currents=list(common.CURRENTS), tseps=list(common.TSEPS), momenta=momenta, momenta_z=momenta,
                channels=list(getattr(common, 'CHANNELS', ('unpolarized','helicity','transversity'))),
                basis_schema=getattr(common, 'BASIS_SCHEMA', 'selected_output_flowed_basis_l32x96_matched_radius_B_v1'),
                weight_scheme=common.SCHEME, weight_applied_count=1, tasks=rows,
                configuration_inputs=ready if request['ensemble']=='E32' else {}, input_blocked=blocked,
                expected_case_count=252*len(rows), per_column_squared_norm_gate=1e-6,
                algebra_relative_l2_gate=2e-10, files_sha256=request['files_sha256'],
                request_sha256=common.sha256(root/'request.json'), persistent_basis_cache=True,
                persistent_vertex_cache=False, persistent_VVV_cache=False,
                transversity_status='pending_external_physics_confirmation')
    if request['ensemble'] == 'C24':
        info.update(quda_library_path=str(common.QUDA_LIBRARY.resolve()),
                    quda_library_sha256=request['backend_library_sha256'])
    common.atomic_json(destination, info)
    common.atomic_json(root / 'status/cpu_gate.json', dict(status='PASS', job=os.environ['SLURM_JOB_ID'],
                       manifest_sha256=common.sha256(destination), ready=len(ready), blocked=len(blocked)))
    with (root / 'task_manifest.csv').open('w') as stream:
        fields = ['task', 'conf', 'tau64', 'flow_time_a2', 'epsilon', 'nstep', 'window', 'radius_fm', 'tag']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
