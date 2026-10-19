#!/usr/bin/env python3
"""Metadata-only staging. Reuse upstream basis/controls read-only, never reflow."""
import copy
import json
from common import HERE, UPSTREAM, FlowConfig, fingerprint, sha256, atomic_json


def main():
    if (HERE/'manifest.json').exists():
        raise RuntimeError('Already frozen: use existing submission, not a second freeze')
    upstream = json.loads((UPSTREAM/'manifest.json').read_text())
    if sha256(UPSTREAM/'manifest.json') != '48eb91a7c1e5474088e8f5e5bb74300bb28046706099aa9032ee0ea21259713f':
        raise RuntimeError('Unexpected upstream P4 manifest')
    for name, digest in upstream['files_sha256'].items():
        if sha256(UPSTREAM/name) != digest:
            raise RuntimeError('Upstream frozen code changed: '+name)
    m = copy.deepcopy(upstream)
    m.update(schema='L32x64_P4_schemeB_memory_successor_v1',
        upstream_root=str(UPSTREAM), upstream_manifest_sha256=sha256(UPSTREAM/'manifest.json'),
        selected_tasks_to_compute=[1,2], reused_control_task=0,
        memory_policy=dict(c2_before_vertices=True,c2_momentum_batch=1,
            full_spin_oracle_retained=True, physical_H_sigma_b_batch=1,
            GPU_pool_limit_GiB=70, unchanged_physical_scope=True))
    for row in m['tasks']:
        cfg = FlowConfig(row['conf'], row['flow_time_a2'], .01, max(row['windows']),
                         work_root=UPSTREAM/'basis_cache')
        row['basis_inputs'] = []
        for sigma in row['source_times']:
            done = cfg.basis_done_path(sigma)
            meta = json.loads(done.read_text())
            if (meta['schema'] != 'selected_output_flowed_basis_l32x64_P4_B_pilot_v1'
                    or meta['manifest_sha256'] != m['upstream_manifest_sha256']
                    or meta['flow_time'] != cfg.flow_time or meta['epsilon'] != .01
                    or meta['n_steps'] != cfg.n_steps):
                raise RuntimeError('Incompatible reusable basis: '+str(done))
            row['basis_inputs'].append(fingerprint(done))
            for tj in meta['stored_t_current']:
                item = fingerprint(cfg.basis_path(sigma, tj))
                if item['shape'] != [400,32,32,32,3] or item['dtype'] != 'complex64':
                    raise RuntimeError('Reusable basis header mismatch')
                row['basis_inputs'].append(item)
        row['inputs'] += row['basis_inputs']
    marker = UPSTREAM/'status/task0_comparison.json'
    if json.loads(marker.read_text())['status'] != 'PASS':
        raise RuntimeError('tau .2 upstream control not PASS')
    m['upstream_control_inputs'] = [fingerprint(marker)]
    tag = FlowConfig('10000',.2,.01,5,work_root=UPSTREAM/'basis_cache').tag
    base = UPSTREAM/'C3_schemeB_Pz3456_tsep7to12'/tag/'10000'
    for tj in m['currents']:
        for sep in m['tseps']:
            for name in ('receipt.json','window_comparison.npz'):
                m['upstream_control_inputs'].append(fingerprint(base/f'tJ{tj:03d}'/f'tsep{sep:02d}'/name))
    files = sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.slurm'))+sorted((HERE/'reference').glob('*.py'))
    m['files_sha256']={str(p.relative_to(HERE)):sha256(p) for p in files}
    atomic_json(HERE/'manifest.json', m)
    print('MEMORY_SUCCESSOR_FROZEN; basis reused, GPU missing-only tasks 1,2',flush=True)


if __name__ == '__main__':
    main()
