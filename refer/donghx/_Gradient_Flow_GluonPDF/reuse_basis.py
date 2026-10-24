"""CPU-only, read-only validation of old caches, then hardlink into successor.

Never edit the frozen predecessor or copy its incomplete .partial arrays.
New receipts bind the new manifest and preserve the old receipt hash.
"""
from dataclasses import replace
import json
import os
import numpy as np
from common import (HERE,PREVIOUS,config_for,load_manifest,load_request,
    source_times,selected_currents,sha256,atomic_json,fingerprint,require_allocation)


def link_existing(source,target):
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(('not a regular predecessor cache',str(source)))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if not os.path.samefile(source,target):
            raise RuntimeError(('refuse to replace unrelated successor cache',str(target)))
    else:
        os.link(source,target)


def main():
    require_allocation()
    manifest=load_manifest()
    old_request=json.loads((PREVIOUS/'request.json').read_text())
    for name,digest in old_request['files_sha256'].items():
        if sha256(PREVIOUS/name)!=digest:
            raise RuntimeError(('predecessor frozen source changed',name))
        if name.startswith('reference/') and sha256(HERE/name)!=digest:
            raise RuntimeError(('reference implementation changed; cannot reuse',name))
    old=json.loads((PREVIOUS/'manifest.json').read_text())
    old_digest=sha256(PREVIOUS/'manifest.json')
    new_digest=sha256(HERE/'manifest.json')
    assert old['request_sha256']==sha256(PREVIOUS/'request.json')
    for key,value in old.items():
        if key not in ('date','request_sha256'):
            assert manifest[key]==value,('changed production definition',key)
    ledger=[]
    absent=0
    for row in manifest['tasks']:
        cfg=config_for(row)
        previous=replace(cfg,work_root=PREVIOUS/'basis_cache')
        for sigma in source_times(row):
            source_done=previous.basis_done_path(sigma)
            if not source_done.exists():
                absent+=1
                continue
            meta=json.loads(source_done.read_text())
            assert meta['manifest_sha256']==old_digest
            expected=dict(schema=manifest['basis_schema'],conf=row['conf'],original_time=sigma,
                flow_time=row['flow_time_a2'],epsilon=row['epsilon'],n_steps=row['nstep'],
                fermion_flow_dimension=4,temporal_window=row['window'],
                stored_t_current=selected_currents(row,sigma),exact_temporal_support=False,
                nev=400,dtype='complex64',input_eigen_path=str(cfg.eigen_path(sigma)))
            for key,value in expected.items():
                assert meta[key]==value,(str(source_done),key)
            norm_source=source_done.with_name('norm_diagnostics.npz')
            with np.load(norm_source,allow_pickle=False) as d:
                norm=d['norm_by_mode_time']
                assert norm.shape==(400,96) and np.isfinite(norm).all()
                assert (norm>=0).all() and (norm.sum(axis=1)>0).all()
                assert int(d['original_time'])==sigma and int(d['window'])==row['window']
                distance=np.asarray([min((t-sigma)%96,(sigma-t)%96) for t in range(96)])
                tail=norm[:,distance>row['window']].sum(axis=1)/norm.sum(axis=1)
                assert float(tail.max())<=manifest['per_column_squared_norm_gate']
            files=[]
            for tj in selected_currents(row,sigma):
                source,target=previous.basis_path(sigma,tj),cfg.basis_path(sigma,tj)
                header=fingerprint(source)
                assert header['shape']==[400,32,32,32,3] and header['dtype']=='complex64'
                assert sha256(source)==meta['slice_sha256'][str(tj)]
                files.append((source,target))
            for source,target in files:
                link_existing(source,target)
            target_done=cfg.basis_done_path(sigma)
            link_existing(norm_source,target_done.with_name('norm_diagnostics.npz'))
            updated=dict(meta,manifest_sha256=new_digest,reused_from=dict(
                receipt=str(source_done),receipt_sha256=sha256(source_done),
                manifest_sha256=old_digest,method='validated_regular_file_hardlink_no_data_copy'))
            if target_done.exists():
                assert json.loads(target_done.read_text())==updated
            else:
                atomic_json(target_done,updated)
            ledger.append(dict(task=row['task'],original_time=sigma,slices=len(files)))
            print('REUSED_VERIFIED_BASIS',row['task'],sigma,len(files),flush=True)
    load_request()
    atomic_json(HERE/'status/basis_reuse.json',dict(status='PASS',manifest_sha256=new_digest,
        predecessor=str(PREVIOUS),predecessor_manifest_sha256=old_digest,
        reused_original_time_count=len(ledger),reused_slice_count=sum(r['slices'] for r in ledger),
        missing_original_time_count=absent,ledger=ledger,
        no_predecessor_mutation=True,incomplete_caches_reused=False,
        scope='validated finite-flow basis only; no predecessor C3/C2 imported'))
    print('BASIS_REUSE_PASS',len(ledger),flush=True)


if __name__=='__main__':
    main()

