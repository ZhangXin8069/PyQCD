"""Verify and read only needed B correlator members, not cached basis/vertices."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from common import *


@lru_cache(maxsize=1)
def upstream_manifest_hash():
    return sha256(UPSTREAM/'manifest.json')


def case_times(current, sep, dt):
    source=(current-dt)%NT
    return source, (source+sep)%NT, -1 if source+sep>=NT else 1


def case_location(row, current, sep, dt):
    if dt==sep//2:
        directory=(Path(row['midpoint_result_root'])/'C3_schemeB_Pz3456_tsep7to12'/
                   row['tag']/row['conf']/f'tJ{current:03d}'/f'tsep{sep:02d}')
        return directory/'window_comparison.npz', directory/'receipt.json', True
    directory=(UPSTREAM/'C3_C2_schemeB'/row['tag']/row['conf']/f'tJ{current:03d}'/
               f'tsep{sep:02d}'/f'dt{dt:02d}')
    return directory/'correlators.npz', directory/'receipt.json', False


def read_case(row, current, sep, dt):
    allocation()
    path, receipt_path, midpoint=case_location(row,current,sep,dt)
    before=stat_identity(path); receipt=json.loads(receipt_path.read_text())
    if receipt['status']!='PASS' or receipt['conf']!=row['conf'] or receipt['flow_time_a2']!=row['flow_time_a2']:
        raise ValueError(('bad B receipt',str(receipt_path)))
    if midpoint:
        assert receipt['manifest_sha256']==row['midpoint_manifest_sha256']
        expected=row['midpoint_npz_sha256'][f'{current}_{sep}']
    else:
        assert receipt['schema']=='L32x64_B_all_insertions_four_currents_v1'
        assert receipt['manifest_sha256']==upstream_manifest_hash()
        assert (receipt['task'],receipt['t_current'],receipt['tsep'],receipt['dt'],receipt['window'])==(row['task'],current,sep,dt,row['window'])
        expected=receipt['npz_sha256']
    actual=sha256(path)
    if actual!=expected:
        raise ValueError(('changed raw B payload',str(path)))
    with np.load(path,allow_pickle=False) as data:
        c3=data['c3_by_window_flavor'][-1] if midpoint else data['c3_flavor']
        c2=data['c2_common']
        assert str(data['weight_scheme'])==SCHEME and int(data['weight_applied_count'])==1
        assert int(data['dt'])==dt and int(data['t_current'])==current and int(data['tsep'])==sep
        source,sink,boundary=case_times(current,sep,dt)
        assert int(data['t_source'])==source and int(data['t_sink'])==sink
        if not midpoint:
            assert int(data['baryon_wrap_sign'])==boundary
        np.testing.assert_array_equal(data['momenta_z'],MOMENTA)
    if c3.shape!=(2,2,4,8,21,4,4) or c2.shape!=(4,8):
        raise ValueError(('wrong raw axes',str(path),c3.shape,c2.shape))
    if not np.isfinite(c3).all() or not np.isfinite(c2).all() or stat_identity(path)!=before:
        raise ValueError(('nonfinite or changed raw case',str(path)))
    return c3,c2,dict(conf=row['conf'],tau=row['flow_time_a2'],current=current,tsep=sep,dt=dt,
        source=source,sink=sink,midpoint_reused=midpoint,sha256=actual,**before)
