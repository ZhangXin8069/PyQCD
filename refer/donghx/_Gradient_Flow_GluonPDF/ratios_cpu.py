#!/usr/bin/env python3
"""Compact current projection, shared configuration bootstrap, then VDV symmetry."""
import argparse
import json
import os
import time
import numpy as np
from common import *
from raw_io import member
from conventions import make_channel_specs,PHASE_FACTORS,validate_gamma_basis

SPECS=make_channel_specs(['Pp3_z','Pp4_z','Pp5_z'],['z']*3,['nopol','pol15','pol25','pol35'])


def project_case(path,tsep):
    """C3(flavor,channel,sign,p,z,insert), C2(p); average exactly 64 sources."""
    before=stat_identity(path)
    with np.load(path,allow_pickle=False) as d:
        if list(map(str,d['mom_labels']))!=['Pp3_z','Pp4_z','Pp5_z']:raise ValueError('momentum order')
        if list(map(str,d['pol_labels']))!=['nopol','pol15','pol25','pol35']:raise ValueError('pol order')
        if list(map(str,d['vdv_sign_labels']))!=['minus_dz','plus_dz']:raise ValueError('VDV order')
    sources=np.arange(NT)
    raw2=member(path,'corr_2pt')
    # Two-stage indexing is essential: preserve momentum rather than averaging it away.
    c2=raw2[0][:,sources,(sources+tsep)%NT].mean(axis=-1)
    if c2.shape!=(3,) or not np.isfinite(c2).all():raise ValueError('invalid C2')
    result=np.empty((2,3,2,3,21,tsep+1),complex)
    for ifl,flavor in enumerate(('u','d')):
        raw=member(path,f'corr_3pt_{flavor}')
        for ip in range(3):
            projected=[]
            for ch in range(4):
                base=raw[:,int(SPECS.pol_indices[ip,ch]),ip]
                # First contract only nonzero gamma elements, then gather each
                # physical insertion/source diagonal. No new full raw cache.
                out=np.zeros((2,21,tsep+1),complex)
                for m,o in zip(*np.nonzero(SPECS.gamma[ip,ch])):
                    block=base[:,:,m,o]
                    for q in range(tsep+1):
                        values=block[:,:,sources,(sources+q)%NT]
                        if not np.isfinite(values).all():raise ValueError('nonfinite selected raw C3')
                        out[:,:,q]+=SPECS.gamma[ip,ch,m,o]*values.mean(axis=-1)
                projected.append(out)
            result[ifl,0,:,ip]=projected[0]
            result[ifl,1,:,ip]=PHASE_FACTORS['helicity']*projected[1]
            result[ifl,2,:,ip]=.5*PHASE_FACTORS['transversity']*(projected[2]+projected[3])
        del raw
    if not np.isfinite(result).all() or stat_identity(path)!=before:
        raise ValueError('nonfinite projected C3 or source changed during read')
    return result,c2


def ratio_from_projected(c3,c2,indices):
    """Return original(ch,fl,vdv,p,z,T,q) and boot(b,ch,fl,vdv,p,z,T,q)."""
    weights=np.stack([np.bincount(x,minlength=NCONF)/NCONF for x in indices])
    # Input c3: cfg,ch,fl,sign,p,z,T,q. u-d formed before any ratio/selection.
    c3=np.concatenate((c3,(c3[:,:,0]-c3[:,:,1])[:,:,None]),axis=2)
    c3mean=c3.mean(axis=0);c2mean=c2.mean(axis=0)
    c3boot=np.tensordot(weights,c3,axes=(1,0))
    c2boot=np.tensordot(weights,c2,axes=(1,0))
    if np.any(c2mean==0) or np.any(c2boot==0) or not np.isfinite(c2boot).all():
        raise FloatingPointError('zero/nonfinite signed complex nopol denominator; no clipping')
    r=c3mean/c2mean[None,None,None,:,None,:,None]
    b=c3boot/c2boot[:,None,None,None,:,None,:,None]
    # Stored orientations minus,plus; do this after the bootstrap ratio.
    ro=np.stack((.5*(r[:,:,0]+r[:,:,1]),.5*(r[:,:,0]-r[:,:,1])),axis=2)
    rb=np.stack((.5*(b[:,:,:,0]+b[:,:,:,1]),.5*(b[:,:,:,0]-b[:,:,:,1])),axis=3)
    return ro,rb,c2mean,c2boot


def main():
    p=argparse.ArgumentParser();p.add_argument('--flow-index',type=int,required=True);a=p.parse_args()
    allocation();m=manifest();indices=shared_indices();tau=FLOWS[a.flow_index][0]
    marker=HERE/'status'/f'ratio_{tau_tag(tau)}.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if (record['status']=='PASS' and record['input_manifest_sha256']==sha256(HERE/'input_manifest.json')
                and all(sha256(x['path'])==x['sha256'] for x in record['products'])):
            print('VALIDATED_RATIO_SKIP',tau,flush=True);return
        raise ValueError('incompatible existing ratio completion')
    started=time.monotonic()
    c3=np.full((NCONF,3,2,2,3,21,len(TSEPS),11),np.nan+1j*np.nan)
    c2=np.empty((NCONF,3,len(TSEPS)),complex)
    records={(r['conf'],r['tsep']):r for r in m['raw_records'] if r['tau']==tau}
    for ic,conf in enumerate(CONFS):
        for it,tsep in enumerate(TSEPS):
            r=records[conf,tsep];path=Path(r['path'])
            if stat_identity(path)!={k:r[k] for k in ('path','size','mtime_ns')}:raise ValueError('raw input changed')
            nums,den=project_case(path,tsep)
            c3[ic,:,:,:,:,:,it,:tsep+1]=nums.transpose(1,0,2,3,4,5)
            c2[ic,:,it]=den
        print('PROJECTED_CONF',tau,conf,ic+1,'/',NCONF,flush=True)
    # Preserve configuration-level source means for reanalysis without rereading TB raw.
    projected=HERE/'projected'/f'projected_{tau_tag(tau)}.npz'
    atomic_npz(projected,c3_source_mean=c3,c2_source_mean=c2,**common_metadata(tau),
        axes_c3=np.asarray('conf,channel,flavor_u_d,sign,p,z,tsep,insertion'))
    ro,rb,den,denboot=ratio_from_projected(c3,c2,indices)
    time_mask=np.arange(11)[None,:]<=np.asarray(TSEPS)[:,None]
    expected=(3,3,2,3,21,6,11)
    if ro.shape!=expected or rb.shape!=(100,)+expected:raise ValueError((ro.shape,rb.shape))
    products=[]
    for ic,channel in enumerate(CHANNELS):
        for jf,flavor in enumerate(FLAVORS):
            r=ro[ic,jf];b=rb[:,ic,jf]
            valid=np.broadcast_to(time_mask,r.shape).copy()
            if np.any(~np.isfinite(r[valid])) or np.any(~np.isfinite(b[:,valid])):
                raise ValueError('nonfinite physical ratio entries')
            if np.any(np.isfinite(r[~valid])) or np.any(np.isfinite(b[:,~valid])):
                raise ValueError('invalid padding is not NaN')
            path=ratio_path(tau,channel,flavor)
            atomic_npz(path,ratio_original=r,ratio_boot=b,ratio_valid_mask=valid,valid_tau_mask=time_mask,
                c2_original=den,c2_boot=denboot,channel=np.asarray(channel),flavor_combo=np.asarray(flavor),
                vdv_labels=np.asarray(['even','odd']),axes_original=np.asarray('vdv,p,z,tsep,insertion'),
                axes_boot=np.asarray('boot,vdv,p,z,tsep,insertion'),c2_mode=np.asarray('complex_nopol'),
                gamma_matrices=SPECS.gamma,gamma_labels=SPECS.gamma_labels,
                selected_pol_indices=SPECS.pol_indices,phase_factors=np.asarray([1,-1j,1j]),
                pipeline_version=np.asarray('L32x64_A_flow_ratio_compact_v1_not_six_direction'),**common_metadata(tau))
            products.append(dict(path=str(path),sha256=sha256(path)))
    for ic in range(3):
        np.testing.assert_allclose(rb[:,ic,2],rb[:,ic,0]-rb[:,ic,1],rtol=2e-12,atol=1e-12,equal_nan=True)
    atomic_json(marker,dict(status='PASS',tau=tau,products=products,projected=str(projected),
        projected_sha256=sha256(projected),input_manifest_sha256=sha256(HERE/'input_manifest.json'),
        selected_source_payload_finite=True,full_raw_payload_crc_scan=False,
        seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID']))
    print('RATIO_PASS',tau,flush=True)


if __name__=='__main__':main()
