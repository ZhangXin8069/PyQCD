"""Explicit current-spin and post-ratio displacement/momentum projections."""
import numpy as np
from common import MOMENTA, PZ, NCONF, NBOOT
from conventions import make_channel_specs,PHASE_FACTORS

SPECS=make_channel_specs([f'P{"p" if p>=0 else "m"}{abs(p)}_z' for p in MOMENTA],
                         ['z']*len(MOMENTA),['nopol','pol15','pol25','pol35'])


def project_spin(c3):
    assert c3.shape[:4]==(2,2,4,len(MOMENTA)) and c3.shape[-2:]==(4,4)
    out=np.empty((3,2,2,len(MOMENTA),c3.shape[4]),complex)
    for ip in range(len(MOMENTA)):
        blocks=[np.einsum('fszmo,mo->fsz',c3[:,:,int(SPECS.pol_indices[ip,ch]),ip],
                          SPECS.gamma[ip,ch],optimize=True) for ch in range(4)]
        out[0,:,:,ip]=blocks[0]
        out[1,:,:,ip]=PHASE_FACTORS['helicity']*blocks[1]
        out[2,:,:,ip]=.5*PHASE_FACTORS['transversity']*(blocks[2]+blocks[3])
    return out


def ratio_from_projected(c3,c2,indices):
    assert c3.shape[0]==c2.shape[0]==NCONF
    assert c3.shape[3:5]==(2,len(MOMENTA)) and c2.shape[1]==len(MOMENTA)
    assert c3.shape[-2:]==c2.shape[-2:] and indices.shape==(NBOOT,NCONF)
    weights=np.stack([np.bincount(row,minlength=NCONF)/NCONF for row in indices])
    full=np.concatenate((c3,(c3[:,:,0]-c3[:,:,1])[:,:,None]),axis=2)
    mean=full.mean(axis=0); den=c2.mean(axis=0)
    boots=np.tensordot(weights,full,axes=(1,0));denboot=np.tensordot(weights,c2,axes=(1,0))
    physical=np.isfinite(c2).all(axis=0)
    if (np.any(den[physical]==0) or np.any(denboot[:,physical]==0)
            or not np.isfinite(denboot[:,physical]).all()):
        raise ValueError('zero/nonfinite complex matched C2; no clipping or real-only replacement')
    raw=np.full_like(mean,np.nan+1j*np.nan);rawboot=np.full_like(boots,np.nan+1j*np.nan)
    np.divide(mean,den[None,None,None,:,None,:,:],out=raw,
              where=physical[None,None,None,:,None,:,:])
    np.divide(boots,denboot[:,None,None,None,:,None,:,:],out=rawboot,
              where=physical[None,None,None,None,:,None,:,:])
    directional=np.stack((.5*(raw[:,:,0]+raw[:,:,1]),.5*(raw[:,:,0]-raw[:,:,1])),axis=2)
    directionboot=np.stack((.5*(rawboot[:,:,:,0]+rawboot[:,:,:,1]),
                            .5*(rawboot[:,:,:,0]-rawboot[:,:,:,1])),axis=3)
    parity=np.asarray([1.,-1.])
    values,samples=[],[]
    for p in PZ:
        ip=MOMENTA.index(p)
        if p==0:
            values.append(directional[:,:,:,ip])
            samples.append(directionboot[:,:,:,:,ip])
        else:
            im=MOMENTA.index(-p)
            values.append(.5*(directional[:,:,:,ip]+
                directional[:,:,:,im]*parity[None,None,:,None,None,None]))
            samples.append(.5*(directionboot[:,:,:,:,ip]+
                directionboot[:,:,:,:,im]*parity[None,None,None,:,None,None,None]))
    return np.stack(values,axis=3),np.stack(samples,axis=4),den,denboot,directional,directionboot
