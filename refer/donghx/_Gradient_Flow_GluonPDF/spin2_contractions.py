"""Exact external parity reduction for the existing flowed C2/C3 equations.

P[k,a]=U[k,s] q[P,s,r] U*[a,r].  Only external baryon indices a,k
are reduced. Current m,o and internal diquark spin indices remain four.
All contractions retain the original six topology signs and open-current axes.
"""
from functools import lru_cache
import numpy as np
from opt_einsum import contract_expression


class Contractor:
    def __init__(self,optimize='optimal'):
        self.optimize=optimize
        self.expressions={}

    def __call__(self,eq,*arrays):
        key=(eq,tuple(a.shape for a in arrays))
        if key not in self.expressions:
            self.expressions[key]=contract_expression(eq,*key[1],optimize=self.optimize)
        return self.expressions[key](*arrays)


def parity_basis(xp=np):
    return xp.asarray(np.vstack([np.eye(2),np.eye(2)])/np.sqrt(2),dtype=xp.complex128)


def reduced_projectors(projectors,basis,contract):
    return contract('ks,Pka,ar->Psr',basis.conj(),projectors,basis)


def external_perambulators(p,basis,contract):
    left=contract('ar,agbh->rgbh',basis.conj(),p)
    right=contract('ckdl,ks->csdl',p,basis)
    both=contract('rkbl,ks->rsbl',left,basis)
    return left,right,both


def c2_reference(p,vs,vr,g,q,contract):
    return (contract('eofp,ambn,cgdh,ce,mo,Mbdf,Mnph,Pga->PM',p,p,p,g,g,vs,vr,q)
            -contract('eofp,agbh,cmdn,ce,mo,Mbdf,Mnph,Pga->PM',p,p,p,g,g,vs,vr,q))


def c2_reduced(p,vs,vr,g,q,contract,basis=None):
    xp=__import__(type(p).__module__.split('.')[0])
    basis=parity_basis(xp) if basis is None else basis
    q2=reduced_projectors(q,basis,contract)
    left,right,both=external_perambulators(p,basis,contract)
    return (contract('eofp,rmbn,csdh,ce,mo,Mbdf,Mnph,Psr->PM',p,left,right,g,g,vs,vr,q2)
            -contract('eofp,rsbh,cmdn,ce,mo,Mbdf,Mnph,Psr->PM',p,both,p,g,g,vs,vr,q2))


# name, flavor, coefficient, right equation, operand keys, final equation, H type
SPECS=(
 ('u1','u',-1,'eifj,cgdh,uospl,ce,gi,Mbdf,Mhjl->Mbuosp',
  ('p','p','br','g','g','vs','vr'),'uSLprmb,Mbuosp,Psr->PSMLmo','hl'),
 ('u2','u',+1,'eifj,csdl,uogph,ce,gi,Mbdf,Mhjl->Mbsuop',
  ('p','pr','b','g','g','vs','vr'),'uSLprmb,Mbsuop,Psr->PSMLmo','hl'),
 ('u3','u',+1,'eifj,rgbh,uospl,ce,gi,Mbdf,Mhjl,Psr->McdPuop',
  ('p','pl','br','g','g','vs','vr','q'),'uSLpcmd,McdPuop->PSMLmo','h'),
 ('u4','u',-1,'eifj,rsbl,uogph,ce,gi,Mbdf,Mhjl,Psr->cMPduop',
  ('p','pb','b','g','g','vs','vr','q'),'uSLpcmd,cMPduop->PSMLmo','h'),
 ('d1','d',+1,'uoipj,rgbh,csdl,ce,gi,Mbdf,Mhjl,Psr->eMfPuop',
  ('b','pl','pr','g','g','vs','vr','q'),'uSLpemf,eMfPuop->PSMLmo','h'),
 ('d2','d',-1,'uoipj,rsbl,cgdh,ce,gi,Mbdf,Mhjl,Psr->eMPfuop',
  ('b','pb','p','g','g','vs','vr','q'),'uSLpemf,eMPfuop->PSMLmo','h'),
)


def c3_reduced(h,p,b,g,q,vs,vr,contract,basis=None,return_terms=False):
    """One direction, arbitrary momentum batch. Output [sign,pol,mom,z,m,o].

    h [tau_b,sign,z,eigen_current,sink_spin,current_m,eigen_src]
    p [sink_spin,source_spin,eigen_sink,eigen_src]
    b [tau_b,current_o,source_spin,eigen_current,eigen_src]
    vs,vr [momentum,eigen,eigen,eigen]; vr already conjugated.
    Costs of all parity projections are included in this callable.
    """
    xp=__import__(type(p).__module__.split('.')[0])
    basis=parity_basis(xp) if basis is None else basis
    pl,pr,pb=external_perambulators(p,basis,contract)
    br=contract('uokpl,ks->uospl',b,basis)
    hl=contract('uSLpamb,ar->uSLprmb',h,basis.conj())
    q2=reduced_projectors(q,basis,contract)
    arrays=dict(p=p,b=b,pl=pl,pr=pr,pb=pb,br=br,g=g,q=q2,vs=vs,vr=vr,h=h,hl=hl)
    shape=(h.shape[1],q.shape[0],vs.shape[0],h.shape[2],4,4)
    u=xp.zeros(shape,dtype=xp.complex128);d=xp.zeros_like(u);terms={}
    for name,flavor,sign,eq,keys,final,hkey in SPECS:
        right=contract(eq,*(arrays[k] for k in keys))
        for im in range(vs.shape[0]):
            axis=eq.split('->')[1].index('M')
            slices=[slice(None)]*right.ndim;slices[axis]=slice(im,im+1)
            args=(arrays[hkey],right[tuple(slices)])
            if 'Psr' in final:args+= (q2,)
            value=contract(final,*args)[:,:,0].transpose(1,0,2,3,4)
            (u if flavor=='u' else d)[:,:,im]+=sign*value
            if return_terms:
                if name not in terms:terms[name]=xp.empty(shape,dtype=xp.complex128)
                terms[name][:,:,im]=value
            del value
        del right
    return (u,d,terms) if return_terms else (u,d)


def accumulate_triplet_reduced(flowed_by_direction, momentum_directions,
                              p, b, g, q, vs, vr, u_accum, d_accum,
                              momentum_indices, contract):
    """Production interface; reuse each right factor across the momentum batch.

    Preserve direction mapping and arbitrary global momentum indices exactly.
    Only one direction's projected H is retained at a time to bound memory.
    """
    if len(momentum_directions) != len(momentum_indices) or vs.shape[0] != len(momentum_indices):
        raise ValueError('momentum direction/index/VVV sizes disagree')
    xp=__import__(type(p).__module__.split('.')[0])
    basis=parity_basis(xp)
    pl,pr,pb=external_perambulators(p,basis,contract)
    br=contract('uokpl,ks->uospl',b,basis)
    q2=reduced_projectors(q,basis,contract)
    arrays=dict(p=p,b=b,pl=pl,pr=pr,pb=pb,br=br,g=g,q=q2,vs=vs,vr=vr)
    for name,flavor,sign,eq,keys,final,hkey in SPECS:
        right=contract(eq,*(arrays[k] for k in keys))
        for im,(direction,global_im) in enumerate(zip(momentum_directions,momentum_indices)):
            h=flowed_by_direction[direction]
            hh=contract('uSLpamb,ar->uSLprmb',h,basis.conj()) if hkey=='hl' else h
            slices=[slice(None)]*right.ndim
            slices[eq.split('->')[1].index('M')]=slice(im,im+1)
            args=(hh,right[tuple(slices)])
            if 'Psr' in final: args+=(q2,)
            value=contract(final,*args)[:,:,0].transpose(1,0,2,3,4)
            (u_accum if flavor=='u' else d_accum)[:,:,global_im]+=sign*value
            del value,hh,args
        del right
