"""Independent small-tensor gates before a production-size GPU comparison."""
import sys
from pathlib import Path
import numpy as np
from opt_einsum import contract
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from flow_common_l32x96 import spin_objects, gamma_matrices
from flowed_perambulator_factorization import DIRECT_TIME_EQUATIONS, build_flowed_sequential
from spin2_contractions import Contractor,parity_basis,reduced_projectors,c2_reference,c2_reduced,c3_reduced

def main():
    rng=np.random.default_rng(20260922)
    def rand(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    c=Contractor('greedy');g,_,q=spin_objects();U=parity_basis()
    np.testing.assert_allclose(U.conj().T@U,np.eye(2),atol=1e-14)
    np.testing.assert_allclose(U@U.conj().T,q[0],atol=1e-14)
    # Complex arbitrary unitary basis ensures hidden transpose errors do not pass.
    rotation=np.linalg.qr(rand((2,2)))[0]
    for basis in [U,U@rotation]:
        q2=reduced_projectors(q,basis,c)
        np.testing.assert_allclose(c('ks,Psr,ar->Pka',basis,q2,basis.conj()),q,atol=1e-14)
        # A nontrivial complex external spin tensor reproduces pol25 too.
        raw=rand((4,4));small=c('ar,ak,ks->rs',basis.conj(),raw,basis)
        np.testing.assert_allclose(c('Pka,ak->P',q,raw),c('Psr,rs->P',q2,small),atol=1e-13)
        nt,ns,nc,nmom=2,3,2,2
        p=rand((4,4,ns,ns));a=rand((nt,4,4,ns,nc));b=rand((nt,4,4,nc,ns))
        j=rand((nt,nt,2,3,nc,nc));vs=rand((nmom,ns,ns,ns));vr=rand(vs.shape)
        np.testing.assert_allclose(c2_reference(p,vs,vr,g,q,c),c2_reduced(p,vs,vr,g,q,c,basis),rtol=1e-11,atol=1e-9)
        h=build_flowed_sequential(j,a,c)
        u,d,terms=c3_reduced(h,p,b,g,q,vs,vr,c,basis,True)
        arrays=dict(p=p,a=a,b=b,j=j,g=g,q=q,vs=vs,vr=vr)
        ref={'u':np.zeros_like(u),'d':np.zeros_like(d)}
        for name,flavor,sign,(eq,keys) in DIRECT_TIME_EQUATIONS:
            expected=contract(eq,*(arrays[k] for k in keys),optimize='greedy').transpose(1,0,2,3,4,5)
            np.testing.assert_allclose(terms[name],expected,rtol=2e-11,atol=1e-8)
            ref[flavor]+=sign*expected
        for name,value in [('u',u),('d',d)]:
            err=np.linalg.norm(value-ref[name])/np.linalg.norm(ref[name])
            assert err<2e-12,err
            print(name,'relative L2',err)
    print('PASS: projector identities, complex basis rotation, C2, each of six direct double-time C3 terms, both flavors/all polarizations.')

if __name__=='__main__':main()
