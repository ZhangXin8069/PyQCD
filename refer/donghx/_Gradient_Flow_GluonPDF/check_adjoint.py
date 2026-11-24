"""Independent 8x8 adjoint matrices versus local SU(3) character formula."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'numerics require CPU allocation'
    import numpy as np
    rng=np.random.default_rng(2026111196)
    def hermitian_basis(i,j,imag=False):
        m=np.zeros((3,3),complex)
        m[i,j]=-1j if imag else 1
        m[j,i]=1j if imag else 1
        return m/2
    generators=[hermitian_basis(0,1),hermitian_basis(0,1,True),np.diag([1.,-1.,0.])/2,
                hermitian_basis(0,2),hermitian_basis(0,2,True),
                hermitian_basis(1,2),hermitian_basis(1,2,True),np.diag([1.,1.,-2.])/(2*np.sqrt(3))]
    t=np.asarray(generators)
    assert np.allclose(np.einsum('aij,bji->ab',t,t),np.eye(8)/2,rtol=0,atol=1e-15)
    def su3():
        q,r=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
        q=q*np.diag(r).conj()[None,:]/abs(np.diag(r))[None,:]
        q[:,0]/=np.linalg.det(q)
        assert np.max(abs(q@q.conj().T-np.eye(3)))<2e-15
        return q
    def adjoint(u):
        # U T^b U^dagger = sum_a T^a D_ab(U), Tr(T^a T^b)=delta_ab/2.
        return 2*np.einsum('aij,jk,bkl,il->ab',t,u,t,u.conj())
    max_character,max_product=0.,0.
    for length in (0,1,2,5,12):
        for _ in range(16):
            uf,ua=np.eye(3,dtype=complex),np.eye(8,dtype=complex)
            for __ in range(length):
                link=su3();uf=uf@link;ua=ua@adjoint(link)
            product_error=float(np.max(abs(ua-adjoint(uf))))
            error=float(abs(np.trace(ua)-(abs(np.trace(uf))**2-1)))
            assert product_error<2e-13 and error<2e-13
            max_character=max(max_character,error);max_product=max(max_product,product_error)
    local_traces=np.asarray([np.trace(su3()) for _ in range(40)])
    correct=float(np.mean((abs(local_traces)**2-1)/8))
    incorrect=float((abs(local_traces.mean())**2-1)/8)
    variance=float(np.mean(abs(local_traces-local_traces.mean())**2)/8)
    assert np.isclose(correct-incorrect,variance,atol=1e-15,rtol=1e-13)
    assert variance>1e-3
    result=dict(status='PASS_EXPLICIT_ADJOINT_CHARACTER_AND_PATH_PRODUCT',
        max_unnormalized_character_error=max_character,max_adjoint_path_matrix_error=max_product,
        zero_length_F=1.,zero_length_A=1.,local_average=correct,wrong_average_then_square=incorrect,
        difference_is_local_trace_variance_over_8=variance,seed=2026111196,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        job=os.environ['SLURM_JOB_ID'])
    (ROOT/'adjoint_contract.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
