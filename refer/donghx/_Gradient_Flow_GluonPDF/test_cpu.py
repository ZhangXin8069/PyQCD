"""Small independent gamma, complex-bootstrap, GLS and fallback oracles."""
from common import *
from conventions import validate_gamma_basis
from ratios_cpu import compute_ratios, project_raw, SPECS
from fit_cpu import candidate_fit, select, fallback, weights, STATS
from barefit.covariance import estimate_inverse_covariance


def main():
    allocation(); manifest(); gamma = validate_gamma_basis()
    rng = np.random.default_rng(20261101)
    raw = rng.normal(size=(2,2,4,len(MOMENTA),21,4,4)) + 1j*rng.normal(size=(2,2,4,len(MOMENTA),21,4,4))
    projected = project_raw(raw, MOMENTA)
    for ch in range(3):
        for ip in (0,len(MOMENTA)-1):
            val=0j
            entries=((0,1.),) if ch==0 else (((1,-1j),) if ch==1 else ((2,.5j),(3,.5j)))
            for entry,phase in entries:
                for left in range(4):
                    for right in range(4):
                        val += phase*SPECS.gamma[ip,entry,left,right]*raw[1,0,SPECS.pol_indices[ip,entry],ip,2,left,right]
            np.testing.assert_allclose(projected[ch,1,0,ip,2],val,rtol=3e-14,atol=3e-14)
    shape=(NCONF,3,2,2,len(MOMENTA),3,6,13)
    c3=rng.normal(size=shape)+1j*rng.normal(size=shape)
    c2=3+rng.uniform(.1,1.,size=(NCONF,len(MOMENTA),6,13))+.2j
    physical=np.arange(13)[None,:]<=np.asarray(TSEPS)[:,None]
    c3[...,~physical]=np.nan+1j*np.nan;c2[...,~physical]=np.nan+1j*np.nan
    c3[:,:,:,1,:,0]=c3[:,:,:,0,:,0]
    ind=shared_indices();ro,rb,_,_=compute_ratios(c3,c2,ind)
    full=np.concatenate((c3,(c3[:,:,0]-c3[:,:,1])[:,:,None]),axis=2)
    for boot in range(NBOOT):
        with np.errstate(invalid='ignore'):
            raw_ratio=full[ind[boot]].mean(axis=0)/c2[ind[boot]].mean(axis=0)[None,None,None,:,None,:,:]
        expected=np.empty_like(rb[boot])
        for co in range(2):
            pair=.5*(raw_ratio[:,:,0]+(1 if co==0 else -1)*raw_ratio[:,:,1])
            for ip,p in enumerate(PZ):
                expected[:,:,co,ip]=pair[:,:,MOMENTA.index(p)] if p==0 else .5*(
                    pair[:,:,MOMENTA.index(p)]+(1 if co==0 else -1)*pair[:,:,MOMENTA.index(-p)])
        np.testing.assert_allclose(rb[boot],expected,rtol=2e-13,atol=3e-14,equal_nan=True)
    assert (rb[:,:,:,1,:,0,...][...,physical]==0).all()
    np.testing.assert_allclose(rb[:,:,2],rb[:,:,0]-rb[:,:,1],rtol=2e-13,atol=3e-14,equal_nan=True)
    samples=1.7+rng.normal(scale=.2,size=(NBOOT,9));central=samples.mean(axis=0)
    result=candidate_fit(central,samples);cov=estimate_inverse_covariance(samples,STATS)
    one=np.ones(9);linear=(one@cov.inverse)/(one@cov.inverse@one)
    np.testing.assert_allclose(result['M'],linear@central,rtol=2e-13)
    np.testing.assert_allclose(result['boot'],samples@linear,rtol=2e-13)
    chosen=select(np.arange(1.,5.),np.arange(1.,5.),np.arange(1.,5.),
        np.array([True,False,False,False]),np.array([.3,.1,.2,.1]),[np.array([0,1]),np.array([2,3])])
    assert chosen['fallback_indices']==[3] and chosen['value']==2.5
    assert fallback(np.array([0,1,2]),np.array([.2,.1,.1]),np.array([0.,3.,2.]))==2
    np.testing.assert_allclose(weights([2.,4.]).sum(),1.)
    try:
        candidate_fit(np.ones(3),np.ones((NBOOT,3)))
    except ValueError:
        pass
    else:
        raise AssertionError('Zero-variance data must not become a statistical fit')
    assert TMIN*A_FM>.6 and (TMIN-1)*A_FM<=.6
    assert all(ts[0]==TMIN for _,ts,_ in CANDIDATES)
    atomic_json(HERE/'status/cpu_tests.json',dict(status='PASS',job=os.environ['SLURM_JOB_ID'],gamma=gamma,
        checks=['explicit spin-index gamma oracle','all200 explicit configuration-gather complex-ratio replicas',
          'P0 once; signed-P odd alignment; NaN padding; exact odd z0; paired u-d identity',
          'independent GLS projector','AIC normalization and min-statistical-error fallback','strict physical Tmin'],
        bootstrap_indices_sha256=sha256(index_path())))
    print('CPU_TESTS_PASS',KEY,flush=True)


if __name__=='__main__':
    main()
