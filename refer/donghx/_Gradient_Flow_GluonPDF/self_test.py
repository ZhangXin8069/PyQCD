"""CPU gate for covariance, shared block jackknife and slope conventions."""
import numpy as np
from analysis import jackknife,jk_error,gls

def test():
 rng=np.random.default_rng(20261114);n=np.arange(2,9);design=np.column_stack([np.ones(len(n)),-n])
 p=np.array([.13,.21]);mean=design@p
 noise=rng.normal(size=(100,len(n)));noise-=noise.mean(axis=0)
 x=mean+noise*.002
 for block in [1,2,5]:
  rep=jackknife(x,block);fit,pr,diag=gls(x.mean(axis=0),rep,n)
  assert np.max(abs(fit-p))<1e-12
  assert diag['dof']>0 and diag['model_rank']==2
  assert np.all(jk_error(pr)>0)
  manual=np.stack([np.delete(x,np.arange(i*block,(i+1)*block),axis=0).mean(axis=0) for i in range(100//block)])
  assert np.allclose(manual,rep,atol=1e-14,rtol=0)
 # Deliberately rank-one features must not yield a reportable two-parameter fit.
 rankone=np.arange(100)[:,None]*np.ones((1,len(n)))
 try:gls(mean,jackknife(rankone,1),n)
 except AssertionError:pass
 else:raise AssertionError('rank-deficient covariance accepted')
 return dict(status='PASS',tests=['GLS slope sign/units','delete-block jackknife normalization','shared replicas','rank-deficient rejection'])
if __name__=='__main__': print(test())
