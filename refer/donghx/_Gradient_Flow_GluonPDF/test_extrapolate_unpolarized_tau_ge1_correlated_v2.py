import argparse
import unittest

import numpy as np

from extrapolate_unpolarized_tau_ge1_correlated_v2 import (
    FIT_TAUS, aic_weights, average_candidates, correlated_linear, extrapolate,
)


class ExtrapolationTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(9016)
        self.x=FIT_TAUS.copy()
        self.y=-.8+.15*self.x
        L=np.eye(8)*.03+np.ones((8,8))*.02
        self.boot=self.y+self.rng.normal(size=(1000,8))@L.T

    def test_original_central_and_generalized_least_squares(self):
        fit=correlated_linear(self.x,self.y,self.boot)
        np.testing.assert_allclose(fit['beta'],[-.8,.15],atol=1e-12)
        X=np.column_stack([np.ones(8),self.x])
        inv=np.linalg.inv(np.cov(self.boot,rowvar=False,ddof=1))
        direct=np.linalg.solve(X.T@inv@X,X.T@inv)
        np.testing.assert_allclose(fit['coef'],direct,atol=1e-11)
        np.testing.assert_allclose(fit['beta_boot'],self.boot@direct.T,atol=1e-11)
        self.assertEqual(fit['raw_rank'],8)
        self.assertEqual(fit['retained_rank'],8)
        self.assertEqual(fit['dof'],6)

    def test_paired_replica_intercept_shift(self):
        first=correlated_linear(self.x,self.y,self.boot)
        second=correlated_linear(self.x,self.y,self.boot+.5)
        np.testing.assert_allclose(first['beta'],second['beta'],atol=1e-12)
        np.testing.assert_allclose(second['beta_boot'][:,0]-first['beta_boot'][:,0],.5,atol=1e-11)
        np.testing.assert_allclose(second['beta_boot'][:,1],first['beta_boot'][:,1],atol=1e-11)

    def test_rank_gate_and_nonfinite_input(self):
        single=self.rng.normal(size=(1000,1))
        with self.assertRaisesRegex(ValueError,'rank'):
            correlated_linear(self.x,self.y,self.y+single)
        bad=self.boot.copy();bad[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            correlated_linear(self.x,self.y,bad)
        with self.assertRaises(ValueError):
            correlated_linear(self.x[:2],self.y[:2],self.boot[:,:2])

    def test_aic_stability_and_replica_weights(self):
        score=np.array([10000.,10002.])
        w=aic_weights(score)
        np.testing.assert_allclose(w.sum(),1.)
        np.testing.assert_allclose(w[1]/w[0],np.exp(-1))
        beta=np.array([[1.,2.],[3.,4.]])
        boot=np.broadcast_to(beta,(5,2,2)).copy()+.2
        scores=np.broadcast_to(score,(5,2)).copy()
        scores[0]=scores[0,::-1]
        central,replicas,w,wb,spread,fixed=average_candidates(beta,boot,score,scores)
        np.testing.assert_allclose(central,w@beta)
        self.assertFalse(np.allclose(wb[0],w))
        self.assertFalse(np.allclose(replicas.mean(axis=0),central))
        self.assertTrue((spread>0).all())
        np.testing.assert_allclose(fixed[1],central+.2)

    def test_contiguous_windows_and_no_silent_fallback(self):
        data={'y':np.broadcast_to(self.y[:,None,None],(8,3,25)).copy(),
              'boot':np.broadcast_to(self.boot[:,:,None,None],(1000,8,3,25)).copy(),
              'usable':np.ones((8,3,25),bool)}
        data['usable'][1,0,1]=False
        data['usable'][:,0,2]=False
        r=extrapolate(data,argparse.Namespace(svdcut=1e-10,qmin=.01,chi2max=3.))
        self.assertEqual(len(r['windows']),21)
        for iw,(lo,hi) in enumerate(r['windows']):
            if lo<=1<hi:
                self.assertFalse(r['candidate_identifiable'][iw,0,1])
        self.assertFalse(r['valid'][0,2])
        self.assertTrue(np.isnan(r['beta_boot'][:,0,2]).all())
        np.testing.assert_allclose(r['weights'].sum(axis=0)[r['valid']],1.,atol=1e-12)
        np.testing.assert_allclose(r['weights_boot'].sum(axis=1)[:,r['valid']],1.,atol=1e-12)
        np.testing.assert_allclose(r['beta'][r['valid']],np.broadcast_to([-.8,.15],(r['valid'].sum(),2)),atol=1e-10)


if __name__=='__main__':
    unittest.main()
