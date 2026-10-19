"""CPU checks of reconstruction, GLS and Fourier conventions."""
import unittest
from types import SimpleNamespace
import numpy as np
from lambda_fourier_quasi_matching_v2 import (
    amplitude_fit, shape, reconstruction_basis, quadrature_nodes,
    fit_tail, coefficients, transform,
)


class PipelineTests(unittest.TestCase):
    def test_correlated_gls_and_original_center(self):
        rng=np.random.default_rng(141)
        phi=np.array([1.,.8,.5,.3])
        boot=2.1*phi+rng.normal(size=(1000,4))*.03
        y=2.*phi
        r=amplitude_fit(phi,y,boot)
        cov=np.cov(boot,rowvar=False,ddof=1)
        inv=np.linalg.inv(cov)
        coef=(phi@inv)/(phi@inv@phi)
        np.testing.assert_allclose(r['coef'],coef,rtol=1e-12)
        np.testing.assert_allclose(r['Ab'],boot@coef,rtol=1e-12)
        self.assertAlmostEqual(r['A'],2.)
        self.assertGreater(abs(r['Ab'].mean()-r['A']),.05)
        self.assertEqual(r['dof'],3)
        self.assertLess(r['chi'],1e-20)

    def test_singular_covariance_is_reported(self):
        rng=np.random.default_rng(6)
        phi=np.array([1.,.7,.3])
        u=rng.normal(size=(1000,2))
        boot=u@np.array([[1.,.3,.2],[.2,1.,.7]])
        r=amplitude_fit(phi,phi,boot)
        self.assertEqual(r['raw_rank'],2)
        self.assertEqual(r['rank'],2)
        self.assertEqual(r['dof'],1)

    def test_fourier_normalization_against_analytic_exponential(self):
        nodes,w=quadrature_nodes(np.arange(25)*.6,160,8,1.5,2.,.25,8)
        x=np.linspace(-3,3,91);decay=.7
        actual=(np.cos(x[:,None]*nodes)*w)@np.exp(-decay*nodes)/np.pi
        expected=decay/(np.pi*(decay**2+x*x))
        np.testing.assert_allclose(actual,expected,atol=2e-14)

    def test_reconstruction_local_value_and_tail_boundary(self):
        lam=np.arange(25)*.6;registry=np.array([[4.,1.,8.]])
        h=np.exp(-lam/4);amp=.2
        nodes=np.array([0.,7.,8.-1e-6,8.,8.+1e-6,50.])
        D,T=reconstruction_basis(nodes,lam,registry)
        v=D@h+T[:,0]*amp
        self.assertEqual(v[0],h[0])
        np.testing.assert_allclose(v[3:],amp*shape(nodes[3:],1.,8.))
        self.assertLess(abs((v[3]-v[2])/1e-6-(v[4]-v[3])/1e-6),1e-5)

    def test_selection_and_tree_level_matching(self):
        rng=np.random.default_rng(13)
        lam=np.arange(25)*2*np.pi*3/32;z=np.arange(25)
        h0=np.exp(-lam/5)
        boot=h0+rng.normal(size=(40,25))*.05;boot[:,0]=1.
        h=np.vstack([h0,boot])
        args=SimpleNamespace(svdcut=1e-10,qmin=1e-6,chi2max=10.,
                             mu=2.,zs_fm=.25,alpha_s=0.,xgap=.02)
        r=fit_tail(lam,z,h,args)
        np.testing.assert_allclose(r['weights_boot'].sum(axis=1),1.)
        self.assertEqual(coefficients(r).shape,(41,27))
        x=np.array([-.3,-.1,0.,.1,.3]);y=np.linspace(-3,3,601)
        out=transform(h,r,lam,1.5,x,y,args)
        mask=out['computed_mask']
        np.testing.assert_allclose(out['matched'][:,mask],out['quasi_ms'][:,mask])
        self.assertTrue(np.isnan(out['matched'][:,~mask]).all())
        np.testing.assert_allclose(out['quasi_ms'],out['quasi_ms'][:,::-1],atol=1e-14)


if __name__=='__main__':
    unittest.main()
