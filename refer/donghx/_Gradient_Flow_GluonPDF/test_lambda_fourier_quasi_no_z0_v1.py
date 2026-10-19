"""Small independent algebra/numerics checks; production runs separately."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad
from gluon_absolute_normalization_v1 import kinematic_factor,local_spin2_coefficient,fit_c2_window
from lambda_fourier_quasi_no_z0_v1 import coordinate_basis,transform,c0_coefficient
from lambda_fourier_quasi_matching_v2 import fit_tail


class Normalization(unittest.TestCase):
    def test_even_trace_and_kinematics(self):
        E=np.array([.74,.91,1.1]);g2=4*np.pi*.25;I=.43
        # Oadj=-2 E^2 I; even-sum 2 times fundamental trace 1/2.
        ratio=2*.5*g2*(-2*E**2*I)/(2*E)
        np.testing.assert_allclose(kinematic_factor(E,.25)*ratio,2*I)

    def test_local_poles_and_msbar_log(self):
        eps=1e-4;t=1.4*(.0775/.1973269804)**2;mu=2.;alpha=.25
        scale=alpha*3/(4*np.pi)
        # zeta*Z at first order, never multiplying two truncated terms exactly.
        delta_zeta=scale*((11/3)*(1/eps+np.log(2*mu*mu*t)+np.euler_gamma)+7/3)
        delta_Z=scale*(-11/3)/eps
        self.assertAlmostEqual(1+delta_zeta+delta_Z,local_spin2_coefficient(1.4,.0775,mu,alpha),places=11)
        self.assertAlmostEqual(local_spin2_coefficient(1.4,.0775,mu,0),1.)

    def test_full_line_fourier_factor_two(self):
        # H(lambda)=2 A exp[-s^2 lambda^2/2]. Its 1/pi cosine
        # transform integrates over positive x to A, not 2A or 1.
        A=.43;s=.4
        for x in [.1,.3,.7]:
            v=quad(lambda l:2*A*np.exp(-s*s*l*l/2)*np.cos(x*l)/np.pi,0,40,epsabs=1e-11)[0]
            expected=2*A/(np.sqrt(2*np.pi)*s)*np.exp(-x*x/(2*s*s))
            self.assertAlmostEqual(v,expected,places=10)

    def test_local_boundary_is_not_forced_to_one(self):
        args=SimpleNamespace(mu=2.,zs_fm=.25,alpha_s=.25)
        lam=np.arange(25)*.6;nodes=np.array([0.,lam[1]/2,lam[1]])
        D,T,DC,TC=coordinate_basis(nodes,lam,np.array([[4,.5,4]]),1.5,args)
        H=np.linspace(.86,.01,25);b=D@H-args.alpha_s*3/(2*np.pi)*(DC@H)
        self.assertAlmostEqual(b[0],.86)
        self.assertAlmostEqual(b[-1],H[1]*(1-args.alpha_s*3/(2*np.pi)*c0_coefficient(lam[1]/1.5,2.,.25)))
        self.assertTrue(np.isfinite(b).all())

    def test_correlated_c2(self):
        rng=np.random.default_rng(724);t=np.arange(8,15);central=1.3*np.exp(-.77*(t-8))
        b=central[None]*(1+rng.normal(size=(80,7))*.01)
        f=fit_c2_window(t,np.vstack([central,b]))
        self.assertAlmostEqual(f['beta'][0,0],.77,places=6)
        self.assertAlmostEqual(f['beta'][0,1],1.3,places=6)
        self.assertEqual(f['rank'],7)

    def test_pipeline_keeps_amplitude(self):
        rng=np.random.default_rng(982);z=np.arange(25);lam=z*.6
        central=.7*np.exp(-lam/6);b=central[None]*(1+rng.normal(size=(60,25))*.08)
        h=np.vstack([central,b]);args=SimpleNamespace(mu=2.,zs_fm=.25,alpha_s=.25,
            svdcut=1e-10,qmin=1e-6,chi2max=10.,xgap=.02)
        tail=fit_tail(lam,z,h,args);twice=fit_tail(lam,z,2*h,args)
        np.testing.assert_allclose(tail['weights_boot'],twice['weights_boot'],atol=1e-10)
        x=np.array([-.4,-.2,.2,.4]);y=np.linspace(-3,3,121)
        a=transform(h,tail,lam,1.5,x,y,args,stop=40.,order=4,kernel_order=12)
        b=transform(2*h,twice,lam,1.5,x,y,args,stop=40.,order=4,kernel_order=12)
        np.testing.assert_allclose(b['matched'],2*a['matched'],atol=1e-10)
        np.testing.assert_allclose(a['matched'],a['matched'][:,::-1],atol=1e-10)


if __name__=='__main__':unittest.main()
