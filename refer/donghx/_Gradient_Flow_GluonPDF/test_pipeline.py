import unittest
import numpy as np
from .pipeline import paired


class PairingTests(unittest.TestCase):
    def test_common_resamples_preserve_exact_cancellation(self):
        values=-np.array([[1.,2.,4.],[2.,3.,7.]])
        idx=np.array([[0,0,1],[0,1,2],[2,2,1],[1,1,0]])
        flow=np.array([.2,.5])
        eb=values[:,idx].mean(axis=2)
        central=values.mean(axis=1)[:,None,None]
        boot=eb[:,:,None,None]
        d=paired(values,flow,idx,central,boot)
        expected=-6/((4*np.pi)**2*flow**2)
        np.testing.assert_allclose(d['ringed_central'][:,0,0],expected)
        np.testing.assert_allclose(d['ringed_error'],0,atol=1e-14)
        wrong=paired(values,flow,idx,central,boot[:,::-1])
        self.assertGreater(float(wrong['ringed_error'].max()),.01)
        self.assertFalse(np.allclose(d['Zchi'],(-6/((4*np.pi)**2*flow**2))*(1/values).mean(axis=1)))

    def test_invalid_indices_sign_and_missing_configuration_are_rejected(self):
        v=-np.ones((2,3));f=np.array([.2,.5]);i=np.array([[0,1,2],[2,1,0]])
        m=np.ones((2,1,1));b=np.ones((2,2,1,1))
        for bad in (i.astype(float),np.array([[0,1,3],[0,1,2]]),i[:,:2]):
            with self.assertRaises(ValueError):paired(v,f,bad,m,b)
        with self.assertRaises(ValueError):paired(-v,f,i,m,b)


if __name__=='__main__':unittest.main()
