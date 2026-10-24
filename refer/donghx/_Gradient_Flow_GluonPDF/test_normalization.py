import os
import unittest
import numpy as np
from normalization import local_normalize


class PairedLocalNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.m = np.arange(1., 25.).reshape(2, 3, 4)
        self.m[1, :, 0] = 0
        self.b = np.stack([self.m * (1 + j*.03) + j*.001 for j in range(8)], axis=1)
        self.b[1, :, :, 0] = 0

    def test_independent_scalar_oracle_and_ddof(self):
        result = local_normalize(self.m, self.b)
        oracle = np.empty_like(self.b)
        for c in range(2):
            for b in range(8):
                for p in range(3):
                    for z in range(4):
                        oracle[c,b,p,z] = self.b[c,b,p,z]/self.b[0,b,p,0]
        np.testing.assert_array_equal(result['H_local_boot'], oracle)
        np.testing.assert_array_equal(result['statistical_error'], np.std(oracle, axis=1, ddof=1))
        np.testing.assert_array_equal(result['H_local'], self.m/self.m[0,:,0][None,:,None])

    def test_replica_dependent_common_Z_cancels(self):
        original = local_normalize(self.m, self.b)
        factor = np.linspace(.3, 7., 8)[None,:,None,None]
        transformed = local_normalize(self.m*2.1, self.b*factor)
        np.testing.assert_allclose(original['H_local'], transformed['H_local'], atol=1e-14)
        np.testing.assert_allclose(original['H_local_boot'], transformed['H_local_boot'], atol=1e-14)

    def test_anchor_is_constraint_and_odd_zero_not_divisor(self):
        result = local_normalize(self.m, self.b)
        np.testing.assert_array_equal(result['H_local'][0,:,0], 1)
        np.testing.assert_array_equal(result['H_local_boot'][0,:,:,0], 1)
        np.testing.assert_array_equal(result['statistical_error'][:,:,0], 0)
        np.testing.assert_array_equal(result['H_local'][1,:,0], 0)

    def test_zero_and_nonfinite_not_dropped_or_clipped(self):
        self.b[0, 2, 1, 0] = 0
        self.b[0, 4, 2, 0] = np.nan
        result = local_normalize(self.m, self.b)
        self.assertTrue(np.isnan(result['H_local_boot'][:,2,1,:]).all())
        self.assertTrue(np.isnan(result['H_local_boot'][:,4,2,:]).all())
        self.assertFalse(result['selected_available'][:,1:,:].any())
        self.assertTrue(np.isnan(result['statistical_error'][:,1:,:]).all())

    def test_sign_flips_kept(self):
        self.b[0, 1, 0, 0] *= -1
        result = local_normalize(self.m, self.b)
        self.assertLess(result['H_local_boot'][0,1,0,1], 0)
        self.assertTrue(result['selected_available'].all())

    def test_shapes_rejected(self):
        for b in [self.b.transpose(1,0,2,3), self.b[:,:,:2], self.b[:,:1]]:
            with self.assertRaises(ValueError):
                local_normalize(self.m, b)

    def test_does_not_impose_flavor_difference_after_normalization(self):
        u, d = self.m.copy(), self.m.copy()*.3
        d[:,:,1:] *= .5
        ub, db = self.b.copy(), self.b.copy()*.3
        db[:,:,:,1:] *= .5
        ud = local_normalize(u-d, ub-db)['H_local']
        wrong = local_normalize(u,ub)['H_local']-local_normalize(d,db)['H_local']
        self.assertFalse(np.allclose(ud, wrong))


if __name__ == '__main__':
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Numerical tests must run in a CPU allocation')
    unittest.main(verbosity=2)
