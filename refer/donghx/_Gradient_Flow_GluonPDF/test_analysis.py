import unittest
import numpy as np

from .analyze import observables, paired_noise_statistics


class PairedMassStatistics(unittest.TestCase):
    def setUp(self):
        self.tau = np.array([.05, .5, 1.])
        self.base = -np.array([2., .12, .03])[:, None]*np.array([.8, .9, 1., 1.1, 1.2])[None, :]
        self.values = np.stack([self.base, self.base*1.02, self.base*.98, self.base*1.04,
                                self.base*1.1]).astype(complex)

    def test_shared_noise_cancels_in_mass_ratio(self):
        stats, errors, replicas = paired_noise_statistics(self.values, self.tau)
        np.testing.assert_allclose(stats['Z_relative_plus_sigma'], 1/.98-1)
        np.testing.assert_allclose(errors['Z_relative_plus_sigma'], 0, atol=1e-14)
        self.assertGreater(errors['Z_single_conf_diagnostic'][0, 0], 0)
        self.assertEqual(replicas['Z_single_conf_diagnostic'].shape, (5, 5, 3))

    def test_inverse_after_mean_and_sem(self):
        stats, errors, _ = paired_noise_statistics(self.values, self.tau)
        expected = -6/((4*np.pi)**2*self.tau**2*self.base.mean(axis=-1))
        np.testing.assert_allclose(stats['Z_single_conf_diagnostic'][0], expected)
        wrong = (-6/((4*np.pi)**2*self.tau[:, None]**2*self.base)).mean(axis=-1)
        self.assertFalse(np.allclose(expected, wrong))
        np.testing.assert_allclose(errors['E_real_mean'][0], self.base.std(axis=-1, ddof=1)/np.sqrt(5))

    def test_invalid_or_missing_samples_rejected(self):
        bad = self.values.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            observables(bad, self.tau)
        with self.assertRaises(ValueError):
            paired_noise_statistics(self.values[:, :, :2], self.tau)
        with self.assertRaises(ValueError):
            observables(-self.values, self.tau)


if __name__ == '__main__':
    unittest.main()
