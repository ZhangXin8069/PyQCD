import unittest
from pathlib import Path

import numpy as np


RESULT = Path(__file__).resolve().parents[1] / "results_v1/flow_time_tau_gt_0p1_v1.npz"


class TauGreaterThanPointOneContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(RESULT, allow_pickle=False)

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_shared_resampling_metadata(self):
        self.assertEqual(int(self.data["nconf"]), 406)
        self.assertEqual(int(self.data["nboot"]), 1500)
        self.assertEqual(int(self.data["seed"]), 1115)
        self.assertEqual(self.data["bootstrap_indices"].shape, (1500, 406))

    def test_tau_point_one_is_never_fitted(self):
        tau_index = int(np.flatnonzero(np.isclose(self.data["flow_taus"], 0.1))[0])
        self.assertFalse(np.any(self.data["fit_tau_mask"][..., tau_index]))

    def test_geometry_examples(self):
        modes = self.data["mode_names"].astype(str)
        windows = self.data["window_names"].astype(str)
        im = int(np.flatnonzero(modes == "geometry_guarded")[0])
        iw = int(np.flatnonzero(windows == "tau0p2_3p8")[0])
        iz2 = int(np.flatnonzero(self.data["z_values"] == 2)[0])
        iz6 = int(np.flatnonzero(self.data["z_values"] == 6)[0])
        self.assertEqual(self.data["flow_taus"][self.data["fit_tau_mask"][im, iw, iz2]].tolist(),
                         [0.2, 0.3, 0.4])
        self.assertEqual(np.count_nonzero(self.data["fit_tau_mask"][im, iw, iz6]), 13)

    def test_estimator_branches_remain_separate(self):
        self.assertEqual(self.data["estimator_names"].astype(str).tolist(),
                         ["accepted_only", "filled_fallback_diagnostic"])
        self.assertEqual(self.data["intercept"].shape[:2], (2, 2))


if __name__ == "__main__":
    unittest.main()
