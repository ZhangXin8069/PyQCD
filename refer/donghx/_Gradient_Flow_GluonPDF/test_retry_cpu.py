"""No-GPU regression for the observed native observation-buffer overrun."""
from types import SimpleNamespace
import unittest
from common import HERE,rows,require_allocation,atomic_json
from flow_safety import configure_initial_only,validate_initial_only


class ObservationSafety(unittest.TestCase):
    def test_reproduce_old_failure(self):
        old=SimpleNamespace(n_steps=40,epsilon=.006698097814776275,meas_interval=41)
        validate_initial_only(old)
        old.n_steps*=2
        old.epsilon/=2
        self.assertEqual(old.n_steps//old.meas_interval,1)
        with self.assertRaises(RuntimeError):
            validate_initial_only(old)

    def test_every_physical_flow_and_halved_step(self):
        for row in rows()[:7]:
            smear=SimpleNamespace()
            for factor in (1,2,1):
                configure_initial_only(smear,factor*row['nstep'],row['epsilon']/factor)
                self.assertEqual(smear.n_steps//smear.meas_interval,0)
                self.assertAlmostEqual(smear.n_steps*smear.epsilon,row['flow_time_a2'],places=14)

    def test_invalid_parameters(self):
        for steps,eps in ((0,.01),(-1,.01),(1.5,.01),(True,.01),(40,0.),(40,float('nan'))):
            with self.assertRaises(ValueError):
                configure_initial_only(SimpleNamespace(),steps,eps)


if __name__=='__main__':
    require_allocation()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ObservationSafety))
    atomic_json(HERE/'status/retry_tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        scope='CPU reproduction of stale meas_interval and single-observable guard; not a GPU regression'))
    raise SystemExit(0 if result.wasSuccessful() else 1)

