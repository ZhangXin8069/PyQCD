"""Safety contract for the single-observable performGFlowQuda binding.

The native routine indexes obs_param[0] initially and obs_param[++n] at
each meas_interval. GaugeDirac supplies ONE observable, not an array. Keep
meas_interval strictly larger than the actual n_steps on EVERY call.
"""
import math


def configure_initial_only(smear, n_steps, epsilon):
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError(('positive integral n_steps required', n_steps))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(('positive finite epsilon required', epsilon))
    smear.n_steps = int(n_steps)
    smear.epsilon = float(epsilon)
    smear.meas_interval = int(n_steps) + 1
    validate_initial_only(smear)


def validate_initial_only(smear):
    if smear.n_steps < 1 or smear.meas_interval <= smear.n_steps:
        raise RuntimeError(('single-observable QUDA binding would index out of bounds',
                            smear.n_steps, smear.meas_interval))
    if not math.isfinite(smear.epsilon) or smear.epsilon <= 0:
        raise RuntimeError(('invalid flow step', smear.epsilon))

