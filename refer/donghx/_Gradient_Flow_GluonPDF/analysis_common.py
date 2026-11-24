"""Immutable B-scheme ratio/plateau diagnostic contract, not a bare-state fit."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2'
RAW20 = ROOT / 'quark_l32x64_B_insertions_20conf_8flow_20261017'
NEW = ROOT / 'quark_l32x64_B_expand50_P456_20261026'
CONFS = tuple(map(str, range(10000, 12000, 40)))
FLOWS = (.1, .2, .3, .4, .5, .6, .7, .8)
WINDOWS = (4, 5, 5, 6, 6, 7, 8, 8)
TSEPS = (7, 8, 9, 10, 11, 12)
MOMENTA = (4, -4, 5, -5, 6, -6)
PZ = (4, 5, 6)
CURRENTS = (0, 16, 32, 63)
CHANNELS = ('unpolarized', 'helicity', 'transversity')
FLAVORS = ('u', 'd', 'u_minus_d')
A_FM, NCONF, NBOOT, SEED, DZ = .0897, 50, 200, 1115, 21
SCHEME = 'same_time_second_cross_time_independent_product'
ESTIMATOR = 'four_current_mean_shared_configuration_bootstrap_directional_complex_ratio_then_zpair_vdv_v1'
TRANS_STATUS = 'pending_external_physics_confirmation'


def allocation():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical data processing is restricted to a CPU allocation')


def tag(tau):
    return f'tau{tau:.2f}'.replace('.', 'p')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **payload):
    import numpy as np
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    os.replace(tmp, path)


def code_check():
    lock = json.loads((HERE / 'code_lock.json').read_text())
    for key, digest in lock['files_sha256'].items():
        if sha(HERE / key) != digest:
            raise RuntimeError(('frozen analysis source changed', key))
    for path, digest in lock['upstream_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError(('frozen upstream source/receipt changed', path))
    return lock


def indices():
    import numpy as np
    gate = json.loads((HERE / 'status/gate.json').read_text())
    assert gate['status'] == 'PASS'
    path = HERE / 'bootstrap_indices_Nboot200_Nconf50_seed1115.npy'
    assert sha(path) == gate['bootstrap_indices_sha256']
    value = np.load(path, allow_pickle=False)
    assert value.shape == (NBOOT, NCONF) and value.min() >= 0 and value.max() < NCONF
    return value


def production():
    sys.path.insert(0, str(NEW / 'reference'))
    spec = importlib.util.spec_from_file_location('frozen_B50_production', NEW / 'common.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_manifest()
    return module


def ratio_path(tau, channel, flavor):
    return HERE / 'ratios' / f'ratio_{tag(tau)}_{channel}_{flavor}_Nboot200_seed1115.npz'


def metadata(tau):
    import numpy as np
    return dict(confs=np.asarray(CONFS), nconf=np.asarray(NCONF), nboot=np.asarray(NBOOT),
        seed=np.asarray(SEED), resample_size=np.asarray(NCONF), boot_indices=indices(),
        bootstrap_indices_sha256=np.asarray(sha(HERE / 'bootstrap_indices_Nboot200_Nconf50_seed1115.npy')),
        tau_a2=np.asarray(tau), a_fm=np.asarray(A_FM), epsilon=np.asarray(.01),
        nstep=np.asarray(round(tau / .01)), temporal_window=np.asarray(WINDOWS[FLOWS.index(tau)]),
        pz_values=np.asarray(PZ), signed_pz_values=np.asarray(MOMENTA),
        delta_z_values=np.arange(DZ), tseps=np.asarray(TSEPS), current_anchors=np.asarray(CURRENTS),
        weight_scheme=np.asarray(SCHEME), weight_applied_count=np.asarray(1), estimator_id=np.asarray(ESTIMATOR),
        c2_mode=np.asarray('complex_nopol_dt_matched'),
        sampling_scope=np.asarray('four fixed current anchors; source=(current-dt)%64; not all64 sources'),
        error_definition=np.asarray('bootstrap std(ddof=1), separately Re-even and Im-odd; no sqrt(Nboot) division'),
        transversity_convention_status=np.asarray(TRANS_STATUS),
        physical_result_status=np.asarray('finite_flow_connected_bare_ratio_diagnostic; ground_state_not_established'),
        code_lock_sha256=np.asarray(sha(HERE / 'code_lock.json')))


def project_spin(c3):
    import numpy as np
    from reference.conventions import make_channel_specs, PHASE_FACTORS
    nm = c3.shape[3]
    if c3.shape != (2, 2, 4, nm, DZ, 4, 4) or nm not in (6, 8):
        raise ValueError(('invalid raw C3 axes', c3.shape))
    moms = MOMENTA if nm == 6 else (3, -3, *MOMENTA)
    specs = make_channel_specs([f'P{"p" if p > 0 else "m"}{abs(p)}_z' for p in moms],
                              ['z'] * nm, ['nopol', 'pol15', 'pol25', 'pol35'])
    result = np.empty((3, 2, 2, nm, DZ), complex)
    for ip in range(nm):
        v = [np.einsum('fszmo,mo->fsz', c3[:, :, int(specs.pol_indices[ip, ch]), ip],
                       specs.gamma[ip, ch], optimize=True) for ch in range(4)]
        result[0, :, :, ip] = v[0]
        result[1, :, :, ip] = PHASE_FACTORS['helicity'] * v[1]
        result[2, :, :, ip] = .5 * PHASE_FACTORS['transversity'] * (v[2] + v[3])
    return result


def compute_ratios(c3, c2, ind):
    """Average configs in each shared bootstrap before complex division; preserve dt."""
    import numpy as np
    assert c3.shape == (NCONF, 3, 2, 2, 6, DZ, 6, 13)
    assert c2.shape == (NCONF, 6, 6, 13)
    physical = np.arange(13)[None, :] <= np.asarray(TSEPS)[:, None]
    assert np.isfinite(c3[..., physical]).all() and np.isfinite(c2[..., physical]).all()
    assert np.isnan(c3[..., ~physical]).all() and np.isnan(c2[..., ~physical]).all()
    weights = np.stack([np.bincount(row, minlength=NCONF) / NCONF for row in ind])
    full = np.concatenate((c3, (c3[:, :, 0] - c3[:, :, 1])[:, :, None]), axis=2)
    num, den = full.mean(axis=0), c2.mean(axis=0)
    nums, dens = np.tensordot(weights, full, axes=(1, 0)), np.tensordot(weights, c2, axes=(1, 0))
    assert np.isfinite(den[..., physical]).all() and np.isfinite(dens[..., physical]).all()
    if np.any(den[..., physical] == 0) or np.any(dens[..., physical] == 0):
        raise ValueError('Exactly zero complex C2; no clipping, replica selection or absolute value allowed')
    with np.errstate(invalid='ignore', divide='raise'):
        raw = num / den[None, None, None, :, None, :, :]
        rawb = nums / dens[:, None, None, None, :, None, :, :]
    # Direct algebraic combination, sign is +/-z and even/odd negative-P parity is +/-1.
    ratio = np.stack([.25 * (raw[:, :, 0, 0::2] + s * raw[:, :, 1, 0::2]
                              + s * raw[:, :, 0, 1::2] + raw[:, :, 1, 1::2])
                      for s in (1, -1)], axis=2)
    boot = np.stack([.25 * (rawb[:, :, :, 0, 0::2] + s * rawb[:, :, :, 1, 0::2]
                             + s * rawb[:, :, :, 0, 1::2] + rawb[:, :, :, 1, 1::2])
                     for s in (1, -1)], axis=3)
    assert ratio.shape == (3, 3, 2, 3, DZ, 6, 13)
    assert boot.shape == (NBOOT,) + ratio.shape
    np.testing.assert_allclose(boot[:, :, 2], boot[:, :, 0] - boot[:, :, 1],
                               rtol=3e-10, atol=1e-10, equal_nan=True)
    return ratio, boot, den, dens
