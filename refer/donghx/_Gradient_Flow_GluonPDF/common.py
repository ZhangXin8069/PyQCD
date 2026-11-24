"""Frozen paired-configuration bootstrap200 contracts for the two B ensembles."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
KEY = os.environ.get('GF_ANALYSIS_ENSEMBLE')
if KEY not in ('L32x64', 'L32x96'):
    raise RuntimeError('Set GF_ANALYSIS_ENSEMBLE=L32x64 or L32x96 explicitly')
HERE = ROOT / KEY
sys.path.insert(0, str(ROOT / 'reference'))
PARENT64 = WORK / 'quark_l32x64_B_ratio50_boot200_plateau_20261030'
PARENT96 = WORK / 'quark_l32x96_B_ratio20_boot100_T8_rfmatched_20261029'
OLD64 = WORK / 'quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2'
P064 = WORK / 'quark_l32x64_B_Pz0_fill50_8flow_20261029'
RAW64 = WORK / 'quark_l32x64_B_expand50_P456_20261026'
RAW96 = WORK / 'quark_l32x96_B_matched_radius20_7flow_retry_v2_20261021'
SCHEME = 'same_time_second_cross_time_independent_product'
ESTIMATOR = 'four_current_mean_shared_configuration_bootstrap_directional_complex_ratio_then_zpair_v1'
TRANS_STATUS = 'pending_external_physics_confirmation'
CHANNELS = ('unpolarized', 'helicity', 'transversity')
FLAVORS = ('u', 'd', 'u_minus_d')
COMPONENTS = ('even_real', 'odd_imag')
POLS = ('nopol', 'pol15', 'pol25', 'pol35')
NBOOT, SEED, DZ = 200, 1115, 21
TSEPS = tuple(range(7, 13))
if KEY == 'L32x64':
    ENSEMBLE = 'beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64'
    NT, A_FM, NCONF, EPSILON = 64, .0897, 50, .01
    CONFS = tuple(map(str, range(10000, 12000, 40)))
    FLOWS64 = (.1, .2, .3, .4, .5, .6, .7, .8)
    FLOWS = tuple(zip(FLOWS64, (4, 5, 5, 6, 6, 7, 8, 8)))
    PZ, MOMENTA, CURRENTS = (0, 4, 5, 6), (0, 4, -4, 5, -5, 6, -6), (0, 16, 32, 63)
else:
    ENSEMBLE = 'beta6.41_mu-0.2295_ms-0.2050_L32x96'
    NT, A_FM, NCONF = 96, .0775, 20
    CONFS = tuple(map(str, tuple(range(10000, 11200, 100)) + tuple(range(12300, 13100, 100))))
    EPSILON = .005 * (.0897 / A_FM)**2
    FLOWS64 = (.2, .3, .4, .5, .6, .7, .8)
    FLOWS = tuple((round(t / .005) * EPSILON, w)
                  for t, w in zip(FLOWS64, (6, 6, 7, 7, 9, 10, 10)))
    PZ, MOMENTA, CURRENTS = (0, 3, 4, 5, 6), (0, 3, -3, 4, -4, 5, -5, 6, -6), (0, 24, 48, 95)
TMIN_FM_THRESHOLD = .6
TMIN = int(np.floor(TMIN_FM_THRESHOLD / A_FM)) + 1
Q_MIN, CHI_MAX, CUTS = 1e-3, 1e3, (1, 2, 3)
TMAXS = tuple(range(TMIN + 1, 13))
CANDIDATES = tuple((f'T{TMIN}-{top}_L{cut}_R{cut}', tuple(range(TMIN, top + 1)), cut)
                   for cut in CUTS for top in TMAXS)


def allocation():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All payload reads, bootstrap, fits, plots and numerical checks require CPU allocation')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    os.replace(tmp, path)


def tau_tag(tau):
    return f'tau{tau:.8f}'.replace('.', 'p')


def parent_tag(tau):
    return f'tau{tau:.2f}'.replace('.', 'p') if KEY == 'L32x64' else tau_tag(tau)


def ratio_path(tau, channel, flavor):
    return HERE / 'ratios' / f'ratio_{tau_tag(tau)}_{channel}_{flavor}_Nboot200_seed1115.npz'


def fit_path(tau, channel, flavor):
    return HERE / 'fits' / f'bare_ratio_AIC_{tau_tag(tau)}_{channel}_{flavor}.npz'


def index_path():
    return HERE / f'bootstrap_indices_Nboot200_Nconf{NCONF}_seed1115.npy'


@lru_cache(maxsize=1)
def code_check():
    lock = json.loads((ROOT / 'code_lock.json').read_text())
    for rel, digest in lock['files_sha256'].items():
        if sha256(ROOT / rel) != digest:
            raise RuntimeError(('frozen analysis source changed', rel))
    for path, digest in lock['parent_metadata_sha256'].items():
        if sha256(path) != digest:
            raise RuntimeError(('frozen parent provenance changed', path))
    return lock


@lru_cache(maxsize=1)
def manifest():
    code_check()
    m = json.loads((HERE / 'input_manifest.json').read_text())
    assert m['confs'] == list(CONFS) and m['nboot'] == NBOOT and m['Tmin'] == TMIN
    assert m['flows'] == [t for t, _ in FLOWS] and m['weight_scheme'] == SCHEME
    return m


@lru_cache(maxsize=1)
def shared_indices():
    m = manifest(); path = index_path()
    assert sha256(path) == m['bootstrap_indices_sha256']
    a = np.load(path, allow_pickle=False)
    assert a.shape == (NBOOT, NCONF) and a.dtype == np.int32
    assert ((a >= 0) & (a < NCONF)).all()
    return a


def common_metadata(tau):
    i = [t for t, _ in FLOWS].index(tau)
    radius = float(A_FM * np.sqrt(8 * tau))
    return dict(ensemble=np.asarray(ENSEMBLE), ensemble_key=np.asarray(KEY), confs=np.asarray(CONFS),
        nconf=NCONF, nboot=NBOOT, seed=SEED, resample_size=NCONF, boot_indices=shared_indices(),
        bootstrap_indices_sha256=np.asarray(manifest()['bootstrap_indices_sha256']),
        tau_a2=tau, tau64_a2=FLOWS64[i], a_fm=A_FM, tf_fm2=tau*A_FM**2, rf_fm=radius,
        rf_window_selected=.0897 < radius < .3, rf_lower_fm=.0897, rf_upper_fm=.3,
        epsilon=EPSILON, nstep=round(tau/EPSILON), temporal_window=FLOWS[i][1],
        pz_values=np.asarray(PZ), signed_pz_values=np.asarray(MOMENTA), delta_z_values=np.arange(DZ),
        tseps=np.asarray(TSEPS), current_anchors=np.asarray(CURRENTS),
        weight_scheme=np.asarray(SCHEME), weight_applied_count=1, estimator_id=np.asarray(ESTIMATOR),
        c2_mode=np.asarray('complex_nopol_dt_matched'),
        momentum_average_stage=np.asarray('after complex signed-P division and z pairing; negative-P odd times -1; P0 once'),
        sampling_scope=np.asarray(f'four fixed current anchors; source=(current-dt)%{NT}; z axis only'),
        transversity_convention_status=np.asarray(TRANS_STATUS),
        input_manifest_sha256=np.asarray(sha256(HERE/'input_manifest.json')),
        code_lock_sha256=np.asarray(sha256(ROOT/'code_lock.json')))


def load_isolated(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
