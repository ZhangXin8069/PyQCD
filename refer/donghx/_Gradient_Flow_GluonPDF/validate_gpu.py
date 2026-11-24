"""Single-GPU Landau validation; all numerical work requires gpu-debug."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def identity(path):
    p = Path(path)
    s = p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                inode=s.st_ino, ctime_ns=s.st_ctime_ns)


def dagger(a):
    return a.swapaxes(-1, -2).conj()


def translated(a, offsets):
    # a[t,z,y,x,color,color], physical direction mu maps to axis 3-mu.
    for mu, displacement in enumerate(offsets):
        if displacement:
            a = cp.roll(a, -displacement, axis=3-mu)
    return a


def path_matrix(u, steps):
    result = cp.broadcast_to(cp.eye(3, dtype=cp.complex128), u.shape[1:]).copy()
    offset = [0, 0, 0, 0]
    for signed_mu in steps:
        mu = abs(signed_mu)-1
        if signed_mu < 0:
            offset[mu] -= 1
            link = dagger(translated(u[mu], offset))
        else:
            link = translated(u[mu], offset)
            offset[mu] += 1
        result = result @ link
    return result


LOOPS = [(mu, nu, 1, 1) for mu in range(4) for nu in range(mu+1, 4)]
LOOPS += [(mu, 3, 2, 2) for mu in range(3)]


def closed_loops(u):
    values = []
    for mu, nu, r, t in LOOPS:
        p = [mu+1]*r + [nu+1]*t + [-(mu+1)]*r + [-(nu+1)]*t
        tr = cp.trace(path_matrix(u, p), axis1=-2, axis2=-1)
        values.append([float(tr.real.mean()/3), float(((cp.abs(tr)**2-1)/8).mean())])
    return np.asarray(values)


def group_errors(u):
    eye = cp.eye(3, dtype=cp.complex128)
    unitarity, determinant = 0., 0.
    for mu in range(4):
        unitarity = max(unitarity, float(cp.max(cp.abs(u[mu] @ dagger(u[mu])-eye))))
        a = u[mu]
        det = (a[..., 0, 0]*(a[..., 1, 1]*a[..., 2, 2]-a[..., 1, 2]*a[..., 2, 1])
               -a[..., 0, 1]*(a[..., 1, 0]*a[..., 2, 2]-a[..., 1, 2]*a[..., 2, 0])
               +a[..., 0, 2]*(a[..., 1, 0]*a[..., 2, 1]-a[..., 1, 1]*a[..., 2, 0]))
        determinant = max(determinant, float(cp.max(cp.abs(det-1))))
    return dict(unitarity_max=unitarity, determinant_max=determinant)


def quality(u):
    # Matches QUDA: Delta = sum_mu[U_mu(x-mu)-U_mu(x)] - h.c., traceless.
    delta = cp.zeros(u.shape[1:], dtype=cp.complex128)
    action = 0.
    for mu in range(4):
        delta += cp.roll(u[mu], 1, axis=3-mu)-u[mu]
        action += float(cp.trace(u[mu], axis1=-2, axis2=-1).real.mean())/12
    delta -= dagger(delta)
    tr = cp.trace(delta, axis1=-2, axis2=-1)/3
    for color in range(3):
        delta[..., color, color] -= tr
    local = cp.sum(cp.abs(delta)**2, axis=(-2, -1))/3
    return dict(theta=float(local.mean()), theta_local_max=float(local.max()),
                gauge_functional=action)


def lines(u, nmax):
    fund = np.empty((3, 2, nmax+1), np.complex128)
    adj = np.empty((3, 2, nmax+1), np.float64)
    for mu in range(3):
        for si, sign in enumerate([1, -1]):
            product = cp.broadcast_to(cp.eye(3, dtype=cp.complex128), u.shape[1:]).copy()
            fund[mu, si, 0] = adj[mu, si, 0] = 1
            for n in range(1, nmax+1):
                if sign == 1:
                    link = cp.roll(u[mu], -(n-1), axis=3-mu)
                else:
                    link = dagger(cp.roll(u[mu], n, axis=3-mu))
                product = product @ link
                tr = cp.trace(product, axis1=-2, axis2=-1)
                fund[mu, si, n] = complex(tr.mean()/3)
                adj[mu, si, n] = float(((cp.abs(tr)**2-1)/8).mean())
    reversal = max(float(np.max(abs(fund[:, 1]-fund[:, 0].conj()))),
                   float(np.max(abs(adj[:, 1]-adj[:, 0]))))
    assert np.isfinite(fund).all() and np.isfinite(adj).all()
    assert reversal < CONTRACT['line_reversal_atol'], reversal
    return fund, adj, reversal


def random_su3(shape, seed):
    # Product of three normalized random SU(2) embeddings; reproducible, not Haar SU(3).
    rng = cp.random.RandomState(seed)
    g = cp.broadcast_to(cp.eye(3, dtype=cp.complex128), (*shape, 3, 3)).copy()
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        q = rng.standard_normal((*shape, 4), dtype=cp.float64)
        q /= cp.sqrt(cp.sum(q*q, axis=-1))[..., None]
        alpha, beta = q[..., 0]+1j*q[..., 1], q[..., 2]+1j*q[..., 3]
        col_i, col_j = g[..., :, i].copy(), g[..., :, j].copy()
        g[..., :, i] = col_i*alpha[..., None]-col_j*beta.conj()[..., None]
        g[..., :, j] = col_i*beta[..., None]+col_j*alpha.conj()[..., None]
    return g


def transform(u, g):
    result = cp.empty_like(u)
    for mu in range(4):
        result[mu] = g @ u[mu] @ dagger(cp.roll(g, -1, axis=3-mu))
    return result


def prepare(gauge):
    gauge.toHost()
    dr = gauge.gauge_dirac
    dr.setPrecision(cuda=QudaPrecision.QUDA_DOUBLE_PRECISION)
    dr.setReconstruct(cuda=QudaReconstructType.QUDA_RECONSTRUCT_NO)
    dr.gauge_param.make_resident_gauge = 0
    assert gauge.data.dtype == np.dtype('complex128')
    return gauge


def to_lex(gauge):
    return cp.asarray(gauge.latt_info.lexico(gauge.data, True))


def from_lex(u, li):
    return prepare(LatticeGauge(li, li.evenodd(cp.asnumpy(u), True)))


def fix(gauge, method, tolerance, max_steps):
    prepare(gauge)
    started = time.monotonic()
    print('FIX_START', method, tolerance, max_steps, flush=True)
    if method == 'OVR':
        gauge.fixingOVR(4, max_steps, 100, 1.7, tolerance, 10, 1)
    else:
        gauge.fixingFFT(4, max_steps, 100, .08, 1, tolerance, 1)
    cp.cuda.get_current_stream().synchronize()
    print('FIX_RETURN', method, time.monotonic()-started, flush=True)
    return time.monotonic()-started


def tiny_contract():
    li = LatticeInfo([4, 4, 4, 4])
    unit = cp.broadcast_to(cp.eye(3, dtype=cp.complex128), (4, 4, 4, 4, 4, 3, 3)).copy()
    assert quality(unit)['theta'] == 0
    assert np.max(abs(closed_loops(unit)-1)) < 1e-14
    f, a, _ = lines(unit, 5)
    assert np.max(abs(f-1)) < 1e-14 and np.max(abs(a-1)) < 1e-14
    g = random_su3((4, 4, 4, 4), 11011)
    pure = transform(unit, g)
    assert np.max(abs(closed_loops(pure)-1)) < 2e-13
    expected = g @ dagger(cp.roll(g, -3, axis=3))
    assert float(cp.max(abs(path_matrix(pure, [1, 1, 1])-expected))) < 2e-13
    out = dict(unit_and_pure_gauge_paths='PASS', algorithms={})
    for method in ['OVR', 'FFT']:
        field = from_lex(pure, li)
        seconds = fix(field, method, 1e-12, 4000)
        u = to_lex(field)
        q = quality(u)
        err = float(np.max(abs(closed_loops(u)-1)))
        assert q['theta'] < 1.1e-12 and err < CONTRACT['closed_loop_atol'], (method, q, err)
        out['algorithms'][method] = dict(**q, loop_error=err, seconds=seconds)
        del field, u
    return out


def measure(gauge, output, label, reference=None):
    u = to_lex(gauge)
    q = quality(u)
    q.update(group_errors(u))
    closed = closed_loops(u)
    fund, adj, reversal = lines(u, CONTRACT['line_nmax'])
    plaq = np.asarray(gauge.plaquette(), dtype=float)
    independent_plaq = np.mean(closed[:6, 0])
    assert abs(plaq[0]-independent_plaq) < CONTRACT['closed_loop_atol'], (plaq, independent_plaq)
    assert max(q['unitarity_max'], q['determinant_max']) < CONTRACT['su3_atol'], q
    q.update(line_reversal_error=reversal, quda_plaquette=plaq.tolist(), label=label)
    if reference is not None:
        q['closed_loop_max_change'] = float(np.max(abs(closed-reference)))
        assert q['closed_loop_max_change'] < CONTRACT['closed_loop_atol'], q
    target = output/(label+'.npz')
    np.savez_compressed(target, fundamental=fund, adjoint=adj, closed_loops=closed,
                        directions=np.array(['x', 'y', 'z']), signs=np.array([1, -1]),
                        lengths_a=np.arange(CONTRACT['line_nmax']+1), loop_geometry=np.array(LOOPS))
    with np.load(target, allow_pickle=False) as saved:
        assert np.array_equal(saved['fundamental'], fund)
    q['npz_sha256'] = sha(target)
    write_json(output/(label+'.json'), q)
    print('MEASURE', label, json.dumps(q), flush=True)
    del u
    cp.get_default_memory_pool().free_all_blocks()
    return q, closed, fund


def run(args, output):
    global cp, np, LatticeInfo, LatticeGauge, QudaPrecision, QudaReconstructType, CONTRACT
    import numpy as np
    import cupy as cp
    import pyquda
    from pyquda.field import LatticeInfo, LatticeGauge
    from pyquda.enum_quda import QudaPrecision, QudaReconstructType
    from pyquda_utils.io import readILDGGauge
    request = json.loads((ROOT/'request.json').read_text())
    CONTRACT = request['contract']
    item = request['ensembles'][args.task]
    assert identity(item['gauge']) == item['identity']
    assert cp.cuda.runtime.getDeviceCount() == 1
    pyquda.init(grid_size=[1, 1, 1, 1], backend='cupy', resource_path=os.environ['QUDA_RESOURCE_PATH'])
    assert pyquda.getMPISize() == 1
    write_json(output/'environment.json', dict(job=os.environ['SLURM_JOB_ID'],
        partition=os.environ.get('SLURM_JOB_PARTITION'), host=socket.gethostname(),
        pyquda_version=pyquda.__version__, pyquda_path=pyquda.__file__, cupy_version=cp.__version__,
        numpy_version=np.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True),
        source_lock_sha256=sha(ROOT/'source_lock.json'), request=item))
    write_json(output/'tiny_contract.json', tiny_contract())
    cp.get_default_memory_pool().free_all_blocks()
    print('READ_GAUGE', item['gauge'], flush=True)
    raw = prepare(readILDGGauge(item['gauge'], checksum=True))
    assert list(raw.latt_info.global_size) == item['lattice_xyzt']
    assert identity(item['gauge']) == item['identity']
    raw_hash = hashlib.sha256(memoryview(raw.data).cast('B')).hexdigest()
    flow = prepare(raw.copy())
    raw_quality, _, _ = measure(raw, output, 'raw')
    records = []
    tau_previous = 0.
    for flow_index, tau in enumerate(item['taus']):
        nstep = round((tau-tau_previous)/item['epsilon'])
        epsilon = (tau-tau_previous)/nstep
        print('FLOW', tau_previous, tau, nstep, epsilon, flush=True)
        flow.wilsonFlow(nstep, epsilon, compute_plaquette=False, compute_qcharge=False)
        flow_hash = hashlib.sha256(memoryview(flow.data).cast('B')).hexdigest()
        tag = f'flow{flow_index}'
        before, closed, _ = measure(flow, output, tag+'_before')
        gauge = prepare(flow.copy())
        elapsed = fix(gauge, 'OVR', CONTRACT['theta_primary'], CONTRACT['max_steps'])
        after, _, f_primary = measure(gauge, output, tag+'_ovr', closed)
        assert after['theta'] <= CONTRACT['theta_primary']*1.1, after
        assert after['gauge_functional'] >= before['gauge_functional']-1e-12
        record = dict(tau=tau, rf_fm=item['a_fm']*(8*tau)**.5,
                      before=before, after=after, fixing_seconds=elapsed)
        if flow_index == len(item['taus'])-1:
            fix(gauge, 'OVR', CONTRACT['theta_tight'], CONTRACT['refinement_steps'])
            tighter, _, f_tight = measure(gauge, output, tag+'_ovr_tight', closed)
            assert tighter['theta'] <= CONTRACT['theta_tight']*1.1, tighter
            record['tighter_tolerance'] = tighter
            record['tight_minus_primary_line_max_abs'] = float(np.max(abs(f_tight-f_primary)))
            u = to_lex(flow)
            g = random_su3(tuple(u.shape[1:5]), item['random_seed'])
            randomized = transform(u, g)
            transform_error = float(np.max(abs(closed_loops(randomized)-closed)))
            assert transform_error < CONTRACT['closed_loop_atol'], transform_error
            alternate = from_lex(randomized, flow.latt_info)
            del u, g, randomized
            cp.get_default_memory_pool().free_all_blocks()
            fix(alternate, 'OVR', CONTRACT['theta_primary'], CONTRACT['max_steps'])
            alternate_result, _, f_alt = measure(alternate, output, tag+'_random_start_ovr', closed)
            assert alternate_result['theta'] <= CONTRACT['theta_primary']*1.1, alternate_result
            record['random_start'] = alternate_result
            record['random_gauge_covariance_closed_error'] = transform_error
            record['random_start_minus_primary_line_max_abs'] = float(np.max(abs(f_alt-f_primary)))
            record['copy_sensitivity_interpretation'] = 'diagnostic_only_no_equality_gate_not_a_complete_Gribov_study'
            del alternate
        assert hashlib.sha256(memoryview(flow.data).cast('B')).hexdigest() == flow_hash
        records.append(record)
        write_json(output/'progress.json', dict(status='RUNNING', completed_flows=records))
        tau_previous = tau
        del gauge
    assert hashlib.sha256(memoryview(raw.data).cast('B')).hexdigest() == raw_hash
    assert identity(item['gauge']) == item['identity']
    cp.cuda.get_current_stream().synchronize()
    write_json(output/'complete.json', dict(status='PASS_GPU_NUMERICAL_VALIDATION',
        ensemble=item['name'], conf_id=item['conf_id'], job=os.environ['SLURM_JOB_ID'],
        raw=raw_quality, flow_results=records, input_unchanged=True,
        configuration_count=1, bootstrap_performed=False, d_fitted=False,
        final_d_F=None, final_d_A=None, source_lock_sha256=sha(ROOT/'source_lock.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gpu-debug':
        raise RuntimeError('Numerical validation requires a gpu-debug allocation')
    lock = json.loads((ROOT/'source_lock.json').read_text())
    for name, digest in lock['sha256'].items():
        assert sha(ROOT/name) == digest, name
    output = ROOT/'runs'/f"task{args.task}_job{os.environ['SLURM_JOB_ID']}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'started.json', dict(job=os.environ['SLURM_JOB_ID'], task=args.task,
                                           start_unix=time.time(), host=socket.gethostname()))
    try:
        run(args, output)
    except BaseException as error:
        write_json(output/'failed.json', dict(status='FAILED', error=repr(error), traceback=traceback.format_exc()))
        raise
