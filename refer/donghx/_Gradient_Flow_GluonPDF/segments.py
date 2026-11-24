"""Segment engine derived from frozen v2; strict configurable residual gate."""
import hashlib
import os
from pathlib import Path
import time
import json
import core as c
ROOT=Path(__file__).resolve().parent

def data_hash(gauge):
    return hashlib.sha256(memoryview(gauge.data).cast('B')).hexdigest()

def residual(gauge):
    u = c.to_lex(gauge)
    value = c.quality(u)
    del u
    c.cp.get_default_memory_pool().free_all_blocks()
    if not all(c.np.isfinite(v) for v in value.values()):
        raise FloatingPointError(value)
    return value

def save_checkpoint(gauge, out, phase, history, stop_reason):
    path = out/'last_gauge.npy'
    tmp = out/'last_gauge.npy.tmp'
    with tmp.open('wb') as f:
        c.np.save(f, gauge.data, allow_pickle=False)
    os.replace(tmp, path)
    # Verify bytes and metadata without reading a second full array into memory.
    view = c.np.load(path, mmap_mode='r', allow_pickle=False)
    assert view.shape == gauge.data.shape and view.dtype == gauge.data.dtype
    assert hashlib.sha256(memoryview(view).cast('B')).hexdigest() == data_hash(gauge)
    c.write_json(out/'checkpoint.json', dict(phase=phase, reason=stop_reason,
        last=history[-1], file=path.name, sha256=c.sha(path), data_sha256=data_hash(gauge),
        shape=list(gauge.data.shape), dtype=str(gauge.data.dtype),
        source_lock_sha256=c.sha(ROOT/'source_lock.json')))

def phase(gauge, method, target, max_steps, out, name, deadline):
    contract = c.CONTRACT
    q = residual(gauge)
    history = [dict(iterations=0, kernel_seconds=0., elapsed_seconds=0., **q)]
    started = time.monotonic()
    total, seconds = 0, 0.
    per_step = None
    reason = 'MAX_ITERATIONS'
    while q['theta'] > target*contract['theta_acceptance_factor'] and total < max_steps:
        remaining_time = deadline-time.monotonic()
        if remaining_time < 90:
            reason = 'WALLTIME_BUDGET'
            break
        limit = contract['ovr_chunk_steps'] if method == 'OVR' else contract['fft_chunk_steps']
        chunk = min(limit, max_steps-total)
        if method == 'FFT' and per_step is None:
            chunk = min(chunk, contract['fft_first_chunk_steps'])
        if per_step:
            chunk = min(chunk, max(10, int(min(contract['estimated_chunk_budget_seconds'],
                                            remaining_time-60)/per_step)//10*10))
        c.prepare(gauge)
        print('SEGMENT_START', name, method, total, chunk, target, flush=True)
        t0 = time.monotonic()
        # tolerance=0 runs exactly chunk iterations. Convergence is checked independently
        # between segments, so QUDA's early-stop/iteration-report semantics are irrelevant.
        if method == 'OVR':
            gauge.fixingOVR(4, chunk, min(100, chunk), 1.7, 0., 10, 1)
        else:
            gauge.fixingFFT(4, chunk, min(100, chunk), contract['fft_alpha_reset_per_chunk'], contract['fft_autotune'], 0., 1)
        c.cp.cuda.get_current_stream().synchronize()
        dt = time.monotonic()-t0
        per_step = dt/chunk
        seconds += dt
        total += chunk
        q = residual(gauge)
        history.append(dict(iterations=total, chunk_steps=chunk, chunk_seconds=dt,
                            kernel_seconds=seconds, elapsed_seconds=time.monotonic()-started, **q))
        c.write_json(out/(name+'_convergence.json'), dict(target=target, method=method,
            max_steps=max_steps, history=history, converged=q['theta'] <= target*contract['theta_acceptance_factor'],
            count_definition='exact fixed QUDA iterations; independent checks between chunks',
            fft_alpha_policy='fixed alpha=.04 with autotune=0 at every FFT segment'))
        print('SEGMENT_END', name, json.dumps(history[-1]), flush=True)
    passed = q['theta'] <= target*contract['theta_acceptance_factor']
    reason = 'CONVERGED' if passed else reason
    result = dict(status=reason, converged=passed, theta_target=target,
                  iterations=total, kernel_seconds=seconds, final=q, history=history)
    c.write_json(out/(name+'_phase.json'), result)
    if not passed:
        save_checkpoint(gauge, out, name, history, reason)
    return result

