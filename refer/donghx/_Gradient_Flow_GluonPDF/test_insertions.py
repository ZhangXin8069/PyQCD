#!/usr/bin/env python3
"""New dt/APBC/streamed Wick checks, run exclusively on CPU nodes."""
import json
import os
import numpy as np
from common import HERE, CURRENTS, TSEPS, SCHEME, case_times, atomic_json
from insertion_kernels import one_right_time_h
from flow_common_l32x64 import spin_objects, seq_peram
from spin2_contractions import Contractor, c3_reduced, SPECS
from weights_and_terms import iter_reduced_right, current_block_h, signed_block_term
from flowed_perambulator_factorization import build_flowed_sequential


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('CPU allocation required')
    keys = set()
    wraps = set()
    for tj in CURRENTS:
        for sep in TSEPS:
            for dt in range(sep+1):
                src, sink, saved_dt, boundary = case_times(tj, sep, dt)
                assert (src+dt) % 64 == tj and (src+sep) % 64 == sink
                assert saved_dt == dt and boundary == (-1 if src+sep >= 64 else 1)
                key = (tj, sep, dt)
                assert key not in keys
                keys.add(key)
                wraps.add(boundary)
    assert len(keys) == 252 and wraps == {-1, 1}
    assert case_times(0, 7, 0) == (0, 7, 0, 1)
    assert case_times(0, 7, 7) == (57, 0, 7, -1)
    assert case_times(63, 12, 0) == (63, 11, 0, -1)
    rng = np.random.default_rng(20261017)
    def z(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    contract = Contractor()
    g, g5, q = spin_objects(np)
    p = z((4,4,3,3))
    a = z((3,4,4,3,4))
    b = z((3,4,4,4,3))
    j = z((3,3,2,3,4,4))
    vs, vr = z((2,3,3,3)), z((2,3,3,3))
    full = build_flowed_sequential(j, a, contract)
    streamed = np.zeros((6,2,4,2,3,4,4),complex)
    ordered = np.zeros_like(streamed)
    for ib in range(3):
        h = one_right_time_h(j,a,ib,contract)
        np.testing.assert_allclose(h,full[ib:ib+1],rtol=3e-14,atol=1e-12)
        uu, dd, terms = c3_reduced(h,p,b[ib:ib+1],g,q,vs,vr,contract,return_terms=True)
        for it,spec in enumerate(SPECS):
            streamed[it] += spec[2]*terms[spec[0]]
        hh = current_block_h(j[:,ib],a,contract)
        for it,(spec,right,q2,basis) in enumerate(iter_reduced_right(
                p,b[ib:ib+1],g,q,vs,vr,contract,np)):
            ordered[it] += signed_block_term(hh,spec,right,q2,basis,contract).sum(axis=0)
    np.testing.assert_allclose(streamed,ordered,rtol=5e-12,atol=2e-10)
    # Sequential reversal uses the explicitly checked gamma5-adjoint ordering.
    original=z((3,4,4,4,3))
    seq=seq_peram(original,np)
    gold=np.empty_like(seq)
    for t in range(3):
        for a_spin in range(4):
            for b_spin in range(4):
                for src_mode in range(3):
                    for dst_mode in range(4):
                        gold[t,a_spin,b_spin,src_mode,dst_mode]=sum(
                            g5[x,b_spin]*original[t,x,y,dst_mode,src_mode].conjugate()*g5[a_spin,y]
                            for x in range(4) for y in range(4))
    np.testing.assert_allclose(seq,gold,rtol=1e-14,atol=1e-14)
    atomic_json(HERE/'status/insertion_tests.json',dict(status='PASS',seed=20261017,
        case_count_per_conf_flow=252,weight_scheme=SCHEME,
        checks=['all dt=0..tsep unique and periodic source/current/sink identity',
                'APBC baryon wrap at both temporal boundaries and contact endpoints',
                'one-sigma_b H vs full factorization; all six signed Wick vs ordered sum',
                'gamma5 sequential spin/color-mode adjoint explicit-loop oracle'],
        max_streamed_ordered_abs=float(np.max(abs(streamed-ordered)))))
    print('INSERTION_ALGEBRA_PASS',flush=True)


if __name__ == '__main__':
    main()
